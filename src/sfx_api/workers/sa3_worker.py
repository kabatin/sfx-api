"""Stable Audio 3 (MLX) engine worker. Runs inside the Stable Audio 3 checkout's own Python
(``optimized/mlx/.venv``), never inside sfx-api's environment: it imports only the standard
library, numpy, ``mlx`` and the checkout's own modules (``optimized/mlx`` and its ``scripts/``).

Protocol: see :mod:`sfx_api.workers.sfx_worker_protocol`. Requests carry ``prompt``,
``negative_prompt``, ``duration``, ``seed`` and ``out``.

All MLX work happens on this process's main thread: from MLX 0.31 GPU streams are per-thread.

Facts measured on the target Mac (M5, 32 GB, sm-sfx, 8 steps): models load in ~0.6 s, the
first clip takes ~2.6 s (kernel compilation), later clips 0.37 s (2 s) to 0.53 s (10 s); one
loaded DiT serves every length (bit-identical to a DiT loaded for that length); the same seed
gives the same clip.
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path
from typing import Any

# -P keeps this directory off the front of sys.path (it must not shadow the engine's modules);
# appended last, it only provides the uniquely named protocol module
sys.path.append(str(Path(__file__).resolve().parent))
import sfx_worker_protocol as protocol

# Stable Audio 3's recommendation for the distilled sm-sfx model
STEPS = 8
SIGMA_MAX = 1.0  # text-to-audio: start from pure noise
DIT_NAME = "sm-sfx"
DECODER_NAME = "same-s"
T5_MAX_TOKENS = 256
# a load-time shape only (one loaded DiT serves every length)
LOAD_SECONDS = 10.0


class Engine:
    def __init__(self, mlx_dir: Path, *, negative_cfg: float) -> None:
        sys.path[:0] = [str(mlx_dir), str(mlx_dir / "scripts")]
        import mlx.core as mx  # pyright: ignore[reportMissingImports]
        from models.defs.sa3_pipeline import (  # pyright: ignore[reportMissingImports]
            apply_prompt_padding,
            build_pingpong_schedule,
            load_conditioner_from_npz,
            patched_decode,
            sample_flow_pingpong,
        )
        from models.defs.t5gemma_mlx import T5Gemma  # pyright: ignore[reportMissingImports]
        from sa3_mlx import (  # pyright: ignore[reportMissingImports]
            DIT_CHOICES,
            SAMPLE_RATE,
            SAMPLES_PER_LATENT,
            T5GEMMA_NPZ_REL,
            load_decoder,
            load_dit,
        )
        from weights import ensure_local  # pyright: ignore[reportMissingImports]

        self.mx = mx
        self.apply_prompt_padding = apply_prompt_padding
        self.build_schedule = build_pingpong_schedule
        self.sample = sample_flow_pingpong
        self.patched_decode = patched_decode
        self.sample_rate = int(SAMPLE_RATE)
        self.samples_per_latent = int(SAMPLES_PER_LATENT)
        # cfg 1.0 (the distilled default) runs no unconditional pass: a negative prompt needs > 1
        self.negative_cfg = negative_cfg
        mx.set_default_device(mx.gpu)
        self.t5 = T5Gemma.from_npz(str(ensure_local(T5GEMMA_NPZ_REL)))
        self.padding_emb, self.secs_embedder = load_conditioner_from_npz(
            str(ensure_local(DIT_CHOICES[DIT_NAME]["ckpt"])), prefix="cond."
        )
        t_lat = math.ceil(LOAD_SECONDS * self.sample_rate / self.samples_per_latent)
        self.dit, _ = load_dit(DIT_NAME, T_lat=t_lat, dtype=mx.float16)
        self.decoder, self.chunk_fn, (self.chunk, self.overlap) = load_decoder(
            DECODER_NAME, mx.float32
        )
        mx.eval(self.dit.parameters())

    def _embed(self, text: str, seconds_embed: Any) -> Any:
        mx = self.mx
        embeds, mask = self.t5.encode([text], max_len=T5_MAX_TOKENS)
        padded = self.apply_prompt_padding(
            embeds.astype(mx.float16), mask, self.padding_emb.astype(mx.float16)
        )
        return mx.concatenate([padded, seconds_embed], axis=1)

    def generate(
        self, prompt: str, negative: str, seconds: float, seed: int, progress: Any
    ) -> tuple[Any, dict[str, Any]]:
        """``(channels, frames)`` float32 audio at 44.1 kHz, trimmed to ``seconds``."""
        import numpy as np  # pyright: ignore[reportMissingImports]

        mx = self.mx
        t_lat = max(1, math.ceil(seconds * self.sample_rate / self.samples_per_latent))
        seconds_embed = self.secs_embedder(seconds).astype(mx.float16)
        cross = self._embed(prompt, seconds_embed)
        global_cond = seconds_embed[:, 0, :]
        cfg = self.negative_cfg if negative and self.negative_cfg > 1.0 else 1.0
        null = self._embed(negative, seconds_embed) if cfg != 1.0 else None
        dit = self.dit

        def model_fn(x: Any, t: Any) -> Any:
            if null is None:
                return dit(x, t, cross, global_cond)
            v = dit(
                mx.concatenate([x, x]),
                mx.concatenate([t, t]),
                mx.concatenate([cross, null]),
                mx.concatenate([global_cond, global_cond]),
            )
            cond_v, uncond_v = mx.split(v, 2, axis=0)
            return _apg(mx, x, t, cond_v, uncond_v, cfg)

        sigmas = self.build_schedule(STEPS, sigma_max=SIGMA_MAX, use_logsnr_shift=True)
        noise = mx.random.normal((1, 256, t_lat), dtype=mx.float16, key=mx.random.key(seed))

        def on_step(index: int, total: int) -> None:
            progress(0.9 * (index + 1) / max(total, 1))

        latents = self.sample(model_fn, noise, sigmas, seed=seed + 1, on_step=on_step)
        audio = self.patched_decode(self._decode(latents.astype(mx.float32), t_lat), 256, 2)
        mx.eval(audio)
        data = np.array(audio.astype(mx.float32))[0]
        return data[..., : round(seconds * self.sample_rate)], {"cfg": cfg}

    def _decode(self, latents: Any, t_lat: int) -> Any:
        """SAME-S decode, as the checkout's own generator does it for each latent length."""
        mx = self.mx
        if t_lat > self.chunk + 2 * self.overlap:
            patches = self.chunk_fn(self.decoder, latents, self.chunk, self.overlap)
        elif t_lat % 2 == 0:
            patches = self.decoder(latents)
        elif t_lat > 6:
            patches = self.chunk_fn(self.decoder, latents, 2, 2)
        else:
            even = mx.concatenate([latents, latents[..., -1:]], axis=-1)
            patches = self.decoder(even)[..., : t_lat * 16]
        mx.eval(patches)
        return patches


def _apg(mx: Any, x: Any, t: Any, cond_v: Any, uncond_v: Any, cfg: float) -> Any:
    """Classifier-free guidance with adaptive projection (the checkout's default, apg = 1)."""
    sigma = t.reshape(-1, 1, 1).astype(mx.float32)
    xf = x.astype(mx.float32)
    cond_d = xf - cond_v.astype(mx.float32) * sigma
    diff = cond_d - (xf - uncond_v.astype(mx.float32) * sigma)
    norm = mx.sqrt((cond_d * cond_d).sum(axis=(-2, -1), keepdims=True))
    unit = cond_d / mx.maximum(norm, 1e-8)
    diff = diff - (diff * unit).sum(axis=(-2, -1), keepdims=True) * unit
    return ((xf - (cond_d + (cfg - 1.0) * diff)) / sigma).astype(x.dtype)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--proto-fd", type=int, required=True)
    parser.add_argument("--mlx-dir", required=True)
    parser.add_argument("--negative-cfg", type=float, default=1.0)
    args = parser.parse_args()
    channel = protocol.open_channel(args.proto_fd)
    started = time.monotonic()
    try:
        engine = Engine(Path(args.mlx_dir), negative_cfg=args.negative_cfg)
    except Exception as exc:
        return protocol.fatal(channel, exc)
    protocol.send(channel, event="ready", load_s=round(time.monotonic() - started, 2), device="mlx")

    def generate(request: dict[str, Any], progress: Any) -> tuple[Any, int, dict[str, Any]]:
        data, extra = engine.generate(
            request["prompt"],
            request.get("negative_prompt") or "",
            float(request["duration"]),
            int(request["seed"]),
            progress,
        )
        return data.T, engine.sample_rate, extra

    return protocol.serve(channel, generate)


if __name__ == "__main__":
    sys.exit(main())
