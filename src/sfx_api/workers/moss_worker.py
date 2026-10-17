"""MOSS-SoundEffect v2.0 engine worker. Runs inside its own virtual environment (Python 3.12,
torch with MPS, the pins of the upstream ``moss_soundeffect_v2/pyproject.toml``), never inside
sfx-api's: it imports only the standard library, numpy, torch and the MOSS-TTS checkout's
``moss_soundeffect_v2`` package.

Protocol: see :mod:`sfx_api.workers.sfx_worker_protocol`.

The upstream code assumes CUDA; :func:`mps_patches` makes it run on Apple Silicon without
editing the checkout (see its docstring). Output is 48 kHz mono.
"""

from __future__ import annotations

import os

# the sampler is wrapped in torch.compile(cudagraphs) at import time: run it eagerly
os.environ["TORCHDYNAMO_DISABLE"] = "1"

import argparse
import sys
import time
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Any

# -P keeps this directory off the front of sys.path (it must not shadow the engine's modules);
# appended last, it only provides the uniquely named protocol module
sys.path.append(str(Path(__file__).resolve().parent))
import sfx_worker_protocol as protocol


def mps_patches(device: str) -> None:
    """Three upstream spots fail or lose precision off CUDA; patch them in memory:

    1. RoPE tables are built in float64 and become complex128, which MPS cannot hold
       (``wan_audio_dit.precompute_freqs_cis``): cast them to complex64.
    2. The timestep embedding computes in float64 on the device
       (``sinusoidal_embedding_1d``): compute it in float32.
    3. ``torch.autocast("cuda", float32)`` blocks (time embedding, VAE decode) mean "full
       precision here" but do nothing on MPS, where the pipeline's bf16 autocast stays on:
       turn autocast off inside them instead.
    """
    import torch  # pyright: ignore[reportMissingImports]
    from moss_soundeffect_v2.diffsynth.models import (  # pyright: ignore[reportMissingImports]
        wan_audio_dit,
        wan_video_dit,
    )
    from moss_soundeffect_v2.diffsynth.pipelines import (  # pyright: ignore[reportMissingImports]
        wan_audio,
    )

    original = wan_audio_dit.precompute_freqs_cis

    def freqs_cis_complex64(*args: Any, **kwargs: Any) -> Any:
        return original(*args, **kwargs).to(torch.complex64)

    wan_audio_dit.precompute_freqs_cis = freqs_cis_complex64

    def sinusoidal_embedding_1d(dim: int, position: Any) -> Any:
        exponent = torch.arange(dim // 2, dtype=torch.float32, device=position.device)
        sinusoid = torch.outer(
            position.type(torch.float32), torch.pow(10000, -exponent.div(dim // 2))
        )
        return torch.cat([torch.cos(sinusoid), torch.sin(sinusoid)], dim=1).to(position.dtype)

    for module in (wan_audio_dit, wan_video_dit, wan_audio):
        if hasattr(module, "sinusoidal_embedding_1d"):
            module.sinusoidal_embedding_1d = sinusoidal_embedding_1d

    real_autocast = torch.autocast

    def autocast(
        device_type: str, dtype: Any = None, enabled: bool = True, cache_enabled: Any = None
    ) -> Any:
        if device_type == "cuda" and dtype == torch.float32 and not torch.cuda.is_available():
            return real_autocast(device, enabled=False)
        return real_autocast(device_type, dtype=dtype, enabled=enabled, cache_enabled=cache_enabled)

    torch.autocast = autocast  # this process runs nothing else


class Engine:
    def __init__(
        self,
        code_dir: Path,
        model_dir: Path,
        *,
        device: str,
        steps: int,
        cfg: float,
        latent_seconds: int,
    ) -> None:
        sys.path.insert(0, str(code_dir))
        import torch  # pyright: ignore[reportMissingImports]

        mps_patches(device)
        from moss_soundeffect_v2 import (  # pyright: ignore[reportMissingImports]
            MossSoundEffectPipeline,
        )

        self.torch = torch
        self.device = device
        self.steps = steps
        self.cfg = cfg
        # 0: the model's own fixed latent (30 s); shorter latents are faster
        self.latent_seconds = latent_seconds or None
        self.pipe = MossSoundEffectPipeline.from_pretrained(
            str(model_dir), torch_dtype=torch.bfloat16, device=device
        )
        self.sample_rate = int(self.pipe.sample_rate)

    def generate(
        self, prompt: str, negative: str, seconds: float, seed: int, progress: Any
    ) -> tuple[Any, dict[str, Any]]:
        """``(channels, frames)`` float32 audio at 48 kHz."""
        steps = self.steps

        def bar(items: Iterable[Any]) -> Iterator[Any]:
            for index, item in enumerate(items):
                yield item
                progress(0.95 * (index + 1) / steps)

        audio = self.pipe(
            prompt=prompt,
            seconds=seconds,
            num_inference_steps=steps,
            cfg_scale=self.cfg,
            seed=seed,
            negative_prompt=negative,
            max_inference_seconds=self.latent_seconds,
            progress_bar_cmd=bar,
        )
        if self.device == "mps":
            self.torch.mps.synchronize()
        data = audio.float().cpu().numpy()[0]
        return data, {
            "steps": steps,
            "cfg": self.cfg,
            "latent_seconds": self.latent_seconds or self.pipe.max_inference_seconds,
        }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--proto-fd", type=int, required=True)
    parser.add_argument("--code-dir", required=True)  # the MOSS-TTS checkout
    parser.add_argument("--model-dir", required=True)  # the downloaded model repository
    parser.add_argument("--device", default="mps")
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--cfg", type=float, default=4.0)
    parser.add_argument("--latent-seconds", type=int, default=0)
    args = parser.parse_args()
    channel = protocol.open_channel(args.proto_fd)
    started = time.monotonic()
    try:
        engine = Engine(
            Path(args.code_dir),
            Path(args.model_dir),
            device=args.device,
            steps=args.steps,
            cfg=args.cfg,
            latent_seconds=args.latent_seconds,
        )
    except Exception as exc:
        return protocol.fatal(channel, exc)
    protocol.send(
        channel, event="ready", load_s=round(time.monotonic() - started, 2), device=args.device
    )

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
