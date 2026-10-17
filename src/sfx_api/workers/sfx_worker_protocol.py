"""The worker side of the engine protocol, shared by every engine worker. Standard library and
numpy only (both engine environments have numpy); the WAV is written with ``wave``.

One JSON object per line. Requests arrive on stdin; replies go to the file descriptor given by
``--proto-fd`` (engines print logs to stdout, so stdout is pointed at stderr and never carries
the protocol).

    -> {"event": "ready", "load_s": 0.6, "device": "mlx"}
    <- {"cmd": "generate", "id": "...", "prompt": "...", "negative_prompt": "...",
        "duration": 2.5, "seed": 7, "out": "/abs/raw-0.wav"}
    -> {"event": "progress", "id": "...", "value": 0.52}
    -> {"event": "done", "id": "...", "seconds": 0.4, "sample_rate": 44100, "frames": 110250,
        "channels": 2, "gain_db": -1.3, ...engine extras}
    -> {"event": "error", "id": "...", "message": "..."}
    <- {"cmd": "quit"}             (or stdin closed: the worker exits)

A failed load sends ``{"event": "fatal", "message": ...}`` and exits with status 1.
"""

from __future__ import annotations

import json
import os
import sys
import time
import traceback
import wave
from collections.abc import Callable
from pathlib import Path
from typing import Any, TextIO

# engines' float output can exceed full scale; 16-bit PCM would clip it for good
PEAK_CEILING_DB = -1.0

# (request, progress) -> ((frames, channels) float audio, sample rate, extras for the reply)
Generate = Callable[[dict[str, Any], Callable[[float], None]], tuple[Any, int, dict[str, Any]]]


def open_channel(fd: int) -> TextIO:
    """The protocol channel, then fd 1 -> stderr so nothing else can write to the channel."""
    channel = os.fdopen(os.dup(fd), "w", buffering=1, encoding="utf-8")
    if fd > 2:
        os.close(fd)
    os.dup2(2, 1)
    sys.stdout = sys.stderr
    return channel


def send(channel: TextIO, **message: Any) -> None:
    channel.write(json.dumps(message, ensure_ascii=False) + "\n")
    channel.flush()


def fatal(channel: TextIO, exc: BaseException) -> int:
    traceback.print_exc()
    send(channel, event="fatal", message=f"{type(exc).__name__}: {exc}")
    return 1


def fit_peak(data: Any, ceiling_db: float = PEAK_CEILING_DB) -> tuple[Any, float]:
    """Turn the whole clip down (never up) so its peak stays under ``ceiling_db``. A limiter
    would reshape the attack of an impact; one gain keeps the sound as made. Returns the audio
    and the gain in dB (0.0 when nothing was done)."""
    import numpy as np  # pyright: ignore[reportMissingImports]

    audio = np.nan_to_num(np.asarray(data, dtype=np.float32), nan=0.0, posinf=1.0, neginf=-1.0)
    ceiling = 10 ** (ceiling_db / 20)
    peak = float(np.abs(audio).max()) if audio.size else 0.0
    if peak <= ceiling:
        return audio, 0.0
    gain = ceiling / peak
    return audio * np.float32(gain), round(float(20 * np.log10(gain)), 2)


def write_wav(path: Path, data: Any, rate: int) -> None:
    """``(frames, channels)`` float audio in [-1, 1] as 16-bit PCM."""
    import numpy as np  # pyright: ignore[reportMissingImports]

    pcm = np.clip(np.round(np.asarray(data) * 32767.0), -32768, 32767).astype("<i2")
    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(int(pcm.shape[1]))
        wav.setsampwidth(2)
        wav.setframerate(int(rate))
        wav.writeframes(pcm.tobytes())


def _run(channel: TextIO, generate: Generate, request: dict[str, Any]) -> dict[str, Any]:
    job = request["id"]
    last = [-1.0]

    def progress(value: float) -> None:
        if 0.0 <= value <= 1.0 and value - last[0] >= 0.01:
            last[0] = value
            send(channel, event="progress", id=job, value=round(value, 3))

    started = time.monotonic()
    data, rate, extra = generate(request, progress)
    if data.ndim != 2 or data.shape[0] == 0:
        raise RuntimeError(f"the engine returned audio of shape {tuple(data.shape)}")
    data, gain_db = fit_peak(data)
    out = Path(request["out"])
    partial = out.with_suffix(".partial")
    write_wav(partial, data, rate)
    partial.replace(out)
    return {
        "seconds": round(time.monotonic() - started, 3),
        "sample_rate": int(rate),
        "frames": int(data.shape[0]),
        "channels": int(data.shape[1]),
        "gain_db": gain_db,
        **extra,
    }


def serve(channel: TextIO, generate: Generate) -> int:
    for line in sys.stdin:
        try:
            request = json.loads(line)
        except ValueError:
            continue
        if not isinstance(request, dict) or request.get("cmd") == "quit":
            break
        if request.get("cmd") != "generate":
            continue
        try:
            reply = _run(channel, generate, request)
        except Exception as exc:
            traceback.print_exc()
            send(
                channel, event="error", id=request.get("id"), message=f"{type(exc).__name__}: {exc}"
            )
            continue
        send(channel, event="done", id=request["id"], **reply)
    return 0
