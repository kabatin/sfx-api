"""A stand-in engine worker for the tests: speaks the real protocol
(``sfx_api/workers/sfx_worker_protocol.py``) and fakes the engine.

FAKE_WORKER_MODE: ok, noisy (prints to stdout), load_fail, gen_error, hang, crash.
FAKE_WORKER_RATE / FAKE_WORKER_CHANNELS: the audio it makes (default 44100 Hz stereo).
FAKE_WORKER_RECORD: a file the worker's argv is written to.
Each request is also written next to its output as ``<out>.params.json``.
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

WORKERS = Path(__file__).resolve().parent.parent / "src" / "sfx_api" / "workers"
sys.path.append(str(WORKERS))
import sfx_worker_protocol as protocol  # noqa: E402

MODE = os.environ.get("FAKE_WORKER_MODE", "ok")
RATE = int(os.environ.get("FAKE_WORKER_RATE", "44100"))
CHANNELS = int(os.environ.get("FAKE_WORKER_CHANNELS", "2"))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--proto-fd", type=int, required=True)
    args, rest = parser.parse_known_args()
    channel = protocol.open_channel(args.proto_fd)
    record = os.environ.get("FAKE_WORKER_RECORD")
    if record:
        Path(record).write_text(json.dumps(rest))
    if MODE == "load_fail":
        return protocol.fatal(channel, FileNotFoundError("checkpoint missing"))
    protocol.send(channel, event="ready", load_s=0.01, device="fake")

    def generate(request, progress):
        import numpy as np

        progress(0.1)
        if MODE == "noisy":
            print("Sampling... 52%", flush=True)
        progress(0.52)
        if MODE == "hang":
            time.sleep(3600)
        if MODE == "crash":
            os._exit(3)
        if MODE == "gen_error":
            raise RuntimeError("MPS backend out of memory")
        Path(request["out"] + ".params.json").write_text(
            json.dumps({k: v for k, v in request.items() if k not in ("cmd", "id", "out")})
        )
        frames = round(float(request["duration"]) * RATE)
        # a full-scale square wave: the protocol has to bring it under the ceiling
        square = np.where((np.arange(frames) // 50) % 2 == 0, 1.0, -1.0).astype(np.float32)
        progress(0.99)
        return np.repeat(square[:, None], CHANNELS, axis=1), RATE, {"cfg": 1.0}

    return protocol.serve(channel, generate)


if __name__ == "__main__":
    sys.exit(main())
