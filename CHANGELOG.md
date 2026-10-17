# Changelog

## 0.1.0 — unreleased

First version, built on the job server of [ace-step-api](https://github.com/kabatin/ace-step-api) 0.1.0.

- HTTP job API per [docs/api.md](docs/api.md): health (ace-step-api compatible, plus `workflows`
  with `available` / `loaded` / `reason` per workflow), capabilities, workflow listing with JSON
  Schemas, typed submit routes, job status with progress over all clips, queue position,
  `waiting_for`, estimate and timings, per-index WAV download, cancel. One error envelope;
  `503 workflow_unavailable` for a workflow whose engine is missing.
- Workflows `stable-audio-3-sfx` (Stable Audio 3 sm-sfx, Stability AI's MLX build; 44.1 kHz stereo)
  and `moss-sfx-v2` (MOSS-SoundEffect v2.0, PyTorch on MPS; 48 kHz mono): `prompt`,
  `negative_prompt`, `duration_sec` 0.5–10, `count` 1–4, `seed` (clip *i* uses `seed + 2i`); every clip
  read back as 16-bit PCM WAV before the job succeeds.
- One resident worker per engine in that engine's own Python (`-P`, own process group, requests on
  stdin, replies on a private pipe, offline), released when idle, when a coordinated server is busy,
  after failed jobs and on shutdown. The server starts when at least one engine is available.
- MOSS on Apple Silicon: the upstream CUDA assumptions (complex128 RoPE tables, float64 timestep
  embedding, CUDA-only full-precision autocast blocks) are patched in memory by the worker.
- Output level: a clip whose peak exceeds −1 dBFS is turned down as a whole (no limiter).
- Coordination with vpipe-api, klein-api and ace-step-api (`waiting_for_peer` yields, so servers that
  wait for each other cannot deadlock).
