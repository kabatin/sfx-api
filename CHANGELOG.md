# Changelog

## 0.1.1 — 2026-10-28
- `sfx-api setup`: the install command shared by the AI servers. `--models sa3-medium,sa3-small,moss`
  (default all), `--check` (changes nothing: machine, per-component state and download bytes),
  `--yes`, `--json` (one object on stdout), `--prefix`, `--lan` (token from `SFX_API_TOKEN`, the
  config or generated; never printed), `--port`, `--coordinate-with NAME=URL`, `--service` (launchd
  plist, `bootout` + `bootstrap`, waits for `/v1/health`), `--label-suffix`. Stable Audio 3 runs its
  own `install.sh` at a pinned commit; MOSS gets a pinned checkout, the README's pinned environment and
  the model at a pinned revision. Downloads resume, a stalled one is restarted, free disk is checked
  first. The config is merged (existing values kept), written atomically with mode 0600 and validated
  by the server's own loader.
- The Stable Audio 3 weights are fetched at a pinned revision before `install.sh` runs offline;
  both engines' dependencies resolve with `UV_EXCLUDE_NEWER`. Peer URLs and tokens and the LAN
  token are checked before any download; a known peer gets its new URL. `--label-suffix` needs
  `--port`. Usage errors with `--json` still print the JSON object. The plist is replaced atomically,
  launchd is given time to drop the old agent, and directories setup creates are 0700.

## 0.1.0 — 2026-10-19

First version, built on the job server of [ace-step-api](https://github.com/kabatin/ace-step-api) 0.1.0.

- HTTP job API per [docs/api.md](docs/api.md): health (ace-step-api compatible, plus `workflows`
  with `available` / `loaded` / `reason` per workflow), capabilities, workflow listing with JSON
  Schemas, typed submit routes, job status with progress over all clips, queue position,
  `waiting_for`, estimate and timings, per-index WAV download, cancel. One error envelope;
  `503 workflow_unavailable` for a workflow whose engine is missing.
- Workflows `stable-audio-3-medium-sfx` (Stable Audio 3 medium, prompt tagged `TrackType: SFX`),
  `stable-audio-3-sfx` (Stable Audio 3 sm-sfx, generated at whole seconds and cut), both Stability
  AI's MLX build at 44.1 kHz stereo, and `moss-sfx-v2` (MOSS-SoundEffect v2.0, PyTorch on MPS; 48 kHz
  mono): `prompt` (500 chars), `negative_prompt`, `duration_sec` 0.5–10, `count` 1–4, `seed` (clip *i*
  uses `seed + 2i`); every clip read back as 16-bit PCM WAV before the job succeeds.
- Workflow `stable-audio-3-medium-music`: instrumental music from the same medium engine, prompt
  tagged `TrackType: Music, VocalType: Instrumental`, `prompt` up to 1000 chars, `duration_sec`
  0.5–190 (default 30).
- One resident worker per engine in that engine's own Python (`-P`, own process group, requests on
  stdin, replies on a private pipe, offline), released when idle, when a coordinated server is busy,
  after failed jobs and on shutdown; only the running job's engine stays in memory. The server starts
  when at least one engine is available.
- MOSS on Apple Silicon: the upstream CUDA assumptions (complex128 RoPE tables, float64 timestep
  embedding, CUDA-only full-precision autocast blocks) are patched in memory by the worker.
- Output level: a clip whose peak exceeds −1 dBFS is turned down as a whole (no limiter).
- Coordination with vpipe-api, klein-api and ace-step-api (`waiting_for_peer` yields, so servers that
  wait for each other cannot deadlock). The peer probe follows no redirects and never goes through a
  proxy, so a peer token only reaches that peer's `health_url`.
