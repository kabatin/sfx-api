# sfx-api

**Make sound effects (and instrumental music) on your Apple Silicon Mac, as an HTTP job API.**

`sfx-api` is a small, safe job server around two text-to-sound-effect engines:
[Stable Audio 3](https://github.com/Stability-AI/stable-audio-3) (its medium and sm-sfx models,
Stability AI's own MLX build) and [MOSS-SoundEffect v2.0](https://huggingface.co/OpenMOSS-Team/MOSS-SoundEffect-v2.0)
(PyTorch on the Mac's GPU), plus instrumental music from the same Stable Audio 3 medium engine. Send
an English description, a length and a count, get back 1–4 WAV clips:

```
POST /v1/workflows/stable-audio-3-medium-sfx/jobs  → 202 {id}
GET  /v1/jobs/{id}                                   → queued → running (progress) → succeeded
GET  /v1/jobs/{id}/output?index=N                    → audio/wav
```

It speaks the same job protocol as [ace-step-api](https://github.com/kabatin/ace-step-api),
[klein-api](https://github.com/kabatin/klein-api) and [vpipe-api](https://github.com/kabatin/vpipe-api)
(job lifecycle, error envelope, `429` + `Retry-After`, `Idempotency-Key`), so a client that drives them
needs only a different workflow id, body and output type. The full contract is [docs/api.md](docs/api.md).

| workflow | engine | output | speed (loaded, per clip) |
|---|---|---|---|
| `stable-audio-3-medium-sfx` | Stable Audio 3 medium, MLX, prompt tagged `TrackType: SFX` | 44.1 kHz stereo | **1.1–1.7 s** for 1–10 s of sound |
| `stable-audio-3-medium-music` | the same medium engine, prompt tagged `TrackType: Music, VocalType: Instrumental`; up to 190 s | 44.1 kHz stereo | **3.4–15.8 s** for 32–190 s of music |
| `stable-audio-3-sfx` | Stable Audio 3 sm-sfx, MLX | 44.1 kHz stereo | **0.4–0.6 s** for 2–10 s of sound |
| `moss-sfx-v2` | MOSS-SoundEffect v2.0, PyTorch MPS | 48 kHz mono | ~90 s whatever the length |

Use `stable-audio-3-medium-sfx`. In a blind listening test on the nine sounds of a car-chase trailer
(four setups, two seeds each), its sound was picked for eight; for the ninth no setup fitted. For
instrumental background music, `stable-audio-3-medium-music` was picked over ACE-Step 1.5 for three of
four briefs; it makes no vocals.

- **Closed workflows.** Clients send `prompt`, `negative_prompt`, `duration_sec` (0.5–10; up to 190
  for music), a seed and a count — never model names, steps, guidance or paths.
- **Checked output.** Every clip is a 16-bit PCM WAV with the workflow's rate and channels, peaks
  under −1 dBFS, read back before the job succeeds; the result reports each clip's real length.
- **Honest availability.** `/v1/health` says per workflow whether its engine is installed and whether
  its models are loaded; a workflow that cannot run answers `503 workflow_unavailable` with the reason,
  and the other one keeps working.
- **One GPU slot, honest backpressure.** One job at a time (its clips run one after another), a small
  waiting queue, `429 busy` with `Retry-After` when full.
- **Plays nicely with other GPU servers.** With `coordinate_with`, a job starts only once vpipe-api /
  klein-api / ace-step-api report idle; while waiting it yields (`waiting_for_peer`), so servers that
  wait for each other cannot deadlock.
- **No needless model loads, no hoarding.** Each engine stays loaded in a resident worker between its
  jobs; only the engine of the running job is kept, and it is released when idle (300 s) or as soon as
  another GPU server is busy.
- **Restart-safe.** Jobs live on disk; queued jobs run again, interrupted ones fail as retryable, and a
  worker left behind by a crashed server is stopped.

## Requirements

- Apple Silicon Mac, macOS 14+. 32 GB of memory recommended (Stable Audio 3 peaks at ~11 GB with
  medium, sound effects or music, and ~4 GB with sm-sfx; MOSS at ~22 GB).
- Python 3.12+ and [uv](https://docs.astral.sh/uv/) for sfx-api itself.
- Either engine, or both, each in its own environment (sfx-api runs it as a separate worker process;
  the environments never mix):

| engine | code | weights | download |
|---|---|---|---|
| Stable Audio 3 | [Stability-AI/stable-audio-3](https://github.com/Stability-AI/stable-audio-3) `optimized/mlx` (MIT) | [stabilityai/stable-audio-3-optimized](https://huggingface.co/stabilityai/stable-audio-3-optimized) `MLX/` (Stability AI Community License, not gated) | 1.9 GB (sm-sfx, SAME-S, T5Gemma); medium adds 4.6 GB (its DiT and SAME-L) |
| MOSS-SoundEffect v2.0 | [OpenMOSS/MOSS-TTS](https://github.com/OpenMOSS/MOSS-TTS) `moss_soundeffect_v2` (Apache-2.0) | [OpenMOSS-Team/MOSS-SoundEffect-v2.0](https://huggingface.co/OpenMOSS-Team/MOSS-SoundEffect-v2.0) (Apache-2.0, not gated) | 11.2 GB |

### Measurements

On the target Mac (MacBook Pro M5, 32 GB, macOS 26.5). Peak memory is macOS's `phys_footprint` of the
worker (sampled every second, a lower bound).

**Stable Audio 3** (sm-sfx, 8 steps, no guidance), through this server, seconds per clip
(`timings.backend_seconds`):

| | 2 s clip | 5 s clip | 10 s clip | count 2 (2–4 s clips) |
|---|---|---|---|---|
| models loaded | **0.37** | **0.42** | **0.59** | 0.73–0.80 per job |
| first job (load 0.3–0.6 s + clip) | 1.0 | | | |

Peak memory ~4 GB. A brand-new process also compiles its kernels once (the first clip after
installing took 2.6 s). The same seed gave a bit-identical clip; one loaded model serves every length
(bit-identical to a model loaded for that length). With guidance on (`negative_prompt_cfg = 3.0`) a clip
took 0.48–0.54 s and peaked 1–4 dB hotter before the level stage.

**Stable Audio 3 medium** (8 steps, no guidance), with the worker's code outside the server, three
clips each, GPU otherwise idle:

| | 1 s clip | 2 s clip | 5 s clip | 10 s clip |
|---|---|---|---|---|
| models loaded | 1.10 | 1.10 | 1.23 | 1.73 |

Loading took ~1 s with the files in the OS cache (reading them from disk was not measured). Peak
memory 11 GB. Unlike sm-sfx, medium is fine at fractional lengths: 92 clips at 0.7–9.5 s (3 prompts,
2–3 seeds) peaked at 1.7 at most, so it makes the requested length itself. Through the running
server, `count: 1` jobs with the same prompt and seed gave the same clips bit for bit (5 clips of
2–3 s).

**Stable Audio 3 medium, music** (the same engine with `TrackType: Music, VocalType: Instrumental, `),
one track each, with the worker's code outside the server, models loaded:

| | 15 s | 20 s | 32 s | 62 s | 120 s | 190 s | 380 s |
|---|---|---|---|---|---|---|---|
| seconds | 2.2 | 2.6 | 3.4 | 5.3 | 10.3 | 15.8 | 32.8 |

Peak memory 11 GB with tracks up to 62 s, 14 GB in the run that included 380 s, and 11 GB for a 190 s
track through the server (the worker sampled every 0.5 s). Every length kept
in range (raw peak 1.0–1.65) and had sound until its last second or two; the level of the last 30 s
was within 4 dB of the whole track. The workflow stops at 190 s, the longest length Stability AI's
paper evaluated; 380 s (the model's own limit) measured as cleanly but was not listened to.

**MOSS-SoundEffect v2.0** (cfg 4, PyTorch 2.9.1 on MPS), seconds per clip with the models loaded and
the GPU otherwise idle:

| setting | 2 s clip | sound |
|---|---|---|
| 100 steps, 30 s latent (upstream defaults) | 168 | full length, tonal screech |
| **50 steps, 30 s latent (this server's default)** | **86–88** | same character as 100 steps (spectral centroid 3.07 vs 2.95 kHz, envelope correlation 0.99) |
| 50 steps, 10 s latent | 27 | **broken**: ~1.5 s of sound, then silence, different pitch |

Through this server (`timings.backend_seconds`, 50 steps): 2 s / 5 s / 10 s clips took 87.5 / 87.1 /
87.2 s with the models loaded; the first job took 104.5 s (load 16.2 s + clip 88.3 s). Short and long
clips cost the same (the latent is fixed), so a job of `count` n takes about n × 88 s. Loading takes
15–16 s with the files in the OS cache and ~28 s without. Memory: 21–22 GB `phys_footprint` (20 GB of it
on the GPU). While another GPU job was running on the Mac, a 50-step clip took 266 s.

## Setup on the Mac

```sh
# 1a. Stable Audio 3 (MLX): its own installer makes .venv and downloads medium and sm-sfx
git clone https://github.com/Stability-AI/stable-audio-3 ~/.local/opt/stable-audio-3/src
cd ~/.local/opt/stable-audio-3/src/optimized/mlx
HF_HOME=~/.local/share/sfx-api/hf ./install.sh -y --download medium,sm-sfx

# 1b. MOSS-SoundEffect v2.0 (PyTorch, MPS): code, an environment, the model repository
git clone https://github.com/OpenMOSS/MOSS-TTS ~/.local/opt/moss-sfx/src
cd ~/.local/opt/moss-sfx && uv venv --python 3.12 .venv
VIRTUAL_ENV=$PWD/.venv uv pip install torch==2.9.1 torchaudio==2.9.1 numpy==1.26.4 einops==0.8.2 \
  pillow==12.2.0 tqdm==4.67.3 safetensors==0.7.0 transformers==4.57.1 diffusers==0.37.1 \
  ftfy==6.3.1 regex==2026.4.4 soundfile==0.13.1 descript-audiotools==0.7.2
uvx --from 'huggingface_hub[cli]' hf download OpenMOSS-Team/MOSS-SoundEffect-v2.0 \
  --local-dir ~/.local/share/sfx-api/moss/MOSS-SoundEffect-v2.0

# 2. the server
uv tool install git+https://github.com/kabatin/sfx-api
```

The MOSS pins are the upstream `moss_soundeffect_v2/pyproject.toml` ones without the CUDA extra and
without gradio (only the demo needs it). `HF_HUB_DISABLE_XET=1` can speed up the downloads. The server
itself never downloads anything: its workers run with `HF_HUB_OFFLINE=1`.

```toml
# ~/.config/sfx-api/config.toml   (chmod 600: it holds tokens)
host = "0.0.0.0"          # LAN; requires a token
port = 8769
token = "…openssl rand -hex 32…"

[backends.sa3-medium]
mlx_dir = "~/.local/opt/stable-audio-3/src/optimized/mlx"
hf_home = "~/.local/share/sfx-api/hf"        # where install.sh put the weights

[backends.sa3]
mlx_dir = "~/.local/opt/stable-audio-3/src/optimized/mlx"
hf_home = "~/.local/share/sfx-api/hf"

[backends.moss]
code_dir = "~/.local/opt/moss-sfx/src"
model_dir = "~/.local/share/sfx-api/moss/MOSS-SoundEffect-v2.0"

# wait for the other GPU servers on this Mac before each job
[[coordinate_with]]
name = "vpipe-api"
health_url = "http://127.0.0.1:8765/v1/health"
token = "…vpipe-api's bearer token…"

[[coordinate_with]]
name = "ace-step-api"
health_url = "http://127.0.0.1:8768/v1/health"
token = "…ace-step-api's bearer token…"
```

### `doctor`

```sh
sfx-api doctor           # platform, macOS, Python, config permissions, every engine (checkout,
                         # Python, model files), disk, power, memory, every coordinate_with server
sfx-api doctor --smoke   # + one real 1-second clip per available workflow (stop the server first)
```

### `serve`

```sh
sfx-api serve            # http://127.0.0.1:8769   (OpenAPI: /openapi.json, docs: /docs)
```

`serve` starts when at least one engine is available, warns about the other, and exits with a clear
message if neither can generate.

### Run at login (launchd)

[`examples/com.github.kabatin.sfx-api.plist`](examples/com.github.kabatin.sfx-api.plist):

```sh
cp examples/com.github.kabatin.sfx-api.plist ~/Library/LaunchAgents/   # then edit the paths
launchctl bootstrap gui/$(id -u) ~/Library/LaunchAgents/com.github.kabatin.sfx-api.plist
launchctl bootout gui/$(id -u)/com.github.kabatin.sfx-api                # stop
```

Logs are JSON lines on stderr (job id, state transitions, queue/coordination/backend/postprocess/total
seconds, model loads, failure category) — never prompts, audio or tokens. Each engine worker's own
output goes to `data_dir/engine/<engine>/worker.log`.

## Configuration

`~/.config/sfx-api/config.toml` (or `$SFX_API_CONFIG`); every scalar can also be set as
`SFX_API_<NAME>`.

| key | default | |
|---|---|---|
| `host` / `port` | `127.0.0.1` / `8769` | a non-loopback host **requires** `token` |
| `token` | — | when set, every request needs `Authorization: Bearer <token>` (≥ 32 printable ASCII) |
| `max_waiting` | `2` | jobs allowed to wait behind the running one |
| `retention_days` | `7` | finished jobs and outputs are deleted after this |
| `data_dir` | `~/.local/share/sfx-api` | `jobs/<id>/` (record, outputs), `engine/<engine>/` (worker log, pid) |
| `job_timeout_factor` | `3.0` | a job is stopped after `estimate × factor + 5 min` |
| `max_body_mb` | `64` | request size limit |
| `coordinate_with` | `[]` | `[[coordinate_with]]` tables: `name`, `health_url`, optional `token` (file only) |
| `settle_seconds` | `30` | after a coordinated server was seen busy, a job starts no sooner than this |
| `coordination_poll_seconds` | `5` | how often a busy server is asked again |

`[backends.sa3-medium]` (medium) and `[backends.sa3]` (sm-sfx) take the same keys; a workflow whose
table is missing reports itself unavailable:

| key | default | |
|---|---|---|
| `mlx_dir` | — | the checkout's `optimized/mlx` directory (`install.sh` run in it) |
| `python` | `<mlx_dir>/.venv/bin/python` | the engine's interpreter |
| `hf_home` | — | `HF_HOME` the weights were downloaded into |
| `negative_prompt_cfg` | `1.0` | guidance used when a job has a negative prompt; `1.0` ignores negative prompts (the distilled model's own sound); `3.0` is the checkout's example and costs ~30 % more |
| `resident_idle_seconds` | `300` | keep the models loaded this long after a job (`0`: load per job) |
| `start_timeout_s`, `stop_grace_s` | `120`, `10` | |
| `load_seconds`, `clip_overhead_seconds`, `seconds_per_audio_second` | measured | estimate model |

`[backends.moss]`:

| key | default | |
|---|---|---|
| `code_dir` | — | the MOSS-TTS checkout (contains `moss_soundeffect_v2/`) |
| `model_dir` | — | the downloaded model repository |
| `python` | `<code_dir>/../.venv/bin/python` | the engine's interpreter |
| `device` | `mps` | `cpu` works but is far slower |
| `steps` | `50` | sampler steps (upstream's API default is 100; 50 sounded the same at half the time) |
| `cfg` | `4.0` | guidance, must be > 1 |
| `resident_idle_seconds` | `300` | |
| `start_timeout_s`, `stop_grace_s` | `300`, `15` | |
| `load_seconds`, `clip_overhead_seconds`, `seconds_per_step` | measured | estimate per clip: `overhead + seconds_per_step × steps` |

### Coordination with other GPU servers

When `coordinate_with` is set, the job at the head of the queue starts only after a check in which
**every** listed server's `/v1/health` reports `running == 0 && waiting == 0` (and `outside_runs == 0`
when present). If one was seen busy within the last `settle_seconds`, the job checks again
`settle_seconds` after that sighting. Until then it is `queued` with `waiting_for: "<name>"`. A server
that cannot be reached counts as idle and is logged as a warning. A running job is never paused.
Redirects are not followed, so a peer token only ever goes to its own `health_url`.

**Yielding.** While its head job waits for a peer, sfx-api's own `/v1/health` reports the queue as
`waiting_for_peer` and `waiting: 0`, so a server that lists sfx-api in its own `coordinate_with` goes
first instead of waiting forever. vpipe-api 0.1.4 and later can list sfx-api too and yields the same
way: when both have jobs waiting, the first final check to pass starts its job and the other sees
`running: 1` and waits. Both start only if the two final checks overlap, a window as long as one check.

### Resident engines

The first job for an engine starts its worker (`<python> -P <engine>_worker.py …` in the engine's
checkout, its own process group, stdin for requests and a private pipe for replies, so the engine's
progress output on stdout can never corrupt the protocol). It loads the models once; later jobs reuse
them. A worker is stopped:

- after `resident_idle_seconds` without a job;
- when a job for the other engine starts (one engine in memory at a time);
- immediately when a `coordinate_with` server is seen busy while it idles;
- after a job that did not succeed (canceled, timed out, failed), and on shutdown.

A worker left behind by a crashed sfx-api is stopped at the next start (pid file per engine, matched by
pid, script name and start time).

**MOSS on Apple Silicon.** The upstream pipeline assumes CUDA. The worker patches three spots in memory,
without editing the checkout: the RoPE tables become complex64 (MPS has no complex128), the timestep
embedding is computed in float32 (no float64 on MPS), and the pipeline's CUDA-only "full precision
here" autocast blocks (time embedding, VAE decode) turn autocast off on MPS instead of doing nothing.
`torch.compile` is disabled (`TORCHDYNAMO_DISABLE=1`).

### Output

Each clip comes from its engine as float audio and is written as 16-bit PCM. A clip whose peak exceeds
−1 dBFS is turned down as a whole (never up); there is no limiter, so the attack of an impact keeps its
shape. Each clip's gain is in `result.details.generation.engine.clips[]`.

## API examples

Full reference: [docs/api.md](docs/api.md); more in [examples/curl.md](examples/curl.md).

```sh
API=http://127.0.0.1:8769
AUTH=()                                   # or: AUTH=(-H "Authorization: Bearer $SFX_API_TOKEN")

JOB=$(curl -s "${AUTH[@]}" -X POST "$API/v1/workflows/stable-audio-3-sfx/jobs" \
        -H 'Content-Type: application/json' -H 'Idempotency-Key: sfx-001' \
        -d '{"prompt": "tyre screech on wet asphalt, close, short", "duration_sec": 2.5,
             "count": 2, "seed": 1234}' | jq -r .id)

curl -s "${AUTH[@]}" "$API/v1/jobs/$JOB" | jq '{status, progress, queue_position, waiting_for, timings}'
curl -s "${AUTH[@]}" -o screech-0.wav "$API/v1/jobs/$JOB/output?index=0"
curl -s "${AUTH[@]}" -o screech-1.wav "$API/v1/jobs/$JOB/output?index=1"
curl -s "${AUTH[@]}" -X DELETE "$API/v1/jobs/$JOB"          # cancel (200 canceled / 202 stopping)
```

## Security

- Binds `127.0.0.1` by default. A non-loopback `host` without a `token` refuses to start.
- Without a token, only loopback `Host` headers are served (blocks DNS rebinding) and proxy headers are
  refused. With a token, every endpoint (including `/docs`, `/openapi.json`, `/healthz`) needs it.
- **No TLS.** On a LAN the token, the prompts and the audio travel in clear text: use a trusted network
  or a TLS reverse proxy (keep the token set). `doctor` warns if a config file holding tokens is readable
  by others.
- Body limit (`max_body_mb`; any `Transfer-Encoding` is refused with 411), at most 64 concurrent
  connections, control characters refused in prompts, strict job ids, no client-chosen paths, models or
  settings. The engine workers run without a shell and without sfx-api's environment (the token),
  offline (`HF_HUB_OFFLINE=1`), and receive prompts as JSON on a pipe. Client-facing errors and the
  `reason` in `/v1/health` never contain paths. Job files are private to the server's user (umask 077).
  Do not add CORS: the API relies on browsers not being able to send cross-site JSON.
- MOSS's VAE file (`vae_128d_48k.pth`) is a pickled PyTorch object, loaded with its own code; take the
  model repository only from OpenMOSS-Team.

## Verification status

Checked on the target Mac (2026-10-18) with `doctor`, the real-engine test (both engines, `count` 2,
seeds `seed` and `seed + 2`), and a run through the server on loopback for each workflow: health,
capabilities, a first job that loads the models, 2 / 5 / 10 s clips with them loaded, four example
sounds (tyre screech, engine rev, whoosh, impact; Stable Audio 3 with `count` 2, MOSS with `count` 1)
and an idempotent resubmit. Every job succeeded. Output: 16-bit WAV, 44.1 kHz stereo (Stable Audio 3)
and 48 kHz mono (MOSS), exactly the requested length, peaks at or under −1.0 dBFS. A 10 s MOSS clip
has sound to the end. While MOSS ran, `/v1/health` showed it loaded and Stable Audio 3 unloaded. The
Stable Audio 3 models were released when vpipe-api became busy and loaded again for the next job. The
protocol, cancel/timeout, crashes, idle release and the level stage are covered by tests against a
stand-in worker.

The clips were checked by measurement (length, level, loudness per second, spectrum), not by
listening.

Stable Audio 3 sm-sfx at fractional lengths (found by the first client, 2026-10-18): most lengths
that are not whole seconds (0.7, 0.9, 1.2, 2.8 s …) came out 8–40 times over full scale and sounded
like noise, for every prompt and seed tried; the official MLX script gives the same numbers. sm-sfx
clips are now generated at the next whole second and cut (see the worker's `conditioning_seconds`). Checked
afterwards: 504 whole-second clips (1–12 s, 7 prompts, 6 seeds) peak at 1.8 at most, and the
client's 36 requests (0.7–8 s) replayed peak under 1.0. A second machine on the LAN has run jobs
with the token.

Stable Audio 3 medium (2026-10-19): the user compared nine trailer sounds in a blind listening test,
each made four ways (sm-sfx or medium, with or without `TrackType: SFX, ` in front of the prompt) with
two seeds, loudness-matched for playback. Medium with the tag was picked for eight sounds; for the
ninth (an exhaust backfire) none fitted. A second round on the backfire (eight wordings, medium with
the tag) found a usable clip for seven of them, so the wording was the problem, not the length. The
real-engine tests ran medium through the backend (exact 1.4 s, the tag sent) with a config that has
`[backends.sa3-medium]`.

Music (2026-10-19): the user compared four instrumental briefs in a blind listening test (a 62 s
trailer cue, a 32 s anime-opening style track, 20 s lo-fi piano, 15 s lofi hip hop), each with the
same number of tracks from ACE-Step 1.5 turbo (ace-step-api) and from Stable Audio 3 medium with the
music tag, loudness-matched for playback. Stable Audio 3 medium was picked for three; ACE-Step for the
lo-fi piano. Through the running server, a 190 s track (`count` 1) took 17.7 s including a 1.4 s
load, came back as a 190.0 s 44.1 kHz stereo WAV with the tag in `prompt_used`, and was turned down
1.8 dB to the ceiling.

Not checked: cancelling a job on a real engine, `count` 3–4 in one job, Stable Audio 3 with
`negative_prompt_cfg` > 1 through the server (measured outside it), MOSS on `device = "cpu"`.

## Development

```sh
uv sync
uv run pytest            # stand-in worker (tests/fake_worker.py); 80%+ coverage gate; `real` tests skipped
uv run ruff check && uv run ruff format --check && uv run pyright
SFX_API_REAL_TESTS=1 uv run pytest -m real --no-cov   # on the Mac with the engines installed
```

`tests/test_engine.py` runs a stand-in worker that speaks the real protocol module in a child process,
so the protocol, cancel/timeout, crashes, idle release and the output level are covered without a GPU.
The engine workers themselves are covered by the `real` tests.

## License

Apache-2.0 (this repository; it ships no weights and no engine code). Stable Audio 3's code is MIT; its
weights are under the [Stability AI Community License](https://huggingface.co/stabilityai/stable-audio-3-optimized/blob/main/LICENSE.md)
(free below US$1M annual revenue; commercial use requires registering with Stability AI; see the
license for attribution and other terms). MOSS-SoundEffect v2.0's code and weights are Apache-2.0
(its text encoder is Qwen3-1.7B, Apache-2.0). Check the license of every model you use.
