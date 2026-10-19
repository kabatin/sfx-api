# sfx-api HTTP API (v1)

Request and response bodies are JSON (UTF-8) unless noted. The machine-readable schema is served at
`/openapi.json`; interactive docs at `/docs`. The shapes deliberately match
[ace-step-api](https://github.com/kabatin/ace-step-api), [klein-api](https://github.com/kabatin/klein-api)
and [vpipe-api](https://github.com/kabatin/vpipe-api) (same job lifecycle, error envelope,
`429` + `Retry-After`, `Idempotency-Key`), so a client that drives them needs only a different
workflow id, body and output type.

Four workflows on three engines: three for sound effects and one for instrumental music.

| workflow | engine | output |
|---|---|---|
| `stable-audio-3-medium-sfx` (recommended) | [Stable Audio 3](https://github.com/Stability-AI/stable-audio-3) medium, Stability AI's MLX build; the server tags the prompt `TrackType: SFX` | WAV, 16-bit, 44.1 kHz, stereo |
| `stable-audio-3-medium-music` | the same Stable Audio 3 medium engine; the server tags the prompt `TrackType: Music, VocalType: Instrumental` | WAV, 16-bit, 44.1 kHz, stereo, up to 190 s |
| `stable-audio-3-sfx` | [Stable Audio 3](https://github.com/Stability-AI/stable-audio-3) sm-sfx, Stability AI's MLX build | WAV, 16-bit, 44.1 kHz, stereo |
| `moss-sfx-v2` | [MOSS-SoundEffect v2.0](https://huggingface.co/OpenMOSS-Team/MOSS-SoundEffect-v2.0), PyTorch on MPS | WAV, 16-bit, 48 kHz, mono |

**Nothing about the engines (model files, steps, guidance) is part of this contract.**

## Authentication

Identical to ace-step-api:

- If the server has a token configured (`SFX_API_TOKEN`, ≥ 32 printable ASCII characters), **every**
  request — including `/healthz`, `/docs` and `/openapi.json` — must send `Authorization: Bearer <token>`;
  otherwise `401`.
- A server bound to a non-loopback address refuses to start without a token.
- Without a token the server only answers requests whose `Host` is `127.0.0.1`, `localhost` or `[::1]`
  and that carry no proxy headers (`Forwarded`, `X-Forwarded-*`, `X-Real-IP`).
- There is no TLS. On a LAN the token, the prompts and the audio travel in clear text.

## Error envelope

```json
{ "error": { "code": "busy", "message": "human readable", "retryable": true, "details": null } }
```

| HTTP | `code` | `retryable` | When |
|---|---|---|---|
| 401 | `unauthorized` | false | missing / wrong bearer token |
| 403 | `forbidden_host` / `proxy_requires_token` | false | token-less server reached via a foreign Host or a proxy |
| 404 | `not_found` | false | unknown workflow, job, output index or path |
| 409 | `conflict` | false | output requested before the job succeeded, cancel of a finished job |
| 409 | `idempotency_conflict` | false | `Idempotency-Key` reused with different params |
| 409 | `idempotency_in_flight` | true | another request with the same key is being accepted right now |
| 411 | `length_required` | false | any `Transfer-Encoding` — send a plain `Content-Length` |
| 413 | `payload_too_large` | false | request body over `max_body_mb` (default 64 MB) |
| 422 | `invalid_params` | false | params failed validation; `details` lists `{loc, msg}` (inputs are never echoed) |
| 429 | `busy` | true | the GPU slot and the waiting queue are full. `Retry-After` header (seconds) is set |
| 500 | `internal` | true | unexpected server error (details only in the server log) |
| 503 | `workflow_unavailable` | false | this workflow's engine is not installed or misses model files on this server; `message` says why. The other workflow may still work (see `/v1/health`) |
| 503 | `insufficient_storage` | true | less than 2 GB free in `data_dir`; nothing was queued |

## Endpoints

### `GET /healthz` and `GET /v1/health`

```json
{
  "status": "ok", "version": "0.1.0", "running": 0, "waiting": 0, "max_waiting": 2,
  "waiting_for_peer": 0,
  "workflows": {
    "stable-audio-3-sfx": { "available": true, "loaded": true, "reason": null },
    "stable-audio-3-medium-sfx": { "available": true, "loaded": false, "reason": null },
    "moss-sfx-v2": { "available": true, "loaded": false, "reason": null }
  }
}
```

- `running` / `waiting` / `waiting_for_peer`: as in ace-step-api (see *Coordination*).
- `workflows.<id>.available`: its engine is installed with its model files, so jobs are accepted.
  When `false`, `reason` says why and a submit answers `503 workflow_unavailable`.
- `workflows.<id>.loaded`: its models are in memory right now. **`available: true, loaded: false` is
  the normal idle state**: the next job loads them (see `typical_seconds` and *Timings*).
- `503` with `"status": "degraded"` if the server's job worker has stopped.

### `GET /v1/capabilities`

What a client may ask for, without engine details. **This is the source of truth for client
descriptors** — a client must not assume values that are not listed here.

```json
{
  "version": "0.1.0",
  "queue": { "max_running": 1, "max_waiting": 2 },
  "workflows": [{
    "id": "stable-audio-3-sfx",
    "output": { "media_type": "audio/wav", "sample_rate": 44100, "channels": 2, "sample_format": "pcm_s16le" },
    "prompt": { "max_chars": 500, "language": "en" },
    "negative_prompt": { "max_chars": 500, "applied": false, "note": "…" },
    "duration_sec": { "min": 0.5, "max": 10.0, "default": 2.0 },
    "count": { "min": 1, "max": 4 },
    "seed": { "min": 0, "max": 4294967295, "per_clip": "seed + 2i" },
    "typical_seconds": { "2s_clip": 1.1 }
  }, {
    "id": "stable-audio-3-medium-sfx",
    "prompt": { "max_chars": 500, "language": "en", "prefix": "TrackType: SFX, " },
    "…": "…"
  }, {
    "id": "stable-audio-3-medium-music",
    "prompt": { "max_chars": 1000, "language": "en", "prefix": "TrackType: Music, VocalType: Instrumental, " },
    "duration_sec": { "min": 0.5, "max": 190.0, "default": 30.0 },
    "…": "…"
  }, { "id": "moss-sfx-v2", "…": "…" }],
  "coordination": { "waits_for": ["vpipe-api", "klein-api", "ace-step-api"] }
}
```

`typical_seconds` is per 2-second clip with the models already loaded, and only a hint.
`negative_prompt.applied` says whether that workflow's engine uses the negative prompt at all.
`prompt.prefix`, when present, is put in front of the prompt by the server (see the workflow).
`prompt.max_chars` and `duration_sec` differ per workflow (the music workflow takes longer ones).

### `GET /v1/workflows`

Lists workflows with the JSON Schema of their POST body (`params_schema`).

### `POST /v1/workflows/{workflow}/jobs`

`{workflow}` is `stable-audio-3-medium-sfx`, `stable-audio-3-medium-music`, `stable-audio-3-sfx` or
`moss-sfx-v2`; all take the same body (the music workflow allows a longer prompt and length),
`Content-Type: application/json`. Optional header `Idempotency-Key: <1–128 of [A-Za-z0-9._:-]>`.

```json
{ "prompt": "tyre screech on wet asphalt, close, short", "negative_prompt": "music, speech",
  "duration_sec": 2.5, "count": 2, "seed": 1234 }
```

| field | type | default | notes |
|---|---|---|---|
| `prompt` | string, 1–500 chars (1–1000 for `stable-audio-3-medium-music`) | — | English description of the sound. No control characters (tab/newline are fine). |
| `negative_prompt` | string, 0–500 chars | `""` | What the sound should not contain. Used by `moss-sfx-v2`; the Stable Audio 3 workflows accept it and ignore it unless the server enables guidance for them (`capabilities` → `negative_prompt.applied`). |
| `duration_sec` | number 0.5–10 (0.5–190 for the music workflow) | `2.0` (`30.0` for music) | Requested length. The real length of each clip is reported in the result. |
| `seed` | int 0 … 2³²−1 \| null | random | Clip *i* (0-based) uses **`seed + 2·i`**; `seed + 2·(count − 1)` must not exceed 2³²−1. |
| `count` | int 1–4 | `1` | Number of clips. They run one after another inside the one job. |

Anything else in the body — model names, steps, guidance, file paths — is rejected with `422`.

**Seeds.** Clips are two seeds apart because Stable Audio 3 draws its starting noise from `seed` and
its sampler noise from `seed + 1`: consecutive seeds would share randomness. To make one clip again,
send its `seed_used` with `count: 1`. With `stable-audio-3-sfx` and `stable-audio-3-medium-sfx` the
same seed and prompt give the same clip (verified bit-identical); `moss-sfx-v2` draws its noise on the
CPU from the seed as well.

Success is `202 Accepted`:

```json
{ "id": "job_01J9ZK3N2Q8V7W6X5Y4Z3A2B1C", "workflow": "stable-audio-3-sfx", "status": "queued", "created_at": "2026-10-18T02:00:00Z" }
```

`429 busy`: nothing was queued; try again after `Retry-After` seconds. Answered before the body is read.
`503 workflow_unavailable`: that workflow cannot run on this server; nothing was queued.

**Idempotency** works exactly as in ace-step-api: same key + same params → the existing job with
`200 OK`; same key + different params → `409 idempotency_conflict`; concurrent → `409 idempotency_in_flight`.

### `GET /v1/jobs/{job_id}`

```json
{
  "id": "job_…",
  "workflow": "stable-audio-3-sfx",
  "status": "succeeded",
  "progress": 1.0,
  "queue_position": null,
  "waiting_for": null,
  "created_at": "…", "started_at": "…", "finished_at": "…",
  "estimate_seconds": 7.2,
  "cancel_requested": false,
  "timings": { "queue_seconds": 0.1, "coordination_seconds": 0.0, "backend_seconds": 3.9,
               "load_seconds": 0.9, "postprocess_seconds": 0.0, "total_seconds": 4.0 },
  "result": {
    "outputs": [
      { "index": 0, "media_type": "audio/wav", "duration_sec": 2.5, "sample_rate": 44100,
        "channels": 2, "seed_used": 1234 },
      { "index": 1, "media_type": "audio/wav", "duration_sec": 2.5, "sample_rate": 44100,
        "channels": 2, "seed_used": 1236 }
    ],
    "seed_used": 1234,
    "details": { "generation": { "requested_duration_sec": 2.5,
                                 "prompt_used": "tyre screech on wet asphalt, close, short",
                                 "negative_prompt_applied": false,
                                 "engine": { "load_seconds": 0.9, "clips": [
                                   { "index": 0, "seconds": 2.6, "gain_db": 0.0, "cfg": 1.0 },
                                   { "index": 1, "seconds": 0.4, "gain_db": -1.2, "cfg": 1.0 } ] } } }
  },
  "error": null
}
```

- `status`: `queued` → `running` → `succeeded` | `failed` | `canceled` — **one "l"** (as in
  ace-step-api, klein-api, vpipe-api and wan-api; `cancelled` never appears).
- `running` only while the job owns the GPU slot. A job that is first in line but is waiting for another
  GPU user on the machine (see *Coordination*) stays `queued` with `waiting_for: "vpipe-api"`.
- `progress`: `0.0..0.99` while running, over all clips of the job; `1.0` when succeeded; `null` when
  unknown.
- `queue_position`: 1-based while `queued`, else `null`.
- `result.seed_used` is the seed of clip 0; each output has its own `seed_used`.
- `result.outputs[N].sample_rate` / `channels` are read back from the delivered WAV.
- `timings.load_seconds` is present only when the job had to load the models.
- `details` is informational and may change; clients should not depend on it.
- `error` (only when `failed`): `{code, message, retryable}`; e.g. `generation_failed`, `timeout`,
  `invalid_output`, `backend_unavailable`, `engine_failed`, `server_restarted`, `server_stopped`.
- Restart behaviour is identical to ace-step-api (`server_stopped` / `server_restarted`, retryable;
  queued jobs run again).

### `GET /v1/jobs/{job_id}/output` and `GET /v1/jobs/{job_id}/output?index=N`

The WAV (`audio/wav`) of clip `N` (default `0`). `409 conflict` unless the job has succeeded,
`404 not_found` for an index outside `0 … count-1`. Outputs are kept for `retention_days` (default 7).

Every output is checked before the job succeeds: a readable 16-bit PCM WAV with the workflow's sample
rate and channels, not empty. Its real length is `result.outputs[N].duration_sec`.

### `DELETE /v1/jobs/{job_id}`

As in ace-step-api: `200` `canceled`, `202` while the engine is stopping, `409` if already finished.

## Coordination with other GPU servers

The Mac has one GPU and one pool of unified memory. When `coordinate_with` is configured (a list of
`{name, health_url, token}`), the job at the head of the queue starts only after a check in which
**every** listed server reports idle (`running == 0 && waiting == 0`, and `outside_runs == 0` when
present), with the same `settle_seconds` rule as ace-step-api. While waiting the job stays `queued`
with `waiting_for` set. An unreachable server counts as idle (logged as a warning). Coordination is
checked only before a job starts; a running job is never paused.

Servers that list each other cannot deadlock: while sfx-api's head job waits for a peer,
`/v1/health` reports the queued jobs as `waiting_for_peer` and `waiting: 0`, so the peer sees it idle
and goes first. vpipe-api never waits for anyone; sfx-api waits for it.

**Memory.** Between jobs each engine keeps its models loaded for `resident_idle_seconds` (default
300). An engine is released at once when a listed server is seen busy while it idles, after any job of
it that did not succeed, and on shutdown.

## Workflows

All: peaks are kept under −1 dBFS before the clip is written as 16-bit PCM. A clip whose peak is
higher is turned down as a whole (no limiter, so the attack of an impact keeps its shape); quieter
clips are left exactly as the engine made them. The gain is `gain_db` in
`result.details.generation.engine.clips[]`; a few dB is normal for loud sounds, while −20 dB or lower
means the engine's output went far past full scale and the clip is most likely noise.

### `stable-audio-3-medium-sfx` (recommended)

- Stable Audio 3 medium (1.4 B), 8 steps, no guidance. 44.1 kHz stereo, the requested length made
  directly (92 clips at fractional lengths from 0.7 to 9.5 s stayed in range, unlike sm-sfx). Same
  params, limits and seed rule as the others.
- The server puts `TrackType: SFX, ` in front of the prompt, as Stable Audio 3's own prompt
  assistant does for sound effects; do not add it yourself. A prompt that already starts with a
  `TrackType:` tag is sent as it is. `result.details.generation.prompt_used` shows what was sent,
  and `capabilities` → `prompt.prefix` names the tag.
- `negative_prompt` is ignored by default (`applied: false`), as with `stable-audio-3-sfx`.
- About 1.1 s per clip for 1–2 s of sound and 1.7 s for 10 s once loaded; loading takes ~1 s
  (longer when the files are not in the OS cache). Peak memory ~11 GB.
- Chosen for 8 of 9 sounds of a car-chase trailer in a blind listening test against sm-sfx, with and
  without the tag.
- Weights: [Stability AI Community License](https://huggingface.co/stabilityai/stable-audio-3-optimized/blob/main/LICENSE.md).

### `stable-audio-3-medium-music`

- Instrumental music from the same Stable Audio 3 medium engine (8 steps, no guidance), so it loads
  no other model; music and `stable-audio-3-medium-sfx` jobs share one queue, and a long music job
  (up to 4 × 190 s) holds it for a while before the next sound effect starts. 44.1 kHz stereo.
- The server puts `TrackType: Music, VocalType: Instrumental, ` in front of the prompt, the tag
  Stable Audio 3's prompting guide recommends for instrumental music; a prompt that already starts
  with a `TrackType:` tag is sent as it is (so a prompt with its own tag also replaces the
  `VocalType: Instrumental` part). The model makes no intelligible vocals or lyrics either way; use
  [ace-step-api](https://github.com/kabatin/ace-step-api) for songs.
- `prompt` up to 1000 characters. The text encoder reads the first 256 tokens of the tagged prompt;
  a 741-character trailer brief came to 174.
- `duration_sec` 0.5–190 (default 30). 190 s is the longest length Stability AI evaluated in its
  paper; the model's own interface goes to 380 s.
- `negative_prompt` is ignored by default (`applied: false`), as with the other Stable Audio 3
  workflows. The same seed and prompt give the same track.
- About 3.4 s for a 32 s track, 5.3 s for 62 s, 10.3 s for 120 s and 15.8 s for 190 s once loaded
  (M5 32 GB). Peak memory ~11 GB for a 62 s track and up to ~14 GB for the longest.
- In a blind listening test on four instrumental briefs (a trailer cue, an anime-opening style
  track, lo-fi piano, lofi hip hop) against ACE-Step 1.5 turbo, it was picked for three; ACE-Step
  won the lo-fi piano.
- Weights: [Stability AI Community License](https://huggingface.co/stabilityai/stable-audio-3-optimized/blob/main/LICENSE.md).

### `stable-audio-3-sfx`

- Stable Audio 3 sm-sfx (the sound-effects model), 8 steps, no guidance (the model is distilled for
  that). 44.1 kHz stereo.
- sm-sfx only knows whole seconds: it was trained on lengths rounded up to the second, and its
  demos offer 1, 2, 3 … s. Each clip is generated at the next whole second (0.7 s → 1 s, 1.4 s → 2 s)
  and cut to the requested length. Generated at the fractional length itself, most lengths (0.9,
  1.2, 2.8 s …) came out many times over full scale and sounded like noise, whatever the prompt.
- `negative_prompt` is ignored by default (`applied: false`): it only acts through guidance, which
  doubles the cost and changes the distilled model's sound. The server can enable it.
- Weights: [Stability AI Community License](https://huggingface.co/stabilityai/stable-audio-3-optimized/blob/main/LICENSE.md).

### `moss-sfx-v2`

- MOSS-SoundEffect v2.0 with guidance on, so `negative_prompt` is used. 48 kHz mono.
- Much slower than `stable-audio-3-sfx` (about 90 s per clip on an M5 32 GB, plus ~16 s to load;
  see `typical_seconds`): the model always denoises a 30-second latent and cuts the requested length
  out of it, so a 0.5 s clip costs as much as a 10 s one. A shorter latent is not offered: it broke
  the model (only ~1.5 s of sound, then silence).
- Loudness varies from clip to clip. Most come out near full scale and are turned down to the −1 dBFS
  ceiling; some come out quieter (one of eight in the check peaked at −11 dBFS) and stay as they are.
  Even out loudness on the client if clips are used side by side.
- Weights: Apache-2.0.

## Writing prompts

The prompt is what shapes the sound; `seed` only picks a variation of it. Change the wording first.

- Write one or two plain English sentences about one scene: the source, what it does, the material,
  the space (close or distant, dry or reverberant), the attack and decay, and any motion (approaching,
  passing by). Stable Audio 3's own prompt assistant asks for exactly this
  ([`reprompt.py`](https://github.com/Stability-AI/stable-audio-3/blob/main/stable_audio_3/interface/reprompt.py)).
  Its examples: "Car speeding past at high velocity, doppler effect, realistic whoosh", "Punch impact
  with deep bass hit, cinematic trailer style", "Heavy rain on a tin roof with distant thunder rolls".
- Do not write the length into the prompt; send `duration_sec`. MOSS adds its own duration tag to the
  prompt, and Stable Audio 3's interface strips "Length: …" before the model sees it.
- Keep clips short: Stable Audio 3's prompting guide recommends a short duration; its assistant
  suggests 1–3 s for impacts and 3–6 s for actions.
- Stable Audio 3: its [prompting guide](https://github.com/Stability-AI/stable-audio-3/blob/main/docs/guides/prompting.md)
  also asks for the recording and processing (mic placement, room, effects), and lists a
  `TrackType: SFX` prefix as a tag that "tends to produce more semantically reasonable sound effects".
  `stable-audio-3-medium-sfx` adds it for you; `stable-audio-3-sfx` sends the prompt as it is.
  `negative_prompt` has no effect by default (see above).
- Music (`stable-audio-3-medium-music`): describe the genre, instruments, mood, tempo (BPM) and
  how the track develops, as Stable Audio 3's prompting guide does for music ("lo-fi piano, calm",
  "anime opening, j-pop instrumental, cinematic orchestral pop, 128 bpm, …"). Sections can be placed
  in time ("a breakdown at around 38 seconds"); whether the model keeps to them is not checked.
- MOSS-SoundEffect v2.0 was trained on English and Chinese captions; its examples are full sentences
  ("The crisp, rhythmic click-clack of fast typing on a mechanical keyboard."). This server accepts
  English prompts.
