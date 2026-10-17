# curl examples

```sh
API=http://127.0.0.1:8769
AUTH=()                                   # or: AUTH=(-H "Authorization: Bearer $SFX_API_TOKEN")
```

## Is it up? Which workflows can run?

```sh
curl -s "${AUTH[@]}" "$API/v1/health" | jq
# {"status":"ok", …, "workflows":{"stable-audio-3-sfx":{"available":true,"loaded":false,"reason":null},
#                                  "moss-sfx-v2":{"available":true,"loaded":false,"reason":null}}}
curl -s "${AUTH[@]}" "$API/v1/capabilities" | jq '.workflows[] | {id, output, duration_sec, negative_prompt, typical_seconds}'
```

`loaded: false` is the normal idle state: the next job loads that engine's models.

## Two takes of a tyre screech (Stable Audio 3)

```sh
curl -s "${AUTH[@]}" -X POST "$API/v1/workflows/stable-audio-3-sfx/jobs" \
  -H 'Content-Type: application/json' -H 'Idempotency-Key: sfx-001' \
  -d '{"prompt": "tyre screech on wet asphalt, close, short", "duration_sec": 2.5, "count": 2, "seed": 1234}'
# → 202 {"id":"job_…","workflow":"stable-audio-3-sfx","status":"queued","created_at":"…"}
# Sending the same Idempotency-Key + body again returns the same job (200), never a duplicate.
```

Clip *i* uses `seed + 2i` (1234, 1236). To make clip 1 again: `"seed": 1236, "count": 1`.

## With a negative prompt (MOSS-SoundEffect)

```sh
curl -s "${AUTH[@]}" -X POST "$API/v1/workflows/moss-sfx-v2/jobs" \
  -H 'Content-Type: application/json' \
  -d '{"prompt": "heavy deep boom impact, cinematic hit", "negative_prompt": "music, speech", "duration_sec": 2}'
```

`moss-sfx-v2` is much slower (see `typical_seconds`) and returns 48 kHz mono.

## Poll, download, cancel

```sh
JOB=job_…
curl -s "${AUTH[@]}" "$API/v1/jobs/$JOB" | jq '{status, progress, queue_position, waiting_for, timings, error}'
curl -s "${AUTH[@]}" "$API/v1/jobs/$JOB" | jq '.result.outputs[] | {index, duration_sec, sample_rate, channels, seed_used}'
curl -s "${AUTH[@]}" -o clip-0.wav "$API/v1/jobs/$JOB/output"           # index 0; 409 until succeeded
curl -s "${AUTH[@]}" -o clip-1.wav "$API/v1/jobs/$JOB/output?index=1"   # 404 outside 0 … count-1
curl -s "${AUTH[@]}" -X DELETE "$API/v1/jobs/$JOB"                       # 200 canceled / 202 stopping
```

## A workflow this server cannot run

```
HTTP/1.1 503 Service Unavailable
{"error":{"code":"workflow_unavailable","message":"moss-sfx-v2 cannot run on this server: model files missing: …","retryable":false,"details":null}}
```

Nothing was queued; `/v1/health` → `workflows` shows the same reason.

## Waiting for another GPU server

With `[[coordinate_with]]` configured, the first job in line stays queued while e.g. vpipe-api is busy:

```json
{ "status": "queued", "queue_position": 1, "waiting_for": "vpipe-api", "progress": null }
```

Meanwhile `/v1/health` reports the queue as `waiting_for_peer`, not `waiting`, so servers that wait
for sfx-api go first instead of waiting for each other.

## Handling `429 busy`

One job runs and `max_waiting` (default 2) wait. When full, the server answers before reading the body:

```
HTTP/1.1 429 Too Many Requests
Retry-After: 30
{"error":{"code":"busy","message":"the GPU slot and the waiting queue are full","retryable":true,"details":null}}
```

Wait `Retry-After` seconds and submit again. Nothing was queued.
