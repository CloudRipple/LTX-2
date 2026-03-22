# ltx-service

`ltx-service` is a standalone FastAPI wrapper around the official `ltx-pipelines` package.

It keeps one or more official single-device pipeline instances warm in memory, queues requests, writes generated media to disk, and exposes a small HTTP API for submission, health checks, and job polling.

Unlike the earlier in-package experiments, this package keeps `ltx-service` as the service entrypoint and applies only the minimal upstream `ltx-pipelines` hooks needed for runtime behavior such as progress reporting.

---

## Features

- standalone service package under `packages/ltx-service`
- wraps official `ltx_pipelines` pipeline classes directly
- persistent in-memory pipeline instance(s)
- shared internal queue with single-runner or multi-GPU worker execution
- image conditioning from local paths, URLs, base64 strings, and multipart file uploads
- optional stage-boundary GPU weight retention for `ti2vid-two-stages`, plus optional inter-stage GPU cache retention for `distilled`
- optional cross-request GPU model retention inside a single `ltx-service` process
- supports these official backends:
  - `ti2vid-one-stage`
  - `distilled`
  - `ti2vid-two-stages`
- supports official transformer quantization flags:
  - `fp8-cast`
  - `fp8-scaled-mm`

---

## Current execution model

`ltx-service` currently runs the **official single-device pipeline paths**.

- `--execution-mode auto` resolves to `single`
- `--execution-mode data-parallel` starts one official single-device runner per configured GPU and balances requests through one shared queue
- `--execution-mode sharded` is intentionally rejected in this standalone package

This is deliberate: the goal of `ltx-service` is to stay a thin wrapper around the upstream pipeline implementations instead of carrying custom cross-GPU runtime modifications.

---

## Required model artifacts

`ltx-service` does **not** hardcode model paths.

You should pass the required paths at startup. The relevant artifacts are:

- full checkpoint: `ltx-2.3-22b-dev.safetensors`
- distilled checkpoint: `ltx-2.3-22b-distilled.safetensors`
- distilled LoRA: `ltx-2.3-22b-distilled-lora-384.safetensors`
- spatial upsampler: `ltx-2.3-spatial-upscaler-x2-1.1.safetensors`

`--gamma-path` is also a required startup parameter.

---

## Installation

From the repository root:

```bash
python -m venv .venv
./.venv/bin/pip install -e packages/ltx-core -e packages/ltx-pipelines -e packages/ltx-service
```

If your environment is already prepared, installing just the standalone package is enough:

```bash
./.venv/bin/pip install -e packages/ltx-service
```

---

## Start the service

### 1) Default two-stage service

```bash
CHECKPOINT_PATH="<full-checkpoint-path>"
DISTILLED_LORA_PATH="<distilled-lora-path>"
SPATIAL_UPSAMPLER_PATH="<spatial-upsampler-path>"
GAMMA_PATH="<gamma-path>"

./.venv/bin/python -m ltx_service \
  --pipeline-type ti2vid-two-stages \
  --checkpoint-path "${CHECKPOINT_PATH}" \
  --distilled-lora "${DISTILLED_LORA_PATH}" \
  --spatial-upsampler-path "${SPATIAL_UPSAMPLER_PATH}" \
  --gamma-path "${GAMMA_PATH}" \
  --execution-mode single \
  --gpu-count 1 \
  --quantization fp8-cast \
  --output-dir outputs/ltx-service
```

### 2) One-stage service

```bash
CHECKPOINT_PATH="<full-checkpoint-path>"
GAMMA_PATH="<gamma-path>"

./.venv/bin/python -m ltx_service \
  --pipeline-type ti2vid-one-stage \
  --checkpoint-path "${CHECKPOINT_PATH}" \
  --gamma-path "${GAMMA_PATH}" \
  --execution-mode single \
  --gpu-count 1 \
  --quantization fp8-cast \
  --output-dir outputs/ltx-service
```

### 3) Distilled service

```bash
DISTILLED_CHECKPOINT_PATH="<distilled-checkpoint-path>"
SPATIAL_UPSAMPLER_PATH="<spatial-upsampler-path>"
GAMMA_PATH="<gamma-path>"

./.venv/bin/python -m ltx_service \
  --pipeline-type distilled \
  --distilled-checkpoint-path "${DISTILLED_CHECKPOINT_PATH}" \
  --spatial-upsampler-path "${SPATIAL_UPSAMPLER_PATH}" \
  --gamma-path "${GAMMA_PATH}" \
  --execution-mode single \
  --gpu-count 1 \
  --quantization fp8-cast \
  --output-dir outputs/ltx-service
```

### 4) Data-parallel two-stage service

```bash
CHECKPOINT_PATH="<full-checkpoint-path>"
DISTILLED_LORA_PATH="<distilled-lora-path>"
SPATIAL_UPSAMPLER_PATH="<spatial-upsampler-path>"
GAMMA_PATH="<gamma-path>"

./.venv/bin/python -m ltx_service \
  --pipeline-type ti2vid-two-stages \
  --checkpoint-path "${CHECKPOINT_PATH}" \
  --distilled-lora "${DISTILLED_LORA_PATH}" \
  --spatial-upsampler-path "${SPATIAL_UPSAMPLER_PATH}" \
  --gamma-path "${GAMMA_PATH}" \
  --execution-mode data-parallel \
  --gpu-count 2 \
  --quantization fp8-cast \
  --output-dir outputs/ltx-service
```

By default the service binds to:

- host: `127.0.0.1`
- port: `8000`

Use `--host 0.0.0.0 --port 8000` if you want to expose it outside localhost.

---

## CLI options

Show all options:

```bash
./.venv/bin/python -m ltx_service --help
```

Current options:

- `--pipeline-type {ti2vid-one-stage,distilled,ti2vid-two-stages}`
- `--checkpoint-path`
- `--distilled-checkpoint-path`
- `--distilled-lora`
- `--spatial-upsampler-path`
- `--gamma-path` (required at startup)
- `--quantization POLICY [AMAX_PATH ...]`
- `--output-dir`
- `--host`
- `--port`
- `--execution-mode {auto,single,data-parallel,sharded}`
- `--gpu-ids [GPU_IDS ...]`
- `--gpu-count`
- `--keep-stage-weights-on-gpu`
- `--keep-model-weights-on-gpu`

Notes:

- `auto` currently resolves to `single`
- `data-parallel` starts one service worker per selected GPU and balances requests through a shared FIFO queue
- `sharded` is not supported in this package
- `--gpu-ids` takes precedence over `--gpu-count`
- if `--gpu-ids` is omitted and `--gpu-count` is set, the service picks the first N visible GPUs automatically
- if neither `--gpu-ids` nor `--gpu-count` is set, the service auto-detects all visible GPUs
- `fp8-cast` takes no extra argument
- `fp8-scaled-mm` can take an optional amax file path only when the installed `ltx-core` version supports it
- `--keep-stage-weights-on-gpu` keeps stage weights resident for `ti2vid-two-stages`; for `distilled` it skips the inter-stage GPU cache cleanup because the same stage models are already reused
- `--keep-model-weights-on-gpu` caches model instances on GPU across requests in the same process, avoiding repeated weight loads until the service shuts down
- when `--keep-model-weights-on-gpu` is enabled, weight residency already spans requests, so `--keep-stage-weights-on-gpu` mainly affects whether extra inter-stage cache cleanup runs

---

## HTTP API

### Health check

```bash
curl http://127.0.0.1:8000/health
```

Example response:

```json
{
  "status": "ok",
  "pipeline_loaded": false,
  "queue_depth": 0,
  "pipeline_type": "ti2vid-two-stages",
  "execution_mode": "single",
  "primary_device": "cuda:0",
  "gpu_ids": [0],
  "worker_count": 1,
  "loaded_runner_count": 0
}
```

### Submit a job

`POST /v1/videos` accepts both `application/json` and `multipart/form-data`.

#### JSON request

```bash
curl -X POST http://127.0.0.1:8000/v1/videos \
  -H 'content-type: application/json' \
  -d '{
    "prompt": "A beautiful sunset over the ocean",
    "seed": 10,
    "height": 768,
    "width": 1280,
    "num_frames": 193,
    "frame_rate": 24.0,
    "num_inference_steps": 40,
    "enhance_prompt": true
  }'
```

#### Multipart request with uploaded image conditioning

Use a `payload` form field containing JSON, and reference uploaded files with `images[].source = "upload:<field-name>"`.

```bash
curl -X POST http://127.0.0.1:8000/v1/videos \
  -F 'payload={
    "prompt": "Animate this reference frame into a cinematic shot",
    "num_frames": 193,
    "images": [
      {
        "source": "upload:reference_image",
        "frame_idx": 0,
        "strength": 1.0
      }
    ]
  }' \
  -F 'reference_image=@/absolute/path/to/image.png;type=image/png'
```

Example response:

```json
{
  "id": "1748cae271254678b5698bbe130b3b04",
  "object": "video.generation",
  "status": "queued"
}
```

### Poll a job

```bash
curl http://127.0.0.1:8000/v1/videos/<video-id>
```

Example terminal loop:

```bash
VIDEO_ID="1748cae271254678b5698bbe130b3b04"
watch -n 5 "curl -s http://127.0.0.1:8000/v1/videos/${VIDEO_ID}"
```

Possible statuses:

- `queued`
- `running`
- `succeeded`
- `failed`

When a generation succeeds, the response exposes a single `id`. The same id is used for `/v1/videos/{id}` and `/v1/files/{id}`.
If a generation finishes with `failed`, the related `/v1/files/{file_id}` endpoints return `410 Gone` because no output video exists.

### Get file metadata

```bash
curl http://127.0.0.1:8000/v1/files/<file-id>
```

### Download the generated video

```bash
curl http://127.0.0.1:8000/v1/files/<file-id>/content --output result.mp4
```

---

## Request schema

`POST /v1/videos` accepts `application/json` and `multipart/form-data` payloads. The JSON body inside `payload` uses this schema:

```json
{
  "prompt": "string",
  "negative_prompt": "string",
  "seed": 10,
  "height": 768,
  "width": 1280,
  "num_frames": 193,
  "frame_rate": 24.0,
  "num_inference_steps": 40,
  "enhance_prompt": true,
  "images": [],
  "video_guidance": {
    "cfg_scale": 3.0,
    "stg_scale": 1.0,
    "rescale_scale": 0.7,
    "modality_scale": 3.0,
    "skip_step": 0,
    "stg_blocks": [28]
  },
  "audio_guidance": {
    "cfg_scale": 7.0,
    "stg_scale": 1.0,
    "rescale_scale": 0.7,
    "modality_scale": 3.0,
    "skip_step": 0,
    "stg_blocks": [28]
  }
}
```

### Validation rules

- `height > 0`
- `width > 0`
- `height % 32 == 0`
- `width % 32 == 0`
- for `ti2vid-two-stages` and `distilled`, `height` and `width` must also be divisible by 64
- `num_frames > 0`
- `num_frames % 8 == 1`
- `frame_rate > 0`
- `num_inference_steps > 0`
- image conditioning frame indices must be `< num_frames`
- each image entry must provide exactly one `source` string
- `source` can represent a local path, remote URL, base64 payload, or `upload:<field-name>` multipart reference
- remote URL sources must resolve to public IP addresses; localhost and private/link-local addresses are rejected
- `upload:<field-name>` is only valid for `multipart/form-data` requests and must match an uploaded file field name

---

## Image conditioning format

Each item in `images` must contain one unified `source` field plus the shared conditioning fields.

Shared fields:

- `frame_idx`: target frame index
- `strength`: conditioning strength
- `crf`: optional compression quality, default `33`

Supported `source` forms:

- local path: existing local image path (backward compatible with earlier requests)
- URL: remote `http`/`https` image URL that resolves to public IP addresses
- base64: raw base64 image data or a `data:image/...;base64,...` data URL
- multipart upload: `upload:<field-name>` referencing a file part from the same request

Resolution order:

1. `upload:<field-name>`
2. `http://` / `https://`
3. `data:image/...;base64,...`
4. local path-like strings
5. raw base64 image data

Examples:

### Local path

```json
{
  "source": "/absolute/path/to/image.png",
  "frame_idx": 0,
  "strength": 1.0,
  "crf": 33
}
```

### URL

```json
{
  "source": "https://www.baidu.com/img/flexible/logo/pc/result.png",
  "frame_idx": 0,
  "strength": 1.0,
  "crf": 33
}
```

### Base64

```json
{
  "source": "data:image/png;base64,iVBORw0KGgoAAAANSUhEUg...",
  "frame_idx": 0,
  "strength": 1.0,
  "crf": 33
}
```

### Multipart upload reference

```json
{
  "source": "upload:reference_image",
  "frame_idx": 0,
  "strength": 1.0,
  "crf": 33
}
```

For non-path sources, `ltx-service` materializes the image into a temporary job-scoped local file before calling the upstream pipeline, then removes those temporary inputs after the job finishes.

---

## Pipeline selection guide

### `ti2vid-one-stage`

Use when:

- you want the thinnest wrapper over the official one-stage path
- a single 48GB GPU is enough
- you want the simplest service deployment
- you are running with CUDA-enabled PyTorch

### `ti2vid-two-stages`

Use when:

- you want the official higher-quality two-stage path
- you need the spatial upsampler refinement stage
- you are willing to spend more VRAM and latency
- you are running with CUDA-enabled PyTorch
- you want to optionally keep stage weights resident on GPU between stages

### `distilled`

Use when:

- you want the official distilled path
- you want fewer denoising steps than the full model
- you are running with CUDA-enabled PyTorch
- you want to optionally skip inter-stage GPU cache cleanup while reusing the same stage models

---

## Runtime behavior

`ltx-service` keeps one pipeline instance per worker alive in each process.

- the first request pays model-load cost
- later requests reuse the same in-memory pipeline on that worker
- `single` mode processes jobs **serially**
- `data-parallel` mode runs one worker per selected GPU and pulls jobs from the same FIFO queue
- if all workers are busy, later requests remain queued

This is intentional because LTX inference is VRAM-heavy: `single` mode stays conservative on one device, while `data-parallel` scales out by replicating official single-device runners across GPUs.

When the service receives `Ctrl+C`, it first gives shutdown a short chance to release runners and cached weights. If the process is still stuck in in-flight work after that window, it force-exits so the service does not hang indefinitely.

During server-side inference, `ltx-service` suppresses per-worker tqdm progress bars and emits aggregated `info` logs instead, so one log line can show the current progress of all active workers at once.

---

## Verified examples

The following standalone service path has been verified in this workspace:

- `ti2vid-two-stages`
- `seed=10`
- `height=768`
- `width=1280`
- `num_frames=193`
- `frame_rate=24.0`
- `num_inference_steps=40`
- `enhance_prompt=true`
- `quantization=fp8-cast`

---

## Development

Run tests:

```bash
./.venv/bin/python -m pytest packages/ltx-service/tests/test_service.py
```

Check CLI:

```bash
./.venv/bin/python -m ltx_service --help
```

---

## Package layout

```text
packages/ltx-service/
├── pyproject.toml
├── README.md
├── src/ltx_service/
│   ├── __init__.py
│   ├── __main__.py
│   ├── app.py
│   ├── backend.py
│   ├── config.py
│   └── models.py
└── tests/
    └── test_service.py
```

---

## Important limitation

This package is intentionally conservative:

- it does **not** patch `ltx-core`
- it does **not** patch `ltx-pipelines`
- it does **not** implement custom multi-GPU runtime surgery

If you need aggressive cross-GPU model partitioning, that should live in a separate package evolution of `ltx-service`, not in upstream dependency packages.
