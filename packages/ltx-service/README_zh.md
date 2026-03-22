# ltx-service

`ltx-service` 是一个基于官方 `ltx-pipelines` 的独立 FastAPI 服务封装。

它会在进程内常驻一个官方 pipeline 实例，串行排队处理请求，把生成结果写入磁盘，并提供一个简单的 HTTP API 用于：

- 提交生成任务
- 健康检查
- 查询任务状态

与之前把服务直接塞进依赖包内部的做法不同，这个包把 `ltx-core` 和 `ltx-pipelines` 视为普通依赖，**不需要修改它们的源码**。

---

## 功能特性

- 独立服务包，位置在 `packages/ltx-service`
- 直接包装官方 `ltx_pipelines` pipeline 类
- 单进程内持久化保留一个 pipeline 实例
- 内部任务队列串行执行
- 图像条件输入支持本地路径、URL、base64 字符串和 multipart 上传文件
- 支持以下官方 pipeline 后端：
  - `ti2vid-one-stage`
  - `distilled`
  - `ti2vid-two-stages`
- 支持以下官方量化选项：
  - `fp8-cast`
  - `fp8-scaled-mm`

---

## 当前执行模型

`ltx-service` 当前运行的是**官方单设备 pipeline 路径**。

- `--execution-mode auto` 会解析为 `single`
- `--execution-mode sharded` 在这个独立包中会被明确拒绝

这是有意为之：`ltx-service` 的目标是做一个尽可能薄的服务封装层，而不是在上游 pipeline 之外再维护一套自定义多卡运行时逻辑。

---

## 必需模型文件

`ltx-service` **不会硬编码任何模型路径**。

你需要在启动服务时，通过参数传入对应路径。相关模型文件包括：

- full checkpoint：`ltx-2.3-22b-dev.safetensors`
- distilled checkpoint：`ltx-2.3-22b-distilled.safetensors`
- distilled LoRA：`ltx-2.3-22b-distilled-lora-384.safetensors`
- spatial upsampler：`ltx-2.3-spatial-upscaler-x2-1.1.safetensors`

另外，`--gamma-path` 也是启动时必须提供的参数。

---

## 安装方式

在仓库根目录执行：

```bash
python -m venv .venv
./.venv/bin/pip install -e packages/ltx-core -e packages/ltx-pipelines -e packages/ltx-service
```

如果你的环境已经准备好了，也可以只安装服务包本身：

```bash
./.venv/bin/pip install -e packages/ltx-service
```

---

## 启动服务

### 1）默认 two-stage 服务

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
  --gpu-ids 0 \
  --quantization fp8-cast \
  --output-dir outputs/ltx-service
```

### 2）one-stage 服务

```bash
CHECKPOINT_PATH="<full-checkpoint-path>"
GAMMA_PATH="<gamma-path>"

./.venv/bin/python -m ltx_service \
  --pipeline-type ti2vid-one-stage \
  --checkpoint-path "${CHECKPOINT_PATH}" \
  --gamma-path "${GAMMA_PATH}" \
  --execution-mode single \
  --gpu-ids 0 \
  --quantization fp8-cast \
  --output-dir outputs/ltx-service
```

### 3）distilled 服务

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
  --gpu-ids 0 \
  --quantization fp8-cast \
  --output-dir outputs/ltx-service
```

默认绑定地址：

- host：`127.0.0.1`
- port：`8000`

如果你希望从本机外访问，可以加上：

```bash
--host 0.0.0.0 --port 8000
```

---

## CLI 参数

查看完整参数：

```bash
./.venv/bin/python -m ltx_service --help
```

当前支持的参数包括：

- `--pipeline-type {ti2vid-one-stage,distilled,ti2vid-two-stages}`
- `--checkpoint-path`
- `--distilled-checkpoint-path`
- `--distilled-lora`
- `--spatial-upsampler-path`
- `--gamma-path`（启动时必传）
- `--quantization POLICY [AMAX_PATH ...]`
- `--output-dir`
- `--host`
- `--port`
- `--execution-mode {auto,single,sharded}`
- `--gpu-ids [GPU_IDS ...]`

说明：

- `auto` 当前会解析为 `single`
- `sharded` 在当前独立包中不支持
- `fp8-cast` 不接受额外参数
- `fp8-scaled-mm` 只有在当前安装的 `ltx-core` 版本支持时，才可以额外传一个可选的 amax 文件路径

---

## HTTP API

### 健康检查

```bash
curl http://127.0.0.1:8000/health
```

返回示例：

```json
{
  "status": "ok",
  "pipeline_loaded": false,
  "queue_depth": 0,
  "pipeline_type": "ti2vid-two-stages",
  "execution_mode": "single",
  "primary_device": "cuda:0",
  "gpu_ids": [0]
}
```

### 提交任务

`POST /v1/videos` 同时支持 `application/json` 和 `multipart/form-data`。

#### JSON 请求

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

#### 携带上传图片的 multipart 请求

把 JSON 放进 `payload` 表单字段，并用 `images[].source = "upload:<字段名>"` 引用同一个请求里的上传文件。

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

返回示例：

```json
{
  "id": "1748cae271254678b5698bbe130b3b04",
  "object": "video.generation",
  "status": "queued",
  "output_file_id": "1748cae271254678b5698bbe130b3b04"
}
```

### 查询任务状态

```bash
curl http://127.0.0.1:8000/v1/videos/<video-id>
```

终端轮询示例：

```bash
VIDEO_ID="1748cae271254678b5698bbe130b3b04"
watch -n 5 "curl -s http://127.0.0.1:8000/v1/videos/${VIDEO_ID}"
```

可能的状态：

- `queued`
- `running`
- `succeeded`
- `failed`

任务成功后，生成接口不会返回本地文件路径，而是返回 `output_file_id`。
如果任务最终是 `failed`，对应的 `/v1/files/{file_id}` 接口会返回 `410 Gone`，因为实际上没有输出视频文件。

### 查询文件元数据

```bash
curl http://127.0.0.1:8000/v1/files/<file-id>
```

### 下载生成视频

```bash
curl http://127.0.0.1:8000/v1/files/<file-id>/content --output result.mp4
```

---

## 请求体结构

`POST /v1/videos` 接受 `application/json` 和 `multipart/form-data`。其中 `payload` 里的 JSON 结构如下：

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

### 校验规则

- `height > 0`
- `width > 0`
- `height % 32 == 0`
- `width % 32 == 0`
- 对 `ti2vid-two-stages` 和 `distilled`，`height` 与 `width` 还必须能被 `64` 整除
- `num_frames > 0`
- `num_frames % 8 == 1`
- `frame_rate > 0`
- `num_inference_steps > 0`
- 图像条件输入里的 `frame_idx` 必须 `< num_frames`
- 每个 `images` 条目必须提供一个 `source` 字符串
- `source` 可以表示本地路径、远端 URL、base64 数据，或 `upload:<字段名>` 的 multipart 上传引用
- 远端 URL 必须解析到公网 IP；`localhost`、私网地址和链路本地地址会被拒绝
- `upload:<字段名>` 只能用于 `multipart/form-data` 请求，且必须对应实际上传的文件字段名

---

## 图像条件输入格式

`images` 中的每一项都必须包含一个统一的 `source` 字段，再加上通用条件参数。

通用字段：

- `frame_idx`：作用到哪个帧
- `strength`：条件强度
- `crf`：可选压缩质量，默认 `33`

`source` 支持的形式：

- 本地路径：已有本地图像路径（与旧版用法保持兼容）
- URL：远端 `http`/`https` 图像地址，且必须解析到公网 IP
- base64：原始 base64 图像数据，或 `data:image/...;base64,...` 数据 URL
- multipart 上传引用：`upload:<字段名>`，引用同一个请求里的上传文件

解析优先级：

1. `upload:<字段名>`
2. `http://` / `https://`
3. `data:image/...;base64,...`
4. 看起来像本地路径的字符串
5. 原始 base64 图像数据

示例：

### 本地路径

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

### Multipart 上传引用

```json
{
  "source": "upload:reference_image",
  "frame_idx": 0,
  "strength": 1.0,
  "crf": 33
}
```

对于非 `path` 来源，`ltx-service` 会在调用上游 pipeline 之前先把输入物化成当前任务专属的临时本地文件；任务结束后，这些临时输入会被自动清理。

---

## Pipeline 选择建议

### `ti2vid-one-stage`

适合：

- 想尽量薄地包装官方 one-stage 路径
- 单张 48GB 显卡已经够用
- 希望服务部署尽量简单
- 当前运行环境使用的是支持 CUDA 的 PyTorch

### `ti2vid-two-stages`

适合：

- 想使用官方质量更高的 two-stage 路径
- 需要 spatial upsampler 精修阶段
- 可以接受更高显存和更长耗时
- 当前运行环境使用的是支持 CUDA 的 PyTorch

### `distilled`

适合：

- 想使用官方 distilled 路径
- 希望 denoising 步数少一些
- 当前运行环境使用的是支持 CUDA 的 PyTorch

---

## 运行行为

`ltx-service` 每个进程只会持有一个 pipeline 实例。

- 第一个请求会承担模型加载开销
- 后续请求会复用同一个常驻 pipeline
- 所有任务**串行执行**
- 当前任务运行时，后续任务会停留在队列里

这是刻意设计的，因为 LTX 推理显存占用很高，在单设备上并发运行通常不稳定。

---

## 已验证示例

以下独立服务路径已经在当前工作区里真实验证通过：

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

## 开发与测试

运行测试：

```bash
./.venv/bin/python -m pytest packages/ltx-service/tests/test_service.py
```

查看 CLI：

```bash
./.venv/bin/python -m ltx_service --help
```

---

## 包结构

```text
packages/ltx-service/
├── pyproject.toml
├── README.md
├── README_zh.md
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

## 重要限制

这个包是刻意保守设计的：

- **不会** patch `ltx-core`
- **不会** patch `ltx-pipelines`
- **不会**在内部实现自定义多卡 runtime 手术式改造

如果你后续确实需要更激进的跨卡模型切分，建议继续在 `ltx-service` 这个独立包里演进，而不是回头修改上游依赖包。
