import asyncio
import base64
from importlib import import_module
import json
from email.message import Message
from pathlib import Path
import threading
import time
from types import SimpleNamespace

import pytest
import torch
from fastapi.testclient import TestClient
from pydantic import ValidationError

backend_module = import_module("ltx_service.backend")
create_app = import_module("ltx_service.app").create_app
JobRecord = import_module("ltx_service.backend").JobRecord
PipelineRunner = import_module("ltx_service.backend").PipelineRunner
PipelineServiceBackend = import_module("ltx_service.backend").PipelineServiceBackend
ExecutionMode = import_module("ltx_service.config").ExecutionMode
parse_service_config = import_module("ltx_service.config").parse_service_config
ServiceConfig = import_module("ltx_service.config").ServiceConfig
ServingPipelineType = import_module("ltx_service.config").ServingPipelineType
GenerateJobRequest = import_module("ltx_service.models").GenerateJobRequest
JobStatus = import_module("ltx_service.models").JobStatus


SAMPLE_PNG_BYTES = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO7ZC6sAAAAASUVORK5CYII="
)


class _FakeURLResponse:
    def __init__(self, content: bytes, content_type: str) -> None:
        self._content = content
        self.headers = Message()
        self.headers["Content-Type"] = content_type

    def read(self) -> bytes:
        return self._content

    def __enter__(self) -> "_FakeURLResponse":
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        return False


def test_service_config_auto_prefers_single_when_gpus_are_visible(monkeypatch) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 2)

    config = _make_test_config(Path("/tmp/ltx-service-test"), execution_mode=ExecutionMode.AUTO)

    assert config.visible_gpu_ids() == (0, 1)
    assert config.resolved_execution_mode() is ExecutionMode.SINGLE
    assert config.primary_device() == torch.device("cuda:0")


def test_official_backend_requires_cuda(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    with pytest.raises(ValueError, match="require CUDA-enabled PyTorch"):
        PipelineServiceBackend(config=_make_test_config(tmp_path))


def test_parse_service_config_accepts_fp8_scaled_mm_without_amax_path(monkeypatch, tmp_path: Path) -> None:
    sentinel_policy = object()

    class DummyQuantizationPolicy:
        @classmethod
        def fp8_scaled_mm(cls):
            return sentinel_policy

        @classmethod
        def fp8_cast(cls):
            return object()

    class DummyQuantizationModule:
        QuantizationPolicy = DummyQuantizationPolicy

    monkeypatch.setattr(import_module("ltx_service.config"), "_quantization_module", lambda: DummyQuantizationModule)

    config = parse_service_config(
        _required_service_argv(tmp_path)
        + [
            "--quantization",
            "fp8-scaled-mm",
        ]
    )

    assert config.quantization is sentinel_policy


def test_parse_service_config_enables_keep_stage_weights_on_gpu(tmp_path: Path) -> None:
    config = parse_service_config(_required_service_argv(tmp_path) + ["--keep-stage-weights-on-gpu"])

    assert config.keep_stage_weights_on_gpu is True


def test_parse_service_config_enables_keep_model_weights_on_gpu(tmp_path: Path) -> None:
    config = parse_service_config(_required_service_argv(tmp_path) + ["--keep-model-weights-on-gpu"])

    assert config.keep_model_weights_on_gpu is True


@pytest.mark.parametrize(
    ("pipeline_type", "expected_constructor"),
    [
        (ServingPipelineType.TI2VID_ONE_STAGE, "one_stage"),
        (ServingPipelineType.DISTILLED, "distilled"),
        (ServingPipelineType.TI2VID_TWO_STAGES, "two_stage"),
    ],
)
def test_build_default_runner_passes_keep_stage_weights_flag(monkeypatch, tmp_path: Path, pipeline_type, expected_constructor) -> None:
    seen: dict[str, dict[str, object]] = {}

    class DummyPipeline:
        def __init__(self, **kwargs):
            seen[expected_constructor] = kwargs

    class DummyLora:
        def __init__(self, *args):
            self.args = args

    modules = {
        "ltx_core.loader": SimpleNamespace(
            LTXV_LORA_COMFY_RENAMING_MAP=object(),
            LoraPathStrengthAndSDOps=DummyLora,
        ),
        "ltx_pipelines.ti2vid_one_stage": SimpleNamespace(TI2VidOneStagePipeline=DummyPipeline),
        "ltx_pipelines.distilled": SimpleNamespace(DistilledPipeline=DummyPipeline),
        "ltx_pipelines.ti2vid_two_stages": SimpleNamespace(TI2VidTwoStagesPipeline=DummyPipeline),
    }

    monkeypatch.setattr(backend_module.importlib, "import_module", lambda name: modules[name])

    config = _make_test_config(tmp_path, pipeline_type=pipeline_type, keep_stage_weights_on_gpu=True)
    runner = backend_module.build_default_runner(
        config,
        execution_mode=ExecutionMode.SINGLE,
        gpu_ids=(0,),
        primary_device=torch.device("cuda:0"),
    )

    assert runner is not None
    assert seen[expected_constructor]["keep_stage_weights_on_gpu"] is True


@pytest.mark.parametrize(
    ("pipeline_type", "expected_constructor"),
    [
        (ServingPipelineType.TI2VID_ONE_STAGE, "one_stage"),
        (ServingPipelineType.DISTILLED, "distilled"),
        (ServingPipelineType.TI2VID_TWO_STAGES, "two_stage"),
    ],
)
def test_build_default_runner_passes_keep_model_weights_flag(monkeypatch, tmp_path: Path, pipeline_type, expected_constructor) -> None:
    seen: dict[str, dict[str, object]] = {}

    class DummyPipeline:
        def __init__(self, **kwargs):
            seen[expected_constructor] = kwargs

    class DummyLora:
        def __init__(self, *args):
            self.args = args

    modules = {
        "ltx_core.loader": SimpleNamespace(
            LTXV_LORA_COMFY_RENAMING_MAP=object(),
            LoraPathStrengthAndSDOps=DummyLora,
        ),
        "ltx_pipelines.ti2vid_one_stage": SimpleNamespace(TI2VidOneStagePipeline=DummyPipeline),
        "ltx_pipelines.distilled": SimpleNamespace(DistilledPipeline=DummyPipeline),
        "ltx_pipelines.ti2vid_two_stages": SimpleNamespace(TI2VidTwoStagesPipeline=DummyPipeline),
    }

    monkeypatch.setattr(backend_module.importlib, "import_module", lambda name: modules[name])

    config = _make_test_config(tmp_path, pipeline_type=pipeline_type, keep_model_weights_on_gpu=True)
    runner = backend_module.build_default_runner(
        config,
        execution_mode=ExecutionMode.SINGLE,
        gpu_ids=(0,),
        primary_device=torch.device("cuda:0"),
    )

    assert runner is not None
    assert seen[expected_constructor]["keep_model_weights_on_gpu"] is True


def test_model_ledger_caches_video_encoder_when_enabled(monkeypatch) -> None:
    model_ledger_module = import_module("ltx_pipelines.utils.model_ledger")
    ModelLedger = model_ledger_module.ModelLedger

    class FakeModule:
        def __init__(self, token: object):
            self.token = token

        def to(self, device):
            _ = device
            return self

        def eval(self):
            return self

    class FakeBuilder:
        def __init__(self):
            self.calls = 0

        def build(self, **kwargs):
            _ = kwargs
            self.calls += 1
            return FakeModule(object())

    ledger = ModelLedger(dtype=torch.bfloat16, device=torch.device("cpu"), cache_models=True)
    builder = FakeBuilder()
    ledger.vae_encoder_builder = builder

    first = ledger.video_encoder()
    second = ledger.video_encoder()

    assert first is second
    assert builder.calls == 1


def test_model_ledger_does_not_cache_video_encoder_by_default() -> None:
    model_ledger_module = import_module("ltx_pipelines.utils.model_ledger")
    ModelLedger = model_ledger_module.ModelLedger

    class FakeModule:
        def to(self, device):
            _ = device
            return self

        def eval(self):
            return self

    class FakeBuilder:
        def __init__(self):
            self.calls = 0

        def build(self, **kwargs):
            _ = kwargs
            self.calls += 1
            return FakeModule()

    ledger = ModelLedger(dtype=torch.bfloat16, device=torch.device("cpu"))
    builder = FakeBuilder()
    ledger.vae_encoder_builder = builder

    first = ledger.video_encoder()
    second = ledger.video_encoder()

    assert first is not second
    assert builder.calls == 2


def test_backend_shutdown_releases_cached_pipeline_models(tmp_path: Path) -> None:
    release_calls: list[str] = []

    class FakePipeline:
        def release_cached_models(self) -> None:
            release_calls.append("released")

    backend = PipelineServiceBackend(
        config=_make_test_config(tmp_path),
        runner_factory=lambda: backend_module.OneStagePipelineRunner(pipeline=FakePipeline()),
    )

    async def scenario() -> None:
        await backend.start()
        job = await backend.submit(GenerateJobRequest(prompt="hello"))
        _ = await _wait_for_job(backend, job.job_id, JobStatus.FAILED)
        await backend.shutdown()

    asyncio.run(scenario())

    assert release_calls == ["released"]


def test_backend_serializes_jobs(tmp_path: Path) -> None:
    first_job_started = threading.Event()
    release_first_job = threading.Event()
    execution_order: list[str] = []

    class FakeRunner:
        def generate(self, request, output_path: Path) -> None:
            execution_order.append(request.prompt)
            if request.prompt == "first":
                first_job_started.set()
                release_first_job.wait(timeout=1.0)
            output_path.write_text(request.prompt)

    backend = PipelineServiceBackend(config=_make_test_config(tmp_path), runner_factory=lambda: FakeRunner())

    async def scenario() -> None:
        await backend.start()
        try:
            first_job = await backend.submit(GenerateJobRequest(prompt="first"))
            second_job = await backend.submit(GenerateJobRequest(prompt="second"))

            await asyncio.to_thread(first_job_started.wait, 1.0)
            assert backend.get_job(second_job.job_id).status is JobStatus.QUEUED

            release_first_job.set()
            first_record = await _wait_for_job(backend, first_job.job_id, JobStatus.SUCCEEDED)
            second_record = await _wait_for_job(backend, second_job.job_id, JobStatus.SUCCEEDED)

            assert first_record.output_path.read_text() == "first"
            assert second_record.output_path.read_text() == "second"
            assert execution_order == ["first", "second"]
        finally:
            await backend.shutdown()

    asyncio.run(scenario())


def test_fastapi_endpoints_report_health_and_job_status(tmp_path: Path) -> None:
    runner_builds = 0

    class FakeRunner:
        def generate(self, request, output_path: Path) -> None:
            output_path.write_text(request.prompt)

    def build_runner():
        nonlocal runner_builds
        runner_builds += 1
        return FakeRunner()

    config = _make_test_config(tmp_path)
    backend = PipelineServiceBackend(config=config, runner_factory=build_runner)
    app = create_app(config, backend=backend)

    with TestClient(app) as client:
        health_response = client.get("/health")
        assert health_response.status_code == 200
        assert health_response.json()["pipeline_loaded"] is False

        first_submit = client.post("/v1/videos", json={"prompt": "hello world"}).json()
        first_job_id = str(first_submit["id"])
        first_job = _wait_for_generation_via_api(client, first_job_id)
        assert first_job["id"] == first_job_id
        first_file_metadata = client.get(f"/v1/files/{first_job_id}")
        assert first_file_metadata.status_code == 200
        assert first_file_metadata.json()["id"] == first_job_id
        first_file_download = client.get(f"/v1/files/{first_job_id}/content")
        assert first_file_download.status_code == 200
        assert first_file_download.text == "hello world"

        second_submit = client.post("/v1/videos", json={"prompt": "goodbye"}).json()
        second_job_id = str(second_submit["id"])
        second_job = _wait_for_generation_via_api(client, second_job_id)
        assert second_job["id"] == second_job_id
        second_file_download = client.get(f"/v1/files/{second_job_id}/content")
        assert second_file_download.status_code == 200
        assert second_file_download.text == "goodbye"

        assert runner_builds == 1

        health_response = client.get("/health")
        assert health_response.status_code == 200
        assert health_response.json()["pipeline_loaded"] is True

        missing_generation_response = client.get("/v1/videos/missing-job")
        assert missing_generation_response.status_code == 404
        missing_file_response = client.get("/v1/files/file-missing/content")
        assert missing_file_response.status_code == 404


def test_fastapi_rejects_non_utf8_json_payloads(tmp_path: Path) -> None:
    config = _make_test_config(tmp_path)
    backend = PipelineServiceBackend(config=config, runner_factory=lambda: _NeverCalledRunner())
    app = create_app(config, backend=backend)

    with TestClient(app) as client:
        response = client.post(
            "/v1/videos",
            content=b"\xff\xfe",
            headers={"content-type": "application/json"},
        )

    assert response.status_code == 422
    assert "request body must be UTF-8" in response.text


def test_fastapi_rejects_non_utf8_multipart_payloads(tmp_path: Path) -> None:
    config = _make_test_config(tmp_path)
    backend = PipelineServiceBackend(config=config, runner_factory=lambda: _NeverCalledRunner())
    app = create_app(config, backend=backend)
    boundary = "boundary123"
    multipart_body = (
        f"--{boundary}\r\n"
        'Content-Disposition: form-data; name="payload"\r\n'
        "Content-Type: application/json; charset=latin-1\r\n\r\n"
    ).encode("utf-8") + b'{"prompt":"\xff"}\r\n' + f"--{boundary}--\r\n".encode("utf-8")

    with TestClient(app) as client:
        response = client.post(
            "/v1/videos",
            content=multipart_body,
            headers={"content-type": f"multipart/form-data; boundary={boundary}"},
        )

    assert response.status_code == 422
    assert "payload' form field must be valid UTF-8 JSON text" in response.text


def test_fastapi_rejects_two_stage_resolution_not_divisible_by_64(tmp_path: Path) -> None:
    config = _make_test_config(tmp_path, pipeline_type=ServingPipelineType.TI2VID_TWO_STAGES)
    backend = PipelineServiceBackend(config=config, runner_factory=lambda: _NeverCalledRunner())
    app = create_app(config, backend=backend)

    with TestClient(app) as client:
        response = client.post(
            "/v1/videos",
            json={
                "prompt": "bad resolution",
                "height": 544,
                "width": 960,
                "num_frames": 9,
            },
        )

    assert response.status_code == 422
    assert "height and width must be divisible by 64" in response.text


def test_distilled_pipeline_skips_inter_stage_cleanup_when_keep_weights_enabled(monkeypatch) -> None:
    distilled_module = import_module("ltx_pipelines.distilled")
    cleanup_calls: list[str] = []

    class DummyLedger:
        def __init__(self, **kwargs):
            self.kwargs = kwargs
            self.video_encoder_calls = 0
            self.transformer_calls = 0

        def video_encoder(self):
            self.video_encoder_calls += 1
            return object()

        def transformer(self):
            self.transformer_calls += 1
            return object()

        def spatial_upsampler(self):
            return object()

        def video_decoder(self):
            return object()

        def audio_decoder(self):
            return object()

        def vocoder(self):
            return object()

    monkeypatch.setattr(distilled_module, "ModelLedger", DummyLedger)
    monkeypatch.setattr(distilled_module, "PipelineComponents", lambda **kwargs: SimpleNamespace(**kwargs))
    monkeypatch.setattr(distilled_module, "assert_resolution", lambda **kwargs: None)
    monkeypatch.setattr(
        distilled_module,
        "encode_prompts",
        lambda *args, **kwargs: (SimpleNamespace(video_encoding="video", audio_encoding="audio"),),
    )
    monkeypatch.setattr(distilled_module, "combined_image_conditionings", lambda **kwargs: [])
    monkeypatch.setattr(
        distilled_module,
        "denoise_audio_video",
        lambda **kwargs: (SimpleNamespace(latent=torch.zeros(1)), SimpleNamespace(latent=torch.zeros(1))),
    )
    monkeypatch.setattr(distilled_module, "upsample_video", lambda **kwargs: torch.zeros(1))
    monkeypatch.setattr(distilled_module, "vae_decode_video", lambda *args, **kwargs: iter(()))
    monkeypatch.setattr(distilled_module, "vae_decode_audio", lambda *args, **kwargs: torch.zeros(1))
    monkeypatch.setattr(distilled_module, "cleanup_memory", lambda: cleanup_calls.append("cleanup"))
    monkeypatch.setattr(torch.cuda, "synchronize", lambda: None)

    pipeline = distilled_module.DistilledPipeline(
        distilled_checkpoint_path="checkpoint.safetensors",
        gemma_root="gemma",
        spatial_upsampler_path="upsampler.safetensors",
        loras=(),
        device=torch.device("cpu"),
        keep_stage_weights_on_gpu=True,
    )
    _ = pipeline(
        prompt="hello",
        seed=1,
        height=512,
        width=512,
        num_frames=9,
        frame_rate=24.0,
        images=[],
    )

    assert cleanup_calls == ["cleanup"]
    assert pipeline.model_ledger.video_encoder_calls == 1
    assert pipeline.model_ledger.transformer_calls == 1


def test_distilled_pipeline_default_behavior_runs_inter_stage_cleanup(monkeypatch) -> None:
    distilled_module = import_module("ltx_pipelines.distilled")
    cleanup_calls: list[str] = []

    class DummyLedger:
        def __init__(self, **kwargs):
            self.kwargs = kwargs
            self.video_encoder_calls = 0
            self.transformer_calls = 0

        def video_encoder(self):
            self.video_encoder_calls += 1
            return object()

        def transformer(self):
            self.transformer_calls += 1
            return object()

        def spatial_upsampler(self):
            return object()

        def video_decoder(self):
            return object()

        def audio_decoder(self):
            return object()

        def vocoder(self):
            return object()

    monkeypatch.setattr(distilled_module, "ModelLedger", DummyLedger)
    monkeypatch.setattr(distilled_module, "PipelineComponents", lambda **kwargs: SimpleNamespace(**kwargs))
    monkeypatch.setattr(distilled_module, "assert_resolution", lambda **kwargs: None)
    monkeypatch.setattr(
        distilled_module,
        "encode_prompts",
        lambda *args, **kwargs: (SimpleNamespace(video_encoding="video", audio_encoding="audio"),),
    )
    monkeypatch.setattr(distilled_module, "combined_image_conditionings", lambda **kwargs: [])
    monkeypatch.setattr(
        distilled_module,
        "denoise_audio_video",
        lambda **kwargs: (SimpleNamespace(latent=torch.zeros(1)), SimpleNamespace(latent=torch.zeros(1))),
    )
    monkeypatch.setattr(distilled_module, "upsample_video", lambda **kwargs: torch.zeros(1))
    monkeypatch.setattr(distilled_module, "vae_decode_video", lambda *args, **kwargs: iter(()))
    monkeypatch.setattr(distilled_module, "vae_decode_audio", lambda *args, **kwargs: torch.zeros(1))
    monkeypatch.setattr(distilled_module, "cleanup_memory", lambda: cleanup_calls.append("cleanup"))
    monkeypatch.setattr(torch.cuda, "synchronize", lambda: None)

    pipeline = distilled_module.DistilledPipeline(
        distilled_checkpoint_path="checkpoint.safetensors",
        gemma_root="gemma",
        spatial_upsampler_path="upsampler.safetensors",
        loras=(),
        device=torch.device("cpu"),
    )
    _ = pipeline(
        prompt="hello",
        seed=1,
        height=512,
        width=512,
        num_frames=9,
        frame_rate=24.0,
        images=[],
    )

    assert cleanup_calls == ["cleanup", "cleanup"]
    assert pipeline.model_ledger.video_encoder_calls == 1
    assert pipeline.model_ledger.transformer_calls == 1


def test_two_stage_pipeline_keeps_stage_models_when_enabled(monkeypatch) -> None:
    two_stage_module = import_module("ltx_pipelines.ti2vid_two_stages")
    cleanup_calls: list[str] = []
    ledgers: dict[str, object] = {}

    class DummyLedger:
        def __init__(self, **kwargs):
            self.kwargs = kwargs
            self.label = "stage1"
            self.video_encoder_calls = 0
            self.transformer_calls = 0

        def with_additional_loras(self, loras):
            stage2 = DummyLedger.__new__(DummyLedger)
            stage2.kwargs = {"loras": loras}
            stage2.label = "stage2"
            stage2.video_encoder_calls = 0
            stage2.transformer_calls = 0
            ledgers["stage2"] = stage2
            return stage2

        def video_encoder(self):
            self.video_encoder_calls += 1
            return object()

        def transformer(self):
            self.transformer_calls += 1
            return object()

        def spatial_upsampler(self):
            return object()

        def video_decoder(self):
            return object()

        def audio_decoder(self):
            return object()

        def vocoder(self):
            return object()

    monkeypatch.setattr(two_stage_module, "ModelLedger", DummyLedger)
    monkeypatch.setattr(two_stage_module, "PipelineComponents", lambda **kwargs: SimpleNamespace(**kwargs))
    monkeypatch.setattr(two_stage_module, "assert_resolution", lambda **kwargs: None)
    monkeypatch.setattr(
        two_stage_module,
        "encode_prompts",
        lambda *args, **kwargs: (
            SimpleNamespace(video_encoding="video+", audio_encoding="audio+"),
            SimpleNamespace(video_encoding="video-", audio_encoding="audio-"),
        ),
    )
    monkeypatch.setattr(two_stage_module, "combined_image_conditionings", lambda **kwargs: [])
    monkeypatch.setattr(
        two_stage_module,
        "denoise_audio_video",
        lambda **kwargs: (SimpleNamespace(latent=torch.zeros(1)), SimpleNamespace(latent=torch.zeros(1))),
    )
    monkeypatch.setattr(two_stage_module, "upsample_video", lambda **kwargs: torch.zeros(1))
    monkeypatch.setattr(two_stage_module, "vae_decode_video", lambda *args, **kwargs: iter(()))
    monkeypatch.setattr(two_stage_module, "vae_decode_audio", lambda *args, **kwargs: torch.zeros(1))
    monkeypatch.setattr(two_stage_module, "cleanup_memory", lambda: cleanup_calls.append("cleanup"))
    monkeypatch.setattr(two_stage_module, "create_multimodal_guider_factory", lambda **kwargs: object())
    monkeypatch.setattr(two_stage_module, "multi_modal_guider_factory_denoising_func", lambda **kwargs: object())
    monkeypatch.setattr(two_stage_module, "simple_denoising_func", lambda **kwargs: object())
    monkeypatch.setattr(two_stage_module, "LTX2Scheduler", lambda: SimpleNamespace(execute=lambda steps: torch.ones(steps)))
    monkeypatch.setattr(torch.cuda, "synchronize", lambda: None)

    pipeline = two_stage_module.TI2VidTwoStagesPipeline(
        checkpoint_path="checkpoint.safetensors",
        distilled_lora=(),
        spatial_upsampler_path="upsampler.safetensors",
        gemma_root="gemma",
        loras=(),
        device=torch.device("cpu"),
        keep_stage_weights_on_gpu=True,
    )
    ledgers["stage1"] = pipeline.stage_1_model_ledger
    _ = pipeline(
        prompt="hello",
        negative_prompt="world",
        seed=1,
        height=512,
        width=512,
        num_frames=9,
        frame_rate=24.0,
        num_inference_steps=4,
        video_guider_params=object(),
        audio_guider_params=object(),
        images=[],
    )

    stage1 = ledgers["stage1"]
    stage2 = ledgers["stage2"]
    assert isinstance(stage1, DummyLedger)
    assert isinstance(stage2, DummyLedger)
    assert cleanup_calls == ["cleanup"]
    assert stage1.video_encoder_calls == 1
    assert stage1.transformer_calls == 1
    assert stage2.transformer_calls == 1


def test_backend_materializes_url_images_and_cleans_up(tmp_path: Path, monkeypatch) -> None:
    seen_path = ""
    seen_content = b""
    seen_url = ""
    seen_timeout = 0.0

    class FakeRunner:
        def generate(self, request, output_path: Path) -> None:
            nonlocal seen_path, seen_content
            [image] = request.to_pipeline_images()
            seen_path = image.path
            seen_content = Path(image.path).read_bytes()
            _ = output_path.write_text(request.prompt)

    def fake_fetch_public_url(url: str, *, timeout: float, redirect_count: int = 0):
        nonlocal seen_url, seen_timeout
        _ = redirect_count
        seen_url = url
        seen_timeout = timeout
        headers = Message()
        headers["Content-Type"] = "image/png"
        return SAMPLE_PNG_BYTES, headers, url

    monkeypatch.setattr(backend_module, "_fetch_public_url", fake_fetch_public_url)

    backend = PipelineServiceBackend(config=_make_test_config(tmp_path), runner_factory=lambda: FakeRunner())

    async def scenario() -> None:
        await backend.start()
        try:
            job = await backend.submit(
                GenerateJobRequest(
                    prompt="url image",
                    num_frames=9,
                    images=[{"source": "https://www.baidu.com/img/flexible/logo/pc/result.png", "frame_idx": 0, "strength": 1.0}],
                )
            )
            record = await _wait_for_job(backend, job.job_id, JobStatus.SUCCEEDED)

            materialized_path = Path(seen_path)
            assert record.output_path.read_text() == "url image"
            assert seen_url == "https://www.baidu.com/img/flexible/logo/pc/result.png"
            assert seen_timeout == 10
            assert seen_content == SAMPLE_PNG_BYTES
            assert materialized_path.suffix == ".png"
            assert not materialized_path.exists()
            assert not (tmp_path / ".job_inputs" / job.job_id).exists()
        finally:
            await backend.shutdown()

    asyncio.run(scenario())


def test_backend_materializes_base64_images_without_touching_path_inputs(tmp_path: Path) -> None:
    seen_paths: list[str] = []
    seen_contents: list[bytes] = []
    existing_image_path = tmp_path / "existing.png"
    _ = existing_image_path.write_bytes(SAMPLE_PNG_BYTES)

    class FakeRunner:
        def generate(self, request, output_path: Path) -> None:
            nonlocal seen_paths, seen_contents
            pipeline_images = request.to_pipeline_images()
            seen_paths = [image.path for image in pipeline_images]
            seen_contents = [Path(image.path).read_bytes() for image in pipeline_images]
            _ = output_path.write_text(request.prompt)

    backend = PipelineServiceBackend(config=_make_test_config(tmp_path), runner_factory=lambda: FakeRunner())

    async def scenario() -> None:
        await backend.start()
        try:
            job = await backend.submit(
                GenerateJobRequest(
                    prompt="base64 image",
                    num_frames=9,
                    images=[
                        {"source": existing_image_path.as_posix(), "frame_idx": 0, "strength": 1.0},
                        {
                            "source": base64.b64encode(SAMPLE_PNG_BYTES).decode("ascii"),
                            "frame_idx": 8,
                            "strength": 0.75,
                        },
                    ],
                )
            )
            await _wait_for_job(backend, job.job_id, JobStatus.SUCCEEDED)

            materialized_paths = [Path(path) for path in seen_paths]
            assert materialized_paths[0] == existing_image_path
            assert materialized_paths[0].exists()
            assert seen_contents == [SAMPLE_PNG_BYTES, SAMPLE_PNG_BYTES]
            assert materialized_paths[1].suffix == ".png"
            assert not materialized_paths[1].exists()
            assert not (tmp_path / ".job_inputs" / job.job_id).exists()
        finally:
            await backend.shutdown()

    asyncio.run(scenario())


def test_backend_preserves_relative_path_without_extension(tmp_path: Path, monkeypatch) -> None:
    seen_paths: list[str] = []
    seen_contents: list[bytes] = []
    relative_image_path = Path("images/reference")
    absolute_image_path = tmp_path / relative_image_path
    absolute_image_path.parent.mkdir(parents=True, exist_ok=True)
    _ = absolute_image_path.write_bytes(SAMPLE_PNG_BYTES)
    monkeypatch.chdir(tmp_path)

    class FakeRunner:
        def generate(self, request, output_path: Path) -> None:
            nonlocal seen_paths, seen_contents
            pipeline_images = request.to_pipeline_images()
            seen_paths = [image.path for image in pipeline_images]
            seen_contents = [Path(image.path).read_bytes() for image in pipeline_images]
            _ = output_path.write_text(request.prompt)

    backend = PipelineServiceBackend(config=_make_test_config(tmp_path), runner_factory=lambda: FakeRunner())

    async def scenario() -> None:
        await backend.start()
        try:
            job = await backend.submit(
                GenerateJobRequest(
                    prompt="relative path",
                    num_frames=9,
                    images=[{"source": relative_image_path.as_posix(), "frame_idx": 0, "strength": 1.0}],
                )
            )
            await _wait_for_job(backend, job.job_id, JobStatus.SUCCEEDED)
            assert seen_paths == [relative_image_path.as_posix()]
            assert seen_contents == [SAMPLE_PNG_BYTES]
            assert not (tmp_path / ".job_inputs" / job.job_id).exists()
        finally:
            await backend.shutdown()

    asyncio.run(scenario())


def test_backend_expands_home_relative_paths_before_pipeline_execution(tmp_path: Path, monkeypatch) -> None:
    seen_paths: list[str] = []
    home_dir = tmp_path / "home"
    image_path = home_dir / "images" / "reference.png"
    image_path.parent.mkdir(parents=True, exist_ok=True)
    _ = image_path.write_bytes(SAMPLE_PNG_BYTES)
    monkeypatch.setenv("HOME", home_dir.as_posix())

    class FakeRunner:
        def generate(self, request, output_path: Path) -> None:
            nonlocal seen_paths
            seen_paths = [image.path for image in request.to_pipeline_images()]
            _ = output_path.write_text(request.prompt)

    backend = PipelineServiceBackend(config=_make_test_config(tmp_path), runner_factory=lambda: FakeRunner())

    async def scenario() -> None:
        await backend.start()
        try:
            job = await backend.submit(
                GenerateJobRequest(
                    prompt="home path",
                    num_frames=9,
                    images=[{"source": "~/images/reference.png", "frame_idx": 0, "strength": 1.0}],
                )
            )
            await _wait_for_job(backend, job.job_id, JobStatus.SUCCEEDED)
            assert seen_paths == [image_path.as_posix()]
        finally:
            await backend.shutdown()

    asyncio.run(scenario())


def test_fastapi_accepts_multipart_payload_with_uploaded_images(tmp_path: Path) -> None:
    seen_path = ""
    seen_content = b""

    class FakeRunner:
        def generate(self, request, output_path: Path) -> None:
            nonlocal seen_path, seen_content
            [image] = request.to_pipeline_images()
            seen_path = image.path
            seen_content = Path(image.path).read_bytes()
            _ = output_path.write_text(request.prompt)

    config = _make_test_config(tmp_path)
    backend = PipelineServiceBackend(config=config, runner_factory=lambda: FakeRunner())
    app = create_app(config, backend=backend)

    with TestClient(app) as client:
        response = client.post(
            "/v1/videos",
            data={
                "payload": json.dumps(
                    {
                        "prompt": "multipart upload",
                        "num_frames": 9,
                        "images": [{"source": "upload:image_upload", "frame_idx": 0, "strength": 1.0}],
                    }
                )
            },
            files={"image_upload": ("upload.png", SAMPLE_PNG_BYTES, "image/png")},
        )

        assert response.status_code == 202, response.text
        job_id = str(response.json()["id"])
        _ = _wait_for_generation_via_api(client, job_id)

    materialized_path = Path(seen_path)
    assert seen_content == SAMPLE_PNG_BYTES
    assert materialized_path.suffix == ".png"
    assert not materialized_path.exists()
    assert not (tmp_path / ".job_inputs" / job_id).exists()


def test_backend_prefers_image_header_over_non_image_content_type(tmp_path: Path, monkeypatch) -> None:
    seen_path = ""

    class FakeRunner:
        def generate(self, request, output_path: Path) -> None:
            nonlocal seen_path
            [image] = request.to_pipeline_images()
            seen_path = image.path
            _ = output_path.write_text(request.prompt)

    def fake_fetch_public_url(url: str, *, timeout: float, redirect_count: int = 0):
        _ = (timeout, redirect_count)
        headers = Message()
        headers["Content-Type"] = "text/plain"
        return SAMPLE_PNG_BYTES, headers, url

    monkeypatch.setattr(backend_module, "_fetch_public_url", fake_fetch_public_url)

    backend = PipelineServiceBackend(config=_make_test_config(tmp_path), runner_factory=lambda: FakeRunner())

    async def scenario() -> None:
        await backend.start()
        try:
            job = await backend.submit(
                GenerateJobRequest(
                    prompt="header sniff",
                    num_frames=9,
                    images=[{"source": "https://www.baidu.com/img/flexible/logo/pc/result.png", "frame_idx": 0, "strength": 1.0}],
                )
            )
            await _wait_for_job(backend, job.job_id, JobStatus.SUCCEEDED)
            assert Path(seen_path).suffix == ".png"
        finally:
            await backend.shutdown()

    asyncio.run(scenario())


def test_backend_fails_non_image_base64_content(tmp_path: Path) -> None:
    with pytest.raises(ValidationError):
        GenerateJobRequest(
            prompt="bad image",
            num_frames=9,
            images=[
                {
                    "source": base64.b64encode(b"this is not an image").decode("ascii"),
                    "frame_idx": 0,
                    "strength": 1.0,
                }
            ],
        )


def test_generate_job_request_rejects_unsupported_uri_scheme() -> None:
    with pytest.raises(ValidationError):
        GenerateJobRequest(
            prompt="bad scheme",
            num_frames=9,
            images=[{"source": "ftp://ftp.example.com/a.png", "frame_idx": 0, "strength": 1.0}],
        )

    with pytest.raises(ValidationError):
        GenerateJobRequest(
            prompt="bad scheme",
            num_frames=9,
            images=[{"source": "file:/tmp/a.png", "frame_idx": 0, "strength": 1.0}],
        )


def test_backend_rejects_non_public_url_sources(tmp_path: Path) -> None:
    backend = PipelineServiceBackend(config=_make_test_config(tmp_path), runner_factory=lambda: _NeverCalledRunner())

    async def scenario() -> None:
        await backend.start()
        try:
            job = await backend.submit(
                GenerateJobRequest(
                    prompt="ssrf attempt",
                    num_frames=9,
                    images=[{"source": "http://127.0.0.1/private.png", "frame_idx": 0, "strength": 1.0}],
                )
            )
            record = await _wait_for_job(backend, job.job_id, JobStatus.FAILED)
            assert record.error is not None
            assert "public IP addresses" in record.error
        finally:
            await backend.shutdown()

    asyncio.run(scenario())


def test_fetch_public_url_uses_validated_ip_for_connection(monkeypatch) -> None:
    seen_ips: list[str] = []

    monkeypatch.setattr(
        backend_module,
        "_resolve_public_http_url",
        lambda url: (backend_module.urlparse(url), "93.184.216.34"),
    )

    def fake_request_pinned_url(parsed, *, resolved_ip: str, timeout: float):
        _ = (parsed, timeout)
        seen_ips.append(resolved_ip)
        return 200, Message(), SAMPLE_PNG_BYTES

    monkeypatch.setattr(backend_module, "_request_pinned_url", fake_request_pinned_url)

    content, _headers, final_url = backend_module._fetch_public_url(
        "https://www.baidu.com/img/flexible/logo/pc/result.png", timeout=10
    )

    assert content == SAMPLE_PNG_BYTES
    assert final_url == "https://www.baidu.com/img/flexible/logo/pc/result.png"
    assert seen_ips == ["93.184.216.34"]


def test_backend_shutdown_waits_for_inflight_job_completion(tmp_path: Path) -> None:
    job_started = threading.Event()
    release_job = threading.Event()

    class BlockingRunner:
        def generate(self, request, output_path: Path) -> None:
            _ = request
            job_started.set()
            release_job.wait(timeout=1.0)
            _ = output_path.write_text("finished")

    backend = PipelineServiceBackend(config=_make_test_config(tmp_path), runner_factory=lambda: BlockingRunner())

    async def scenario() -> None:
        await backend.start()
        job = await backend.submit(GenerateJobRequest(prompt="blocking job"))
        await asyncio.to_thread(job_started.wait, 1.0)

        shutdown_task = asyncio.create_task(backend.shutdown())
        await asyncio.sleep(0.05)
        assert not shutdown_task.done()
        assert backend.get_job(job.job_id).status is JobStatus.RUNNING

        release_job.set()
        await shutdown_task

        record = backend.get_job(job.job_id)
        assert record is not None
        assert record.status is JobStatus.SUCCEEDED
        assert record.finished_at is not None
        assert record.output_path.read_text() == "finished"

    asyncio.run(scenario())


def test_failed_job_file_endpoints_return_gone(tmp_path: Path) -> None:
    class FailingRunner:
        def generate(self, request, output_path: Path) -> None:
            _ = (request, output_path)
            raise RuntimeError("generation failed")

    config = _make_test_config(tmp_path)
    backend = PipelineServiceBackend(config=config, runner_factory=lambda: FailingRunner())
    app = create_app(config, backend=backend)

    with TestClient(app) as client:
        submit_response = client.post("/v1/videos", json={"prompt": "fail me", "num_frames": 9})
        assert submit_response.status_code == 202
        payload = submit_response.json()
        job_id = str(payload["id"])

        deadline = time.monotonic() + 2.0
        final_job = None
        while time.monotonic() < deadline:
            final_job = client.get(f"/v1/videos/{job_id}").json()
            if final_job["status"] == JobStatus.FAILED.value:
                break
            time.sleep(0.01)

        assert final_job is not None
        assert final_job["status"] == JobStatus.FAILED.value

        file_metadata = client.get(f"/v1/files/{job_id}")
        file_content = client.get(f"/v1/files/{job_id}/content")

    assert file_metadata.status_code == 410
    assert file_content.status_code == 410
    assert "generation failed" in final_job["error"]
    assert "File is unavailable because generation failed." in file_metadata.text
    assert "File is unavailable because generation failed." in file_content.text


@pytest.mark.parametrize(
    "image_payload",
    [
        {"frame_idx": 0, "strength": 1.0},
        {"source": "upload:", "frame_idx": 0, "strength": 1.0},
        {"source": "reference", "path": "/tmp/input.png", "frame_idx": 0, "strength": 1.0},
    ],
)
def test_generate_job_request_rejects_invalid_image_source_combinations(image_payload: dict[str, object]) -> None:
    with pytest.raises(ValidationError):
        GenerateJobRequest(prompt="invalid image sources", num_frames=9, images=[image_payload])


def test_fastapi_rejects_missing_multipart_upload_reference(tmp_path: Path) -> None:
    config = _make_test_config(tmp_path)
    backend = PipelineServiceBackend(config=config, runner_factory=lambda: _NeverCalledRunner())
    app = create_app(config, backend=backend)

    with TestClient(app) as client:
        response = client.post(
            "/v1/videos",
            data={
                "payload": json.dumps(
                    {
                        "prompt": "missing upload",
                        "num_frames": 9,
                        "images": [{"source": "upload:missing_upload", "frame_idx": 0, "strength": 1.0}],
                    }
                )
            },
            files={"unused": ("unused.txt", b"", "text/plain")},
        )

    assert response.status_code == 422
    assert "Multipart upload field 'missing_upload' was not provided." in response.text


def test_fastapi_rejects_upload_source_in_json_requests(tmp_path: Path) -> None:
    config = _make_test_config(tmp_path)
    backend = PipelineServiceBackend(config=config, runner_factory=lambda: _NeverCalledRunner())
    app = create_app(config, backend=backend)

    with TestClient(app) as client:
        response = client.post(
            "/v1/videos",
            json={
                "prompt": "json upload source",
                "num_frames": 9,
                "images": [{"source": "upload:image_upload", "frame_idx": 0, "strength": 1.0}],
            },
        )

    assert response.status_code == 422
    assert "upload: sources are only supported with multipart/form-data requests." in response.text


class _NeverCalledRunner:
    def generate(self, request, output_path: Path) -> None:
        raise AssertionError("Runner should not be called for invalid requests.")


def _make_test_config(
    output_dir: Path,
    execution_mode=ExecutionMode.SINGLE,
    pipeline_type=ServingPipelineType.TI2VID_ONE_STAGE,
    keep_stage_weights_on_gpu: bool = False,
    keep_model_weights_on_gpu: bool = False,
):
    model_dir = output_dir / "models"
    return ServiceConfig(
        pipeline_type=pipeline_type,
        checkpoint_path=model_dir / "ltx-2.3-22b-dev.safetensors",
        distilled_checkpoint_path=model_dir / "ltx-2.3-22b-distilled.safetensors",
        distilled_lora_path=model_dir / "ltx-2.3-22b-distilled-lora-384.safetensors",
        spatial_upsampler_path=model_dir / "ltx-2.3-spatial-upscaler-x2-1.1.safetensors",
        gamma_path=model_dir / "gamma-path",
        output_dir=output_dir,
        execution_mode=execution_mode,
        keep_stage_weights_on_gpu=keep_stage_weights_on_gpu,
        keep_model_weights_on_gpu=keep_model_weights_on_gpu,
    )


def _required_service_argv(output_dir: Path) -> list[str]:
    model_dir = output_dir / "models"
    return [
        "--pipeline-type",
        ServingPipelineType.TI2VID_ONE_STAGE.value,
        "--checkpoint-path",
        (model_dir / "ltx-2.3-22b-dev.safetensors").as_posix(),
        "--gamma-path",
        (model_dir / "gamma-path").as_posix(),
    ]


async def _wait_for_job(backend, job_id: str, status):
    deadline = time.monotonic() + 2.0
    while time.monotonic() < deadline:
        job = backend.get_job(job_id)
        if job is not None and job.status is status:
            return job
        await asyncio.sleep(0.01)
    raise AssertionError(f"Timed out waiting for job {job_id} to reach {status.value}.")


def _wait_for_generation_via_api(client: TestClient, job_id: str) -> dict[str, object]:
    deadline = time.monotonic() + 2.0
    while time.monotonic() < deadline:
        response = client.get(f"/v1/videos/{job_id}")
        payload = response.json()
        if payload["status"] == JobStatus.SUCCEEDED.value:
            return payload
        time.sleep(0.01)
    raise AssertionError(f"Timed out waiting for job {job_id} to complete.")
