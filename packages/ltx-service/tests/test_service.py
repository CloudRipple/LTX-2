import asyncio
import base64
from dataclasses import dataclass
from io import StringIO
from importlib import import_module
import json
import logging
from email.message import Message
from pathlib import Path
import signal
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
    assert config.worker_devices() == (torch.device("cuda:0"),)


def test_service_config_supports_data_parallel_mode(monkeypatch) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 2)

    config = _make_test_config(Path("/tmp/ltx-service-test"), execution_mode=ExecutionMode.DATA_PARALLEL)

    assert config.visible_gpu_ids() == (0, 1)
    assert config.resolved_execution_mode() is ExecutionMode.DATA_PARALLEL
    assert config.worker_devices() == (torch.device("cuda:0"), torch.device("cuda:1"))


def test_service_config_uses_gpu_count_when_ids_are_not_specified(monkeypatch) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 4)

    config = _make_test_config(Path("/tmp/ltx-service-test"), execution_mode=ExecutionMode.DATA_PARALLEL, gpu_count=2)

    assert config.visible_gpu_ids() == (0, 1)
    assert config.worker_devices() == (torch.device("cuda:0"), torch.device("cuda:1"))


def test_service_config_gpu_ids_override_gpu_count(monkeypatch) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 4)

    config = _make_test_config(
        Path("/tmp/ltx-service-test"),
        execution_mode=ExecutionMode.DATA_PARALLEL,
        gpu_ids=(1, 3),
        gpu_count=1,
    )

    assert config.visible_gpu_ids() == (1, 3)
    assert config.worker_devices() == (torch.device("cuda:1"), torch.device("cuda:3"))


def test_official_backend_requires_cuda(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    with pytest.raises(ValueError, match="require CUDA-enabled PyTorch"):
        PipelineServiceBackend(config=_make_test_config(tmp_path))


def test_ctrl_c_server_starts_force_exit_timer_on_first_sigint(monkeypatch) -> None:
    service_module = import_module("ltx_service")
    uvicorn = import_module("uvicorn")
    started_timers: list[float] = []

    async def app(scope, receive, send):
        _ = (scope, receive, send)

    monkeypatch.setattr(service_module, "_start_force_exit_timer", lambda timeout: started_timers.append(timeout))
    server = service_module._ImmediateCtrlCServer(uvicorn.Config(app, host="127.0.0.1", port=0))

    server._server.handle_exit(signal.SIGINT, None)

    assert server._server.should_exit is True
    assert server._server.force_exit is False
    assert started_timers == [service_module.CTRL_C_FORCE_EXIT_TIMEOUT_SECONDS]


def test_ctrl_c_server_forces_exit_on_second_sigint(monkeypatch) -> None:
    service_module = import_module("ltx_service")
    uvicorn = import_module("uvicorn")

    async def app(scope, receive, send):
        _ = (scope, receive, send)

    monkeypatch.setattr(service_module, "_start_force_exit_timer", lambda timeout: None)

    def fake_exit(code: int) -> None:
        raise SystemExit(code)

    monkeypatch.setattr(service_module.os, "_exit", fake_exit)
    server = service_module._ImmediateCtrlCServer(uvicorn.Config(app, host="127.0.0.1", port=0))

    server._server.handle_exit(signal.SIGINT, None)
    with pytest.raises(SystemExit) as exc_info:
        server._server.handle_exit(signal.SIGINT, None)

    assert exc_info.value.code == 130


def test_main_uses_immediate_ctrl_c_server(monkeypatch) -> None:
    service_module = import_module("ltx_service")
    uvicorn = import_module("uvicorn")
    parsed_config = ServiceConfig(gamma_path=Path("gamma"))
    captured: dict[str, object] = {}

    class FakeServer:
        def __init__(self, config) -> None:
            captured["server_config"] = config

        def run(self) -> None:
            captured["ran"] = True

    monkeypatch.setattr(service_module, "parse_service_config", lambda: parsed_config)

    def fake_create_app(config):
        captured["app_config"] = config
        return object()

    monkeypatch.setattr(service_module, "create_app", fake_create_app)
    monkeypatch.setattr(service_module, "_ImmediateCtrlCServer", FakeServer)

    class FakeConfig:
        def __init__(self, app, *, host: str, port: int) -> None:
            captured["uvicorn_app"] = app
            captured["host"] = host
            captured["port"] = port

    monkeypatch.setattr(uvicorn, "Config", FakeConfig)

    service_module.main()

    assert captured["app_config"] is parsed_config
    assert captured["host"] == parsed_config.host
    assert captured["port"] == parsed_config.port
    assert captured["ran"] is True


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


def test_parse_service_config_accepts_gpu_count(tmp_path: Path) -> None:
    config = parse_service_config(_required_service_argv(tmp_path) + ["--gpu-count", "2"])

    assert config.gpu_count == 2


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


def test_data_parallel_backend_balances_jobs_across_gpu_workers(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 2)

    runner_builds: list[tuple[int, str]] = []
    seen_workers: list[tuple[str, int]] = []
    active_workers = 0
    max_active_workers = 0
    active_lock = threading.Lock()
    both_workers_started = threading.Event()
    release_workers = threading.Event()

    class FakeRunner:
        def __init__(self, worker_index: int, device: torch.device):
            self.worker_index = worker_index
            self.device = device

        def generate(self, request, output_path: Path) -> None:
            nonlocal active_workers, max_active_workers
            seen_workers.append((request.prompt, self.worker_index))
            should_block = request.prompt in {"first", "second"}
            if should_block:
                with active_lock:
                    active_workers += 1
                    max_active_workers = max(max_active_workers, active_workers)
                    if active_workers == 2:
                        both_workers_started.set()
                release_workers.wait(timeout=1.0)
                with active_lock:
                    active_workers -= 1
            _ = output_path.write_text(f"{request.prompt}@{self.device}")

    def build_runner(*, device: torch.device, worker_index: int, gpu_id: int | None = None):
        runner_builds.append((worker_index, str(device)))
        assert gpu_id == worker_index
        return FakeRunner(worker_index=worker_index, device=device)

    backend = PipelineServiceBackend(
        config=_make_test_config(tmp_path, execution_mode=ExecutionMode.DATA_PARALLEL),
        runner_factory=build_runner,
    )

    async def scenario() -> None:
        await backend.start()
        try:
            health = backend.health()
            assert health.worker_count == 2
            assert health.loaded_runner_count == 0
            assert health.pipeline_loaded is False
            assert [worker.status for worker in health.workers] == ["idle", "idle"]

            first_job = await backend.submit(GenerateJobRequest(prompt="first"))
            second_job = await backend.submit(GenerateJobRequest(prompt="second"))
            third_job = await backend.submit(GenerateJobRequest(prompt="third"))

            await asyncio.to_thread(both_workers_started.wait, 1.0)
            assert backend.get_job(third_job.job_id).status is JobStatus.QUEUED
            health = backend.health()
            assert health.worker_count == 2
            assert health.loaded_runner_count == 2
            assert health.pipeline_loaded is True
            assert {worker.status for worker in health.workers} == {"running"}
            assert max_active_workers == 2
            assert {worker for prompt, worker in seen_workers if prompt in {"first", "second"}} == {0, 1}

            release_workers.set()
            first_record = await _wait_for_job(backend, first_job.job_id, JobStatus.SUCCEEDED)
            second_record = await _wait_for_job(backend, second_job.job_id, JobStatus.SUCCEEDED)
            third_record = await _wait_for_job(backend, third_job.job_id, JobStatus.SUCCEEDED)

            assert first_record.output_path.read_text().startswith("first@cuda:")
            assert second_record.output_path.read_text().startswith("second@cuda:")
            assert third_record.output_path.read_text().startswith("third@cuda:")
            assert runner_builds == [(0, "cuda:0"), (1, "cuda:1")]
        finally:
            await backend.shutdown()

    asyncio.run(scenario())


def test_data_parallel_backend_shutdown_closes_all_runners(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 2)

    released_workers: list[int] = []

    class FakeRunner:
        def __init__(self, worker_index: int):
            self.worker_index = worker_index

        def generate(self, request, output_path: Path) -> None:
            _ = request
            _ = output_path.write_text(f"worker-{self.worker_index}")

        def close(self) -> None:
            released_workers.append(self.worker_index)

    def build_runner(*, worker_index: int, **kwargs):
        _ = kwargs
        return FakeRunner(worker_index)

    backend = PipelineServiceBackend(
        config=_make_test_config(tmp_path, execution_mode=ExecutionMode.DATA_PARALLEL),
        runner_factory=build_runner,
    )

    async def scenario() -> None:
        await backend.start()
        try:
            first_job = await backend.submit(GenerateJobRequest(prompt="first"))
            second_job = await backend.submit(GenerateJobRequest(prompt="second"))
            await _wait_for_job(backend, first_job.job_id, JobStatus.SUCCEEDED)
            await _wait_for_job(backend, second_job.job_id, JobStatus.SUCCEEDED)
        finally:
            await backend.shutdown()

    asyncio.run(scenario())

    assert sorted(released_workers) == [0, 1]


def test_backend_logs_aggregated_worker_progress(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 2)

    first_updates = threading.Event()
    release_workers = threading.Event()

    class FakeRunner:
        def __init__(self, worker_index: int):
            self.worker_index = worker_index

        def generate(self, request, output_path: Path, progress_callback=None) -> None:
            assert progress_callback is not None
            progress_callback("stage1", 1, 4)
            first_updates.set()
            release_workers.wait(timeout=1.0)
            progress_callback("encode", 1, 1)
            _ = output_path.write_text(request.prompt)

    def build_runner(*, worker_index: int, **kwargs):
        _ = kwargs
        return FakeRunner(worker_index)

    backend = PipelineServiceBackend(
        config=_make_test_config(tmp_path, execution_mode=ExecutionMode.DATA_PARALLEL),
        runner_factory=build_runner,
    )
    progress_logger = backend_module.progress_logger
    stream = StringIO()
    handler = logging.StreamHandler(stream)
    original_level = progress_logger.level
    original_propagate = progress_logger.propagate
    progress_logger.addHandler(handler)
    progress_logger.setLevel(logging.INFO)
    progress_logger.propagate = False

    async def scenario() -> None:
        await backend.start()
        try:
            first_job = await backend.submit(GenerateJobRequest(prompt="first"))
            second_job = await backend.submit(GenerateJobRequest(prompt="second"))
            await asyncio.to_thread(first_updates.wait, 1.0)
            await asyncio.sleep(0.05)
            release_workers.set()
            await _wait_for_job(backend, first_job.job_id, JobStatus.SUCCEEDED)
            await _wait_for_job(backend, second_job.job_id, JobStatus.SUCCEEDED)
        finally:
            await backend.shutdown()

    try:
        asyncio.run(scenario())
    finally:
        progress_logger.removeHandler(handler)
        progress_logger.setLevel(original_level)
        progress_logger.propagate = original_propagate

    log_output = stream.getvalue()
    assert "Worker progress | w0:" in log_output
    assert "| w1:" in log_output


def test_euler_denoising_loop_uses_progress_callback_without_tqdm(monkeypatch) -> None:
    samplers_module = import_module("ltx_pipelines.utils.samplers")

    def fail_tqdm(iterable):
        raise AssertionError("tqdm should not be used when progress_callback is provided")

    monkeypatch.setattr(samplers_module, "tqdm", fail_tqdm)

    class FakeStepper:
        def step(self, sample, denoised_sample, sigmas, step_idx):
            _ = (sigmas, step_idx)
            return sample + denoised_sample

    @dataclass
    class FakeLatentState:
        latent: torch.Tensor
        denoise_mask: torch.Tensor
        clean_latent: torch.Tensor

    state = FakeLatentState(latent=torch.zeros(1), denoise_mask=torch.ones(1), clean_latent=torch.zeros(1))
    progress_updates: list[tuple[int, int]] = []

    video_state, audio_state = samplers_module.euler_denoising_loop(
        sigmas=torch.tensor([1.0, 0.5, 0.0]),
        video_state=state,
        audio_state=state,
        stepper=FakeStepper(),
        denoise_fn=lambda *args, **kwargs: (torch.ones(1), torch.ones(1)),
        progress_callback=lambda current, total: progress_updates.append((current, total)),
    )

    assert torch.equal(video_state.latent, torch.full((1,), 2.0))
    assert torch.equal(audio_state.latent, torch.full((1,), 2.0))
    assert progress_updates == [(0, 2), (1, 2), (2, 2)]


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
        assert health_response.json()["worker_count"] == 1
        assert health_response.json()["loaded_runner_count"] == 0
        first_worker = health_response.json()["workers"][0]
        assert first_worker["worker_index"] == 0
        assert first_worker["device"] == health_response.json()["primary_device"]
        assert first_worker["gpu_id"] == (health_response.json()["gpu_ids"][0] if health_response.json()["gpu_ids"] else None)
        assert first_worker["status"] == "idle"
        assert first_worker["runner_loaded"] is False
        assert first_worker["current_job_id"] is None
        assert first_worker["current_phase"] is None
        assert first_worker["error"] is None

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
        assert health_response.json()["worker_count"] == 1
        assert health_response.json()["loaded_runner_count"] == 1
        first_worker = health_response.json()["workers"][0]
        assert first_worker["worker_index"] == 0
        assert first_worker["device"] == health_response.json()["primary_device"]
        assert first_worker["gpu_id"] == (health_response.json()["gpu_ids"][0] if health_response.json()["gpu_ids"] else None)
        assert first_worker["status"] == "ready"
        assert first_worker["runner_loaded"] is True
        assert first_worker["current_job_id"] is None
        assert first_worker["current_phase"] is None
        assert first_worker["error"] is None

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


def test_fastapi_exposes_data_parallel_worker_errors_in_health_and_failed_job_response(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 2)

    worker_zero_started = threading.Event()
    release_worker_zero = threading.Event()

    class BlockingRunner:
        def generate(self, request, output_path: Path) -> None:
            _ = request
            worker_zero_started.set()
            release_worker_zero.wait(timeout=1.0)
            _ = output_path.write_text("worker-zero")

    def build_runner(*, worker_index: int, device: torch.device, gpu_id: int | None = None):
        _ = device
        if worker_index == 1:
            raise FileNotFoundError("missing checkpoint path /models/missing.safetensors")
        assert gpu_id == 0
        return BlockingRunner()

    config = _make_test_config(tmp_path, execution_mode=ExecutionMode.DATA_PARALLEL)
    backend = PipelineServiceBackend(config=config, runner_factory=build_runner)
    app = create_app(config, backend=backend)

    with TestClient(app) as client:
        first_submit = client.post("/v1/videos", json={"prompt": "first", "num_frames": 9}).json()
        second_submit = client.post("/v1/videos", json={"prompt": "second", "num_frames": 9}).json()
        job_ids = [str(first_submit["id"]), str(second_submit["id"])]

        deadline = time.monotonic() + 2.0
        failed_job: dict[str, object] | None = None
        while time.monotonic() < deadline:
            for job_id in job_ids:
                payload = client.get(f"/v1/videos/{job_id}").json()
                if payload["status"] == JobStatus.FAILED.value:
                    failed_job = payload
                    break
            if failed_job is not None:
                break
            time.sleep(0.01)

        assert failed_job is not None
        assert failed_job["worker_error"] == {
            "worker_index": 1,
            "gpu_id": 1,
            "device": "cuda:1",
            "phase": "runner_init",
            "error_type": "FileNotFoundError",
            "message": "missing checkpoint path /models/missing.safetensors",
        }
        assert "Worker 1 (gpu 1, cuda:1) failed during runner_init" in str(failed_job["error"])
        assert "FileNotFoundError" in str(failed_job["error"])
        assert worker_zero_started.wait(timeout=1.0)

        health_response = client.get("/health")
        assert health_response.status_code == 200
        assert health_response.json()["status"] == "degraded"
        assert health_response.json()["loaded_runner_count"] == 1
        assert health_response.json()["workers"] == [
            {
                "worker_index": 0,
                "gpu_id": 0,
                "device": "cuda:0",
                "status": "running",
                "runner_loaded": True,
                "current_job_id": job_ids[0] if job_ids[0] != failed_job["id"] else job_ids[1],
                "current_phase": "generating",
                "error": None,
            },
            {
                "worker_index": 1,
                "gpu_id": 1,
                "device": "cuda:1",
                "status": "error",
                "runner_loaded": False,
                "current_job_id": None,
                "current_phase": None,
                "error": {
                    "worker_index": 1,
                    "gpu_id": 1,
                    "device": "cuda:1",
                    "phase": "runner_init",
                    "error_type": "FileNotFoundError",
                    "message": "missing checkpoint path /models/missing.safetensors",
                },
            },
        ]

        release_worker_zero.set()
        remaining_job_id = next(job_id for job_id in job_ids if job_id != failed_job["id"])
        remaining_job = _wait_for_generation_via_api(client, remaining_job_id)
        assert remaining_job["status"] == JobStatus.SUCCEEDED.value


def test_data_parallel_worker_runner_init_failure_stops_consuming_future_jobs(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 2)

    worker_zero_started = threading.Event()
    release_worker_zero = threading.Event()
    seen_prompts: list[str] = []

    class BlockingRunner:
        def generate(self, request, output_path: Path) -> None:
            seen_prompts.append(request.prompt)
            worker_zero_started.set()
            release_worker_zero.wait(timeout=1.0)
            _ = output_path.write_text(f"worker-0:{request.prompt}")

    def build_runner(*, worker_index: int, **kwargs):
        _ = kwargs
        if worker_index == 1:
            raise FileNotFoundError("missing checkpoint path /models/missing.safetensors")
        return BlockingRunner()

    backend = PipelineServiceBackend(
        config=_make_test_config(tmp_path, execution_mode=ExecutionMode.DATA_PARALLEL),
        runner_factory=build_runner,
    )

    async def scenario() -> None:
        await backend.start()
        try:
            first_job = await backend.submit(GenerateJobRequest(prompt="first"))
            second_job = await backend.submit(GenerateJobRequest(prompt="second"))
            assert await asyncio.to_thread(worker_zero_started.wait, 1.0)

            failed_job = await _wait_for_one_of_jobs(backend, (first_job.job_id, second_job.job_id), JobStatus.FAILED)
            health = backend.health()
            assert health.status == "degraded"
            assert health.workers[1].status == "error"

            third_job = await backend.submit(GenerateJobRequest(prompt="third"))
            await asyncio.sleep(0.05)
            assert backend.get_job(third_job.job_id).status is JobStatus.QUEUED

            release_worker_zero.set()

            await _wait_for_one_of_jobs(backend, (first_job.job_id, second_job.job_id), JobStatus.SUCCEEDED)
            third_record = await _wait_for_job(backend, third_job.job_id, JobStatus.SUCCEEDED)

            assert third_record.output_path.read_text() == "worker-0:third"
            assert seen_prompts[-1] == "third"
            assert backend.get_job(failed_job.job_id).worker_error is not None
        finally:
            await backend.shutdown()

    asyncio.run(scenario())


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
    gpu_ids: tuple[int, ...] = (),
    gpu_count: int | None = None,
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
        gpu_ids=gpu_ids,
        gpu_count=gpu_count,
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


async def _wait_for_one_of_jobs(backend, job_ids: tuple[str, ...], status):
    deadline = time.monotonic() + 2.0
    while time.monotonic() < deadline:
        for job_id in job_ids:
            job = backend.get_job(job_id)
            if job is not None and job.status is status:
                return job
        await asyncio.sleep(0.01)
    raise AssertionError(f"Timed out waiting for one of {job_ids} to reach {status.value}.")


def _wait_for_generation_via_api(client: TestClient, job_id: str) -> dict[str, object]:
    deadline = time.monotonic() + 2.0
    while time.monotonic() < deadline:
        response = client.get(f"/v1/videos/{job_id}")
        payload = response.json()
        if payload["status"] == JobStatus.SUCCEEDED.value:
            return payload
        time.sleep(0.01)
    raise AssertionError(f"Timed out waiting for job {job_id} to complete.")
