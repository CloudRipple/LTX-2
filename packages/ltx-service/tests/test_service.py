import asyncio
import base64
from importlib import import_module
import json
from email.message import Message
import threading
import time
from pathlib import Path

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
        first_file_id = str(first_submit["output_file_id"])
        assert first_file_id == first_job_id
        first_job = _wait_for_generation_via_api(client, first_job_id)
        assert first_job["output_file_id"] == first_file_id
        first_file_metadata = client.get(f"/v1/files/{first_file_id}")
        assert first_file_metadata.status_code == 200
        assert first_file_metadata.json()["id"] == first_file_id
        first_file_download = client.get(f"/v1/files/{first_file_id}/content")
        assert first_file_download.status_code == 200
        assert first_file_download.text == "hello world"

        second_submit = client.post("/v1/videos", json={"prompt": "goodbye"}).json()
        second_job_id = str(second_submit["id"])
        second_file_id = str(second_submit["output_file_id"])
        assert second_file_id == second_job_id
        second_job = _wait_for_generation_via_api(client, second_job_id)
        assert second_job["output_file_id"] == second_file_id
        second_file_download = client.get(f"/v1/files/{second_file_id}/content")
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
        file_id = str(payload["output_file_id"])

        deadline = time.monotonic() + 2.0
        final_job = None
        while time.monotonic() < deadline:
            final_job = client.get(f"/v1/videos/{job_id}").json()
            if final_job["status"] == JobStatus.FAILED.value:
                break
            time.sleep(0.01)

        assert final_job is not None
        assert final_job["status"] == JobStatus.FAILED.value

        file_metadata = client.get(f"/v1/files/{file_id}")
        file_content = client.get(f"/v1/files/{file_id}/content")

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
