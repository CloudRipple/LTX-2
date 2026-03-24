import asyncio
import base64
import binascii
from collections.abc import Callable
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from functools import partial
import http.client
from http.client import HTTPMessage
import importlib
import inspect
import ipaddress
from io import BytesIO
import logging
import mimetypes
from pathlib import Path
import socket
import ssl
import shutil
import threading
from typing import Any, Protocol
from urllib.parse import ParseResult, urljoin, urlparse
from uuid import uuid4

from PIL import Image, UnidentifiedImageError
import torch

from .config import ExecutionMode, ServiceConfig, ServingPipelineType
from .models import (
    FileObjectResponse,
    GenerateJobRequest,
    HealthResponse,
    ImageConditioningRequest,
    is_local_path_source,
    JobStatus,
    RequestInputError,
    UPLOAD_SOURCE_PREFIX,
    UploadedImagePayload,
    VideoGenerationResponse,
    WorkerErrorResponse,
    WorkerStatusResponse,
)


UploadedImageInput = UploadedImagePayload

logger = logging.getLogger(__name__)
progress_logger = logging.getLogger("uvicorn.error")
_tqdm_patch_lock = threading.Lock()
_tqdm_patch_depths: dict[int, tuple[object, Any, int]] = {}


class PipelineRunner(Protocol):
    def generate(
        self,
        request: GenerateJobRequest,
        output_path: Path,
        progress_callback: Callable[[str, int, int], None] | None = None,
    ) -> None: ...


@dataclass
class WorkerProgressState:
    job_id: str
    phase: str
    current: int
    total: int


@contextmanager
def _suppress_module_tqdm(module: object):
    module_key = id(module)
    with _tqdm_patch_lock:
        patched_state = _tqdm_patch_depths.get(module_key)
        if patched_state is None:
            original_tqdm = getattr(module, "tqdm")
            setattr(module, "tqdm", lambda iterable, *args, **kwargs: iterable)
            _tqdm_patch_depths[module_key] = (module, original_tqdm, 1)
        else:
            stored_module, original_tqdm, depth = patched_state
            _tqdm_patch_depths[module_key] = (stored_module, original_tqdm, depth + 1)
    try:
        yield
    finally:
        with _tqdm_patch_lock:
            stored_module, original_tqdm, depth = _tqdm_patch_depths[module_key]
            if depth == 1:
                setattr(stored_module, "tqdm", original_tqdm)
                del _tqdm_patch_depths[module_key]
            else:
                _tqdm_patch_depths[module_key] = (stored_module, original_tqdm, depth - 1)


def _video_with_progress(
    video: torch.Tensor | Any,
    progress_callback: Callable[[str, int, int], None] | None,
    *,
    total_chunks: int,
) -> torch.Tensor | Any:
    if progress_callback is None:
        return video

    progress_callback("encode", 0, total_chunks)
    if isinstance(video, torch.Tensor):
        progress_callback("encode", 1, total_chunks)
        return video

    def iterator():
        for chunk_index, chunk in enumerate(video, start=1):
            progress_callback("encode", chunk_index, total_chunks)
            yield chunk

    return iterator()


@dataclass
class OneStagePipelineRunner:
    pipeline: Any

    def generate(
        self,
        request: GenerateJobRequest,
        output_path: Path,
        progress_callback: Callable[[str, int, int], None] | None = None,
    ) -> None:
        media_io_module = importlib.import_module("ltx_pipelines.utils.media_io")
        encode_video = getattr(media_io_module, "encode_video")

        with torch.no_grad():
            video, audio = self.pipeline(
                prompt=request.prompt,
                negative_prompt=request.negative_prompt,
                seed=request.seed,
                height=request.height,
                width=request.width,
                num_frames=request.num_frames,
                frame_rate=request.frame_rate,
                num_inference_steps=request.num_inference_steps,
                video_guider_params=request.video_guidance.to_params(),
                audio_guider_params=request.audio_guidance.to_params(),
                images=request.to_pipeline_images(),
                enhance_prompt=request.enhance_prompt,
                progress_callback=progress_callback,
            )
            with _suppress_module_tqdm(media_io_module):
                encode_video(
                    video=_video_with_progress(video, progress_callback, total_chunks=1),
                    fps=int(request.frame_rate),
                    audio=audio,
                    output_path=output_path.as_posix(),
                    video_chunks_number=1,
                )

    def close(self) -> None:
        release = getattr(self.pipeline, "release_cached_models", None)
        if callable(release):
            release()


@dataclass
class DistilledPipelineRunner:
    pipeline: Any

    def generate(
        self,
        request: GenerateJobRequest,
        output_path: Path,
        progress_callback: Callable[[str, int, int], None] | None = None,
    ) -> None:
        video_vae_module = importlib.import_module("ltx_core.model.video_vae")
        TilingConfig = getattr(video_vae_module, "TilingConfig")
        get_video_chunks_number = getattr(video_vae_module, "get_video_chunks_number")
        media_io_module = importlib.import_module("ltx_pipelines.utils.media_io")
        encode_video = getattr(media_io_module, "encode_video")

        with torch.inference_mode():
            tiling_config = TilingConfig.default()
            video_chunks_number = get_video_chunks_number(request.num_frames, tiling_config)
            video, audio = self.pipeline(
                prompt=request.prompt,
                seed=request.seed,
                height=request.height,
                width=request.width,
                num_frames=request.num_frames,
                frame_rate=request.frame_rate,
                images=request.to_pipeline_images(),
                tiling_config=tiling_config,
                enhance_prompt=request.enhance_prompt,
                progress_callback=progress_callback,
            )
            with _suppress_module_tqdm(media_io_module):
                encode_video(
                    video=_video_with_progress(video, progress_callback, total_chunks=video_chunks_number),
                    fps=int(request.frame_rate),
                    audio=audio,
                    output_path=output_path.as_posix(),
                    video_chunks_number=video_chunks_number,
                )

    def close(self) -> None:
        release = getattr(self.pipeline, "release_cached_models", None)
        if callable(release):
            release()


@dataclass
class TwoStagePipelineRunner:
    pipeline: Any

    def generate(
        self,
        request: GenerateJobRequest,
        output_path: Path,
        progress_callback: Callable[[str, int, int], None] | None = None,
    ) -> None:
        video_vae_module = importlib.import_module("ltx_core.model.video_vae")
        TilingConfig = getattr(video_vae_module, "TilingConfig")
        get_video_chunks_number = getattr(video_vae_module, "get_video_chunks_number")
        media_io_module = importlib.import_module("ltx_pipelines.utils.media_io")
        encode_video = getattr(media_io_module, "encode_video")

        with torch.inference_mode():
            tiling_config = TilingConfig.default()
            video_chunks_number = get_video_chunks_number(request.num_frames, tiling_config)
            video, audio = self.pipeline(
                prompt=request.prompt,
                negative_prompt=request.negative_prompt,
                seed=request.seed,
                height=request.height,
                width=request.width,
                num_frames=request.num_frames,
                frame_rate=request.frame_rate,
                num_inference_steps=request.num_inference_steps,
                video_guider_params=request.video_guidance.to_params(),
                audio_guider_params=request.audio_guidance.to_params(),
                images=request.to_pipeline_images(),
                tiling_config=tiling_config,
                enhance_prompt=request.enhance_prompt,
                progress_callback=progress_callback,
            )
            with _suppress_module_tqdm(media_io_module):
                encode_video(
                    video=_video_with_progress(video, progress_callback, total_chunks=video_chunks_number),
                    fps=int(request.frame_rate),
                    audio=audio,
                    output_path=output_path.as_posix(),
                    video_chunks_number=video_chunks_number,
                )

    def close(self) -> None:
        release = getattr(self.pipeline, "release_cached_models", None)
        if callable(release):
            release()


@dataclass
class JobRecord:
    job_id: str
    request: GenerateJobRequest
    uploaded_files: dict[str, UploadedImagePayload]
    output_path: Path
    status: JobStatus
    created_at: datetime
    started_at: datetime | None = None
    finished_at: datetime | None = None
    error: str | None = None
    worker_error: WorkerErrorResponse | None = None

    def to_generation_response(self) -> VideoGenerationResponse:
        return VideoGenerationResponse(
            id=self.job_id,
            status=self.status,
            error=self.error,
            worker_error=self.worker_error,
            created_at=int(self.created_at.timestamp()),
            started_at=int(self.started_at.timestamp()) if self.started_at is not None else None,
            finished_at=int(self.finished_at.timestamp()) if self.finished_at is not None else None,
        )

    def to_file_response(self) -> FileObjectResponse:
        size_bytes = self.output_path.stat().st_size if self.output_path.exists() else 0
        created_unix = int(self.created_at.timestamp())
        return FileObjectResponse(
            id=self.job_id,
            bytes=size_bytes,
            created_at=created_unix,
            filename=self.output_path.name,
        )


def build_default_runner(
    config: ServiceConfig,
    *,
    execution_mode: ExecutionMode,
    gpu_ids: tuple[int, ...],
    primary_device: torch.device,
) -> PipelineRunner:
    loader_module = importlib.import_module("ltx_core.loader")
    distilled_module = importlib.import_module("ltx_pipelines.distilled")
    one_stage_module = importlib.import_module("ltx_pipelines.ti2vid_one_stage")
    two_stage_module = importlib.import_module("ltx_pipelines.ti2vid_two_stages")

    LTXV_LORA_COMFY_RENAMING_MAP = getattr(loader_module, "LTXV_LORA_COMFY_RENAMING_MAP")
    LoraPathStrengthAndSDOps = getattr(loader_module, "LoraPathStrengthAndSDOps")
    DistilledPipeline = getattr(distilled_module, "DistilledPipeline")
    TI2VidOneStagePipeline = getattr(one_stage_module, "TI2VidOneStagePipeline")
    TI2VidTwoStagesPipeline = getattr(two_stage_module, "TI2VidTwoStagesPipeline")

    if execution_mode is ExecutionMode.SHARDED:
        raise ValueError("Standalone ltx-service currently supports single-device official pipelines only.")

    if config.pipeline_type is ServingPipelineType.TI2VID_ONE_STAGE:
        pipeline = TI2VidOneStagePipeline(
            checkpoint_path=config.require_checkpoint_path().as_posix(),
            gemma_root=config.require_gamma_path().as_posix(),
            loras=(),
            device=primary_device,
            quantization=config.quantization,
            keep_stage_weights_on_gpu=config.keep_stage_weights_on_gpu,
            keep_model_weights_on_gpu=config.keep_model_weights_on_gpu,
        )
        return OneStagePipelineRunner(pipeline=pipeline)

    if config.pipeline_type is ServingPipelineType.DISTILLED:
        pipeline = DistilledPipeline(
            distilled_checkpoint_path=config.require_distilled_checkpoint_path().as_posix(),
            gemma_root=config.require_gamma_path().as_posix(),
            spatial_upsampler_path=config.require_spatial_upsampler_path().as_posix(),
            loras=(),
            device=primary_device,
            quantization=config.quantization,
            keep_stage_weights_on_gpu=config.keep_stage_weights_on_gpu,
            keep_model_weights_on_gpu=config.keep_model_weights_on_gpu,
        )
        return DistilledPipelineRunner(pipeline=pipeline)

    distilled_lora = (
        LoraPathStrengthAndSDOps(
            config.require_distilled_lora_path().as_posix(),
            1.0,
            LTXV_LORA_COMFY_RENAMING_MAP,
        ),
    )
    pipeline = TI2VidTwoStagesPipeline(
        checkpoint_path=config.require_checkpoint_path().as_posix(),
        distilled_lora=distilled_lora,
        spatial_upsampler_path=config.require_spatial_upsampler_path().as_posix(),
        gemma_root=config.require_gamma_path().as_posix(),
        loras=(),
        device=primary_device,
        quantization=config.quantization,
        keep_stage_weights_on_gpu=config.keep_stage_weights_on_gpu,
        keep_model_weights_on_gpu=config.keep_model_weights_on_gpu,
    )
    return TwoStagePipelineRunner(pipeline=pipeline)


class PipelineServiceBackend:
    def __init__(
        self,
        config: ServiceConfig,
        runner_factory: Callable[..., PipelineRunner] | None = None,
    ):
        self.config = config
        if runner_factory is None:
            self.config.require_cuda_for_official_pipelines()
        self.output_dir = config.output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self._gpu_ids = config.visible_gpu_ids()
        self._execution_mode = config.resolved_execution_mode()
        self._primary_device = config.primary_device()
        self._worker_devices = config.worker_devices()

        self._custom_runner_factory = runner_factory
        self._runners: dict[int, PipelineRunner] = {}
        self._jobs: dict[str, JobRecord] = {}
        self._queue: asyncio.Queue[str | None] = asyncio.Queue()
        self._worker_tasks: list[asyncio.Task[None]] = []
        self._accepting_jobs = False
        self._progress_lock = threading.Lock()
        self._worker_progress: dict[int, WorkerProgressState | None] = {
            worker_index: None for worker_index, _ in enumerate(self._worker_devices)
        }
        self._worker_errors: dict[int, WorkerErrorResponse | None] = {
            worker_index: None for worker_index, _ in enumerate(self._worker_devices)
        }
        self._last_progress_line: str | None = None

    async def start(self) -> None:
        if not self._worker_tasks:
            self._accepting_jobs = True
            self._worker_tasks = [
                asyncio.create_task(self._run_worker(worker_index, device))
                for worker_index, device in enumerate(self._worker_devices)
            ]

    async def shutdown(self) -> None:
        if not self._worker_tasks:
            return

        self._accepting_jobs = False
        self._clear_all_worker_progress()
        for _ in self._worker_tasks:
            await self._queue.put(None)
        await asyncio.gather(*self._worker_tasks)
        runners = list(self._runners.values())
        self._runners = {}
        for runner in runners:
            close = getattr(runner, "close", None)
            if callable(close):
                await asyncio.to_thread(close)
        self._worker_tasks = []

    async def submit(
        self,
        request: GenerateJobRequest,
        uploaded_files: dict[str, UploadedImagePayload] | None = None,
    ) -> JobRecord:
        if not self._accepting_jobs or not self._worker_tasks:
            raise RuntimeError("Pipeline service backend is not accepting new jobs.")
        self._validate_request(request)
        job_id = uuid4().hex
        job = JobRecord(
            job_id=job_id,
            request=request,
            uploaded_files=dict(uploaded_files or {}),
            output_path=self.output_dir / f"{job_id}.mp4",
            status=JobStatus.QUEUED,
            created_at=_utcnow(),
        )
        self._jobs[job.job_id] = job
        self._queue.put_nowait(job.job_id)
        return job

    async def submit_with_uploads(
        self,
        request: GenerateJobRequest,
        *,
        uploaded_files: dict[str, UploadedImagePayload] | None = None,
    ) -> JobRecord:
        return await self.submit(request, uploaded_files=uploaded_files)

    def get_job(self, job_id: str) -> JobRecord | None:
        return self._jobs.get(job_id)

    def get_file(self, file_id: str) -> JobRecord | None:
        for job in self._jobs.values():
            if job.job_id == file_id:
                return job
        return None

    def health(self) -> HealthResponse:
        loaded_runner_count = len(self._runners)
        workers = self._worker_statuses()
        return HealthResponse(
            status="degraded" if any(worker.status == "error" for worker in workers) else "ok",
            pipeline_loaded=loaded_runner_count > 0,
            queue_depth=self._queue.qsize(),
            pipeline_type=self.config.pipeline_type,
            execution_mode=self._execution_mode,
            primary_device=str(self._primary_device),
            gpu_ids=list(self._gpu_ids),
            worker_count=len(self._worker_devices),
            loaded_runner_count=loaded_runner_count,
            workers=workers,
        )

    async def _run_worker(self, worker_index: int, device: torch.device) -> None:
        while True:
            job_id = await self._queue.get()
            if job_id is None:
                self._set_worker_progress(worker_index, None)
                self._queue.task_done()
                break
            job = self._jobs[job_id]
            job.status = JobStatus.RUNNING
            job.started_at = _utcnow()
            cleanup_dir = self._job_input_dir(job.job_id) if job.request.requires_input_materialization() else None
            generation_error: str | None = None
            worker_error: WorkerErrorResponse | None = None
            failure_phase = "materialize_inputs"
            quarantine_worker = False
            try:
                self._report_progress(worker_index, job.job_id, "preparing", 0, 0)
                materialized_request = await asyncio.to_thread(self._materialize_request_inputs, job)
                job.request = materialized_request
                self._report_progress(worker_index, job.job_id, "loading_runner", 0, 0)
                failure_phase = "runner_init"
                runner = await asyncio.to_thread(self._ensure_runner, worker_index, device)
                self._report_progress(worker_index, job.job_id, "generating", 0, 0)
                failure_phase = "generation"
                progress_callback = self._make_worker_progress_callback(worker_index, job.job_id)
                await asyncio.to_thread(self._run_generation, runner, materialized_request, job.output_path, progress_callback)
            except Exception as exc:
                worker_error = self._build_worker_error(worker_index, device, failure_phase, exc)
                generation_error = self._format_worker_error(worker_error)
                quarantine_worker = failure_phase == "runner_init" and worker_index not in self._runners
                logger.exception(
                    "Worker %s on %s failed during %s for job %s.",
                    worker_index,
                    worker_error.device,
                    failure_phase,
                    job.job_id,
                )
            finally:
                job.uploaded_files = {}
                if cleanup_dir is not None:
                    await asyncio.to_thread(shutil.rmtree, cleanup_dir, True)
                job.finished_at = _utcnow()
                if generation_error is None:
                    job.status = JobStatus.SUCCEEDED
                    job.error = None
                    job.worker_error = None
                    self._set_worker_error(worker_index, None)
                else:
                    job.status = JobStatus.FAILED
                    job.error = generation_error
                    job.worker_error = worker_error
                    self._set_worker_error(worker_index, worker_error)
                self._set_worker_progress(worker_index, None)
                self._queue.task_done()
            if quarantine_worker:
                logger.error(
                    "Worker %s on %s entered persistent error state and will stop consuming new jobs.",
                    worker_index,
                    str(device),
                )
                break

    def _ensure_runner(self, worker_index: int, device: torch.device) -> PipelineRunner:
        if worker_index not in self._runners:
            self._runners[worker_index] = self._build_runner_for_device(worker_index, device)
        return self._runners[worker_index]

    def _build_runner_for_device(self, worker_index: int, device: torch.device) -> PipelineRunner:
        if self._custom_runner_factory is None:
            return build_default_runner(
                self.config,
                execution_mode=self._execution_mode,
                gpu_ids=self._gpu_ids,
                primary_device=device,
            )

        factory = self._custom_runner_factory
        signature = inspect.signature(factory)
        if not signature.parameters:
            return factory()

        available_kwargs = {
            "device": device,
            "gpu_id": self._gpu_ids[worker_index] if worker_index < len(self._gpu_ids) else None,
            "worker_index": worker_index,
        }
        if any(parameter.kind is inspect.Parameter.VAR_KEYWORD for parameter in signature.parameters.values()):
            return factory(**available_kwargs)

        accepted_kwargs = {
            name: value for name, value in available_kwargs.items() if name in signature.parameters
        }
        return factory(**accepted_kwargs)

    def _run_generation(
        self,
        runner: PipelineRunner,
        request: GenerateJobRequest,
        output_path: Path,
        progress_callback: Callable[[str, int, int], None],
    ) -> None:
        signature = inspect.signature(runner.generate)
        if any(parameter.kind is inspect.Parameter.VAR_KEYWORD for parameter in signature.parameters.values()):
            runner.generate(request, output_path, progress_callback=progress_callback)
            return

        if "progress_callback" in signature.parameters:
            runner.generate(request, output_path, progress_callback=progress_callback)
            return

        runner.generate(request, output_path)

    def _make_worker_progress_callback(self, worker_index: int, job_id: str) -> Callable[[str, int, int], None]:
        def callback(phase: str, current: int, total: int) -> None:
            self._report_progress(worker_index, job_id, phase, current, total)

        return callback

    def _gpu_id_for_worker(self, worker_index: int) -> int | None:
        return self._gpu_ids[worker_index] if worker_index < len(self._gpu_ids) else None

    def _build_worker_error(
        self,
        worker_index: int,
        device: torch.device,
        phase: str,
        exc: Exception,
    ) -> WorkerErrorResponse:
        message = str(exc).strip() or type(exc).__name__
        return WorkerErrorResponse(
            worker_index=worker_index,
            gpu_id=self._gpu_id_for_worker(worker_index),
            device=str(device),
            phase=phase,
            error_type=type(exc).__name__,
            message=message,
        )

    def _format_worker_error(self, worker_error: WorkerErrorResponse) -> str:
        location = f"Worker {worker_error.worker_index}"
        if worker_error.gpu_id is not None:
            location += f" (gpu {worker_error.gpu_id}, {worker_error.device})"
        else:
            location += f" ({worker_error.device})"
        return (
            f"{location} failed during {worker_error.phase}: "
            f"{worker_error.error_type}: {worker_error.message}"
        )

    def _report_progress(self, worker_index: int, job_id: str, phase: str, current: int, total: int) -> None:
        self._set_worker_progress(
            worker_index,
            WorkerProgressState(job_id=job_id, phase=phase, current=current, total=total),
        )

    def _set_worker_progress(self, worker_index: int, progress: WorkerProgressState | None) -> None:
        with self._progress_lock:
            self._worker_progress[worker_index] = progress
            line = self._render_worker_progress_line()
            if line == self._last_progress_line:
                return
            self._last_progress_line = line
        progress_logger.info(line)

    def _clear_all_worker_progress(self) -> None:
        with self._progress_lock:
            for worker_index in self._worker_progress:
                self._worker_progress[worker_index] = None
            self._last_progress_line = None

    def _set_worker_error(self, worker_index: int, worker_error: WorkerErrorResponse | None) -> None:
        with self._progress_lock:
            self._worker_errors[worker_index] = worker_error

    def _worker_statuses(self) -> list[WorkerStatusResponse]:
        with self._progress_lock:
            progress_snapshot = dict(self._worker_progress)
            error_snapshot = dict(self._worker_errors)

        workers = []
        for worker_index, device in enumerate(self._worker_devices):
            progress = progress_snapshot[worker_index]
            error = error_snapshot[worker_index]
            runner_loaded = worker_index in self._runners

            if progress is not None:
                status = "running"
            elif error is not None and error.phase == "runner_init" and not runner_loaded:
                status = "error"
            elif runner_loaded:
                status = "ready"
            else:
                status = "idle"

            workers.append(
                WorkerStatusResponse(
                    worker_index=worker_index,
                    gpu_id=self._gpu_id_for_worker(worker_index),
                    device=str(device),
                    status=status,
                    runner_loaded=runner_loaded,
                    current_job_id=progress.job_id if progress is not None else None,
                    current_phase=progress.phase if progress is not None else None,
                    error=error,
                )
            )
        return workers

    def _render_worker_progress_line(self) -> str:
        worker_segments = []
        for worker_index in sorted(self._worker_progress):
            progress = self._worker_progress[worker_index]
            if progress is None:
                worker_segments.append(f"w{worker_index}: idle")
                continue
            job_prefix = progress.job_id[:8]
            if progress.total > 0:
                percentage = int(progress.current * 100 / progress.total)
                worker_segments.append(
                    f"w{worker_index}: {job_prefix} {progress.phase} {progress.current}/{progress.total} ({percentage}%)"
                )
            else:
                worker_segments.append(f"w{worker_index}: {job_prefix} {progress.phase}")
        return "Worker progress | " + " | ".join(worker_segments)

    def _validate_request(self, request: GenerateJobRequest) -> None:
        if self.config.pipeline_type in {ServingPipelineType.DISTILLED, ServingPipelineType.TI2VID_TWO_STAGES}:
            if request.height % 64 != 0 or request.width % 64 != 0:
                raise RequestInputError(
                    f"height and width must be divisible by 64 for the {self.config.pipeline_type.value} pipeline."
                )

    def _materialize_request_inputs(self, job: JobRecord) -> GenerateJobRequest:
        if not job.request.requires_input_materialization():
            return job.request

        input_dir = self._job_input_dir(job.job_id)
        input_dir.mkdir(parents=True, exist_ok=True)
        images = [
            self._materialize_image_input(
                image=image,
                image_index=image_index,
                input_dir=input_dir,
                uploaded_files=job.uploaded_files,
            )
            for image_index, image in enumerate(job.request.images)
        ]
        return job.request.model_copy(update={"images": images})

    def _materialize_image_input(
        self,
        *,
        image: ImageConditioningRequest,
        image_index: int,
        input_dir: Path,
        uploaded_files: dict[str, UploadedImagePayload],
    ) -> ImageConditioningRequest:
        upload_field = _extract_upload_field(image.source)
        if upload_field is not None:
            upload = uploaded_files.get(upload_field)
            if upload is None:
                raise ValueError(f"Multipart upload field '{upload_field}' was not provided.")
            content = upload.content
            content_type = upload.content_type
            source_name = upload.filename
        elif _is_http_url(image.source):
            content, content_type, source_name = _download_image_from_url(image.source)
        elif image.source.startswith("data:"):
            content, content_type = _decode_base64_image(image.source)
            source_name = None
        elif is_local_path_source(image.source):
            return image
        else:
            raw_base64 = _try_decode_base64_image(image.source)
            if raw_base64 is None:
                return image
            content, content_type = raw_base64
            source_name = None

        if not content:
            raise ValueError("Image sources must not be empty.")

        suffix = _guess_image_suffix(source_name=source_name, content_type=content_type, content=content)
        materialized_path = input_dir / f"image-{image_index}{suffix}"
        materialized_path.write_bytes(content)
        return image.to_materialized_path(materialized_path.as_posix())

    def _job_input_dir(self, job_id: str) -> Path:
        return self.output_dir / ".job_inputs" / job_id


def _download_image_from_url(url: str) -> tuple[bytes, str | None, str | None]:
    content, headers, final_url = _fetch_public_url(url, timeout=10)
    parsed = urlparse(final_url)
    content_type = None
    if headers is not None:
        if hasattr(headers, "get_content_type"):
            content_type = headers.get_content_type()
        elif hasattr(headers, "get"):
            content_type = headers.get("Content-Type")
    return content, content_type, parsed.path


def _decode_base64_image(encoded_image: str) -> tuple[bytes, str | None]:
    base64_content = encoded_image.strip()
    content_type = None
    if base64_content.startswith("data:"):
        header, separator, payload = base64_content.partition(",")
        if not separator:
            raise ValueError("base64 image data URL is invalid.")
        if ";base64" not in header:
            raise ValueError("base64 image data URL must include ';base64'.")
        content_type = header[5:].split(";", 1)[0] or None
        base64_content = payload.strip()

    try:
        return base64.b64decode(base64_content, validate=True), content_type
    except (binascii.Error, ValueError) as exc:
        raise ValueError("base64 image data is invalid.") from exc


def _try_decode_base64_image(encoded_image: str) -> tuple[bytes, str | None] | None:
    try:
        return _decode_base64_image(encoded_image)
    except ValueError:
        return None


def _guess_image_suffix(*, source_name: str | None, content_type: str | None, content: bytes) -> str:
    header_suffix = _guess_suffix_from_image_header(content)
    if header_suffix is None:
        raise ValueError("Image source could not be decoded as an image.")

    return (
        _guess_suffix_from_name(source_name)
        or header_suffix
        or _guess_suffix_from_bytes(content)
        or _guess_suffix_from_content_type(content_type)
        or ".img"
    )


def _guess_suffix_from_name(source_name: str | None) -> str | None:
    if not source_name:
        return None
    suffix = Path(source_name).suffix.strip()
    return suffix or None


def _guess_suffix_from_content_type(content_type: str | None) -> str | None:
    if not content_type:
        return None
    normalized = content_type.split(";", 1)[0].strip().lower()
    if not normalized.startswith("image/"):
        return None
    suffix = mimetypes.guess_extension(normalized)
    if suffix == ".jpe":
        return ".jpg"
    return suffix


def _guess_suffix_from_image_header(content: bytes) -> str | None:
    try:
        with Image.open(BytesIO(content)) as image:
            image_format = image.format
    except (UnidentifiedImageError, OSError, ValueError):
        return None

    if image_format is None:
        return None

    normalized = image_format.lower()
    if normalized == "jpeg":
        return ".jpg"
    return f".{normalized}"


def _extract_upload_field(source: str) -> str | None:
    if not source.startswith(UPLOAD_SOURCE_PREFIX):
        return None
    field_name = source[len(UPLOAD_SOURCE_PREFIX) :].strip()
    return field_name or None


def _is_http_url(source: str) -> bool:
    parsed = urlparse(source)
    return parsed.scheme in {"http", "https"} and bool(parsed.netloc)


def _fetch_public_url(url: str, *, timeout: float, redirect_count: int = 0) -> tuple[bytes, HTTPMessage, str]:
    if redirect_count >= 5:
        raise ValueError("Image URL redirected too many times.")

    parsed, resolved_ip = _resolve_public_http_url(url)
    status_code, headers, content = _request_pinned_url(parsed, resolved_ip=resolved_ip, timeout=timeout)

    if 300 <= status_code < 400:
        location = headers.get("Location") if hasattr(headers, "get") else None
        if not isinstance(location, str) or not location.strip():
            raise ValueError("Image URL redirect response was missing a Location header.")
        redirected_url = urljoin(url, location)
        return _fetch_public_url(redirected_url, timeout=timeout, redirect_count=redirect_count + 1)

    if status_code >= 400:
        raise ValueError(f"Image URL request failed with HTTP {status_code}.")

    return content, headers, url


def _resolve_public_http_url(url: str) -> tuple[ParseResult, str]:
    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        raise ValueError("Image URLs must use http or https.")

    try:
        addrinfo = socket.getaddrinfo(parsed.hostname, parsed.port or _default_port_for_scheme(parsed.scheme), type=socket.SOCK_STREAM)
    except socket.gaierror as exc:
        raise ValueError(f"Image URL host '{parsed.hostname}' could not be resolved.") from exc

    resolved_ip: str | None = None
    for _, _, _, _, sockaddr in addrinfo:
        sockaddr_host = str(sockaddr[0])
        ip_address = ipaddress.ip_address(sockaddr_host)
        if not ip_address.is_global:
            raise ValueError("Image URLs must resolve to public IP addresses.")
        if resolved_ip is None:
            resolved_ip = sockaddr_host

    if resolved_ip is None:
        raise ValueError(f"Image URL host '{parsed.hostname}' did not resolve to a usable address.")

    return parsed, resolved_ip


def _default_port_for_scheme(scheme: str) -> int:
    return 443 if scheme == "https" else 80


class _PinnedHTTPConnection(http.client.HTTPConnection):
    def __init__(self, *, connection_host: str, resolved_ip: str, timeout: float):
        super().__init__(host=connection_host, timeout=timeout)
        self._resolved_ip = resolved_ip

    def connect(self) -> None:
        self.sock = socket.create_connection((self._resolved_ip, self.port), self.timeout)


class _PinnedHTTPSConnection(http.client.HTTPSConnection):
    def __init__(self, *, connection_host: str, resolved_ip: str, timeout: float):
        super().__init__(host=connection_host, timeout=timeout)
        self._resolved_ip = resolved_ip
        self._ssl_context = ssl.create_default_context()

    def connect(self) -> None:
        raw_socket = socket.create_connection((self._resolved_ip, self.port), self.timeout)
        self.sock = self._ssl_context.wrap_socket(raw_socket, server_hostname=self.host)


def _request_pinned_url(parsed: ParseResult, *, resolved_ip: str, timeout: float) -> tuple[int, HTTPMessage, bytes]:
    request_target = parsed.path or "/"
    if parsed.query:
        request_target = f"{request_target}?{parsed.query}"

    if parsed.scheme == "https":
        connection = _PinnedHTTPSConnection(
            connection_host=parsed.hostname or "",
            resolved_ip=resolved_ip,
            timeout=timeout,
        )
    else:
        connection = _PinnedHTTPConnection(
            connection_host=parsed.hostname or "",
            resolved_ip=resolved_ip,
            timeout=timeout,
        )

    try:
        connection.request("GET", request_target, headers={"User-Agent": "ltx-service/1.0"})
        response = connection.getresponse()
        return response.status, response.headers, response.read()
    finally:
        connection.close()


def _guess_suffix_from_bytes(content: bytes) -> str | None:
    if content.startswith(b"\x89PNG\r\n\x1a\n"):
        return ".png"
    if content.startswith(b"\xff\xd8\xff"):
        return ".jpg"
    if content.startswith((b"GIF87a", b"GIF89a")):
        return ".gif"
    if content.startswith(b"BM"):
        return ".bmp"
    if content.startswith(b"II*\x00") or content.startswith(b"MM\x00*"):
        return ".tiff"
    if len(content) >= 12 and content.startswith(b"RIFF") and content[8:12] == b"WEBP":
        return ".webp"
    return None


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)
