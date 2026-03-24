import base64
import binascii
from dataclasses import dataclass
from enum import Enum
import importlib
from functools import lru_cache
from io import BytesIO
from pathlib import Path
from urllib.parse import urlparse

from PIL import Image, UnidentifiedImageError
from pydantic import BaseModel, Field, ValidationInfo, field_validator, model_validator

from .config import ExecutionMode, ServingPipelineType


@lru_cache(maxsize=1)
def _constants_module():
    return importlib.import_module("ltx_pipelines.utils.constants")


@lru_cache(maxsize=1)
def _pipeline_args_module():
    return importlib.import_module("ltx_pipelines.utils.args")


@lru_cache(maxsize=1)
def _guiders_module():
    return importlib.import_module("ltx_core.components.guiders")


def _pipeline_params() -> object:
    return getattr(_constants_module(), "LTX_2_3_PARAMS")


def _default_image_crf() -> int:
    return int(getattr(_constants_module(), "DEFAULT_IMAGE_CRF"))


def _default_negative_prompt() -> str:
    return str(getattr(_constants_module(), "DEFAULT_NEGATIVE_PROMPT"))


def _default_seed() -> int:
    return int(getattr(_pipeline_params(), "seed"))


def _default_height() -> int:
    return int(getattr(_pipeline_params(), "stage_2_height"))


def _default_width() -> int:
    return int(getattr(_pipeline_params(), "stage_2_width"))


def _default_num_frames() -> int:
    return int(getattr(_pipeline_params(), "num_frames"))


def _default_frame_rate() -> float:
    return float(getattr(_pipeline_params(), "frame_rate"))


def _default_num_inference_steps() -> int:
    return int(getattr(_pipeline_params(), "num_inference_steps"))


UPLOAD_SOURCE_PREFIX = "upload:"


class GuidanceRequest(BaseModel):
    cfg_scale: float
    stg_scale: float
    rescale_scale: float
    modality_scale: float
    skip_step: int
    stg_blocks: list[int] = Field(default_factory=list)

    @classmethod
    def from_params(cls, params: object) -> "GuidanceRequest":
        return cls(
            cfg_scale=getattr(params, "cfg_scale"),
            stg_scale=getattr(params, "stg_scale"),
            rescale_scale=getattr(params, "rescale_scale"),
            modality_scale=getattr(params, "modality_scale"),
            skip_step=getattr(params, "skip_step"),
            stg_blocks=list(getattr(params, "stg_blocks")),
        )

    def to_params(self) -> object:
        MultiModalGuiderParams = getattr(_guiders_module(), "MultiModalGuiderParams")
        return MultiModalGuiderParams(
            cfg_scale=self.cfg_scale,
            stg_scale=self.stg_scale,
            rescale_scale=self.rescale_scale,
            modality_scale=self.modality_scale,
            skip_step=self.skip_step,
            stg_blocks=list(self.stg_blocks),
        )


def _default_video_guidance() -> GuidanceRequest:
    return GuidanceRequest.from_params(getattr(_pipeline_params(), "video_guider_params"))


def _default_audio_guidance() -> GuidanceRequest:
    return GuidanceRequest.from_params(getattr(_pipeline_params(), "audio_guider_params"))


class ImageConditioningRequest(BaseModel):
    model_config = {"extra": "forbid"}

    source: str
    frame_idx: int
    strength: float
    crf: int = Field(default_factory=lambda: _default_image_crf())

    @field_validator("source")
    @classmethod
    def validate_source(cls, value: str, info: ValidationInfo) -> str:
        stripped = value.strip()
        if not stripped:
            raise ValueError(f"{info.field_name} must not be empty.")

        if stripped.startswith(("http://", "https://")):
            parsed = urlparse(stripped)
            if parsed.scheme not in {"http", "https"} or not parsed.netloc:
                raise ValueError("url must use http or https.")

        if _has_unsupported_source_scheme(stripped):
            raise ValueError("source uses an unsupported URI scheme.")

        if stripped.startswith(UPLOAD_SOURCE_PREFIX) and not stripped[len(UPLOAD_SOURCE_PREFIX) :].strip():
            raise ValueError("upload sources must use the form 'upload:<field-name>'.")

        if stripped.startswith("data:") and not _contains_decodable_base64_image(stripped):
            raise ValueError("data URL sources must contain a decodable base64 image.")

        if not stripped.startswith(("http://", "https://", "data:", UPLOAD_SOURCE_PREFIX)):
            if _contains_decodable_base64_image(stripped):
                return stripped
            if is_local_path_source(stripped):
                return stripped
            raise ValueError(
                "source must be a local path, http/https URL, raw base64 image, data URL, or upload:<field-name>."
            )

        return stripped

    @field_validator("frame_idx")
    @classmethod
    def validate_frame_idx(cls, value: int) -> int:
        if value < 0:
            raise ValueError("frame_idx must be non-negative.")
        return value

    def requires_materialization(self) -> bool:
        return not is_local_path_source(self.source)

    def to_materialized_path(self, path: str) -> "ImageConditioningRequest":
        return self.model_copy(update={"source": path})

    def to_pipeline_input(self) -> object:
        if not is_local_path_source(self.source):
            raise ValueError("Image conditioning inputs must be materialized to local paths before pipeline execution.")
        ImageConditioningInput = getattr(_pipeline_args_module(), "ImageConditioningInput")
        return ImageConditioningInput(
            path=self.expanded_local_path(),
            frame_idx=self.frame_idx,
            strength=self.strength,
            crf=self.crf,
        )

    def expanded_local_path(self) -> str:
        return Path(self.source).expanduser().as_posix()


class GenerateJobRequest(BaseModel):
    model_config = {"extra": "forbid"}

    prompt: str
    negative_prompt: str = Field(default_factory=lambda: _default_negative_prompt())
    seed: int = Field(default_factory=lambda: _default_seed())
    height: int = Field(default_factory=lambda: _default_height())
    width: int = Field(default_factory=lambda: _default_width())
    num_frames: int = Field(default_factory=lambda: _default_num_frames())
    frame_rate: float = Field(default_factory=lambda: _default_frame_rate())
    num_inference_steps: int = Field(default_factory=lambda: _default_num_inference_steps())
    enhance_prompt: bool = False
    images: list[ImageConditioningRequest] = Field(default_factory=list)
    video_guidance: GuidanceRequest = Field(default_factory=_default_video_guidance)
    audio_guidance: GuidanceRequest = Field(default_factory=_default_audio_guidance)

    @field_validator("height", "width")
    @classmethod
    def validate_resolution(cls, value: int) -> int:
        if value <= 0:
            raise ValueError("height and width must be positive.")
        if value % 32 != 0:
            raise ValueError("height and width must be divisible by 32.")
        return value

    @field_validator("num_frames")
    @classmethod
    def validate_num_frames(cls, value: int) -> int:
        if value <= 0:
            raise ValueError("num_frames must be positive.")
        if value % 8 != 1:
            raise ValueError("num_frames must satisfy num_frames % 8 == 1.")
        return value

    @field_validator("frame_rate")
    @classmethod
    def validate_frame_rate(cls, value: float) -> float:
        if value <= 0:
            raise ValueError("frame_rate must be positive.")
        return value

    @field_validator("num_inference_steps")
    @classmethod
    def validate_num_inference_steps(cls, value: int) -> int:
        if value <= 0:
            raise ValueError("num_inference_steps must be positive.")
        return value

    @model_validator(mode="after")
    def validate_images_within_frame_range(self) -> "GenerateJobRequest":
        invalid_frame_indices = [image.frame_idx for image in self.images if image.frame_idx >= self.num_frames]
        if invalid_frame_indices:
            raise ValueError(
                f"image frame indices must be smaller than num_frames; got invalid values {invalid_frame_indices}."
            )
        return self

    def to_pipeline_images(self) -> list[object]:
        return [image.to_pipeline_input() for image in self.images]

    def requires_input_materialization(self) -> bool:
        return any(image.requires_materialization() for image in self.images)


class JobStatus(str, Enum):
    QUEUED = "queued"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"


class WorkerErrorResponse(BaseModel):
    worker_index: int
    gpu_id: int | None = None
    device: str
    phase: str
    error_type: str
    message: str


class WorkerStatusResponse(BaseModel):
    worker_index: int
    gpu_id: int | None = None
    device: str
    status: str
    runner_loaded: bool
    current_job_id: str | None = None
    current_phase: str | None = None
    error: WorkerErrorResponse | None = None


class VideoGenerationResponse(BaseModel):
    id: str
    object: str = "video.generation"
    status: JobStatus
    error: str | None = None
    worker_error: WorkerErrorResponse | None = None
    created_at: int
    started_at: int | None = None
    finished_at: int | None = None


class FileObjectResponse(BaseModel):
    id: str
    object: str = "file"
    bytes: int
    created_at: int
    filename: str
    purpose: str = "generated_video"


class HealthResponse(BaseModel):
    status: str = "ok"
    pipeline_loaded: bool
    queue_depth: int
    pipeline_type: ServingPipelineType
    execution_mode: ExecutionMode
    primary_device: str
    gpu_ids: list[int] = Field(default_factory=list)
    worker_count: int
    loaded_runner_count: int
    workers: list[WorkerStatusResponse] = Field(default_factory=list)


@dataclass(frozen=True)
class UploadedImagePayload:
    filename: str
    content_type: str | None
    content: bytes


class RequestInputError(ValueError):
    pass


def is_local_path_source(source: str) -> bool:
    parsed = urlparse(source)
    if parsed.scheme in {"http", "https"}:
        return False
    if _has_unsupported_source_scheme(source):
        return False
    if source.startswith(("data:", UPLOAD_SOURCE_PREFIX)):
        return False
    if _contains_decodable_base64_image(source):
        return False
    if source.startswith(("/", "./", "../", "~/", "\\", ".\\", "..\\")):
        return True
    if len(source) >= 3 and source[1] == ":" and source[2] in {"/", "\\"}:
        return True
    return "/" in source or "\\" in source or bool(parsed.path and "." in Path(parsed.path).name)


def _has_unsupported_source_scheme(source: str) -> bool:
    parsed = urlparse(source)
    if not parsed.scheme:
        return False
    if source.startswith((UPLOAD_SOURCE_PREFIX, "data:")):
        return False
    if len(source) >= 3 and source[1] == ":" and source[2] in {"/", "\\"}:
        return False
    return source.startswith(f"{parsed.scheme}:") and parsed.scheme not in {"http", "https"}


def _contains_decodable_base64_image(source: str) -> bool:
    try:
        image_bytes = _decode_base64_candidate(source)
    except ValueError:
        return False

    try:
        with Image.open(BytesIO(image_bytes)) as image:
            return image.format is not None
    except (UnidentifiedImageError, OSError, ValueError):
        return False


def _decode_base64_candidate(source: str) -> bytes:
    base64_content = source.strip()
    if base64_content.startswith("data:"):
        header, separator, payload = base64_content.partition(",")
        if not separator:
            raise ValueError("base64 image data URL is invalid.")
        if ";base64" not in header:
            raise ValueError("base64 image data URL must include ';base64'.")
        base64_content = payload.strip()

    return base64.b64decode(base64_content, validate=True)
