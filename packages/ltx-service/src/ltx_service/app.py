from email.parser import BytesParser
from email.policy import default
import json
from contextlib import asynccontextmanager
from json import JSONDecodeError

from fastapi import FastAPI, HTTPException, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse
from pydantic import ValidationError

from .backend import PipelineServiceBackend, UploadedImageInput
from .config import ServiceConfig
from .models import (
    FileObjectResponse,
    GenerateJobRequest,
    HealthResponse,
    JobStatus,
    RequestInputError,
    UPLOAD_SOURCE_PREFIX,
    VideoGenerationResponse,
)


def create_app(config: ServiceConfig, backend: PipelineServiceBackend | None = None) -> FastAPI:
    service_backend = backend or PipelineServiceBackend(config)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        await service_backend.start()
        app.state.backend = service_backend
        try:
            yield
        finally:
            await service_backend.shutdown()

    app = FastAPI(title="LTX Service", lifespan=lifespan)

    @app.get("/health", response_model=HealthResponse)
    async def health(request: Request) -> HealthResponse:
        return _get_backend(request).health()

    @app.post("/v1/videos", response_model=VideoGenerationResponse, status_code=status.HTTP_202_ACCEPTED)
    async def create_video(request: Request) -> VideoGenerationResponse:
        payload, uploaded_files = await _parse_generation_request(request)
        try:
            job = await _get_backend(request).submit_with_uploads(payload, uploaded_files=uploaded_files)
        except RequestInputError as exc:
            raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_CONTENT, detail=str(exc)) from exc
        except RuntimeError as exc:
            raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(exc)) from exc
        return job.to_generation_response()

    @app.get("/v1/videos/{video_id}", response_model=VideoGenerationResponse)
    async def video_status(request: Request, video_id: str) -> VideoGenerationResponse:
        job = _get_backend(request).get_job(video_id)
        if job is None:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Video not found.")
        return job.to_generation_response()

    @app.get("/v1/files/{file_id}", response_model=FileObjectResponse)
    async def file_metadata(request: Request, file_id: str) -> FileObjectResponse:
        job = _get_backend(request).get_file(file_id)
        if job is None:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="File not found.")
        _ensure_file_is_available(job)
        return job.to_file_response()

    @app.get("/v1/files/{file_id}/content")
    async def file_content(request: Request, file_id: str) -> FileResponse:
        job = _get_backend(request).get_file(file_id)
        if job is None:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="File not found.")
        _ensure_file_is_available(job)
        return FileResponse(job.output_path, media_type="video/mp4", filename=job.output_path.name)

    return app


def _get_backend(request: Request) -> PipelineServiceBackend:
    backend = getattr(request.app.state, "backend", None)
    if not isinstance(backend, PipelineServiceBackend):
        raise RuntimeError("Application backend is not initialized.")
    return backend


def _ensure_file_is_available(job) -> None:
    if job.status is JobStatus.FAILED:
        raise HTTPException(status_code=status.HTTP_410_GONE, detail="File is unavailable because generation failed.")
    if job.status is not JobStatus.SUCCEEDED:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="File is not ready yet.")


async def _parse_generation_request(request: Request) -> tuple[GenerateJobRequest, dict[str, UploadedImageInput]]:
    content_type = request.headers.get("content-type", "")
    if content_type.startswith("application/json"):
        payload = _validate_generation_payload(await request.body())
        _reject_json_upload_sources(payload)
        return payload, {}

    if content_type.startswith("multipart/form-data"):
        return await _parse_multipart_generation_request(request)

    raise HTTPException(
        status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
        detail="Supported content types: application/json and multipart/form-data.",
    )


def _validate_generation_payload(raw_payload: bytes | str) -> GenerateJobRequest:
    try:
        raw_payload_text = raw_payload.decode("utf-8") if isinstance(raw_payload, bytes) else raw_payload
    except UnicodeDecodeError as exc:
        raise RequestValidationError(
            [
                _validation_error_detail(
                    loc=("body",),
                    msg="Invalid JSON payload: request body must be UTF-8.",
                    input_value=None,
                )
            ]
        ) from exc
    try:
        payload = json.loads(raw_payload_text)
    except JSONDecodeError as exc:
        raise RequestValidationError(
            [
                _validation_error_detail(
                    loc=("body",),
                    msg=f"Invalid JSON payload: {exc.msg}.",
                    input_value=raw_payload_text,
                )
            ]
        ) from exc

    try:
        return GenerateJobRequest.model_validate(payload)
    except ValidationError as exc:
        raise RequestValidationError(exc.errors()) from exc


async def _parse_multipart_generation_request(
    request: Request,
) -> tuple[GenerateJobRequest, dict[str, UploadedImageInput]]:
    content_type = request.headers.get("content-type", "")
    message = BytesParser(policy=default).parsebytes(
        f"Content-Type: {content_type}\r\nMIME-Version: 1.0\r\n\r\n".encode("utf-8") + await request.body()
    )
    if not message.is_multipart():
        raise RequestValidationError(
            [
                _validation_error_detail(
                    loc=("body", "payload"),
                    msg="Multipart requests require valid multipart/form-data bodies.",
                    input_value=None,
                )
            ]
        )

    payload_field: str | None = None
    uploaded_files: dict[str, UploadedImageInput] = {}
    validation_errors: list[dict[str, object]] = []

    for part in message.iter_parts():
        raw_field_name = part.get_param("name", header="content-disposition")
        if not isinstance(raw_field_name, str):
            continue
        field_name = raw_field_name

        raw_field_bytes = part.get_payload(decode=True)
        field_bytes = raw_field_bytes if isinstance(raw_field_bytes, bytes) else b""
        filename = part.get_filename()
        if filename is None:
            if field_name != "payload":
                continue
            if payload_field is not None:
                validation_errors.append(
                    _validation_error_detail(
                        loc=("body", "payload"),
                        msg="Duplicate 'payload' form fields are not supported.",
                        input_value=field_name,
                    )
                )
                continue
            try:
                payload_field = field_bytes.decode("utf-8")
            except UnicodeDecodeError:
                validation_errors.append(
                    _validation_error_detail(
                        loc=("body", "payload"),
                        msg="The 'payload' form field must be valid UTF-8 JSON text.",
                        input_value=None,
                    )
                )
            continue

        if field_name in uploaded_files:
            validation_errors.append(
                _validation_error_detail(
                    loc=("body", field_name),
                    msg=f"Duplicate upload field '{field_name}' is not supported.",
                    input_value=field_name,
                )
            )
            continue

        uploaded_files[field_name] = UploadedImageInput(
            filename=filename,
            content_type=part.get_content_type(),
            content=field_bytes,
        )

    if payload_field is None:
        validation_errors.append(
            _validation_error_detail(
                loc=("body", "payload"),
                msg="Multipart requests require a JSON string in the 'payload' form field.",
                input_value=None,
            )
        )

    if validation_errors:
        raise RequestValidationError(validation_errors)

    if payload_field is None:
        raise RuntimeError("Multipart payload validation did not produce a payload.")

    payload = _validate_generation_payload(payload_field)
    _validate_uploaded_file_references(payload, uploaded_files)
    return payload, uploaded_files


def _reject_json_upload_sources(payload: GenerateJobRequest) -> None:
    validation_errors = [
        _validation_error_detail(
            loc=("body", "images", index, "source"),
            msg="upload: sources are only supported with multipart/form-data requests.",
            input_value=image.source,
        )
        for index, image in enumerate(payload.images)
        if image.source.startswith(UPLOAD_SOURCE_PREFIX)
    ]
    if validation_errors:
        raise RequestValidationError(validation_errors)


def _validate_uploaded_file_references(
    payload: GenerateJobRequest,
    uploaded_files: dict[str, UploadedImageInput],
) -> None:
    validation_errors = []
    for index, image in enumerate(payload.images):
        if not image.source.startswith(UPLOAD_SOURCE_PREFIX):
            continue
        upload_field = image.source[len(UPLOAD_SOURCE_PREFIX) :].strip()
        if upload_field in uploaded_files:
            continue
        validation_errors.append(
            _validation_error_detail(
                loc=("body", "images", index, "source"),
                msg=f"Multipart upload field '{upload_field}' was not provided.",
                input_value=image.source,
            )
        )
    if validation_errors:
        raise RequestValidationError(validation_errors)


def _validation_error_detail(*, loc: tuple[object, ...], msg: str, input_value: object) -> dict[str, object]:
    return {
        "type": "value_error",
        "loc": loc,
        "msg": msg,
        "input": input_value,
    }
