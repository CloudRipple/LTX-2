import argparse
from collections.abc import Sequence
from dataclasses import dataclass, field
from enum import Enum
import importlib
import inspect
from pathlib import Path
from typing import Any

import torch

QUANTIZATION_POLICIES = ("fp8-cast", "fp8-scaled-mm")

DEFAULT_OUTPUT_DIR = Path("outputs/ltx-service").resolve()


def _resolve_path(path_value: str) -> str:
    return getattr(_pipeline_args_module(), "resolve_path")(path_value)


def _pipeline_args_module():
    return importlib.import_module("ltx_pipelines.utils.args")


def _quantization_module():
    return importlib.import_module("ltx_core.quantization")


class ServingPipelineType(str, Enum):
    TI2VID_ONE_STAGE = "ti2vid-one-stage"
    DISTILLED = "distilled"
    TI2VID_TWO_STAGES = "ti2vid-two-stages"


class ExecutionMode(str, Enum):
    AUTO = "auto"
    SINGLE = "single"
    SHARDED = "sharded"


class ServiceQuantizationAction(argparse.Action):
    def __call__(
        self,
        parser: argparse.ArgumentParser,  # noqa: ARG002
        namespace: argparse.Namespace,
        values: str | Sequence[object] | None,
        option_string: str | None = None,
    ) -> None:
        QuantizationPolicy = getattr(_quantization_module(), "QuantizationPolicy")

        parsed_values = (
            [str(value) for value in values] if isinstance(values, Sequence) and not isinstance(values, str) else []
        )
        if isinstance(values, str):
            parsed_values = [values]
        if not parsed_values:
            raise argparse.ArgumentError(self, f"{option_string} requires a quantization policy name")

        policy_name = parsed_values[0]
        if policy_name not in QUANTIZATION_POLICIES:
            raise argparse.ArgumentError(
                self,
                f"Unknown quantization policy '{policy_name}'. Choose from: {', '.join(QUANTIZATION_POLICIES)}",
            )

        if policy_name == "fp8-cast":
            if len(parsed_values) > 1:
                raise argparse.ArgumentError(self, f"{option_string} fp8-cast does not accept additional arguments")
            policy = QuantizationPolicy.fp8_cast()
        else:
            if len(parsed_values) > 2:
                raise argparse.ArgumentError(self, f"{option_string} fp8-scaled-mm accepts at most one amax path")
            amax_path = _resolve_path(parsed_values[1]) if len(parsed_values) > 1 else None
            fp8_scaled_mm = QuantizationPolicy.fp8_scaled_mm
            supports_amax_path = len(inspect.signature(fp8_scaled_mm).parameters) == 1
            if amax_path is not None and not supports_amax_path:
                raise argparse.ArgumentError(
                    self,
                    f"{option_string} fp8-scaled-mm does not support an amax path with the installed ltx-core version",
                )
            policy = fp8_scaled_mm(amax_path) if supports_amax_path else fp8_scaled_mm()

        setattr(namespace, self.dest, policy)


@dataclass(frozen=True)
class ServiceConfig:
    pipeline_type: ServingPipelineType = ServingPipelineType.TI2VID_TWO_STAGES
    checkpoint_path: Path | None = None
    distilled_checkpoint_path: Path | None = None
    distilled_lora_path: Path | None = None
    spatial_upsampler_path: Path | None = None
    gamma_path: Path | None = None
    quantization: Any | None = None
    output_dir: Path = DEFAULT_OUTPUT_DIR
    host: str = "127.0.0.1"
    port: int = 8000
    execution_mode: ExecutionMode = ExecutionMode.AUTO
    gpu_ids: tuple[int, ...] = field(default_factory=tuple)
    keep_stage_weights_on_gpu: bool = False
    keep_model_weights_on_gpu: bool = False

    @classmethod
    def from_namespace(cls, args: argparse.Namespace) -> "ServiceConfig":
        return cls(
            pipeline_type=ServingPipelineType(args.pipeline_type),
            checkpoint_path=Path(args.checkpoint_path) if args.checkpoint_path else None,
            distilled_checkpoint_path=Path(args.distilled_checkpoint_path) if args.distilled_checkpoint_path else None,
            distilled_lora_path=Path(args.distilled_lora) if args.distilled_lora else None,
            spatial_upsampler_path=Path(args.spatial_upsampler_path) if args.spatial_upsampler_path else None,
            gamma_path=Path(args.gamma_path) if args.gamma_path else None,
            quantization=args.quantization,
            output_dir=Path(args.output_dir),
            host=args.host,
            port=args.port,
            execution_mode=ExecutionMode(args.execution_mode),
            gpu_ids=tuple(args.gpu_ids or ()),
            keep_stage_weights_on_gpu=bool(args.keep_stage_weights_on_gpu),
            keep_model_weights_on_gpu=bool(args.keep_model_weights_on_gpu),
        )

    def visible_gpu_ids(self) -> tuple[int, ...]:
        if not torch.cuda.is_available():
            if self.gpu_ids:
                raise ValueError("GPU ids were provided, but CUDA is not available.")
            return ()

        device_count = torch.cuda.device_count()
        if not self.gpu_ids:
            return tuple(range(device_count))

        invalid_gpu_ids = tuple(gpu_id for gpu_id in self.gpu_ids if gpu_id < 0 or gpu_id >= device_count)
        if invalid_gpu_ids:
            raise ValueError(f"Invalid GPU ids requested: {invalid_gpu_ids}.")

        return self.gpu_ids

    def resolved_execution_mode(self) -> ExecutionMode:
        if self.execution_mode is ExecutionMode.AUTO:
            return ExecutionMode.SINGLE
        if self.execution_mode is ExecutionMode.SHARDED:
            raise ValueError("Standalone ltx-service currently supports single-device official pipelines only.")
        return self.execution_mode

    def primary_device(self) -> torch.device:
        visible_gpu_ids = self.visible_gpu_ids()
        if visible_gpu_ids:
            return torch.device(f"cuda:{visible_gpu_ids[0]}")
        return torch.device("cpu")

    def require_cuda_for_official_pipelines(self) -> None:
        if not torch.cuda.is_available():
            raise ValueError("Standalone ltx-service official pipelines require CUDA-enabled PyTorch.")

    def require_checkpoint_path(self) -> Path:
        if self.checkpoint_path is None:
            raise ValueError("--checkpoint-path is required for this pipeline.")
        return self.checkpoint_path

    def require_distilled_checkpoint_path(self) -> Path:
        if self.distilled_checkpoint_path is None:
            raise ValueError("--distilled-checkpoint-path is required for the distilled pipeline.")
        return self.distilled_checkpoint_path

    def require_distilled_lora_path(self) -> Path:
        if self.distilled_lora_path is None:
            raise ValueError("--distilled-lora is required for the two-stage pipeline.")
        return self.distilled_lora_path

    def require_spatial_upsampler_path(self) -> Path:
        if self.spatial_upsampler_path is None:
            raise ValueError("--spatial-upsampler-path is required for this pipeline.")
        return self.spatial_upsampler_path

    def require_gamma_path(self) -> Path:
        if self.gamma_path is None:
            raise ValueError("--gamma-path is required.")
        return self.gamma_path


def build_service_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Serve LTX-2 inference over HTTP.")
    parser.add_argument(
        "--pipeline-type",
        choices=[pipeline_type.value for pipeline_type in ServingPipelineType],
        default=ServingPipelineType.TI2VID_TWO_STAGES.value,
        help="Serving pipeline backend to load (default: ti2vid-two-stages).",
    )
    parser.add_argument(
        "--checkpoint-path",
        type=_resolve_path,
        default=None,
        help="Path to the full LTX checkpoint used by one-stage or two-stage pipelines.",
    )
    parser.add_argument(
        "--distilled-checkpoint-path",
        type=_resolve_path,
        default=None,
        help="Path to the distilled LTX checkpoint used by the distilled pipeline.",
    )
    parser.add_argument(
        "--distilled-lora",
        type=_resolve_path,
        default=None,
        help="Path to the distilled LoRA used by the two-stage pipeline.",
    )
    parser.add_argument(
        "--spatial-upsampler-path",
        type=_resolve_path,
        default=None,
        help="Path to the spatial upsampler checkpoint.",
    )
    parser.add_argument(
        "--gamma-path",
        type=_resolve_path,
        default=None,
        help="Path to the Gemma model directory. This must be provided at startup.",
    )
    parser.add_argument(
        "--quantization",
        dest="quantization",
        action=ServiceQuantizationAction,
        nargs="+",
        metavar=("POLICY", "AMAX_PATH"),
        default=None,
        help="Optional transformer quantization policy.",
    )
    parser.add_argument(
        "--output-dir",
        type=_resolve_path,
        default=DEFAULT_OUTPUT_DIR.as_posix(),
        help="Directory where generated videos will be written.",
    )
    parser.add_argument("--host", default="127.0.0.1", help="Bind host for the FastAPI service.")
    parser.add_argument("--port", type=int, default=8000, help="Bind port for the FastAPI service.")
    parser.add_argument(
        "--execution-mode",
        choices=[mode.value for mode in ExecutionMode],
        default=ExecutionMode.AUTO.value,
        help="Execution mode. Standalone service currently supports only single-device official pipelines.",
    )
    parser.add_argument(
        "--gpu-ids",
        type=int,
        nargs="*",
        default=None,
        help="Optional subset of visible CUDA device ids to use, for example: --gpu-ids 0.",
    )
    parser.add_argument(
        "--keep-stage-weights-on-gpu",
        action="store_true",
        help=(
            "Keep stage-boundary weights resident on GPU for ti2vid-two-stages, and skip inter-stage GPU cache "
            "cleanup for distilled. This reduces stage-boundary churn but increases VRAM usage."
        ),
    )
    parser.add_argument(
        "--keep-model-weights-on-gpu",
        action="store_true",
        help=(
            "Keep model instances cached on GPU across requests within the same ltx-service process. "
            "This avoids repeated weight loading but increases steady-state VRAM usage."
        ),
    )
    return parser


def parse_service_config(argv: Sequence[str] | None = None) -> ServiceConfig:
    parser = build_service_arg_parser()
    args = parser.parse_args(argv)
    config = ServiceConfig.from_namespace(args)

    missing_arguments: list[str] = []
    if config.gamma_path is None:
        missing_arguments.append("--gamma-path")

    if config.pipeline_type in {ServingPipelineType.TI2VID_ONE_STAGE, ServingPipelineType.TI2VID_TWO_STAGES}:
        if config.checkpoint_path is None:
            missing_arguments.append("--checkpoint-path")

    if config.pipeline_type is ServingPipelineType.DISTILLED:
        if config.distilled_checkpoint_path is None:
            missing_arguments.append("--distilled-checkpoint-path")

    if config.pipeline_type in {ServingPipelineType.DISTILLED, ServingPipelineType.TI2VID_TWO_STAGES}:
        if config.spatial_upsampler_path is None:
            missing_arguments.append("--spatial-upsampler-path")

    if config.pipeline_type is ServingPipelineType.TI2VID_TWO_STAGES and config.distilled_lora_path is None:
        missing_arguments.append("--distilled-lora")

    if missing_arguments:
        parser.error(f"missing required arguments for {config.pipeline_type.value}: {', '.join(missing_arguments)}")

    return config
