from typing import Protocol, TypeVar

import torch

ModelType = TypeVar("ModelType", bound=torch.nn.Module)


class ModelConfigurator(Protocol[ModelType]):
    """Protocol for model loader classes that instantiates models from a configuration dictionary."""

    @classmethod
    def from_config(cls, config: dict[str, object]) -> ModelType: ...
