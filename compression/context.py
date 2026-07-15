from dataclasses import dataclass
from typing import Any


@dataclass
class PipelineContext:
    trainer: Any
    train_loader: Any
    val_loader: Any
    calibration_loader: Any
    device: str
    epochs: int
