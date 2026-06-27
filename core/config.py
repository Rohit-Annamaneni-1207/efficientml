from dataclasses import dataclass
from typing import Optional


@dataclass
class BaseConfig:
    device: str = "cuda"
    dtype: str = "float32"
    save_path: str = "./outputs"
    experiment_name: str = "default_experiment"
    seed: int = 42
    weight_path: Optional[str] = None