from dataclasses import dataclass
import math
from numbers import Integral, Real


@dataclass
class dLLMCacheConfig:
    prompt_interval_steps: int = 1
    gen_interval_steps: int = 1
    transfer_ratio: float = 0.0
    cfg_interval_steps: int = 1

    def __post_init__(self) -> None:
        for name in (
            "prompt_interval_steps",
            "gen_interval_steps",
            "cfg_interval_steps",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, Integral):
                raise TypeError(f"{name} must be an integer, got {type(value).__name__}")
            if value < 1:
                raise ValueError(f"{name} must be at least 1, got {value}")

        if isinstance(self.transfer_ratio, bool) or not isinstance(
            self.transfer_ratio, Real
        ):
            raise TypeError(
                "transfer_ratio must be a real number, "
                f"got {type(self.transfer_ratio).__name__}"
            )
        if not math.isfinite(float(self.transfer_ratio)):
            raise ValueError("transfer_ratio must be finite")
        if not 0.0 <= self.transfer_ratio <= 1.0:
            raise ValueError(
                f"transfer_ratio must be between 0 and 1, got {self.transfer_ratio}"
            )
