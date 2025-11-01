import numpy as np
from dataclasses import dataclass

def set_seed(seed: int | None):
    if seed is None:
        return np.random.default_rng()
    np.random.seed(seed)
    return np.random.default_rng(seed)

@dataclass
class SimClock:
    dt: float
    T_ms: float
    def steps(self) -> int:
        return int(self.T_ms / self.dt)
