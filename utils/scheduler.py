import numpy as np


class EpsilonScheduler:
    def __init__(
            self,
            start: float = 1.0,
            end: float = 0.01,
            decay_steps: int = 5000
    ):
        self.start = start
        self.end = end
        self.decay = decay_steps

    def get_epsilon(self, step: int) -> float:
        return self.end + (self.start - self.end) * np.exp(-step / self.decay)