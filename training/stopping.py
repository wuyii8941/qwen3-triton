"""Shared validation plateau rule; safety termination is never convergence."""
from dataclasses import dataclass
import math


@dataclass
class Plateau:
    min_delta: float = .002
    patience: int = 8
    min_checks: int = 10
    best: float = float('inf')
    stale: int = 0
    checks: int = 0

    def update(self,loss):
        if not math.isfinite(loss):raise ValueError('Nonfinite validation loss')
        self.checks+=1
        improved=loss < self.best-self.min_delta
        if improved:self.best=loss;self.stale=0
        else:self.stale+=1
        return improved,self.checks>=self.min_checks and self.stale>=self.patience
