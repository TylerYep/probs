from dataclasses import dataclass

from probs.continuous.rv import ContinuousRV


@dataclass
class Uniform(ContinuousRV):
    a: float = 0
    b: float = 1

    def median(self) -> float:
        return 0.5 * (self.a + self.b)

    def mode(self) -> float:
        # Any value between (a, b)
        return 0.5 * (self.a + self.b)

    def expectation(self) -> float:
        return 0.5 * (self.a + self.b)

    def variance(self) -> float:
        return ((self.b - self.a) ** 2) / 12

    def pdf(self, x: float) -> float:
        if self.a <= x <= self.b:
            return 1 / (self.b - self.a)
        return 0

    def cdf(self, x: float) -> float:
        if x < self.a:
            return 0
        if x > self.b:
            return 1
        return (x - self.a) / (self.b - self.a)
