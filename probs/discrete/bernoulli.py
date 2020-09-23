from dataclasses import dataclass

from probs.rv import RandomVariable


@dataclass
class Bernoulli(RandomVariable):
    p: float

    def pdf(self, x: float) -> float:
        pass

    def cdf(self, x: float) -> float:
        pass

    def expectation(self) -> float:
        pass

    def variance(self) -> float:
        pass
