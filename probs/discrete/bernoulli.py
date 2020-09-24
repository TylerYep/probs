from probs.discrete.rv import DiscreteRV


class Bernoulli(DiscreteRV):
    def __init__(self, p: float = 1) -> None:
        super().__init__()
        if not 0 <= p <= 1:
            raise ValueError("p must be between 0 and 1.")
        self.p = p

    def median(self) -> float:
        if self.p == 0.5:
            return 0.5
        return 1 if self.p > 0.5 else 0

    def mode(self) -> float:
        if self.p == 0.5:
            return 0
        return 1 if self.p > 0.5 else 0

    def expectation(self) -> float:
        return self.p

    def variance(self) -> float:
        return self.p * (1 - self.p)

    def pdf(self, x: float) -> float:
        k = int(x)
        return self.p if k == 1 else 1 - self.p

    def cdf(self, x: float) -> float:
        k = int(x)
        if k < 0:
            return 0
        if k >= 1:
            return 1
        return 1 - self.p
