import math
from dataclasses import dataclass

from probs.discrete.rv import DiscreteRV


@dataclass(eq=False)
class Poisson(DiscreteRV):
    """
    The Poisson distribution is a discrete probability distribution that
    expresses the probability of a given number of events occurring in a fixed
    interval of time or space if these events occur with a known constant mean
    rate and independently of the time since the last event.
    The Poisson distribution can also be used for the number of events in other
    specified intervals such as distance, area or volume.

    https://en.wikipedia.org/wiki/Poisson_distribution

    :param lambda_: Average rate in which events occur.
    """

    lambda_: float = 0

    def median(self) -> float:
        raise NotImplementedError

    def mode(self) -> float:
        return math.floor(self.lambda_)

    def expectation(self) -> float:
        return self.lambda_

    def variance(self) -> float:
        return self.lambda_

    def pdf(self, x: float) -> float:
        k = int(x)
        return self.lambda_**k * math.exp(-self.lambda_) / math.factorial(k)
