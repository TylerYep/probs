from probs.rv import RandomVariable


class Exponential(RandomVariable):
    """
    The exponential distribution is the probability distribution of the time
    between events in a Poisson point process, i.e., a process in which events
    occur continuously and independently at a constant average rate.
    It is a particular case of the gamma distribution.
    It is the continuous analogue of the geometric distribution,
    and it has the key property of being memory-less.

    https://en.wikipedia.org/wiki/Exponential_distribution

    :param lambda_: The average rate at which events occur.
    """

    lambda_: float = 1

    def __str__(self) -> str:
        return f"Exponential(λ={self.lambda_})"

    def pdf(self, x: float) -> float:
        return 0

    def cdf(self, x: float) -> float:
        return 0
