from dataclasses import dataclass

from probs.discrete.rv import DiscreteRV


@dataclass(eq=False)
class DiceRoll(DiscreteRV):
    sides: int = 6

    def __post_init__(self) -> None:
        if self.sides <= 0:
            raise ValueError("Dice must have a positive number of sides.")
        self.pmf = {i: 1 / self.sides for i in range(1, self.sides + 1)}

    def median(self) -> float:
        return self.sides // 2

    def expectation(self) -> float:
        return (self.sides + 1) / 2

    def variance(self) -> float:
        return (
            sum((i - self.expectation()) ** 2 for i in range(1, self.sides + 1))
            / self.sides
        )
