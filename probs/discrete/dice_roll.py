from dataclasses import dataclass

from probs.discrete.rv import DiscreteRV


@dataclass(init=False, eq=False, repr=False)
class DiceRoll(DiscreteRV):
    def __init__(self, sides: int = 6) -> None:
        super().__init__()
        if sides <= 0:
            raise ValueError("Dice must have a positive number of sides.")
        self.pmf = {i: 1 / sides for i in range(1, sides + 1)}
        self.sides = sides

    def median(self) -> float:
        return self.sides // 2

    def mode(self) -> float:
        return 0  # TODO

    def expectation(self) -> float:
        return (self.sides + 1) / 2

    def variance(self) -> float:
        return (
            sum((i - self.expectation()) ** 2 for i in range(1, self.sides + 1))
            / self.sides
        )
