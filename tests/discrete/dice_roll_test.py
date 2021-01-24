from probs import P
from probs.discrete.dice_roll import DiceRoll


class TestDiceRoll:
    @staticmethod
    def test_die_roll() -> None:
        d = DiceRoll()

        assert d.expectation() == 3.5
        assert d.variance() == 35 / 12
        assert d.mode() == 1  # Any result is technically correct

        assert P(d == 2) == 1 / 6
        assert P(d == 6) == 1 / 6
        assert P(d == 8) == 0

        assert P(d < 5) == 2 / 3
        assert P(d <= 5) == 5 / 6
        assert P(d >= 5) == 1 / 3
        assert P(d > 5) == 1 / 6

    @staticmethod
    def test_two_dice_roll() -> None:
        d = DiceRoll() + DiceRoll()

        assert d.expectation() == 7
        assert d.variance() == 35 / 6
        assert d.mode() == 7
        assert d.pmf == {
            2: 1 / 36,
            3: 1 / 18,
            4: 1 / 12,
            5: 1 / 9,
            6: 5 / 36,
            7: 1 / 6,
            8: 5 / 36,
            9: 1 / 9,
            10: 1 / 12,
            11: 1 / 18,
            12: 1 / 36,
        }
        assert P(d == 2) == 1 / 36
        assert P(d == 8) == 5 / 36
        assert P(d == 60) == 0

    @staticmethod
    def test_repr() -> None:
        d = DiceRoll() + DiceRoll()

        assert str(d) == (
            "DiceRoll(pmf={2: 0.02777778, 3: 0.05555556, 4: 0.08333333, 5: "
            "0.11111111, 6: 0.13888889, 7: 0.16666667, 8: 0.13888889, 9: "
            "0.11111111, 10: 0.08333333, 11: 0.05555556, 12: 0.02777778}, "
            "sides=6)"
        )
