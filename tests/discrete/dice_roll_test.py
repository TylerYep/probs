from probs.discrete.dice_roll import DiceRoll


def test_die_roll() -> None:
    d = DiceRoll()

    assert d.expectation() == 3.5
    assert d.variance() == 105 / 36
    assert d.pdf(2) == 1 / 6
    assert d.pdf(6) == 1 / 6


def test_two_dice_roll() -> None:
    d = DiceRoll() + DiceRoll()

    assert d.expectation() == 7.0
    assert d.variance() == 35 / 6
    assert d.pdf(2) == 1 / 6
    # assert d.pdf(8) == ??
