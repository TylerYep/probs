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
    assert d.pdf(2) == 1 / 36
    assert d.pdf(8) == 5 / 36
    assert d.pdf(60) == 0
