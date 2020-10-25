from probs import DiscreteRV, Event, P


def test_random_variable() -> None:
    a = DiscreteRV()
    a.pmf = {0: 0.4, 1: 0.5, 2: 0.1}
    b = DiscreteRV()
    b.pmf = {0: 0.1, 1: 0.8, 2: 0.1}

    assert (a == b) == Event(0.45)
    assert P(a == 0) == 0.4
    assert P(a != 2) == 0.9
    assert P(a == b) == 0.45
    assert P(a != b) == 0.55

    assert P(0 == a) == 0.4  # type: ignore
    assert P(2 != a) == 0.9  # type: ignore
