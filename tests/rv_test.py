from probs import Event, P, RandomVariable


def test_event() -> None:
    a = Event(0.5)
    b = Event(0.6)

    assert str(a) == "Event(p=0.5)"
    assert P(a & b) == 0.3
    assert P(a | b) == 0.8


def test_random_variable() -> None:
    r = RandomVariable()
    assert isinstance(r + 5, RandomVariable)
    assert isinstance(r + 5.0, RandomVariable)
