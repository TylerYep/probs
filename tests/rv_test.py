from probs.operations import P
from probs.rv import Event


def test_event() -> None:
    a = Event(0.5)
    b = Event(0.6)

    assert str(a) == "Event(prob=0.5)"
    assert P(a & b) == 0.3
    assert P(a | b) == 0.8
