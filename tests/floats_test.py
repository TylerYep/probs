from probs.floats import ApproxFloat, ApproxFloatRtol


def test_floats() -> None:
    x = ApproxFloat(5.000000001)
    assert isinstance(x, float)
    assert x == 5

    y = ApproxFloatRtol(5.01, rtol=1e-2)
    assert isinstance(y, float)
    assert y == 5.03
