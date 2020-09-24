import pytest

from probs import E, RandomVariable, Uniform, Var


def test_uniform() -> None:
    u, v = Uniform(), Uniform()
    z = u + v

    assert isinstance(z, RandomVariable)
    assert z.pdf(0.5) == 2.000000000000003
    assert (u * v).pdf(0.5) == 39.01690085582181
    assert (1 * v).pdf(0.5) == 1.0


def test_uniform_expectation() -> None:
    u, v = Uniform(), Uniform()

    assert E(u) == 0.5
    assert E(u + 1) == 1.5
    assert E(u + v) == 1.0
    assert E(u - v) == 0
    assert isinstance(u / v, RandomVariable)
    assert isinstance(u - 1, Uniform)
    with pytest.raises(NotImplementedError):
        assert E(u / v)


def test_uniform_variance() -> None:
    u, v = Uniform(), Uniform()

    assert Var(u) == 1 / 12
    assert Var(u + v) == 1 / 6
