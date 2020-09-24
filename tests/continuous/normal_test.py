import math

from probs import Event, Normal, P


def test_normal_pdf() -> None:
    # These were confirmed using `scipy.stats.norm(self.mu, self.sigma).pdf(x)`.
    normal = Normal()
    assert normal.pdf(1) == 0.24197072451914337
    assert normal.pdf(15) == 5.530709549844416e-50
    assert normal.pdf(24) == 3.342714441794458e-126
    assert Normal(4, 2).pdf(1) == 0.06475879783294587
    assert Normal(5, 3).pdf(1) == 0.05467002489199788
    assert normal.pdf(10 ** -326) == 0.3989422804014327
    assert Normal(mu=234234, sigma=3425).pdf(2523) == 0.0


def test_normal_event() -> None:
    u = Normal()

    assert isinstance(u > 5, Event)
    assert isinstance(5 < u, Event)
    assert P(u < 0) == 0.5
    assert P(u < 14) == 1


def test_normal_algebra() -> None:
    a = Normal(1, 1) + Normal(5, 6)
    b = Normal(1, 2) - Normal(3, 4)
    c = Normal(1, 2) * Normal(3, 4)
    d = Normal() + 5
    e = 13 * Normal()

    assert a.expectation() == 6
    assert b.expectation() == -2
    assert c.expectation() == 3
    assert d.expectation() == 5
    assert e.expectation() == 0

    assert math.isclose(a.variance(), 37)
    assert math.isclose(b.variance(), 20)
    assert math.isclose(c.variance(), 273)
    assert math.isclose(d.variance(), 1)
    assert math.isclose(e.variance(), 28561)
