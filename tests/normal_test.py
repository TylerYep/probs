from probs import Normal


def test_gaussian() -> None:
    # These were confirmed using `scipy.stats.norm(self.mu, self.sigma).pdf(x)`.
    normal = Normal()
    assert normal.pdf(1) == 0.24197072451914337
    assert normal.pdf(15) == 5.530709549844416e-50
    assert normal.pdf(24) == 3.342714441794458e-126
    assert Normal(4, 2).pdf(1) == 0.06475879783294587
    assert Normal(5, 3).pdf(1) == 0.05467002489199788
    assert normal.pdf(10 ** -326) == 0.3989422804014327
    assert Normal(mu=234234, sigma=3425).pdf(2523) == 0.0
