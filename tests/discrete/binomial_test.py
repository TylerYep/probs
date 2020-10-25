from probs import Binomial


class TestBinomial:
    @staticmethod
    def test_binomial() -> None:
        d = Binomial()

        assert d.expectation() == 0
        assert d.variance() == 0
        # TODO
        # assert P(d == 2) == 1 / 6
        # assert P(d == 6) == 1 / 6
        # assert P(d == 8) == 0

    @staticmethod
    def test_sum() -> None:
        d = Binomial() + Binomial()

        assert d.expectation() == 0
        assert d.variance() == 0
        # TODO
        # assert d.pmf == {}
        # assert P(d == 2) == 1 / 36
        # assert P(d == 8) == 5 / 36
        # assert P(d == 60) == 0

    @staticmethod
    def test_repr() -> None:
        d = Binomial() + Binomial()

        assert str(d) == "Binomial(n=0, p=1)"
