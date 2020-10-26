from __future__ import annotations

from probs.rv import Event, RandomVariable

SYNTAX_WARNING = (
    "The {} syntax is not supported because it adds unnecessary "
    "redundancy. Please use the {} syntax instead."
)


class Probability:
    @staticmethod
    def __call__(event: Event) -> float:
        return event.probabilty

    @staticmethod
    def __getitem__(var: Event) -> float:
        raise NotImplementedError(SYNTAX_WARNING.format("P[X]", "P(X)"))


class Expectation:
    @staticmethod
    def __call__(var: RandomVariable) -> float:
        return var.expectation()

    @staticmethod
    def __getitem__(var: RandomVariable) -> float:
        raise NotImplementedError(SYNTAX_WARNING.format("E[X]", "E(X)"))


class Variance:
    @staticmethod
    def __call__(var: RandomVariable) -> float:
        return var.variance()

    @staticmethod
    def __getitem__(var: RandomVariable) -> float:
        raise NotImplementedError(SYNTAX_WARNING.format("Var[X]", "Var(X)"))


P = Probability()
E = Expectation()
Var = Variance()
