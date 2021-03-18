from __future__ import annotations

# import math
from dataclasses import dataclass

# from typing import Any, Iterable, Optional, Sequence
from typing import cast, no_type_check

import numpy as np

# from matplotlib.axes import Axes
# from mpl_format.axes import AxesFormatter
# from pandas import Series
from scipy.integrate import quad

from probs.rv import RandomVariable


@dataclass(eq=False)
class ContinuousRV(RandomVariable):
    """
    We currently use @no_type_check because there currently isn't any valuable type-info
    for these methods. We will eventually use # type: ignore once there is something
    important to typecheck.
    """

    @no_type_check
    def __add__(self, other: object) -> ContinuousRV:
        if isinstance(other, ContinuousRV):
            other_var = other
            result = type(self)()
            result.pdf = lambda z: quad(
                lambda x: self.pdf(x) * other_var.pdf(z - x),
                -np.inf,
                np.inf,
                full_output=True,
            )[0]
            result.expectation = lambda: self.expectation() + other_var.expectation()
            # Assumes Independence of X and Y, else add (+ 2 * Cov(X, Y)) term
            result.variance = lambda: self.variance() + other_var.variance()
            result.median = lambda: self.median() + other_var.median()
            return result
        return cast(ContinuousRV, super().__add__(other))

    @no_type_check
    def __sub__(self, other: object) -> ContinuousRV:
        if isinstance(other, ContinuousRV):
            result = type(self)()
            result.pdf = lambda z: quad(
                lambda x: self.pdf(x) * other.pdf(z + x),
                -np.inf,
                np.inf,
                full_output=True,
            )[0]
            result.expectation = lambda: self.expectation() - other.expectation()
            # Variances are added regardless of addition/subtraction.
            result.variance = lambda: self.variance() + other.variance()
            result.median = lambda: self.median() - other.median()
            return result
        return cast(ContinuousRV, super().__sub__(other))

    @no_type_check
    def __mul__(self, other: object) -> ContinuousRV:
        if isinstance(other, ContinuousRV):
            result = type(self)()
            result.pdf = lambda z: quad(
                lambda x: (self.pdf(x) * other.pdf(z / x)) / abs(x),
                -np.inf,
                np.inf,
                full_output=True,
            )[0]
            # Assumes Independence of X and Y
            result.expectation = lambda: self.expectation() * other.expectation()
            result.variance = (
                lambda: (self.variance() ** 2 + self.expectation() ** 2)
                + (other.variance() ** 2 + other.expectation() ** 2)
                - (self.expectation() * other.expectation()) ** 2
            )
            result.median = lambda: self.median() * other.median()
            return result
        return cast(ContinuousRV, super().__mul__(other))

    @no_type_check
    def __truediv__(self, other: object) -> ContinuousRV:
        if isinstance(other, ContinuousRV):
            result = type(self)()
            result.pdf = lambda z: quad(
                lambda x: (self.pdf(x) * other.pdf(z * x)) / abs(x),
                -np.inf,
                np.inf,
                full_output=True,
            )[0]
            result.expectation = lambda: (_ for _ in ()).throw(
                NotImplementedError("Expectation cannot be implemented for division.")
            )
            result.variance = lambda: (_ for _ in ()).throw(
                NotImplementedError("Variance cannot be implemented for division.")
            )
            result.median = lambda: self.median() / other.median()
            return result
        return cast(ContinuousRV, super().__truediv__(other))

    def median(self) -> float:
        raise NotImplementedError

    def mode(self) -> float:
        raise NotImplementedError

    def expectation(self) -> float:
        raise NotImplementedError

    def variance(self) -> float:
        raise NotImplementedError

    def pdf(self, x: float) -> float:
        raise NotImplementedError

    def cdf(self, x: float) -> float:
        """
        General implementation of the cdf function, which may be overridden
        in child classes to provide a clearer/more efficient implementation.
        """
        return float(quad(self.pdf, -np.inf, x, full_output=True)[0])

    # def plot(  # pylint: skip-file
    #     self,
    #     x: Optional[Iterable[Any]] = None,
    #     kind: str = "line",
    #     color: str = "C0",
    #     plot_type: str = "pdf",
    #     labels: Sequence[str] = ("mean", "median", "std"),
    #     ax: Optional[Axes] = None,
    #     **kwargs: Any,
    # ) -> Axes:
    #     """
    #     Plot the function.
    #     :param x: Range of values of x to plot p(x) over.
    #     :param kind: Kind of plot e.g. 'bar', 'line'.
    #     :param color: Optional color for the series.
    #     :param mean: Whether to show marker and label for the mean.
    #     :param median: Whether to show marker and label for the median.
    #     :param std: Whether to show marker and label for the standard deviation.
    #     :param ax: Optional matplotlib axes to plot on.
    #     :param kwargs: Additional arguments for the matplotlib plot function.
    #     """
    #     axf = AxesFormatter(axes=ax)
    #     x_mean = self.expectation()
    #     x_median = self.median()
    #     x_std = math.sqrt(self.variance())

    #     if plot_type == "pdf":
    #         fn = self.pdf
    #     elif plot_type == "cdf":
    #         fn = self.cdf
    #     # elif plot_type == 'logpdf':
    #     #   fn = self.logpdf
    #     else:
    #         raise ValueError(f"Plot not implemented for {plot_type}")

    #     vals = (
    #         np.linspace(x_mean - 3 * x_std, x_mean + 3 * x_std, 200)
    #         if x is None
    #         else x
    #     )

    #     data: Series = Series(
    #         index=vals,
    #         data=list(map(fn, vals)),
    #         name=str(self.__class__),
    #     )
    #     data.plot(kind=kind, color=color, ax=axf.axes, **kwargs)

    #     y_min = axf.get_y_min()
    #     y_max = axf.get_y_max()
    #     if "mean" in labels:
    #         axf.add_v_lines(
    #             x=x_mean, y_min=y_min, y_max=y_max, line_styles="--", colors=color
    #         )
    #         axf.add_text(
    #             x=x_mean,
    #             y=self.pdf(x_mean),
    #             text=f"mean={x_mean: 0.3f}",
    #             color=color,
    #             ha="center",
    #             va="bottom",
    #         )
    #     if "median" in labels:
    #         axf.add_v_lines(
    #             x=x_median, y_min=y_min, y_max=y_max, line_styles="-.", colors=color
    #         )
    #         axf.add_text(
    #             x=x_median,
    #             y=self.pdf(x_median),
    #             text=f"median={x_median: 0.3f}",
    #             color=color,
    #             ha="center",
    #             va="bottom",
    #         )
    #     if "std" in labels:
    #         axf.add_v_lines(
    #             x=[x_mean - x_std, x_mean + x_std],
    #             y_min=y_min,
    #             y_max=y_max,
    #             line_styles=":",
    #             colors=color,
    #         )
    #         axf.add_text(
    #             x=x_mean - x_std / 2,
    #             y=self.pdf(x_mean - x_std / 2),
    #             text=f"std={x_std: 0.3f}",
    #             color=color,
    #             ha="center",
    #             va="bottom",
    #         )

    #     if plot_type == "pdf":
    #         axf.axes.set_ylabel("P(X = x)")
    #     elif plot_type == "cdf":
    #         axf.axes.set_ylabel("P(X ≤ x)")
    #     elif plot_type == "logpdf":
    #         axf.axes.set_ylabel("log P(X = x)")
    #     return axf.axes
