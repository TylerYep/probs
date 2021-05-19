from __future__ import annotations

import math


class ApproxFloat(float):
    def __eq__(self, other: object) -> bool:
        if isinstance(other, (int, float)):
            return math.isclose(self, other)
        return super().__eq__(other)

    def __repr__(self) -> str:
        """Removes trailing zeroes from float representation."""
        return str(float(f"{self:.8f}"))


class ApproxFloatRtol(float):
    def __new__(cls, value: float, rtol: float = 1e-9) -> ApproxFloatRtol:
        del rtol
        return float.__new__(cls, value)

    def __init__(self, value: float, rtol: float = 1e-9) -> None:
        float.__init__(value)
        self.value = value
        self.rtol = rtol

    def __eq__(self, other: object) -> bool:
        if isinstance(other, (int, float)):
            return math.isclose(self, other, rel_tol=self.rtol)
        return super().__eq__(other)

    def __repr__(self) -> str:
        """Removes trailing zeroes from float representation."""
        return str(float(f"{self:.8f}"))
