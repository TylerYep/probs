import math


class ApproxFloat(float):
    def __init__(self, value: float):
        float.__init__(value)
        self.value = value

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, (int, float)):
            raise TypeError
        return math.isclose(self.value, other, rel_tol=1e-6)


# x = ApproxFloat(5.000001)
# assert isinstance(x, float)
# assert x == 5
