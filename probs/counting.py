import math


def nCr(n: int, r: int) -> int:
    """
    Equivalent to:

    math.factorial(n) // math.factorial(r) // math.factorial(n - r)

    or

    r = min(r, n - r)
    return reduce(op.mul, range(n, n - r, -1), 1) // reduce(op.mul, range(1, r + 1), 1)
    """
    return math.comb(n, r)


def nPr(n: int, r: int) -> int:
    """
    Equivalent to:

    math.factorial(n) // math.factorial(n - r)
    """
    return math.perm(n, r)
