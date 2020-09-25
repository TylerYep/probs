import math
import operator as op
import sys
from functools import reduce

if sys.version_info >= (3, 8):

    def nCr(n: int, r: int) -> int:
        return math.comb(n, r)  # type: ignore

    def nPr(n: int, r: int) -> int:
        return math.perm(n, r)  # type: ignore


else:

    def nCr(n: int, r: int) -> int:
        r = min(r, n - r)
        numer = reduce(op.mul, range(n, n - r, -1), 1)
        denom = reduce(op.mul, range(1, r + 1), 1)
        return numer // denom

    def nCr_factorial(n: int, r: int) -> int:
        return math.factorial(n) // math.factorial(r) // math.factorial(n - r)

    def nPr(n: int, r: int) -> int:
        return math.factorial(n) // math.factorial(n - r)
