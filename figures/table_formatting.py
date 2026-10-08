"""Rounding/formatting helpers shared by the table printers (half away from zero, as typeset)."""
import math


def rhalf(x: float, d: int) -> float:
    m = 10 ** d
    return math.copysign(math.floor(abs(x) * m + 0.5 + 1e-9) / m, x)


def num(x: float, d: int) -> str:
    """Plain number; negative values in math mode ($-0.05$), as in the SI tables."""
    r = rhalf(x, d)
    return f"$-{abs(r):.{d}f}$" if r < 0 else f"{abs(r):.{d}f}"


def signed(x: float, d: int) -> str:
    """Signed difference in math mode ($+6.7$, $-0.9$, $0.0$)."""
    r = rhalf(x, d)
    return f"${0:.{d}f}$" if r == 0 else f"${r:+.{d}f}$"
