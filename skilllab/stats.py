"""Statistics for skill evaluation: Wilson intervals, exact McNemar, bootstrap-free summaries."""

from __future__ import annotations

import math
from collections.abc import Sequence

Z95 = 1.959963984540054


def wilson(k: int, n: int, z: float = Z95) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion k/n (95% by default)."""
    if n <= 0:
        return (float("nan"), float("nan"))
    if not 0 <= k <= n:
        raise ValueError(f"need 0 <= k <= n, got k={k}, n={n}")
    p = k / n
    den = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / den
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return (max(0.0, centre - half), min(1.0, centre + half))


def mcnemar_exact(b: int, c: int) -> float:
    """Two-sided exact McNemar p-value from the discordant counts b (A ok, B fail) and c (A fail, B ok)."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / 2**n
    return min(1.0, 2 * tail)


def paired_compare(a: Sequence[bool], b: Sequence[bool]) -> dict:
    """McNemar comparison of two success vectors on the same episodes (same seeds, same order)."""
    if len(a) != len(b):
        raise ValueError("paired vectors must have the same length")
    both = sum(1 for x, y in zip(a, b, strict=True) if x and y)
    only_a = sum(1 for x, y in zip(a, b, strict=True) if x and not y)
    only_b = sum(1 for x, y in zip(a, b, strict=True) if y and not x)
    n = len(a)
    return {"n": n, "both": both, "only_a": only_a, "only_b": only_b, "neither": n - both - only_a - only_b,
            "diff": (sum(map(bool, a)) - sum(map(bool, b))) / n if n else float("nan"),
            "p_mcnemar_exact": mcnemar_exact(only_a, only_b)}


def mean_std(xs: Sequence[float]) -> tuple[float, float]:
    n = len(xs)
    if n == 0:
        return (float("nan"), float("nan"))
    m = sum(xs) / n
    if n == 1:
        return (m, 0.0)
    return (m, math.sqrt(sum((x - m) ** 2 for x in xs) / (n - 1)))


def success_summary(success: Sequence[bool], t_success: Sequence[int], dt: float) -> dict:
    """Success rate with Wilson 95% CI and time-to-success over the successful episodes."""
    n = len(success)
    k = int(sum(map(bool, success)))
    lo, hi = wilson(k, n)
    times = [t * dt for s, t in zip(success, t_success, strict=True) if s]
    tm, ts = mean_std(times)
    times.sort()
    med = times[len(times) // 2] if times else float("nan")
    return {"n": n, "successes": k, "rate": k / n if n else float("nan"), "ci95": [lo, hi],
            "time_to_success_s": {"mean": tm, "std": ts, "median": med}}
