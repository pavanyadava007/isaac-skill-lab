import math

import pytest

from skilllab.stats import mcnemar_exact, paired_compare, success_summary, wilson


def test_wilson_known_value():
    lo, hi = wilson(173, 200)
    # hand computation: centre 0.858130, half-width 0.047413 (z = 1.96)
    assert lo == pytest.approx(0.81072, abs=1e-4)
    assert hi == pytest.approx(0.90554, abs=1e-4)


def test_wilson_edges():
    assert wilson(0, 50)[0] == pytest.approx(0.0, abs=1e-12)
    assert wilson(50, 50)[1] == pytest.approx(1.0, abs=1e-12)
    assert 0.0 < wilson(50, 50)[0] < 1.0
    assert all(math.isnan(x) for x in wilson(0, 0))
    with pytest.raises(ValueError):
        wilson(5, 3)


def test_mcnemar_exact():
    assert mcnemar_exact(0, 0) == 1.0
    # b=0, c=6: two-sided exact p = 2 * 0.5**6
    assert mcnemar_exact(0, 6) == pytest.approx(2 * 0.5**6)
    assert mcnemar_exact(5, 5) == 1.0
    assert mcnemar_exact(10, 2) == mcnemar_exact(2, 10)


def test_paired_compare_counts():
    a = [1, 1, 0, 0, 1]
    b = [1, 0, 1, 0, 0]
    r = paired_compare(a, b)
    assert (r["both"], r["only_a"], r["only_b"], r["neither"]) == (1, 2, 1, 1)
    assert r["diff"] == pytest.approx(0.2)
    with pytest.raises(ValueError):
        paired_compare([1], [1, 0])


def test_success_summary_times():
    s = success_summary([True, False, True], [10, -1, 30], dt=0.05)
    assert s["successes"] == 2 and s["n"] == 3
    assert s["time_to_success_s"]["mean"] == pytest.approx(1.0)
