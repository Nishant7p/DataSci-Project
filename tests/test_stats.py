import numpy as np
import pandas as pd
from scipy import stats

from src.stats_tests import (cmh, fe_cluster_diff, holm, practical_verdict, stratified_bootstrap_ci,
                             stratified_diff, stratified_permutation, welch)

RNG = np.random.default_rng(0)


def test_welch_matches_scipy():
    x, y = RNG.normal(4.7, 0.3, 300), RNG.normal(4.8, 0.2, 900)
    ours = welch(x, y)
    ref = stats.ttest_ind(x, y, equal_var=False)
    assert np.isclose(ours["statistic"], ref.statistic)
    assert np.isclose(ours["p"], ref.pvalue)
    # CI must contain the estimate and be symmetric around it
    assert ours["ci_low"] < ours["estimate"] < ours["ci_high"]
    assert np.isclose(ours["estimate"] - ours["ci_low"], ours["ci_high"] - ours["estimate"])


def test_stratified_diff_known_value():
    # stratum A: diff = 1 (n=4), stratum B: diff = 3 (n=4) -> weighted average 2
    y = [1, 1, 2, 2, 5, 5, 8, 8]
    g = [False, False, True, True, False, False, True, True]
    s = ["A"] * 4 + ["B"] * 4
    assert np.isclose(stratified_diff(y, g, s), 2.0)


def test_stratified_diff_skips_strata_missing_a_group():
    y = [1, 2, 10, 10]
    g = [False, True, True, True]
    s = ["A", "A", "B", "B"]
    assert np.isclose(stratified_diff(y, g, s), 1.0)


def test_permutation_null_gives_large_p_and_strong_effect_small_p():
    n = 400
    s = RNG.integers(0, 5, n)
    g = RNG.random(n) < 0.4
    y_null = RNG.normal(0, 1, n) + s          # strata shift the level, groups do not
    res_null = stratified_permutation(y_null, g, s, "two-sided", n_perm=2000, seed=1)
    assert res_null["p"] > 0.01
    y_eff = y_null - 1.0 * g
    res_eff = stratified_permutation(y_eff, g, s, "less", n_perm=2000, seed=1)
    assert res_eff["p"] < 0.001 and res_eff["estimate"] < -0.5
    lo, hi = stratified_bootstrap_ci(y_eff, g, s, n_boot=2000, seed=1)
    assert lo < res_eff["estimate"] < hi


def test_fe_cluster_diff_recovers_effect():
    n_hosts = 300
    rows = []
    for h in range(n_hosts):
        treated = h < 100
        stratum = h % 3
        host_effect = RNG.normal(0, 0.2)
        for _ in range(5 if treated else 1):
            rows.append({"y": 4.8 - 0.1 * treated + 0.05 * stratum + host_effect + RNG.normal(0, 0.05),
                         "t": treated, "s": stratum, "h": h})
    d = pd.DataFrame(rows)
    res = fe_cluster_diff(d, "y", "t", "s", "h")
    assert res["ci_low"] < -0.1 < res["ci_high"]
    assert res["df"] == n_hosts - 1
    # clustering must widen the standard error relative to treating rows as independent
    assert res["se"] > res["iid_se"]


def test_cmh_matches_hand_computation():
    # two strata; MH OR = sum(a d / n) / sum(b c / n)
    d = pd.DataFrame({
        "o": [1] * 10 + [0] * 20 + [1] * 5 + [0] * 25 + [1] * 8 + [0] * 12 + [1] * 6 + [0] * 14,
        "t": [1] * 30 + [0] * 30 + [1] * 20 + [0] * 20,
        "s": ["A"] * 60 + ["B"] * 40,
    })
    res = cmh(d, "o", "t", "s")
    num = 10 * 25 / 60 + 8 * 14 / 40
    den = 20 * 5 / 60 + 12 * 6 / 40
    assert np.isclose(res["estimate"], num / den)
    assert res["df"] == 1 and 0 <= res["p"] <= 1


def test_holm_known_values():
    adj = holm({"a": 0.01, "b": 0.04, "c": 0.03})
    assert np.isclose(adj["a"], 0.03) and np.isclose(adj["c"], 0.06) and np.isclose(adj["b"], 0.06)


def test_practical_verdict():
    assert practical_verdict(-0.05, 0.04, 0.10).startswith("within")
    assert practical_verdict(-0.30, -0.12, 0.10).startswith("beyond")
    assert practical_verdict(-0.15, -0.02, 0.10).startswith("overlaps")
    assert practical_verdict(0.9, 1.1, 1.25, scale="ratio").startswith("within")
