"""Hypothesis tests used in Section 5, each returning a plain dict so results can be saved as JSON.

The two-group tests follow one convention: the first argument (or the True group) is the group
named first in the hypotheses, so a negative difference means "first group lower".
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy import stats
from statsmodels.stats.contingency_tables import StratifiedTable
from statsmodels.stats.proportion import proportions_ztest


# ----------------------------------------------------------------------------- two-sample t tests
def welch(x, y, alternative: str = "two-sided", conf: float = 0.95) -> dict:
    """Welch's unequal-variance t-test for mean(x) - mean(y), with a Welch-df confidence interval."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    nx, ny = len(x), len(y)
    vx, vy = x.var(ddof=1), y.var(ddof=1)
    se = np.sqrt(vx / nx + vy / ny)
    df = (vx / nx + vy / ny) ** 2 / ((vx / nx) ** 2 / (nx - 1) + (vy / ny) ** 2 / (ny - 1))
    diff = x.mean() - y.mean()
    t = diff / se
    res = stats.ttest_ind(x, y, equal_var=False, alternative=alternative)
    crit = stats.t.ppf(0.5 + conf / 2, df)
    return {"test": "Welch t", "n1": nx, "n2": ny, "mean1": x.mean(), "mean2": y.mean(),
            "estimate": diff, "se": se, "statistic": t, "df": df, "p": float(res.pvalue),
            "ci_low": diff - crit * se, "ci_high": diff + crit * se}


def pooled_t(x, y, alternative: str = "two-sided", conf: float = 0.95) -> dict:
    x, y = np.asarray(x, float), np.asarray(y, float)
    nx, ny = len(x), len(y)
    sp2 = ((nx - 1) * x.var(ddof=1) + (ny - 1) * y.var(ddof=1)) / (nx + ny - 2)
    se = np.sqrt(sp2 * (1 / nx + 1 / ny))
    diff = x.mean() - y.mean()
    res = stats.ttest_ind(x, y, equal_var=True, alternative=alternative)
    crit = stats.t.ppf(0.5 + conf / 2, nx + ny - 2)
    return {"test": "pooled t", "estimate": diff, "statistic": float(res.statistic), "df": nx + ny - 2,
            "p": float(res.pvalue), "ci_low": diff - crit * se, "ci_high": diff + crit * se}


def brown_forsythe(x, y) -> dict:
    """Levene's test centred on the median (Brown-Forsythe): robust check of equal variances."""
    res = stats.levene(x, y, center="median")
    return {"test": "Brown-Forsythe", "statistic": float(res.statistic), "df": (1, len(x) + len(y) - 2),
            "p": float(res.pvalue), "sd1": float(np.std(x, ddof=1)), "sd2": float(np.std(y, ddof=1))}


def mann_whitney(x, y, alternative: str = "two-sided") -> dict:
    res = stats.mannwhitneyu(x, y, alternative=alternative, method="asymptotic")
    n1, n2 = len(x), len(y)
    return {"test": "Mann-Whitney U", "statistic": float(res.statistic), "p": float(res.pvalue),
            "prob_superiority": float(res.statistic) / (n1 * n2)}


def permutation_mean_diff(x, y, alternative: str = "two-sided", n_resamples: int = 10_000,
                          seed: int = 558) -> dict:
    res = stats.permutation_test(
        (np.asarray(x, float), np.asarray(y, float)),
        lambda a, b, axis: np.mean(a, axis=axis) - np.mean(b, axis=axis),
        permutation_type="independent", alternative=alternative, n_resamples=n_resamples,
        vectorized=True, random_state=seed)
    return {"test": "permutation (difference in means)", "statistic": float(res.statistic),
            "p": float(res.pvalue), "n_resamples": n_resamples}


# ----------------------------------------------------------------------------- stratified, clustered
def fe_cluster_diff(data: pd.DataFrame, y: str, treat: str, stratum: str, cluster: str,
                    conf: float = 0.95) -> dict:
    """Within-stratum difference with host-clustered standard errors.

    OLS of y on a 0/1 treatment plus one intercept per stratum (fixed effects). The coefficient
    is the average within-stratum gap; the CR1 cluster-robust standard error treats all listings
    of one host as one correlated cluster, and inference uses t with G - 1 degrees of freedom
    (G = number of clusters).
    """
    d = data[[y, treat, stratum, cluster]].dropna().copy()
    d["_t"] = d[treat].astype(float)
    d["_s"] = d[stratum].astype(str)
    d["_g"] = pd.factorize(d[cluster])[0]
    formula = f"{y} ~ _t + C(_s)"
    iid = smf.ols(formula, data=d).fit()
    clu = smf.ols(formula, data=d).fit(cov_type="cluster", cov_kwds={"groups": d["_g"].to_numpy()})
    est = float(clu.params["_t"])
    se = float(clu.bse["_t"])
    g = int(d["_g"].nunique())
    df = g - 1
    t = est / se
    crit = stats.t.ppf(0.5 + conf / 2, df)
    both = d.groupby("_s")["_t"].agg(["min", "max"])
    return {
        "test": "stratum fixed effects + host-clustered SE", "estimate": est, "se": se,
        "statistic": t, "df": df, "p": float(2 * stats.t.sf(abs(t), df)),
        "ci_low": est - crit * se, "ci_high": est + crit * se, "clusters": g,
        "n": int(len(d)), "strata": int(d["_s"].nunique()),
        "strata_with_both_groups": int(((both["min"] == 0) & (both["max"] == 1)).sum()),
        "iid_se": float(iid.bse["_t"]), "iid_statistic": float(iid.tvalues["_t"]),
        "iid_df": float(iid.df_resid), "iid_p": float(iid.pvalues["_t"]),
    }


def _strata_arrays(y, g, s):
    y = np.asarray(y, float)
    g = np.asarray(g, bool)
    s = np.asarray(pd.factorize(pd.Series(s).astype(str))[0])
    groups = []
    for code in np.unique(s):
        idx = np.flatnonzero(s == code)
        m = int(g[idx].sum())
        if 0 < m < len(idx):
            groups.append((y[idx], g[idx], m))
    return groups


def stratified_diff(y, g, s) -> float:
    """Average within-stratum difference mean(g=True) - mean(g=False), weighted by stratum size.

    Strata missing either group carry no information about the difference and are skipped.
    """
    groups = _strata_arrays(y, g, s)
    total = sum(len(ys) for ys, _, _ in groups)
    return float(sum(len(ys) / total * (ys[gs].mean() - ys[~gs].mean()) for ys, gs, _ in groups))


def stratified_permutation(y, g, s, alternative: str, n_perm: int = 10_000, seed: int = 558,
                           chunk: int = 1_000) -> dict:
    """Permute group labels within each stratum; the statistic is `stratified_diff`."""
    groups = _strata_arrays(y, g, s)
    total = sum(len(ys) for ys, _, _ in groups)
    observed = stratified_diff(y, g, s)
    rng = np.random.default_rng(seed)
    null = np.zeros(n_perm)
    for ys, _, m in groups:
        n = len(ys)
        w = n / total
        tot = ys.sum()
        coef = 1 / m + 1 / (n - m)
        for start in range(0, n_perm, chunk):
            b = min(chunk, n_perm - start)
            picks = np.argsort(rng.random((b, n)), axis=1)[:, :m]
            s1 = ys[picks].sum(axis=1)
            null[start:start + b] += w * (s1 * coef - tot / (n - m))
    if alternative == "less":
        p = (1 + np.sum(null <= observed)) / (n_perm + 1)
    elif alternative == "greater":
        p = (1 + np.sum(null >= observed)) / (n_perm + 1)
    else:
        p = (1 + np.sum(np.abs(null) >= abs(observed))) / (n_perm + 1)
    return {"test": "stratified permutation", "estimate": observed, "p": float(p), "n_perm": n_perm,
            "strata_used": len(groups), "n": total, "null_sd": float(null.std())}


def stratified_bootstrap_ci(y, g, s, n_boot: int = 5_000, seed: int = 558, conf: float = 0.95,
                            chunk: int = 1_000) -> tuple[float, float]:
    """Percentile CI for `stratified_diff`, resampling within each stratum and group."""
    groups = _strata_arrays(y, g, s)
    total = sum(len(ys) for ys, _, _ in groups)
    rng = np.random.default_rng(seed)
    boot = np.zeros(n_boot)
    for ys, gs, _ in groups:
        w = len(ys) / total
        a, b_ = ys[gs], ys[~gs]
        for start in range(0, n_boot, chunk):
            b = min(chunk, n_boot - start)
            ma = a[rng.integers(0, len(a), (b, len(a)))].mean(axis=1)
            mb = b_[rng.integers(0, len(b_), (b, len(b_)))].mean(axis=1)
            boot[start:start + b] += w * (ma - mb)
    lo, hi = np.percentile(boot, [50 * (1 - conf), 100 - 50 * (1 - conf)])
    return float(lo), float(hi)


# ----------------------------------------------------------------------------- proportions
def cmh(data: pd.DataFrame, outcome: str, treat: str, stratum: str) -> dict:
    """Cochran-Mantel-Haenszel test of a 0/1 outcome by a 0/1 group across strata.

    Tables are [[treated & outcome, treated & not], [control & outcome, control & not]], so the
    pooled odds ratio compares the treated group's odds of the outcome with the control's.
    """
    d = data[[outcome, treat, stratum]].dropna()
    tables, kept = [], []
    for name, part in d.groupby(stratum):
        t = part[treat].astype(bool)
        o = part[outcome].astype(bool)
        if t.all() or (~t).all():
            continue
        tables.append(np.array([[(t & o).sum(), (t & ~o).sum()], [(~t & o).sum(), (~t & ~o).sum()]], float))
        kept.append(name)
    st = StratifiedTable(tables)
    null = st.test_null_odds(correction=False)
    lo, hi = st.oddsratio_pooled_confint(alpha=0.05)
    bd = st.test_equal_odds(adjust=False)
    pooled = np.sum(tables, axis=0)
    return {"test": "Cochran-Mantel-Haenszel", "statistic": float(null.statistic), "df": 1,
            "p": float(null.pvalue), "estimate": float(st.oddsratio_pooled), "ci_low": float(lo),
            "ci_high": float(hi), "strata_used": len(tables), "breslow_day_stat": float(bd.statistic),
            "breslow_day_p": float(bd.pvalue), "pooled_table": pooled.tolist(),
            "n_in_strata": int(pooled.sum())}


def pooled_2x2(table) -> dict:
    """Unstratified alternatives on the collapsed 2x2 table: chi-square, z-test, Fisher exact."""
    table = np.asarray(table, float)
    chi2, p_chi, dof, _ = stats.chi2_contingency(table, correction=False)
    counts = table[:, 0]
    nobs = table.sum(axis=1)
    z, p_z = proportions_ztest(counts, nobs)
    odds, p_f = stats.fisher_exact(table.astype(int))
    return {"chi2": float(chi2), "chi2_df": int(dof), "chi2_p": float(p_chi), "z": float(z),
            "z_p": float(p_z), "fisher_or": float(odds), "fisher_p": float(p_f),
            "share_treated": float(counts[0] / nobs[0]), "share_control": float(counts[1] / nobs[1])}


# ----------------------------------------------------------------------------- multiple testing
def holm(pvalues: dict) -> dict:
    """Holm-Bonferroni step-down adjusted p-values (family-wise error control)."""
    names = sorted(pvalues, key=pvalues.get)
    m = len(names)
    adjusted, running = {}, 0.0
    for i, name in enumerate(names):
        running = max(running, min(1.0, (m - i) * pvalues[name]))
        adjusted[name] = running
    return adjusted


def practical_verdict(ci_low: float, ci_high: float, threshold: float, scale: str = "difference") -> str:
    """Compare a 95% CI with the smallest difference that matters (+/- threshold, or a ratio band)."""
    if scale == "ratio":
        lo_t, hi_t = 1 / threshold, threshold
    else:
        lo_t, hi_t = -threshold, threshold
    if lo_t < ci_low and ci_high < hi_t:
        return "within the no-practical-difference band"
    if ci_high <= lo_t or ci_low >= hi_t:
        return "beyond the smallest difference that matters"
    return "overlaps the threshold: cannot rule a practical difference in or out"
