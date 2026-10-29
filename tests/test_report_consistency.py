"""The report's numbers must match what the notebook computed (run the notebook first)."""
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
TABLES = ROOT / "outputs" / "tables"
TEX = ROOT / "report" / "report.tex"

pytestmark = pytest.mark.skipif(not (TABLES / "test_results.json").exists(),
                                reason="run notebooks/midterm_analysis.ipynb first")


def n(x: int) -> str:
    """LaTeX thousands format used in the report: 91{,}775."""
    return f"{int(x):,}".replace(",", "{,}")


def f(x: float, d: int) -> str:
    return f"{x:.{d}f}"


def expected_strings():
    raw = json.loads((TABLES / "raw_audit.json").read_text())
    size = json.loads((TABLES / "size_check.json").read_text())
    eda = json.loads((TABLES / "eda_summary.json").read_text())
    res = json.loads((TABLES / "test_results.json").read_text())
    s, t1, t2, t3, t4 = res["samples"], res["T1"], res["T2"], res["T3"], res["T4"]
    conc, rat = eda["concentration"], eda["rating_summary"]
    return {
        "raw rows": n(raw["rows"]), "raw cells": n(raw["cells"]),
        "encoded rows": n(size["rows"]), "encoded cells": n(size["cells"]),
        "encoded columns": f"\\times{size['encoded_columns']}",
        "missing prices": n(raw["price_missing"]), "missing price %": f(raw["price_missing_pct"], 1) + "\\%",
        "missing ratings": n(raw["rating_missing"]),
        "large hosts share of hosts": f(conc["large (10+)"]["share_hosts"], 1) + "\\%",
        "large hosts share of listings": f(conc["large (10+)"]["share_listings"], 1) + "\\%",
        "large below 4.8": f(rat["large (10+)"]["share_below_4_8"], 1) + "\\%",
        "single below 4.8": f(rat["single (1)"]["share_below_4_8"], 1) + "\\%",
        "borough min share": f(eda["large_share_by_borough"]["min"], 1) + "\\%",
        "borough max share": f(eda["large_share_by_borough"]["max"], 1) + "\\%",
        "hype 2+ share": f(eda["hype"]["share_at_least"]["2"], 1) + "\\%",
        "P1 single listings": n(s["p1_single"]), "P1 large listings": n(s["p1_large"]),
        "P1 large hosts": str(s["p1_large_hosts"]), "P1 total": n(s["p1_listings"]),
        "T1 estimate": "$" + f(t1["estimate"], 3).replace("-", "-") + "$",
        "T1 CI": f"${f(t1['ci_low'], 3)}$ to ${f(t1['ci_high'], 3)}$",
        "T1 t": f"t={f(t1['statistic'], 2)}", "T1 df": n(t1["df"]),
        "T2 OR": f(t2["estimate"], 2), "T2 CI": f"{f(t2['ci_low'], 2)} to {f(t2['ci_high'], 2)}",
        "T2 chi2": f(t2["statistic"], 1),
        "T3 estimate": "$" + f(t3["estimate"], 3) + "$",
        "T3 CI": f"${f(t3['ci_low'], 3)}$ to ${f(t3['ci_high'], 3)}$",
        "T3 t": f"t={f(t3['statistic'], 2)}", "T3 df": f(t3["df"], 1),
        "T4 estimate": "$+" + f(t4["estimate"], 3) + "$",
        "T4 p": f(t4["p"], 3), "T4 hyped": n(t4["n_hyped"]), "T4 plain": n(t4["n_plain"]),
    }


@pytest.mark.parametrize("name, text", list(expected_strings().items()) if (TABLES / "test_results.json").exists() else [])
def test_number_appears_in_report(name, text):
    assert text in TEX.read_text(), f"{name}: '{text}' not found in report.tex"


def test_holm_decisions_match_report():
    res = json.loads((TABLES / "test_results.json").read_text())
    assert all(p < res["alpha"] for p in res["holm"].values())   # all four rejected
    assert TEX.read_text().count(r"reject $H_0$") == 4
