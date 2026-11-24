import numpy as np
import pytest
from helpers import RawModel

from combifit import (combine, combine_steps, cumulative, latex_chain_table, latex_pulls_table,
                      latex_summary_table)
from combifit.tables import format_p

POIS = ["mu"]
LAYOUT = [(POIS, ["s1", "s2"], "A"), (POIS, ["s1", "s3"], "A"), (POIS, ["s2", "s3"], "C")]


@pytest.fixture
def model():
    return RawModel(LAYOUT, POIS, seed=9)


def test_pull_table(model):
    res = combine(model.measurements, POIS, auxiliary=model.auxiliary)
    rows = res.pull_table()
    n_entries = sum(len(block) for block in res.pulls.values())
    assert len(rows) == n_entries
    pulls = [abs(r["pull"]) for r in rows]
    assert pulls == sorted(pulls, reverse=True)
    assert all(r["parameter"] != "mu" for r in res.pull_table(nps_only=True))
    assert "measurement" in res.format_pulls(n=3)
    assert "mu =" in res.summary()


@pytest.mark.parametrize("p, text", [(0.5, "0.500"), (2e-4, r"$2.0\times 10^{-4}$"),
                                     (1e-15, r"$<10^{-12}$"), (None, "---")])
def test_format_p(p, text):
    assert format_p(p) == text


def test_latex_tables(model):
    steps = combine_steps(cumulative(model.measurements), POIS)
    assert [len(v) for v in steps.values()] == [2, 2, 2, 2]
    chain = latex_chain_table(steps, "mu", caption="c", label="tab:c", unit="GeV")
    assert chain.startswith(r"\begin{table}") and chain.rstrip().endswith(r"\end{table}")
    assert chain.count(r"\quad+") == 4 and "[GeV]" in chain
    summary = latex_summary_table({"g": {m: rows[-1] for m, rows in steps.items()}}, "mu")
    assert summary.count(r"\\") == 2 + 1 + 4      # 2 header lines, the group line, 4 methods
    res = combine(model.measurements, POIS, auxiliary=model.auxiliary)
    pulls = latex_pulls_table(res, n=4)
    assert pulls.count(" & ") >= 4 * 5


def test_plots(model):
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from combifit import run_toys
    from combifit.plotting import plot_toy_coverage, plot_toy_overview, plot_toys

    res = combine(model.measurements, POIS, auxiliary=model.auxiliary)
    ax = res.plot_pulls(n_max=3)
    assert len(ax.get_yticklabels()) == 3
    toys = {m: run_toys(model.measurements, POIS, res, n_toys=30, seed=1, type=m,
                        with_compatibility=True) for m in ("pois_only", "pois_nps_aux")}
    fig = plot_toys(toys, "mu")
    assert len(fig.axes) == 6
    fig = plot_toys(toys, "mu", rows=("parameter", "chi2"), figsize=(6, 4))
    assert len(fig.axes) == 4
    fig2 = plot_toy_overview({"conf": toys}, "mu")
    assert len(fig2.axes) == 2
    ax = plot_toy_coverage({"conf a": toys, "conf b": toys}, "mu")
    assert len(ax.get_yticklabels()) == 2 and len(ax.legend_handles_toy_coverage) == 2
    plt.close("all")
    assert np.isfinite(res.chi2)
