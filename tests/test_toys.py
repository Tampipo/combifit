import numpy as np
import pytest
from helpers import RawModel

from combifit import ToyResults, combine, generate_toys, joint_model, run_toys

POIS = ["mu"]
LAYOUT = [(POIS, ["s1", "s2"], "A"), (POIS, ["s1", "s3"], "A"), (POIS, ["s2", "s3"], "C")]


@pytest.fixture
def model():
    return RawModel(LAYOUT, POIS, seed=6)


def test_joint_model_follows_the_master_formula(model):
    meas = model.measurements
    mean, cov, shared = joint_model(meas, POIS, truth={"mu": 1.5, "s1": 0.2})
    assert shared == ["s1", "s2", "s3"]
    sizes = np.cumsum([0] + [len(m.names) for m in meas])
    aux = slice(sizes[-1], None)
    np.testing.assert_allclose(cov[aux, aux], np.eye(3))
    np.testing.assert_allclose(mean[aux], [0.2, 0.0, 0.0])
    lam = []
    for k, m in enumerate(meas):
        blk = slice(sizes[k], sizes[k + 1])
        np.testing.assert_allclose(cov[blk, blk], m.cov)
        L = np.zeros((len(m.names), 3))
        for c, x in enumerate(shared):
            if x in m.names:
                L[:, c] = m.cov[:, m.names.index(x)]
        np.testing.assert_allclose(cov[blk, aux], L)
        lam.append(L)
        assert mean[sizes[k]] == 1.5
    np.testing.assert_allclose(cov[sizes[0]:sizes[1], sizes[1]:sizes[2]], lam[0] @ lam[1].T)


def test_toys_are_reproducible_and_type_b_nps_copy_a(model):
    t1 = generate_toys(model.measurements, POIS, n_toys=3, seed=4)
    t2 = generate_toys(model.measurements, POIS, n_toys=3, seed=4)
    for (m1, a1), (m2, a2) in zip(t1, t2):
        assert a1 == a2
        np.testing.assert_array_equal(m1[0].theta, m2[0].theta)
    toy, a = t1[0]
    b = toy[2]
    for x in ("s2", "s3"):
        assert b.get_parameter_value(x) == pytest.approx(a[x], abs=1e-8)


def test_coverage_and_chi2_law(model):
    truth = combine(model.measurements, POIS, auxiliary=model.auxiliary)
    r = run_toys(model.measurements, POIS, truth, n_toys=1500, seed=1)
    c = r.coverage("mu")
    assert abs(c["pull_mean"]) < 0.1
    assert 0.92 < c["pull_std"] < 1.08
    chi2 = r.chi2_check()
    ndf = chi2["ndf"]
    assert abs(chi2["mean_chi2"] - ndf) < 5 * np.sqrt(2 * ndf / 1500)


def test_save_and_load(model, tmp_path):
    truth = combine(model.measurements, POIS, auxiliary=model.auxiliary)
    r = run_toys(model.measurements, POIS, truth, n_toys=20, seed=2, with_compatibility=True)
    r.save(tmp_path / "toys.npz")
    back = ToyResults.load(tmp_path / "toys.npz")
    assert back.method == r.method and back.names == r.names
    np.testing.assert_array_equal(back.theta, r.theta)
    np.testing.assert_array_equal(back.chi2_pois, r.chi2_pois)
