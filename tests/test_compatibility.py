import numpy as np
import pytest
from helpers import RawModel

from combifit import Measurement, compatibility
from combifit.compatibility import free_pois

POIS = ["mu1", "mu2"]
LAYOUT = [(POIS, ["s1", "s2"], "A"), (["mu1"], ["s1", "s3"], "A"), (["mu2"], ["s2", "s3"], "A"),
          (["mu1"], ["s1", "s2"], "C")]


@pytest.mark.parametrize("method", ["weighted_average", "pois_only", "pois_nps", "pois_nps_aux"])
def test_split_adds_up(method):
    model = RawModel(LAYOUT, POIS, seed=2)
    c = compatibility(model.measurements, POIS, type=method, auxiliary=model.auxiliary)
    assert c.chi2 == pytest.approx(c.chi2_pois + c.chi2_nps, rel=1e-12)
    assert c.ndf == c.ndf_pois + c.ndf_nps
    # one copy of each POI per measurement that measures it: 2 + 1 + 1 + 1 = 5 POIs instead of 2
    assert c.ndf_pois == 3
    assert c.chi2_pois >= -1e-9 and c.chi2_nps >= -1e-9


@pytest.mark.parametrize("method", ["weighted_average", "pois_only"])
def test_without_fitted_nps_everything_is_in_the_pois(method):
    model = RawModel(LAYOUT, POIS, seed=2)
    c = compatibility(model.measurements, POIS, type=method)
    assert c.ndf_nps == 0 and c.p_value_nps is None
    assert c.chi2_pois == pytest.approx(c.chi2, rel=1e-12)


def test_free_pois_renames_and_requires_unique_names():
    meas = [Measurement("M1", [1.0, 0.0], np.eye(2), ["mu", "s"]),
            Measurement("M2", [2.0, 0.0], np.eye(2), ["mu", "s"])]
    renamed, free = free_pois(meas, ["mu"])
    assert free == ["mu[M1]", "mu[M2]"]
    assert renamed[0].names == ["mu[M1]", "s"]
    with pytest.raises(ValueError):
        free_pois([meas[0], meas[0]], ["mu"])


def test_summary_and_worst_pulls():
    model = RawModel(LAYOUT, POIS, seed=2)
    c = compatibility(model.measurements, POIS, auxiliary=model.auxiliary)
    assert "total" in c.summary() and "POIs" in c.summary()
    pulls = c.worst_pulls(3)
    assert len(pulls) == 3
    assert abs(pulls[0][2]) >= abs(pulls[1][2]) >= abs(pulls[2][2])
