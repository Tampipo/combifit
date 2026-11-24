"""Combination of measurements from their results and covariance matrices.

Notation (theory notes fitcombi.tex and heterogeneous.tex): theta = (mu, alpha) with mu the
parameters of interest (POIs) and alpha the nuisance parameters (NPs) shared by the measurements,
constrained by one auxiliary measurement a ~ N(alpha, I) (NPs in units of their prior).
Measurement K publishes, according to its type:

    A  the result and covariance on the POIs and the shared NPs, (theta_K, Sigma_K)
       (profile-likelihood fit with its post-fit NPs);
    B  the POI result and its covariance only, (mu_K, Sigma_K^{mu mu}) (total uncertainty);
    C  the POI result, its covariance and the impacts of the shared NPs,
       (mu_K, Sigma_K^{mu mu}, lambda_K) with lambda_K = Sigma_K^{mu alpha} = Cov(mu_K, a).

A type C measurement is a Measurement over (POIs, NPs) whose POI-NP covariance holds the impacts
and whose NP block is the prior (values 0, covariance I), see models.impacts_only; a type B
measurement simply has no NP. A measurement without type is taken as type A.

The measurements are correlated only through a (master formula):
    Cov(theta_K, theta_L) = Sigma_K S S^T Sigma_L = Lambda_K Lambda_L^T,   Lambda_K = Cov(theta_K, a).

Methods (combine(type=...)):
    weighted_average  no correlation (inverse-covariance weighted average of the POIs);
    pois_only         marginalised combination: BLUE on the POIs Y_mu;
    pois_nps          full combination: BLUE on the POIs and shared NPs Y = (theta_1, ..., theta_N);
    pois_nps_aux      likelihood sum: BLUE on the stacked statistic Y~ = (theta_1, ..., theta_N, a),
                      efficient; its chi2 at the minimum is the goodness of fit.
"""

import warnings

import numpy as np
from scipy.linalg import block_diag

from .combifit import CombinedMeasurement
from .measurement import Measurement

def combine(measurements: list[Measurement], pois: list[str], type: str = "pois_nps_aux",
            auxiliary: dict[str, float] | None = None) -> CombinedMeasurement:
    """Combine measurements of the POIs.

    Args:
        measurements: List of measurements to be combined (types A, B, C, see the module).
        pois: List of parameters of interest (POIs).
        type: Combination method:
            - "weighted_average": no correlation; underestimates the uncertainty when the
              measurements share NPs.
            - "pois_only": marginalised combination, the POIs correlated through the impacts
              Lambda_K of the shared NPs.
            - "pois_nps": full combination, the POIs and the shared NPs fitted by type A
              measurements, with their post-fit covariances.
            - "pois_nps_aux": likelihood sum, the full combination with the auxiliary measurement
              counted once; efficient, combines any mix of types A, B and C.
        auxiliary: Observed values of the auxiliary measurement a ~ N(alpha, I) of the shared NPs,
            used by "pois_nps_aux". NPs not listed are at 0 (the usual convention); needed for
            toys, where a fluctuates.

    Returns:
        A CombinedMeasurement: combined values and covariance, weights of each measurement,
        chi2, ndf and p-value of the goodness of fit, pulls.
    """
    # Get the number of measurements
    N = len(measurements)
    if N == 0:
        raise ValueError("No measurements provided for combination.")

    # Get the number of parameters
    M = len(pois)
    if M == 0:
        raise ValueError("No parameters of interest (POIs) provided for combination.")

    for measurement in measurements:
        if not any(poi in measurement.names for poi in pois):
            warnings.warn(f"Measurement '{measurement.name}' does not contain at least one of the specified POIs: {pois}. Maybe it is auxiliary or you should check this.")
    for poi in pois:
        if not any(measurement.has_parameter(poi) for measurement in measurements):
            raise ValueError(f"POI '{poi}' is not measured by any of the measurements.")

    _types = ["weighted_average", "pois_only", "pois_nps", "pois_nps_aux"]
    if type not in _types:
        raise ValueError(f"Invalid combination type '{type}'. Valid options are: {_types}")
    if type == "weighted_average":
        return _combine_weighted_average(measurements, pois)
    if type == "pois_only":
        return _combine_pois_only(measurements, pois)
    if type == "pois_nps":
        return _combine_pois_nps(measurements, pois)
    if type == "pois_nps_aux":
        return _combine_pois_nps_aux(measurements, pois, auxiliary)
    else:
        raise NotImplementedError(f"Combination type '{type}' is not implemented yet.")


def _combine_weighted_average(measurements: list[Measurement], pois: list[str]) -> CombinedMeasurement:
    """Weighted average of the POIs, the measurements taken as uncorrelated.

    With C_K = Sigma_K^{mu mu} and mu_K the POIs of measurement K,
        cov = (sum_K C_K^-1)^-1,    mu = cov sum_K C_K^-1 mu_K.
    A measurement that does not measure some of the POIs contributes only to the POIs it
    measures (its C_K^-1 is embedded in the corresponding rows and columns).

    Only meaningful for POIs. If NPs constrained by the shared auxiliary measurement are passed
    as POIs, each Sigma_K already contains that constraint, which is then counted N times: the
    combined uncertainty is underestimated.

    Args:
        measurements: List of measurements to be combined.
        pois: List of parameters of interest (POIs).

    Returns:
        A CombinedMeasurement of the POIs.
    """

    M = len(pois)
    # Initialize the combined information matrix (inverse covariance) and right-hand side
    combined_info = np.zeros((M, M))
    combined_rhs = np.zeros(M)
    parts = []  # (present, theta_K, C_K^-1) of each measurement

    # Loop over each measurement and combine them
    for measurement in measurements:
        # POIs measured by this measurement: position in the result and in the measurement
        present = [i for i, poi in enumerate(pois) if measurement.has_parameter(poi)]
        indices = [measurement.get_parameter_index(pois[i]) for i in present]

        # Extract the relevant covariance matrix and best fit values
        cov = measurement.cov[np.ix_(indices, indices)]
        theta = measurement.theta[indices]

        # Update the combined information matrix and right-hand side
        weight = np.linalg.inv(cov)
        combined_info[np.ix_(present, present)] += weight
        combined_rhs[present] += weight @ theta
        parts.append((present, theta, weight))

    # Finalize the combined covariance matrix and best fit values
    combined_cov = np.linalg.inv(combined_info)
    combined_theta = combined_cov @ combined_rhs

    # theta = sum_K G_K theta_K with G_K = cov[:, present_K] C_K^-1
    weights = [combined_cov[:, present] @ weight for present, _, weight in parts]

    # chi2 at the minimum, computed from the residuals of each measurement
    chi2 = 0.0
    for present, theta, weight in parts:
        residual = theta - combined_theta[present]
        chi2 += float(residual @ weight @ residual)
    ndf = sum(len(present) for present, _, _ in parts) - M

    # Pulls: Var(theta_K - theta) = C_K - cov[present, present] for the GLS estimate
    pulls = _pulls(measurements, [[pois[i] for i in present] for present, _, _ in parts],
                   np.concatenate([theta - combined_theta[present] for present, theta, _ in parts]),
                   np.concatenate([np.diag(np.linalg.inv(weight)) - np.diag(combined_cov)[present]
                                   for present, _, weight in parts]))

    return CombinedMeasurement(
        measurements=list(measurements),
        pois=list(pois),
        method="weighted_average",
        theta=combined_theta,
        cov=combined_cov,
        weights=weights,
        chi2=chi2,
        ndf=ndf,
        pulls=pulls,
    )



def _get_shared_parameters(measurements: list[Measurement]) -> set[str]:
    """Parameters present in at least two measurements (POIs and NPs).

    Args:
        measurements: List of measurements to be combined.

    Returns:
        The set of their names; minus the POIs, these are the shared NPs alpha.
    """

    parameter_counts = {}
    for measurement in measurements:
        for parameter in measurement.names:
            if parameter in parameter_counts:
                parameter_counts[parameter] += 1
            else:
                parameter_counts[parameter] = 1

    shared_parameters = {param for param, count in parameter_counts.items() if count > 1}
    return shared_parameters


def _combine_pois_only(measurements: list[Measurement], pois: list[str]) -> CombinedMeasurement:
    """Marginalised combination: BLUE on the stacked POIs Y_mu = (mu_1, ..., mu_N).

    E[Y_mu] = U mu and the blocks of V = Cov(Y_mu) are
        V_KK = Sigma_K^{mu mu}                   (private NPs already inside)
        V_KL = lambda_K lambda_L^T   (K != L)    (master formula)
    with lambda_K = Sigma_K^{mu alpha} = Cov(mu_K, a) the impacts of the shared NPs (NPs in
    units of their prior). Solution: mu = (U^T V^-1 U)^-1 U^T V^-1 Y_mu. Types A and C enter the
    same way; type B has no impact, hence no correlation.

    Args:
        measurements: List of measurements to be combined.
        pois: List of parameters of interest (POIs).

    Returns:
        A CombinedMeasurement of the POIs.
    """

    # Nuisance parameters shared by at least two measurements (POIs excluded)
    shared = sorted(_get_shared_parameters(measurements) - set(pois))

    # Each measurement contributes the POIs it measures
    def rows_of(measurement):
        return [poi for poi in pois if measurement.has_parameter(poi)]

    theta, cov, weights, chi2, ndf, pulls = _stacked_gls(measurements, list(pois), shared, rows_of)

    return CombinedMeasurement(
        measurements=list(measurements),
        pois=list(pois),
        method="pois_only",
        theta=theta,
        cov=cov,
        weights=weights,
        chi2=chi2,
        ndf=ndf,
        pulls=pulls,
    )


def _combine_pois_nps(measurements: list[Measurement], pois: list[str]) -> CombinedMeasurement:
    """Full combination: BLUE on Y = (theta_1, ..., theta_N), the POIs and the shared NPs.

    The block of a type A measurement is theta_K = (mu_K, alpha_K) restricted to the POIs it
    measures and the shared NPs it fitted; types B and C contribute their POIs only (the NP
    entries of type C are its impacts, not a measurement of the NPs). The blocks of V = Cov(Y) are
        V_KK = Sigma_K on these parameters        (private NPs profiled out, exactly)
        V_KL = Lambda_K Lambda_L^T   (K != L)      (master formula Sigma_K S S^T Sigma_L)
    with Lambda_K = Cov(theta_K, a) the columns of Sigma_K for the shared NPs (NPs in units of
    their prior). The combined NPs are the shared NPs fitted by at least one type A measurement;
    without any, this is the marginalised combination.

    The auxiliary measurement a is not in Y (see pois_nps_aux). V can be singular, e.g. when a
    measurement reports post-fit NPs equal to the prior (a copy of a): the pseudo-inverse is used
    and ndf = rank(V) - number of combined parameters.

    Args:
        measurements: List of measurements to be combined.
        pois: List of parameters of interest (POIs).

    Returns:
        A CombinedMeasurement of the POIs and the combined NPs (`nps`).
    """

    # Nuisance parameters shared by at least two measurements (POIs excluded), and those that
    # are fitted by at least one measurement, hence combined
    shared = sorted(_get_shared_parameters(measurements) - set(pois))
    nps = [nuisance for nuisance in shared
           if any(m.has_parameter(nuisance) and m.type != "C" for m in measurements)]
    names = list(pois) + nps

    # Each measurement contributes its POIs and, unless of type C, its combined NPs
    def rows_of(measurement):
        usable = pois if measurement.type == "C" else names
        return [x for x in usable if measurement.has_parameter(x)]

    theta, cov, weights, chi2, ndf, pulls = _stacked_gls(measurements, names, shared, rows_of)

    return CombinedMeasurement(
        measurements=list(measurements),
        pois=list(pois),
        nps=nps,
        method="pois_nps",
        theta=theta,
        cov=cov,
        weights=weights,
        chi2=chi2,
        ndf=ndf,
        pulls=pulls,
    )


def _stacked_gls(measurements: list[Measurement], names: list[str], shared: list[str],
                 rows_of, rank_tol: float = 1e-10):
    """BLUE (generalised least squares) of the stacked measurements, E[Y] = U theta, correlated
    through the shared NPs: V_KK = Sigma_K on the rows of K, V_KL = Lambda_K Lambda_L^T.

    Args:
        measurements: List of measurements to be combined.
        names: Combined parameters, in the order of the result.
        shared: NPs shared by at least two measurements, constrained by a common auxiliary
            measurement a ~ N(alpha, I); they correlate the measurements.
        rows_of: Function giving, for a measurement, the parameters of `names` it contributes to
            the stack (in the order of `names`).
        rank_tol: Relative tolerance on the eigenvalues of V for its pseudo-inverse.

    Returns:
        theta, cov, weights (one matrix per measurement, columns = its rows), chi2 at the
        minimum, ndf = rank(V) - len(names), and the pulls {measurement: {parameter: pull}}.
    """
    P = len(names)

    # Stacked vector Y, diagonal blocks of V, impacts Lambda_K and mapping matrix U
    Y = []
    V_blocks = []
    Lambdas = []
    U_rows = []

    for measurement in measurements:
        rows = rows_of(measurement)
        indices = [measurement.get_parameter_index(x) for x in rows]

        # Extract the relevant covariance matrix and best fit values
        Y.append(measurement.theta[indices])
        V_blocks.append(measurement.cov[np.ix_(indices, indices)])

        # Impacts on the shared NPs, Lambda_K = Cov(theta_K, a); zero for NPs it does not have
        lam = np.zeros((len(indices), len(shared)))
        for c, nuisance in enumerate(shared):
            if measurement.has_parameter(nuisance):
                lam[:, c] = measurement.cov[indices, measurement.get_parameter_index(nuisance)]
        Lambdas.append(lam)

        # Mapping matrix U for this measurement
        U_row = np.zeros((len(rows), P))
        U_row[np.arange(len(rows)), [names.index(x) for x in rows]] = 1.0
        U_rows.append(U_row)

    Y_stacked = np.concatenate(Y)
    U_stacked = np.vstack(U_rows)
    # L L^T puts Lambda_K Lambda_L^T in every block; on the diagonal it is replaced by the
    # covariance of the measurement, which already contains the NP contribution
    L = np.vstack(Lambdas)
    V_stacked = block_diag(*V_blocks) + L @ L.T - block_diag(*[lam @ lam.T for lam in Lambdas])
    V_stacked = 0.5 * (V_stacked + V_stacked.T)

    # Pseudo-inverse of V (its inverse when V is regular) and its rank
    eigval, eigvec = np.linalg.eigh(V_stacked)
    keep = eigval > rank_tol * eigval.max()
    W_stacked = (eigvec[:, keep] / eigval[keep]) @ eigvec[:, keep].T
    rank = int(keep.sum())

    # GLS fit: combined best fit values and covariance matrix
    info = U_stacked.T @ W_stacked @ U_stacked
    if np.linalg.matrix_rank(info, tol=rank_tol * np.abs(info).max()) < P:
        raise ValueError("The combined parameters are not all determined by the measurements, "
                         f"e.g. an NP is fitted by none of them: {names}")
    cov = np.linalg.inv(info)
    gain = cov @ U_stacked.T @ W_stacked     # theta = gain @ Y
    theta = gain @ Y_stacked

    # Weights of each measurement: the columns of gain, split per measurement
    weights = np.split(gain, np.cumsum([len(y) for y in Y])[:-1], axis=1)

    # chi2 at the minimum, computed from the residuals
    residual = Y_stacked - U_stacked @ theta
    chi2 = float(residual @ W_stacked @ residual)
    ndf = rank - P

    # Pulls: residual over its own standard deviation, Cov(r) = (1 - U gain) V (1 - U gain)^T
    projector = np.eye(len(Y_stacked)) - U_stacked @ gain
    pulls = _pulls(measurements, [rows_of(m) for m in measurements], residual,
                   np.diag(projector @ V_stacked @ projector.T))

    return theta, cov, weights, chi2, ndf, pulls


def _pulls(measurements, rows, residual, variance, rel_tol: float = 1e-10):
    """{measurement name: {parameter: residual / sqrt(variance)}}, nan when the residual has no
    variance (a parameter determined by this measurement alone)."""
    scale = max(float(np.max(variance)), 0.0) if len(variance) else 0.0
    pulls, start = {}, 0
    for measurement, r in zip(measurements, rows):
        block = {}
        for x in r:
            v = variance[start]
            block[x] = float(residual[start] / np.sqrt(v)) if v > rel_tol * scale else float("nan")
            start += 1
        pulls[measurement.name] = block
    return pulls

def _combine_pois_nps_aux(measurements: list[Measurement], pois: list[str],
                          auxiliary: dict[str, float] | None = None) -> CombinedMeasurement:
    """Likelihood sum: BLUE on the stacked statistic Y~ = (theta_1, ..., theta_N, a).

    The auxiliary measurement a ~ N(alpha, I) of the shared NPs is added once to the stack of the
    full combination, as a pseudo-measurement of type "auxiliary" with covariance I; its
    cross-covariance with measurement K is Lambda_K = Cov(theta_K, a) = Sigma_K S, as it should.
    The estimate is that of the likelihood sum (auxiliary term counted once), efficient for any
    mix of types A, B and C, and chi2_{Y~,min} ~ chi2(rank(V~) - p) is its goodness of fit.

    Args:
        measurements: List of measurements to be combined.
        pois: List of parameters of interest (POIs).
        auxiliary: Observed values of a for the shared NPs; NPs not listed are at 0.
    Returns:
        A CombinedMeasurement containing the POIs and the combined NPs (`nps`).
    """
    auxiliary = auxiliary or {}
    shared = sorted(_get_shared_parameters(measurements) - set(pois))
    unknown = set(auxiliary) - set(shared)
    if unknown:
        raise ValueError(f"Auxiliary values given for NPs that are not shared: {sorted(unknown)}")
    if not shared:   # nothing correlates the measurements: no auxiliary measurement to add
        result = _combine_pois_nps(list(measurements), pois)
        result.method = "pois_nps_aux"
        return result
    a = Measurement(name="auxiliary", theta=[auxiliary.get(x, 0.0) for x in shared],
                    cov=np.eye(len(shared)), names=shared, type="auxiliary")
    result = _combine_pois_nps(list(measurements) + [a], pois)
    result.method = "pois_nps_aux"
    return result