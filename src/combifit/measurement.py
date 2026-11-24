"""Measurement container.

A measurement K is summarised by its fitted parameters theta_K = (mu_K, alpha_K), POIs and NPs,
and their covariance Sigma_K. The parameter names correlate the measurements: an NP with the same
name in two measurements is the same NP, constrained by the same auxiliary measurement
a ~ N(alpha, I), so the NPs must be in units of their prior. The names follow the order of theta
and of the covariance matrix.

Types (theory note heterogeneous.tex):
    A  POIs and shared NPs with their post-fit covariance (profile-likelihood fit);
    B  POIs and their covariance only (total uncertainty);
    C  POIs, their covariance and the impacts lambda_K = Cov(mu_K, a), stored as the POI-NP
       covariance with the NP block at the prior (see models.impacts_only).
"""


from dataclasses import dataclass, field
import numpy as np


@dataclass(frozen=True)
class Measurement:
    """Measurement container.

    Attributes:
        name: Name of the measurement.
        theta: Fitted parameters theta_K.
        cov: Their covariance matrix Sigma_K.
        names: Names of the fitted parameters, in the order of theta.
        type: "A", "B" or "C" (see the module); None is taken as "A".
        chi2, ndof: Goodness of fit of the measurement, if known.
    """

    name: str
    theta: "np.ndarray" = field(repr=False)
    cov: "np.ndarray" = field(repr=False)
    names: list[str] = field(repr=False)
    type: str | None = None
    chi2: float | None = None
    ndof: int | None = None

    def _validate(self) -> None:
        """Validate the measurement.

        Raises:
            ValueError: If theta is not a vector, cov is not a matrix, or if the length of theta,
                cov, and names do not match.
        """
        if self.theta.ndim != 1 or self.cov.ndim != 2:
            raise ValueError(
                f"theta must be 1D and cov 2D, got shapes {self.theta.shape} and {self.cov.shape}."
            )
        Ntheta = len(self.theta)
        Ncovx, Ncovy = self.cov.shape
        Nnames = len(self.names)
        if Ntheta != Ncovx or Ntheta != Ncovy or Ntheta != Nnames:
            raise ValueError(
                f"Length of theta ({Ntheta}), cov ({Ncovx}, {Ncovy}), and names ({Nnames}) do not match."
            )


    def __post_init__(self) -> None:
        """Convert inputs to numpy arrays, then validate.

        The dataclass is frozen, hence object.__setattr__.
        """
        object.__setattr__(self, "theta", np.asarray(self.theta, dtype=float))
        object.__setattr__(self, "cov", np.asarray(self.cov, dtype=float))
        object.__setattr__(self, "names", list(self.names))
        self._validate()

    def has_parameter(self, parameter: str) -> bool:
        """Check if the measurement has a specific parameter.

        Args:
            parameter: Name of the parameter to check.
        """
        return parameter in self.names

    def get_parameter_index(self, parameter: str) -> int:
        """Get the index of a specific parameter.

        Args:
            parameter: Name of the parameter to get the index for.

        Raises:
            ValueError: If the parameter is not found in the measurement.
        """
        if not self.has_parameter(parameter):
            raise ValueError(f"Parameter '{parameter}' not found in measurement '{self.name}'.")
        return self.names.index(parameter)

    def get_parameter_value(self, parameter: str) -> float:
        """Get the value of a specific parameter.

        Args:
            parameter: Name of the parameter to get the value for.

        Raises:
            ValueError: If the parameter is not found in the measurement.
        """
        index = self.get_parameter_index(parameter)
        return self.theta[index]

    def get_covariance(self, parameter1: str, parameter2: str) -> float:
        """Get the covariance between two specific parameters.

        Args:
            parameter1: Name of the first parameter.
            parameter2: Name of the second parameter.

        Raises:
            ValueError: If either parameter is not found in the measurement.
        """
        index1 = self.get_parameter_index(parameter1)
        index2 = self.get_parameter_index(parameter2)
        return self.cov[index1, index2]

    def get_uncertainty(self, parameter: str) -> float:
        """Get the uncertainty (standard deviation) of a specific parameter.

        Args:
            parameter: Name of the parameter to get the uncertainty for.

        Raises:
            ValueError: If the parameter is not found in the measurement.
        """
        index = self.get_parameter_index(parameter)
        return np.sqrt(self.cov[index, index])


