from typing import List, Optional, Tuple, cast

import numpy as np
from scipy.stats import chi2, norm

from cmaes._cma import CMA, _is_valid_bounds


class CMAwM(CMA):
    """CMA-ES with Margin class with ask-and-tell interface.
    The code is adapted from https://github.com/EvoConJP/CMA-ES_with_Margin.

    Example:

        .. code::

            import numpy as np
            from cmaes import CMAwM

            def ellipsoid_onemax(x, n_zdim):
                n = len(x)
                n_rdim = n - n_zdim
                ellipsoid = sum([(1000 ** (i / (n_rdim - 1)) * x[i]) ** 2 for i in range(n_rdim)])
                onemax = n_zdim - (0. < x[(n - n_zdim):]).sum()
                return ellipsoid + 10 * onemax

            dim = 20
            binary_dim = dim//2
            discrete_space = np.tile(np.arange(0, 2, 1), (binary_dim, 1))     # binary variables
            optimizer = CMAwM(mean=0 * np.ones(dim), sigma=2.0, discrete_space=discrete_space)

            evals = 0
            while True:
                solutions = []
                for _ in range(optimizer.population_size):
                    x_for_eval, x_for_tell = optimizer.ask()
                    value = ellipsoid_onemax(x_for_eval, binary_dim)
                    evals += 1
                    solutions.append((x_for_tell, value))
                optimizer.tell(solutions)

                if optimizer.should_stop():
                    break

    Args:

        mean:
            Initial mean vector of multi-variate gaussian distributions.

        sigma:
            Initial standard deviation of covariance matrix.

        discrete_space:
            Discrete space. Both binary and integer are acceptable.

        continuous_space:
            Lower and upper domain boundaries for each parameter (optional).

        n_max_resampling:
            A maximum number of resampling parameters (default: 100).
            If all sampled parameters are infeasible, the last sampled one
            will be clipped with lower and upper bounds.

        seed:
            A seed number (optional).

        population_size:
            A population size (optional).

        cov:
            A covariance matrix (optional).

        margin:
            A margin parameter (optional).
    """

    # Paper: https://arxiv.org/abs/2205.13482

    def __init__(
        self,
        mean: np.ndarray,
        sigma: float,
        discrete_space: np.ndarray,
        continuous_space: Optional[np.ndarray] = None,
        n_max_resampling: int = 100,
        seed: Optional[int] = None,
        population_size: Optional[int] = None,
        cov: Optional[np.ndarray] = None,
        margin: Optional[float] = None,
    ):
        super().__init__(
            mean, sigma, None, n_max_resampling, seed, population_size, cov
        )

        self._n_zdim = len(discrete_space)
        self._n_rdim = self._n_dim - self._n_zdim

        # continuous_space contains low and high of each parameter.
        assert continuous_space is None or _is_valid_bounds(
            continuous_space, mean[: self._n_rdim]
        ), "invalid bounds"
        self._continuous_space = continuous_space
        self._n_max_resampling = n_max_resampling

        # discrete_space
        self.margin = (
            margin if margin is not None else 1 / (self._n_dim * self._popsize)
        )
        assert self.margin > 0, "margin must be non-zero positive value."
        self.z_space = discrete_space
        self.z_lim = (self.z_space[:, 1:] + self.z_space[:, :-1]) / 2
        for i in range(self._n_zdim):
            self.z_space[i][np.isnan(self.z_space[i])] = np.nanmax(self.z_space[i])
            self.z_lim[i][np.isnan(self.z_lim[i])] = np.nanmax(self.z_lim[i])
        self.z_lim_low = np.concatenate(
            [self.z_lim.min(axis=1).reshape([self._n_zdim, 1]), self.z_lim], 1
        )
        self.z_lim_up = np.concatenate(
            [self.z_lim, self.z_lim.max(axis=1).reshape([self._n_zdim, 1])], 1
        )
        m_z = self._mean[self._n_rdim :].reshape(([self._n_zdim, 1]))
        # m_z_lim_low ->|  mean vector    |<- m_z_lim_up
        self.m_z_lim_low = (
            self.z_lim_low
            * np.where(np.sort(np.concatenate([self.z_lim, m_z], 1)) == m_z, 1, 0)
        ).sum(axis=1)
        self.m_z_lim_up = (
            self.z_lim_up
            * np.where(np.sort(np.concatenate([self.z_lim, m_z], 1)) == m_z, 1, 0)
        ).sum(axis=1)

        self._A = np.full(self._n_dim, 1.0)

    def ask(self) -> Tuple[np.ndarray, np.ndarray]:
        """Sample a parameter and return (i) encoded x and (ii) raw x.
        The encoded x is used for the evaluation.
        The raw x is used for updating the distribution."""
        for i in range(self._n_max_resampling):
            x = self._sample_solution()
            if self._is_continuous_feasible(x[: self._n_rdim]):
                x_disc = self._encoding_discrete_params(x[self._n_rdim :])
                return np.append(x[: self._n_rdim], x_disc), x
        x = self._sample_solution()
        x_cont = self._repair_continuous_params(x[: self._n_rdim])
        x_disc = self._encoding_discrete_params(x[self._n_rdim :])
        return np.append(x_cont, x_disc), x

    def _is_continuous_feasible(self, continuous_param: np.ndarray) -> bool:
        if self._continuous_space is None:
            return True
        return cast(
            bool,
            np.all(continuous_param >= self._continuous_space[:, 0])
            and np.all(continuous_param <= self._continuous_space[:, 1]),
        )  # Cast bool_ to bool.

    def _repair_continuous_params(self, continuous_param: np.ndarray) -> np.ndarray:
        if self._continuous_space is None:
            return continuous_param

        # clip with lower and upper bound.
        param = np.where(
            continuous_param < self._continuous_space[:, 0],
            self._continuous_space[:, 0],
            continuous_param,
        )
        param = np.where(
            param > self._continuous_space[:, 1], self._continuous_space[:, 1], param
        )
        return param

    def _encoding_discrete_params(self, discrete_param: np.ndarray) -> np.ndarray:
        """Encode the values into discrete domain."""
        x = (discrete_param - self._mean[self._n_rdim :]) * self._A[
            self._n_rdim :
        ] + self._mean[self._n_rdim :]
        x = x.reshape([self._n_zdim, 1])
        x_enc = (
            self.z_space
            * np.where(np.sort(np.concatenate((self.z_lim, x), axis=1)) == x, 1, 0)
        ).sum(axis=1)
        return x_enc.reshape(self._n_zdim)

    def tell(self, solutions: List[Tuple[np.ndarray, float]]) -> None:
        """Tell evaluation values"""

        super().tell(solutions)

        # margin correction if margin > 0
        if self.margin > 0:
            updated_m_integer = self._mean[self._n_rdim :, np.newaxis]
            self.z_lim_low = np.concatenate(
                [self.z_lim.min(axis=1).reshape([self._n_zdim, 1]), self.z_lim], 1
            )
            self.z_lim_up = np.concatenate(
                [self.z_lim, self.z_lim.max(axis=1).reshape([self._n_zdim, 1])], 1
            )
            self.m_z_lim_low = (
                self.z_lim_low
                * np.where(
                    np.sort(np.concatenate([self.z_lim, updated_m_integer], 1))
                    == updated_m_integer,
                    1,
                    0,
                )
            ).sum(axis=1)
            self.m_z_lim_up = (
                self.z_lim_up
                * np.where(
                    np.sort(np.concatenate([self.z_lim, updated_m_integer], 1))
                    == updated_m_integer,
                    1,
                    0,
                )
            ).sum(axis=1)

            # calculate probability low_cdf := Pr(X <= m_z_lim_low) and up_cdf := Pr(m_z_lim_up < X)
            # sig_z_sq_Cdiag = self.model.sigma * self.model.A * np.sqrt(np.diag(self.model.C))
            z_scale = (
                self._sigma
                * self._A[self._n_rdim :]
                * np.sqrt(np.diag(self._C)[self._n_rdim :])
            )
            updated_m_integer = updated_m_integer.flatten()
            low_cdf = norm.cdf(self.m_z_lim_low, loc=updated_m_integer, scale=z_scale)
            up_cdf = 1.0 - norm.cdf(
                self.m_z_lim_up, loc=updated_m_integer, scale=z_scale
            )
            mid_cdf = 1.0 - (low_cdf + up_cdf)
            # edge case
            edge_mask = np.maximum(low_cdf, up_cdf) > 0.5
            # otherwise
            side_mask = np.maximum(low_cdf, up_cdf) <= 0.5

            if np.any(edge_mask):
                # modify mask (modify or not)
                modify_mask = np.minimum(low_cdf, up_cdf) < self.margin
                # modify sign
                modify_sign = np.sign(self._mean[self._n_rdim :] - self.m_z_lim_up)
                # distance from m_z_lim_up
                dist = (
                    self._sigma
                    * self._A[self._n_rdim :]
                    * np.sqrt(
                        chi2.ppf(q=1.0 - 2.0 * self.margin, df=1)
                        * np.diag(self._C)[self._n_rdim :]
                    )
                )
                # modify mean vector
                self._mean[self._n_rdim :] = self._mean[
                    self._n_rdim :
                ] + modify_mask * edge_mask * (
                    self.m_z_lim_up + modify_sign * dist - self._mean[self._n_rdim :]
                )

            # correct probability
            low_cdf = np.maximum(low_cdf, self.margin / 2.0)
            up_cdf = np.maximum(up_cdf, self.margin / 2.0)
            modified_low_cdf = low_cdf + (1.0 - low_cdf - up_cdf - mid_cdf) * (
                low_cdf - self.margin / 2
            ) / (low_cdf + mid_cdf + up_cdf - 3.0 * self.margin / 2)
            modified_up_cdf = up_cdf + (1.0 - low_cdf - up_cdf - mid_cdf) * (
                up_cdf - self.margin / 2
            ) / (low_cdf + mid_cdf + up_cdf - 3.0 * self.margin / 2)
            modified_low_cdf = np.clip(modified_low_cdf, 1e-10, 0.5 - 1e-10)
            modified_up_cdf = np.clip(modified_up_cdf, 1e-10, 0.5 - 1e-10)

            # modify mean vector and A (with sigma and C fixed)
            chi_low_sq = np.sqrt(chi2.ppf(q=1.0 - 2 * modified_low_cdf, df=1))
            chi_up_sq = np.sqrt(chi2.ppf(q=1.0 - 2 * modified_up_cdf, df=1))
            C_diag_sq = np.sqrt(np.diag(self._C))[self._n_rdim :]

            # simultaneous equations
            self._A[self._n_rdim :] = self._A[self._n_rdim :] + side_mask * (
                (self.m_z_lim_up - self.m_z_lim_low)
                / ((chi_low_sq + chi_up_sq) * self._sigma * C_diag_sq)
                - self._A[self._n_rdim :]
            )
            self._mean[self._n_rdim :] = self._mean[self._n_rdim :] + side_mask * (
                (self.m_z_lim_low * chi_up_sq + self.m_z_lim_up * chi_low_sq)
                / (chi_low_sq + chi_up_sq)
                - self._mean[self._n_rdim :]
            )
