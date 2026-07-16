"""Daily radio-immune forward model."""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ModelParameters:
    """Parameters and numerical grid for a radio-immune simulation."""

    alpha_t: float = 0.05
    beta_t: float = 0.05 / 4.4
    alpha_l: float = 0.182
    beta_l: float = 0.143
    mu: float = np.log(2.0) / 3.2
    lambda_t: float = 1.0 - np.exp(-np.log(2.0) / 15.0)
    lambda_a: float = 1.0 - np.exp(-np.log(2.0) / 15.0)
    lambda_l: float = 1.0 - np.exp(-np.log(2.0) / 15.0)
    rho: float = 0.5
    psi: float = 300.0
    omega: float = 0.135
    gamma: float = 0.0
    r: float = 5.0
    kappa: float = 1.1
    initial_volume: float = 0.03
    treatment_day: int = 10
    dose_gy: float = 10.0
    n_days: int = 181
    n_bins: int = 100
    clearance_mean_days: float = 3.0
    clearance_std_days: float = 1.5
    activation_sensitivity: float = 1.0
    p1: float = 0.0
    c4: float = 0.0

    def __post_init__(self) -> None:
        if self.n_days < 2 or int(self.n_days) != self.n_days:
            raise ValueError("n_days must be an integer of at least 2")
        if self.n_bins < 1 or int(self.n_bins) != self.n_bins:
            raise ValueError("n_bins must be a positive integer")
        if not 0 <= self.treatment_day < self.n_days:
            raise ValueError("treatment_day must lie within the simulation range")
        if not np.isfinite(self.dose_gy) or self.dose_gy < 0.0:
            raise ValueError("dose_gy must be finite and non-negative")
        if not np.isfinite(self.initial_volume) or self.initial_volume <= 0.0:
            raise ValueError("initial_volume must be positive and finite")
        if self.clearance_std_days <= 0.0:
            raise ValueError("clearance_std_days must be positive")


@dataclass(frozen=True)
class SimulationResult:
    """State trajectories produced by :func:`simulate`."""

    days: np.ndarray
    dose: np.ndarray
    sn_t: np.ndarray
    sn_l: np.ndarray
    viable: np.ndarray
    doomed: np.ndarray
    lymphocytes: np.ndarray
    triggering_cells: np.ndarray
    epsilon: np.ndarray
    z_primary: np.ndarray
    z_secondary: np.ndarray

    @property
    def total_volume(self) -> np.ndarray:
        """Total viable plus doomed tumor volume."""
        return self.viable.sum(axis=1) + self.doomed.sum(axis=1)

    @property
    def immune_effect(self) -> np.ndarray:
        """Total primary plus secondary immune effect."""
        return self.z_primary + self.z_secondary


def normal_pdf(x: np.ndarray, mean: float, std: float) -> np.ndarray:
    """Evaluate a normal probability density."""
    if not np.isfinite(std) or std <= 0.0:
        raise ValueError("std must be positive and finite")
    return np.exp(-0.5 * ((x - mean) / std) ** 2) / (std * np.sqrt(2.0 * np.pi))


def clearance_weights(p: ModelParameters) -> np.ndarray:
    """Build delayed tumor-clearance weights."""
    lag = np.arange(p.n_days, dtype=float)
    weights = normal_pdf(lag, mean=p.clearance_mean_days, std=p.clearance_std_days)
    weights[0] = 1.0 - weights[1:].sum()
    if weights[0] < 0.0 or not np.isclose(weights.sum(), 1.0):
        raise ValueError("Invalid delayed-death weights")
    return weights


def build_dose_schedule(p: ModelParameters, coverage: float) -> np.ndarray:
    """Create the treatment-day dose grid for a covered-volume fraction."""
    if not 0.0 <= coverage <= 1.0:
        raise ValueError("coverage must lie in [0, 1]")
    dose = np.zeros((p.n_days, p.n_bins), dtype=float)
    covered_bins = int(round(p.n_bins * coverage))
    dose[p.treatment_day, :covered_bins] = p.dose_gy
    return dose


def survival_fractions(
    dose: np.ndarray, p: ModelParameters
) -> tuple[np.ndarray, np.ndarray]:
    """Return delayed tumor and same-day lymphocyte survival fractions."""
    tumor_hazard = p.alpha_t * dose + p.beta_t * dose**2
    weights = clearance_weights(p)
    sn_t = np.empty_like(tumor_hazard)
    for j in range(p.n_bins):
        delayed_hazard = np.convolve(tumor_hazard[:, j], weights, mode="full")[
            : p.n_days
        ]
        sn_t[:, j] = np.exp(-delayed_hazard)
    lymphocyte_hazard = p.alpha_l * dose + p.beta_l * dose**2
    return sn_t, np.exp(-lymphocyte_hazard)


def simulate_dose_schedule(
    p: ModelParameters,
    dose: np.ndarray,
) -> SimulationResult:
    """Simulate an explicit daily dose-by-volume schedule."""
    dose = np.asarray(dose, dtype=float)
    expected_shape = (p.n_days, p.n_bins)
    if dose.shape != expected_shape:
        raise ValueError(f"dose must have shape {expected_shape}")
    if np.any(dose < 0.0) or not np.all(np.isfinite(dose)):
        raise ValueError("dose must be finite and non-negative")

    days = np.arange(p.n_days)
    sn_t, sn_l = survival_fractions(dose, p)
    t = np.zeros((p.n_days, p.n_bins))
    d = np.zeros_like(t)
    lymphocytes = np.zeros_like(t)
    triggering_cells = np.ones_like(t)
    epsilon = np.zeros(p.n_days)
    z_primary = np.zeros(p.n_days)
    z_secondary = np.zeros(p.n_days)
    t[0, :] = p.initial_volume / p.n_bins

    for n in range(p.n_days - 1):
        next_n = n + 1
        z_now = z_primary[n] + z_secondary[n]
        t[next_n, :] = t[n, :] * sn_t[n, :] * np.exp(p.mu - z_now)

        measured_volume = max(np.sum(t[n, :] + d[n, :]), 1e-15)
        damaged_fraction = np.dot(1.0 - sn_t[n, :], t[n, :]) / measured_volume
        epsilon[next_n] = 0.999 * np.tanh(p.activation_sensitivity * damaged_fraction)
        triggering_cells[next_n, :] = (
            (sn_l[n, :] / np.clip(sn_t[n, :], 1e-15, None)) * triggering_cells[n, :]
            + (1.0 - triggering_cells[n, :]) * p.lambda_a
        ) * (1.0 - epsilon[next_n])
        lymphocytes[next_n, :] = (
            (1.0 - p.lambda_l) * sn_l[n, :] * lymphocytes[n, :]
            + p.rho * t[next_n, :]
            + p.psi * epsilon[next_n] * triggering_cells[next_n, :] * t[next_n, :]
        )

        total_t = t[next_n, :].sum()
        total_l = lymphocytes[next_n, :].sum()
        suppression = 1.0 + p.kappa * total_t ** (2.0 / 3.0) * total_l / (1.0 + p.p1)
        z_primary[next_n] = p.omega * total_l / suppression
        z_secondary[next_n] = (
            z_secondary[n] + p.gamma * (1.0 + p.c4) / (p.r + p.c4) * z_primary[next_n]
        )
        d[next_n, :] = (
            (1.0 - p.lambda_t) * d[n, :]
            + (1.0 - sn_t[n, :]) * t[n, :]
            + sn_t[n, :] * t[n, :] * np.exp(p.mu) * (1.0 - np.exp(-z_now))
        )

    return SimulationResult(
        days=days,
        dose=dose,
        sn_t=sn_t,
        sn_l=sn_l,
        viable=t,
        doomed=d,
        lymphocytes=lymphocytes,
        triggering_cells=triggering_cells,
        epsilon=epsilon,
        z_primary=z_primary,
        z_secondary=z_secondary,
    )


def simulate(p: ModelParameters, coverage: float) -> SimulationResult:
    """Simulate one single-fraction covered-volume treatment arm."""
    return simulate_dose_schedule(p, build_dose_schedule(p, coverage))
