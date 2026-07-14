from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np


@dataclass(frozen=True)
class ModelParameters:
    mu: float = np.log(2.0) / 3.2
    lambda_t: float = 1.0 - np.exp(-np.log(2.0) / 15.0)
    lambda_dc: float = 1.0 - np.exp(-np.log(2.0) / 15.0)
    lambda_ln: float = 1.0 - np.exp(-np.log(2.0) / 15.0)
    rho: float = 0.5
    psi: float = 300.0
    omega: float = 0.135
    gamma: float = 0.0
    r: float = 5.0
    k: float = 1.1
    rs_t_alpha: float = 0.05
    rs_t_beta: float = 0.0114
    rs_l_alpha: float = 0.182
    rs_l_beta: float = 0.143
    initial_volume: float = 0.03
    treatment_day: int = 10
    dose_gy: float = 10.0
    clearance_mean_days: float = 3.0
    clearance_std_days: float = 1.5
    activation_sensitivity: float = 1.0


@dataclass(frozen=True)
class SimulationResult:
    days: np.ndarray
    dose: np.ndarray
    sn_t: np.ndarray
    sn_l: np.ndarray
    total_volume: np.ndarray
    viable_volume: np.ndarray
    doomed_volume: np.ndarray
    immune_effect: np.ndarray
    primary_immune_effect: np.ndarray
    secondary_immune_effect: np.ndarray
    lymphocyte_volume: np.ndarray
    triggering_density: np.ndarray
    epsilon: np.ndarray


def replace_params(params: ModelParameters, **updates: float) -> ModelParameters:
    return replace(params, **updates)


def get_observable_series(result: SimulationResult, observable: str) -> np.ndarray:
    observable_map = {
        "total_volume": result.total_volume,
        "viable_volume": result.viable_volume,
        "doomed_volume": result.doomed_volume,
        "immune_effect": result.immune_effect,
        "primary_immune_effect": result.primary_immune_effect,
        "secondary_immune_effect": result.secondary_immune_effect,
        "lymphocyte_volume": result.lymphocyte_volume,
        "triggering_density": result.triggering_density,
        "epsilon": result.epsilon,
    }
    try:
        return observable_map[observable]
    except KeyError as exc:
        supported = ", ".join(sorted(observable_map))
        raise ValueError(f"Unsupported observable '{observable}'. Supported: {supported}") from exc


def _clearance_weights(num_days: int, mean_days: float, std_days: float) -> np.ndarray:
    grid = np.arange(num_days, dtype=float)
    coeff = 1.0 / (std_days * np.sqrt(2.0 * np.pi))
    weights = coeff * np.exp(-0.5 * ((grid - mean_days) / std_days) ** 2)
    if num_days > 1:
        weights[0] = max(0.0, 1.0 - np.sum(weights[1:]))
    total = np.sum(weights)
    if total <= 0.0:
        raise ValueError("Invalid clearance weights")
    return weights / total


def _tumor_survival(dose: np.ndarray, params: ModelParameters, weights: np.ndarray) -> np.ndarray:
    hazard = params.rs_t_alpha * dose + params.rs_t_beta * dose**2
    sn_t = np.ones_like(hazard)
    num_days, num_voxels = dose.shape
    for voxel in range(num_voxels):
        convolved = np.convolve(hazard[:, voxel], weights, mode="full")[:num_days]
        sn_t[:, voxel] = np.exp(-convolved)
    return sn_t


def _lymphocyte_survival(dose: np.ndarray, params: ModelParameters) -> np.ndarray:
    return np.exp(-(params.rs_l_alpha * dose + params.rs_l_beta * dose**2))


def _build_dose_schedule(num_days: int, n_voxels: int, treatment_day: int, dose_gy: float, coverage: float) -> np.ndarray:
    dose = np.zeros((num_days, n_voxels), dtype=float)
    if not 0.0 <= coverage <= 1.0:
        raise ValueError("coverage must be within [0, 1]")
    if not 0 <= treatment_day < num_days:
        raise ValueError("treatment_day must be within simulation range")
    covered_voxels = int(round(n_voxels * coverage))
    if coverage > 0.0 and covered_voxels == 0:
        covered_voxels = 1
    dose[treatment_day, :covered_voxels] = dose_gy
    return dose


def simulate_experiment(
    params: ModelParameters,
    *,
    coverage: float,
    num_days: int = 181,
    n_voxels: int = 100,
) -> SimulationResult:
    days = np.arange(num_days, dtype=int)
    weights = _clearance_weights(num_days, params.clearance_mean_days, params.clearance_std_days)
    dose = _build_dose_schedule(num_days, n_voxels, params.treatment_day, params.dose_gy, coverage)
    sn_t = _tumor_survival(dose, params, weights)
    sn_l = _lymphocyte_survival(dose, params)

    t = np.zeros((num_days, n_voxels), dtype=float)
    ln = np.zeros((num_days, n_voxels), dtype=float)
    dc = np.ones((num_days, n_voxels), dtype=float)
    d = np.zeros((num_days, n_voxels), dtype=float)
    eps = np.zeros(num_days, dtype=float)
    zp = np.zeros(num_days, dtype=float)
    zs = np.zeros(num_days, dtype=float)

    t[0, :] = params.initial_volume / n_voxels

    for day in range(num_days - 1):
        zmax = zp[day] + zs[day]
        t[day + 1, :] = t[day, :] * sn_t[day, :] * np.exp(params.mu - zmax)

        denominator = np.sum(t[day, :] + d[day, :])
        denominator = max(denominator, 1e-12)
        damaged_fraction = np.dot(1.0 - sn_t[day, :], t[day, :]) / denominator
        eps[day + 1] = 0.999 * np.tanh(params.activation_sensitivity * damaged_fraction)

        ratio = sn_l[day, :] / np.clip(sn_t[day, :], 1e-12, None)
        dc[day + 1, :] = (ratio * dc[day, :] + (1.0 - dc[day, :]) * params.lambda_dc) * (1.0 - eps[day + 1])
        ln[day + 1, :] = (
            (1.0 - params.lambda_ln) * sn_l[day, :] * ln[day, :]
            + params.rho * t[day + 1, :]
            + params.psi * eps[day + 1] * dc[day + 1, :] * t[day + 1, :]
        )

        total_t = np.sum(t[day + 1, :])
        total_ln = np.sum(ln[day + 1, :])
        suppression = 1.0 + params.k * max(total_t, 0.0) ** (2.0 / 3.0) * max(total_ln, 0.0)
        zp[day + 1] = params.omega * total_ln / max(suppression, 1e-12)
        zs[day + 1] = zs[day] + params.gamma * zp[day + 1] / params.r
        d[day + 1, :] = (
            (1.0 - params.lambda_t) * d[day, :]
            + (1.0 - sn_t[day, :]) * t[day, :]
            + sn_t[day, :] * t[day, :] * np.exp(params.mu) * (1.0 - np.exp(-zmax))
        )

    viable_volume = np.sum(t, axis=1)
    doomed_volume = np.sum(d, axis=1)
    total_volume = viable_volume + doomed_volume
    immune_effect = zp + zs
    lymphocyte_volume = np.sum(ln, axis=1)
    triggering_density = np.mean(dc, axis=1)
    return SimulationResult(
        days=days,
        dose=dose,
        sn_t=sn_t,
        sn_l=sn_l,
        total_volume=total_volume,
        viable_volume=viable_volume,
        doomed_volume=doomed_volume,
        immune_effect=immune_effect,
        primary_immune_effect=zp,
        secondary_immune_effect=zs,
        lymphocyte_volume=lymphocyte_volume,
        triggering_density=triggering_density,
        epsilon=eps,
    )
