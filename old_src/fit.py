from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import differential_evolution, least_squares

from .model import ModelParameters, SimulationResult, replace_params, simulate_experiment


@dataclass(frozen=True)
class ObservationSeries:
    name: str
    coverage: float
    absolute_days: np.ndarray
    relative_days: np.ndarray
    clean_volume: np.ndarray
    observed_volume: np.ndarray


@dataclass(frozen=True)
class SyntheticDataset:
    series: tuple[ObservationSeries, ...]
    true_params: ModelParameters
    noise_sigma: float
    seed: int


@dataclass(frozen=True)
class FitParameterSpec:
    name: str
    lower: float
    upper: float


@dataclass(frozen=True)
class FitResult:
    params: ModelParameters
    fitted_values: dict[str, float]
    true_values: dict[str, float]
    parameter_specs: tuple[FitParameterSpec, ...]
    global_cost: float
    local_cost: float
    residual_norm: float
    predictions: dict[str, SimulationResult]


DEFAULT_FIT_SPECS: tuple[FitParameterSpec, ...] = (
    FitParameterSpec("psi", 10.0, 800.0),
    FitParameterSpec("omega", 0.01, 0.3),
    FitParameterSpec("k", 0.05, 2.5),
    FitParameterSpec("initial_volume", 0.005, 0.08),
)

EXTENDED_FIT_SPECS: tuple[FitParameterSpec, ...] = (
    FitParameterSpec("rho", 0.05, 1.0),
    *DEFAULT_FIT_SPECS,
)


def generate_synthetic_dataset(
    true_params: ModelParameters,
    *,
    coverages: tuple[float, ...] = (0.5, 1.0),
    relative_days: np.ndarray | None = None,
    noise_sigma: float = 0.08,
    seed: int = 7,
    num_days: int = 181,
    n_voxels: int = 100,
) -> SyntheticDataset:
    if relative_days is None:
        relative_days = np.arange(0, 31, 3)
    relative_days = np.asarray(relative_days, dtype=int)
    rng = np.random.default_rng(seed)
    series = []
    for coverage in coverages:
        result = simulate_experiment(true_params, coverage=coverage, num_days=num_days, n_voxels=n_voxels)
        absolute_days = true_params.treatment_day + relative_days
        clean = result.total_volume[absolute_days]
        noisy = clean * np.exp(rng.normal(0.0, noise_sigma, size=clean.shape))
        series.append(
            ObservationSeries(
                name=f"coverage_{coverage:.2f}",
                coverage=coverage,
                absolute_days=absolute_days,
                relative_days=relative_days,
                clean_volume=clean,
                observed_volume=noisy,
            )
        )
    return SyntheticDataset(series=tuple(series), true_params=true_params, noise_sigma=noise_sigma, seed=seed)


def _vector_to_params(base_params: ModelParameters, specs: tuple[FitParameterSpec, ...], values: np.ndarray) -> ModelParameters:
    updates = {spec.name: float(value) for spec, value in zip(specs, values, strict=True)}
    return replace_params(base_params, **updates)

def _prior_residuals(
    values: np.ndarray,
    base_params: ModelParameters,
    specs: tuple[FitParameterSpec, ...],
    prior_log_sigma: float | None,
) -> np.ndarray:
    if prior_log_sigma is None:
        return np.array([], dtype=float)
    residuals = []
    for spec, value in zip(specs, values, strict=True):
        center = float(getattr(base_params, spec.name))
        if value <= 0.0 or center <= 0.0:
            residuals.append(1e3)
            continue
        residuals.append((np.log(value) - np.log(center)) / prior_log_sigma)
    return np.asarray(residuals, dtype=float)


def _residual_vector(
    values: np.ndarray,
    dataset: SyntheticDataset,
    base_params: ModelParameters,
    specs: tuple[FitParameterSpec, ...],
    num_days: int,
    n_voxels: int,
    prior_log_sigma: float | None,
) -> np.ndarray:
    try:
        params = _vector_to_params(base_params, specs, values)
        residuals = []
        for series in dataset.series:
            result = simulate_experiment(params, coverage=series.coverage, num_days=num_days, n_voxels=n_voxels)
            prediction = result.total_volume[series.absolute_days]
            if not np.all(np.isfinite(prediction)) or np.any(prediction <= 0.0):
                return np.full(sum(len(s.observed_volume) for s in dataset.series), 1e6)
            residuals.append(np.log(prediction + 1e-12) - np.log(series.observed_volume + 1e-12))
        combined = np.concatenate(residuals)
        priors = _prior_residuals(values, base_params, specs, prior_log_sigma)
        return np.concatenate([combined, priors]) if priors.size else combined
    except Exception:
        return np.full(sum(len(s.observed_volume) for s in dataset.series), 1e6)


def fit_synthetic_dataset(
    dataset: SyntheticDataset,
    *,
    base_params: ModelParameters | None = None,
    fit_specs: tuple[FitParameterSpec, ...] = DEFAULT_FIT_SPECS,
    num_days: int = 181,
    n_voxels: int = 100,
    seed: int = 11,
    prior_log_sigma: float | None = 0.35,
    de_maxiter: int = 60,
    de_popsize: int = 12,
    local_max_nfev: int = 200,
) -> FitResult:
    if base_params is None:
        base_params = ModelParameters()
    bounds = [(spec.lower, spec.upper) for spec in fit_specs]

    objective = lambda x: np.sum(
        _residual_vector(x, dataset, base_params, fit_specs, num_days, n_voxels, prior_log_sigma) ** 2
    )

    global_result = differential_evolution(
        objective,
        bounds=bounds,
        seed=seed,
        polish=False,
        updating="deferred",
        workers=1,
        maxiter=de_maxiter,
        popsize=de_popsize,
    )
    local_result = least_squares(
        lambda x: _residual_vector(x, dataset, base_params, fit_specs, num_days, n_voxels, prior_log_sigma),
        x0=global_result.x,
        bounds=np.array(bounds, dtype=float).T,
        method="trf",
        max_nfev=local_max_nfev,
    )

    fitted_params = _vector_to_params(base_params, fit_specs, local_result.x)
    predictions = {
        series.name: simulate_experiment(fitted_params, coverage=series.coverage, num_days=num_days, n_voxels=n_voxels)
        for series in dataset.series
    }
    fitted_values = {spec.name: float(value) for spec, value in zip(fit_specs, local_result.x, strict=True)}
    true_values = {spec.name: float(getattr(dataset.true_params, spec.name)) for spec in fit_specs}

    return FitResult(
        params=fitted_params,
        fitted_values=fitted_values,
        true_values=true_values,
        parameter_specs=fit_specs,
        global_cost=float(global_result.fun),
        local_cost=float(np.sum(local_result.fun**2)),
        residual_norm=float(np.linalg.norm(local_result.fun)),
        predictions=predictions,
    )
