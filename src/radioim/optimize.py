"""Log-space fitting for volume observations."""

from dataclasses import dataclass, replace
from typing import Sequence

import numpy as np
import pandas as pd
from scipy.optimize import differential_evolution, least_squares

from .data import FittingDataset
from .model import ModelParameters, simulate


@dataclass(frozen=True)
class FitParameterSpec:
    """Natural-scale bounds and display metadata for one fitted parameter."""

    name: str
    lower: float
    upper: float
    label: str

    def __post_init__(self) -> None:
        if self.lower <= 0.0 or self.upper <= self.lower:
            raise ValueError("Fit parameter bounds must satisfy 0 < lower < upper")


DEFAULT_VOLUME_FIT_SPECS = (
    FitParameterSpec("mu", np.log(2.0) / 30.0, np.log(2.0) / 1.5, r"$\mu$"),
    FitParameterSpec("omega", 0.01, 0.30, r"$\omega$"),
    FitParameterSpec("psi", 10.0, 800.0, r"$\psi$"),
    FitParameterSpec("kappa", 0.05, 2.5, r"$\kappa$"),
)


@dataclass(frozen=True)
class FitResult:
    """Result of differential-evolution followed by bounded least squares."""

    params: ModelParameters
    fit_specs: tuple[FitParameterSpec, ...]
    log_values: np.ndarray
    global_cost: float
    local_cost: float
    residuals: np.ndarray
    jacobian: np.ndarray


def log_bounds(fit_specs: Sequence[FitParameterSpec]) -> np.ndarray:
    """Return natural-log lower and upper bounds for fit specifications."""
    if not fit_specs:
        raise ValueError("At least one fit specification is required")
    return np.array([[np.log(spec.lower), np.log(spec.upper)] for spec in fit_specs])


def params_from_log_vector(
    base: ModelParameters, log_values: np.ndarray, fit_specs: Sequence[FitParameterSpec]
) -> ModelParameters:
    """Apply fitted natural-scale values to base parameters."""
    values = np.asarray(log_values, dtype=float)
    if values.ndim != 1 or len(values) != len(fit_specs):
        raise ValueError("log_values must match fit_specs")
    updates = {
        spec.name: float(np.exp(value))
        for spec, value in zip(fit_specs, values, strict=True)
    }
    try:
        return replace(base, **updates)
    except TypeError as error:
        raise ValueError(
            "Fit specification names must be ModelParameters fields"
        ) from error


def _arm_parameters(
    base: ModelParameters, dose_gy: float, treatment_day: int
) -> ModelParameters:
    return replace(base, dose_gy=dose_gy, treatment_day=treatment_day)


def residual_vector(
    log_values: np.ndarray,
    dataset: FittingDataset,
    base_params: ModelParameters,
    fit_specs: Sequence[FitParameterSpec],
) -> np.ndarray:
    """Return concatenated standardized log-volume residuals."""
    p = params_from_log_vector(base_params, log_values, fit_specs)
    residuals: list[np.ndarray] = []
    n_total = sum(len(arm.observed_volume) for arm in dataset.arms)
    for arm in dataset.arms:
        prediction = simulate(
            _arm_parameters(p, arm.dose_gy, arm.treatment_day), arm.coverage
        ).total_volume[arm.absolute_days]
        if np.any(prediction <= 0.0) or not np.all(np.isfinite(prediction)):
            return np.full(n_total, 1e6)
        residuals.append(
            (np.log(prediction) - np.log(arm.observed_volume)) / arm.log_sigma
        )
    return np.concatenate(residuals)


def chi_square(
    log_values: np.ndarray,
    dataset: FittingDataset,
    base_params: ModelParameters,
    fit_specs: Sequence[FitParameterSpec],
) -> float:
    """Return the residual sum of squares in log-volume space."""
    residual = residual_vector(log_values, dataset, base_params, fit_specs)
    return float(np.dot(residual, residual))


def fit_dataset(
    dataset: FittingDataset,
    *,
    base_params: ModelParameters,
    fit_specs: Sequence[FitParameterSpec],
    seed: int,
    de_maxiter: int = 80,
    de_popsize: int = 12,
    local_max_nfev: int = 500,
) -> FitResult:
    """Fit explicit parameters with DE global search then bounded least squares."""
    specs = tuple(fit_specs)
    bounds = log_bounds(specs)
    global_result = differential_evolution(
        lambda u: chi_square(u, dataset, base_params, specs),
        bounds=bounds,
        seed=seed,
        maxiter=de_maxiter,
        popsize=de_popsize,
        polish=False,
        updating="deferred",
        workers=1,
    )
    local_result = least_squares(
        lambda u: residual_vector(u, dataset, base_params, specs),
        x0=global_result.x,
        bounds=(bounds[:, 0], bounds[:, 1]),
        method="trf",
        max_nfev=local_max_nfev,
        x_scale="jac",
    )
    return FitResult(
        params_from_log_vector(base_params, local_result.x, specs),
        specs,
        local_result.x,
        float(global_result.fun),
        float(np.dot(local_result.fun, local_result.fun)),
        local_result.fun,
        local_result.jac,
    )


def parameter_recovery_table(
    fit_result: FitResult,
    true_params: ModelParameters,
    fit_specs: Sequence[FitParameterSpec] | None = None,
) -> pd.DataFrame:
    """Summarize natural-scale recovery for selected fitted parameters."""
    specs = fit_result.fit_specs if fit_specs is None else tuple(fit_specs)
    rows = []
    for spec in specs:
        truth = float(getattr(true_params, spec.name))
        fitted = float(getattr(fit_result.params, spec.name))
        rows.append(
            {
                "parameter": spec.name,
                "true": truth,
                "fitted": fitted,
                "relative_error_pct": 100.0 * (fitted - truth) / truth,
                "abs_relative_error_pct": 100.0 * abs(fitted - truth) / truth,
            }
        )
    return pd.DataFrame(rows)
