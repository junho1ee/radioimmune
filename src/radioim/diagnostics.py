"""Identifiability diagnostics for fitted radio-immune models."""
from typing import Iterable, Sequence

import numpy as np
import pandas as pd
from scipy.optimize import least_squares

from .data import (
    STANDARD_THREE_ARMS,
    TreatmentArm,
    FittingDataset,
    generate_synthetic_dataset,
)
from .model import ModelParameters
from .optimize import (
    DEFAULT_VOLUME_FIT_SPECS,
    FitParameterSpec,
    FitResult,
    fit_dataset,
    log_bounds,
    residual_vector,
)


def local_uncertainty_summary(
    fit_result: FitResult, fit_specs: Sequence[FitParameterSpec] | None = None
) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray]:
    """Estimate log-space uncertainty, correlations, and Jacobian singular values."""
    specs = fit_result.fit_specs if fit_specs is None else tuple(fit_specs)
    if len(specs) != fit_result.jacobian.shape[1]:
        raise ValueError("fit_specs must match the fitted Jacobian columns")
    jac = fit_result.jacobian
    n_data, n_params = jac.shape
    residual_variance = fit_result.local_cost / max(n_data - n_params, 1)
    covariance_log = residual_variance * np.linalg.pinv(jac.T @ jac)
    standard_log = np.sqrt(np.clip(np.diag(covariance_log), 0.0, None))
    denominator = np.outer(standard_log, standard_log)
    correlation = np.divide(
        covariance_log,
        denominator,
        out=np.zeros_like(covariance_log),
        where=denominator > 0.0,
    )
    summary = pd.DataFrame(
        {
            "parameter": [spec.name for spec in specs],
            "estimate": np.exp(fit_result.log_values),
            "approx_log_se": standard_log,
            "approx_relative_se": standard_log,
        }
    )
    names = [spec.name for spec in specs]
    return (
        summary,
        pd.DataFrame(correlation, index=names, columns=names),
        np.linalg.svd(jac, compute_uv=False),
    )


def profile_parameter(
    parameter_index: int,
    *,
    dataset: FittingDataset,
    base_params: ModelParameters,
    fit_result: FitResult,
    fit_specs: Sequence[FitParameterSpec],
    relative_factors: np.ndarray,
    local_max_nfev: int = 250,
) -> pd.DataFrame:
    """Profile a fitted parameter while re-optimizing all remaining parameters."""
    specs = tuple(fit_specs)
    if not 0 <= parameter_index < len(specs) or len(specs) != len(
        fit_result.log_values
    ):
        raise ValueError("parameter_index and fit_specs must match fit_result")
    factors = np.asarray(relative_factors, dtype=float)
    if factors.ndim != 1 or len(factors) == 0 or np.any(factors <= 0.0):
        raise ValueError("relative_factors must be a non-empty positive vector")
    bounds = log_bounds(specs)
    best_log = fit_result.log_values.copy()
    free_indices = np.array(
        [index for index in range(len(specs)) if index != parameter_index]
    )
    best_value = np.exp(best_log[parameter_index])
    natural_values = np.unique(
        np.clip(
            best_value * factors,
            np.exp(bounds[parameter_index, 0]),
            np.exp(bounds[parameter_index, 1]),
        )
    )
    rows = []
    free_start = best_log[free_indices]
    for natural_value in natural_values:
        fixed_log = np.log(natural_value)

        def reduced_residual(free_values: np.ndarray) -> np.ndarray:
            candidate = best_log.copy()
            candidate[parameter_index] = fixed_log
            candidate[free_indices] = free_values
            return residual_vector(candidate, dataset, base_params, specs)

        local = least_squares(
            reduced_residual,
            x0=free_start,
            bounds=(bounds[free_indices, 0], bounds[free_indices, 1]),
            method="trf",
            max_nfev=local_max_nfev,
            x_scale="jac",
        )
        free_start = local.x
        cost = float(np.dot(local.fun, local.fun))
        rows.append(
            {
                "parameter": specs[parameter_index].name,
                "value": natural_value,
                "relative_to_hat": natural_value / best_value,
                "profile_cost": cost,
                "delta_chi2": cost - fit_result.local_cost,
            }
        )
    return pd.DataFrame(rows)


def repeated_noise_recovery(
    seeds: Iterable[int],
    *,
    true_params: ModelParameters,
    relative_days: np.ndarray,
    log_sigma: float | np.ndarray,
    arms: Sequence[TreatmentArm] = STANDARD_THREE_ARMS,
    fit_specs: Sequence[FitParameterSpec] = DEFAULT_VOLUME_FIT_SPECS,
    optimizer_seed_offset: int = 1000,
    de_maxiter: int = 35,
    de_popsize: int = 8,
    local_max_nfev: int = 300,
) -> pd.DataFrame:
    """Repeat synthetic generation and fitting across independent noise seeds."""
    specs = tuple(fit_specs)
    rows = []
    for repeat_index, noise_seed in enumerate(seeds):
        seed = int(noise_seed)
        dataset = generate_synthetic_dataset(
            true_params,
            relative_days=relative_days,
            log_sigma=log_sigma,
            seed=seed,
            arms=arms,
        )
        fit = fit_dataset(
            dataset,
            base_params=true_params,
            fit_specs=specs,
            seed=optimizer_seed_offset + seed,
            de_maxiter=de_maxiter,
            de_popsize=de_popsize,
            local_max_nfev=local_max_nfev,
        )
        for spec in specs:
            truth = float(getattr(true_params, spec.name))
            fitted = float(getattr(fit.params, spec.name))
            rows.append(
                {
                    "repeat": repeat_index,
                    "noise_seed": seed,
                    "parameter": spec.name,
                    "true": truth,
                    "fitted": fitted,
                    "relative_error_pct": 100.0 * (fitted - truth) / truth,
                    "abs_relative_error_pct": 100.0 * abs(fitted - truth) / truth,
                    "cost": fit.local_cost,
                }
            )
    return pd.DataFrame(rows)
