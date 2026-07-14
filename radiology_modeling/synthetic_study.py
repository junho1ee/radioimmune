from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import emcee
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import least_squares

from .fit import (
    DEFAULT_FIT_SPECS,
    FitParameterSpec,
    FitResult,
    SyntheticDataset,
    _residual_vector,
    _vector_to_params,
    fit_synthetic_dataset,
    generate_synthetic_dataset,
)
from .model import ModelParameters, SimulationResult, simulate_experiment


@dataclass(frozen=True)
class PosteriorSummary:
    parameter: str
    mean: float
    std: float
    q05: float
    q50: float
    q95: float


def _spec_bounds(specs: tuple[FitParameterSpec, ...]) -> np.ndarray:
    return np.asarray([(spec.lower, spec.upper) for spec in specs], dtype=float)


def _fit_vector(fit_result: FitResult) -> np.ndarray:
    return np.asarray([fit_result.fitted_values[spec.name] for spec in fit_result.parameter_specs], dtype=float)


def _series_prediction_frame(dataset: SyntheticDataset, predictions: dict[str, SimulationResult]) -> pd.DataFrame:
    rows: list[dict[str, float | str]] = []
    for series in dataset.series:
        prediction = predictions[series.name]
        fitted = prediction.total_volume[series.absolute_days]
        for idx, day in enumerate(series.relative_days):
            rows.append(
                {
                    "series": series.name,
                    "coverage": series.coverage,
                    "relative_day": int(day),
                    "clean_volume": float(series.clean_volume[idx]),
                    "observed_volume": float(series.observed_volume[idx]),
                    "fitted_volume": float(fitted[idx]),
                }
            )
    return pd.DataFrame(rows)


def _param_table_frame(fit_result: FitResult) -> pd.DataFrame:
    rows = []
    for spec in fit_result.parameter_specs:
        truth = fit_result.true_values[spec.name]
        fitted = fit_result.fitted_values[spec.name]
        rows.append(
            {
                "parameter": spec.name,
                "true": truth,
                "fitted": fitted,
                "rel_error_pct": 100.0 * (fitted - truth) / truth,
            }
        )
    return pd.DataFrame(rows)


def compute_local_sensitivity(
    params: ModelParameters,
    *,
    fit_specs: tuple[FitParameterSpec, ...] = DEFAULT_FIT_SPECS,
    coverages: tuple[float, ...] = (0.5, 1.0),
    relative_days: np.ndarray | None = None,
    num_days: int = 181,
    n_voxels: int = 100,
    rel_step: float = 0.05,
) -> pd.DataFrame:
    if relative_days is None:
        relative_days = np.arange(0, 31, 3)
    relative_days = np.asarray(relative_days, dtype=int)
    rows: list[dict[str, float | str]] = []
    bounds = _spec_bounds(fit_specs)
    for spec_index, spec in enumerate(fit_specs):
        center = float(getattr(params, spec.name))
        lower = max(bounds[spec_index, 0], center * (1.0 - rel_step))
        upper = min(bounds[spec_index, 1], center * (1.0 + rel_step))
        if lower <= 0.0 or upper <= 0.0 or lower == upper:
            continue
        plus_params = _vector_to_params(params, (spec,), np.asarray([upper], dtype=float))
        minus_params = _vector_to_params(params, (spec,), np.asarray([lower], dtype=float))
        log_delta = np.log(upper) - np.log(lower)
        for coverage in coverages:
            plus = simulate_experiment(plus_params, coverage=coverage, num_days=num_days, n_voxels=n_voxels)
            minus = simulate_experiment(minus_params, coverage=coverage, num_days=num_days, n_voxels=n_voxels)
            for relative_day in relative_days:
                absolute_day = params.treatment_day + int(relative_day)
                sensitivity = (
                    np.log(plus.total_volume[absolute_day] + 1e-12) - np.log(minus.total_volume[absolute_day] + 1e-12)
                ) / log_delta
                rows.append(
                    {
                        "parameter": spec.name,
                        "coverage": coverage,
                        "relative_day": int(relative_day),
                        "log_sensitivity": float(sensitivity),
                        "abs_log_sensitivity": float(abs(sensitivity)),
                    }
                )
    return pd.DataFrame(rows)


def _profile_cost(
    values: np.ndarray,
    dataset: SyntheticDataset,
    base_params: ModelParameters,
    specs: tuple[FitParameterSpec, ...],
    num_days: int,
    n_voxels: int,
    prior_log_sigma: float | None,
) -> float:
    residual = _residual_vector(values, dataset, base_params, specs, num_days, n_voxels, prior_log_sigma)
    return float(np.sum(residual**2))


def compute_profile_likelihood(
    dataset: SyntheticDataset,
    fit_result: FitResult,
    *,
    base_params: ModelParameters | None = None,
    num_days: int = 181,
    n_voxels: int = 100,
    prior_log_sigma: float | None = 0.35,
    n_grid: int = 11,
    span_ratio: float = 0.35,
) -> pd.DataFrame:
    if base_params is None:
        base_params = ModelParameters()
    specs = fit_result.parameter_specs
    bounds = _spec_bounds(specs)
    theta_hat = _fit_vector(fit_result)
    rows: list[dict[str, float]] = []
    for fixed_index, spec in enumerate(specs):
        center = theta_hat[fixed_index]
        lower = max(bounds[fixed_index, 0], center * (1.0 - span_ratio))
        upper = min(bounds[fixed_index, 1], center * (1.0 + span_ratio))
        grid = np.linspace(lower, upper, n_grid)
        free_indices = [idx for idx in range(len(specs)) if idx != fixed_index]
        free_bounds = bounds[free_indices, :]
        free_start = theta_hat[free_indices]
        for fixed_value in grid:
            def residual_free(free_values: np.ndarray) -> np.ndarray:
                full = theta_hat.copy()
                full[fixed_index] = fixed_value
                full[free_indices] = free_values
                return _residual_vector(full, dataset, base_params, specs, num_days, n_voxels, prior_log_sigma)

            result = least_squares(
                residual_free,
                x0=free_start,
                bounds=free_bounds.T,
                method="trf",
                max_nfev=120,
            )
            free_start = result.x
            full = theta_hat.copy()
            full[fixed_index] = fixed_value
            full[free_indices] = result.x
            cost = float(np.sum(result.fun**2))
            rows.append(
                {
                    "parameter": spec.name,
                    "fixed_value": float(fixed_value),
                    "cost": cost,
                    "delta_cost": cost - fit_result.local_cost,
                    **{f"fit_{specs[idx].name}": float(full[idx]) for idx in range(len(specs))},
                }
            )
    return pd.DataFrame(rows)


def sample_posterior(
    dataset: SyntheticDataset,
    fit_result: FitResult,
    *,
    base_params: ModelParameters | None = None,
    num_days: int = 181,
    n_voxels: int = 100,
    prior_log_sigma: float | None = 0.35,
    n_walkers: int = 24,
    n_steps: int = 900,
    burn_in: int = 300,
    thin: int = 4,
    seed: int = 17,
) -> tuple[np.ndarray, emcee.EnsembleSampler]:
    if base_params is None:
        base_params = ModelParameters()
    specs = fit_result.parameter_specs
    bounds = _spec_bounds(specs)
    ndim = len(specs)
    center = _fit_vector(fit_result)
    rng = np.random.default_rng(seed)
    initial = np.tile(center, (n_walkers, 1)) * np.exp(rng.normal(0.0, 0.02, size=(n_walkers, ndim)))
    initial = np.clip(initial, bounds[:, 0] + 1e-8, bounds[:, 1] - 1e-8)

    def log_prob(values: np.ndarray) -> float:
        if np.any(values <= bounds[:, 0]) or np.any(values >= bounds[:, 1]):
            return -np.inf
        residual = _residual_vector(values, dataset, base_params, specs, num_days, n_voxels, prior_log_sigma)
        if not np.all(np.isfinite(residual)):
            return -np.inf
        return float(-0.5 * np.sum(residual**2))

    sampler = emcee.EnsembleSampler(n_walkers, ndim, log_prob)
    sampler.run_mcmc(initial, n_steps, progress=False)
    flat_samples = sampler.get_chain(discard=burn_in, thin=thin, flat=True)
    return flat_samples, sampler


def summarize_posterior(samples: np.ndarray, fit_specs: tuple[FitParameterSpec, ...]) -> pd.DataFrame:
    rows = []
    for index, spec in enumerate(fit_specs):
        values = samples[:, index]
        rows.append(
            PosteriorSummary(
                parameter=spec.name,
                mean=float(np.mean(values)),
                std=float(np.std(values, ddof=1)),
                q05=float(np.quantile(values, 0.05)),
                q50=float(np.quantile(values, 0.50)),
                q95=float(np.quantile(values, 0.95)),
            ).__dict__
        )
    return pd.DataFrame(rows)


def _plot_fit(output_path: Path, dataset: SyntheticDataset, fit_result: FitResult) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), constrained_layout=True)
    for series in dataset.series:
        prediction = fit_result.predictions[series.name]
        color = "tab:gray" if np.isclose(series.coverage, 0.5) else "black"
        label = "50%" if np.isclose(series.coverage, 0.5) else "100%"
        axes[0].plot(prediction.days - dataset.true_params.treatment_day, prediction.immune_effect, color=color, linewidth=2, label=label)
        axes[1].scatter(series.relative_days, series.observed_volume, color=color, s=28, label=f"{label} noisy")
        axes[1].plot(series.relative_days, series.clean_volume, color=color, linestyle=":", linewidth=1.5, label=f"{label} true")
        axes[1].plot(series.relative_days, prediction.total_volume[series.absolute_days], color=color, linewidth=2, label=f"{label} fit")
    axes[0].axhline(dataset.true_params.mu, color="tab:red", linestyle="--", linewidth=1.5, label="mu")
    axes[0].set_title("Immune effect")
    axes[0].set_xlabel("Days from treatment")
    axes[0].set_ylabel("Immune effect")
    axes[0].set_xlim(0, 40)
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(fontsize=8)
    axes[1].set_title("Total volume fit")
    axes[1].set_xlabel("Days from treatment")
    axes[1].set_ylabel("Tumor volume (T + D) [cc]")
    axes[1].set_xlim(0, 30)
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(fontsize=8, ncol=2)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def _plot_sensitivity(output_path: Path, sensitivity_df: pd.DataFrame) -> None:
    coverages = sorted(sensitivity_df["coverage"].unique())
    fig, axes = plt.subplots(len(coverages), 1, figsize=(10, 3.5 * len(coverages)), constrained_layout=True, sharex=True)
    if len(coverages) == 1:
        axes = [axes]
    for axis, coverage in zip(axes, coverages, strict=True):
        subset = sensitivity_df[sensitivity_df["coverage"] == coverage]
        for parameter, group in subset.groupby("parameter"):
            axis.plot(group["relative_day"], group["log_sensitivity"], marker="o", linewidth=1.8, label=parameter)
        axis.axhline(0.0, color="black", linewidth=0.8)
        axis.set_title(f"Local log-sensitivity at coverage={coverage:.2f}")
        axis.set_ylabel("d log V / d log theta")
        axis.grid(True, alpha=0.3)
        axis.legend(fontsize=8, ncol=2)
    axes[-1].set_xlabel("Days from treatment")
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def _plot_profile(output_path: Path, profile_df: pd.DataFrame) -> None:
    parameters = profile_df["parameter"].unique().tolist()
    ncols = 2
    nrows = int(np.ceil(len(parameters) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(11, 3.8 * nrows), constrained_layout=True)
    axes = np.atleast_1d(axes).ravel()
    for axis, parameter in zip(axes, parameters, strict=True):
        subset = profile_df[profile_df["parameter"] == parameter]
        axis.plot(subset["fixed_value"], subset["delta_cost"], marker="o", linewidth=1.8)
        axis.axhline(0.0, color="black", linewidth=0.8)
        axis.set_title(f"Profile objective: {parameter}")
        axis.set_xlabel(parameter)
        axis.set_ylabel("delta cost")
        axis.grid(True, alpha=0.3)
    for axis in axes[len(parameters):]:
        axis.axis("off")
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def _plot_posterior(output_path: Path, sampler: emcee.EnsembleSampler, samples: np.ndarray, fit_specs: tuple[FitParameterSpec, ...]) -> None:
    ndim = len(fit_specs)
    fig, axes = plt.subplots(ndim, 2, figsize=(12, 2.6 * ndim), constrained_layout=True)
    chain = sampler.get_chain()
    for index, spec in enumerate(fit_specs):
        axes[index, 0].plot(chain[:, :, index], alpha=0.25, linewidth=0.6)
        axes[index, 0].set_title(f"Trace: {spec.name}")
        axes[index, 0].set_ylabel(spec.name)
        axes[index, 0].grid(True, alpha=0.2)
        axes[index, 1].hist(samples[:, index], bins=30, color="tab:blue", alpha=0.8)
        axes[index, 1].set_title(f"Posterior: {spec.name}")
        axes[index, 1].grid(True, alpha=0.2)
    axes[-1, 0].set_xlabel("step")
    axes[-1, 1].set_xlabel("value")
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def _plot_posterior_predictive(
    output_path: Path,
    dataset: SyntheticDataset,
    samples: np.ndarray,
    fit_specs: tuple[FitParameterSpec, ...],
    base_params: ModelParameters,
    *,
    num_days: int = 181,
    n_voxels: int = 100,
    n_draws: int = 100,
    seed: int = 23,
) -> None:
    rng = np.random.default_rng(seed)
    draw_indices = rng.choice(samples.shape[0], size=min(n_draws, samples.shape[0]), replace=False)
    fig, axes = plt.subplots(1, len(dataset.series), figsize=(6 * len(dataset.series), 4), constrained_layout=True, sharey=True)
    if len(dataset.series) == 1:
        axes = [axes]
    for axis, series in zip(axes, dataset.series, strict=True):
        color = "tab:gray" if np.isclose(series.coverage, 0.5) else "black"
        for draw_index in draw_indices:
            params = _vector_to_params(base_params, fit_specs, samples[draw_index])
            prediction = simulate_experiment(params, coverage=series.coverage, num_days=num_days, n_voxels=n_voxels)
            axis.plot(series.relative_days, prediction.total_volume[series.absolute_days], color=color, alpha=0.08, linewidth=1)
        axis.scatter(series.relative_days, series.observed_volume, color=color, s=28, label="noisy obs")
        axis.plot(series.relative_days, series.clean_volume, color="tab:green", linestyle=":", linewidth=1.8, label="true")
        axis.set_title(f"Posterior predictive: coverage={series.coverage:.2f}")
        axis.set_xlabel("Days from treatment")
        axis.set_ylabel("Tumor volume (T + D) [cc]")
        axis.grid(True, alpha=0.3)
        axis.legend(fontsize=8)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def run_synthetic_study(output_dir: str = "artifacts/synthetic_study") -> Path:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    true_params = ModelParameters(
        rho=0.5,
        psi=300.0,
        omega=0.135,
        k=1.1,
        gamma=0.0,
        initial_volume=0.03,
        treatment_day=10,
        dose_gy=10.0,
    )
    base_params = ModelParameters()
    dataset = generate_synthetic_dataset(true_params, noise_sigma=0.08, seed=7)
    fit_result = fit_synthetic_dataset(dataset, base_params=base_params, fit_specs=DEFAULT_FIT_SPECS, seed=11)
    sensitivity_df = compute_local_sensitivity(fit_result.params, fit_specs=DEFAULT_FIT_SPECS)
    profile_df = compute_profile_likelihood(dataset, fit_result, base_params=base_params)
    posterior_samples, sampler = sample_posterior(dataset, fit_result, base_params=base_params)
    posterior_df = summarize_posterior(posterior_samples, DEFAULT_FIT_SPECS)

    _param_table_frame(fit_result).to_csv(output_path / "parameter_recovery.csv", index=False)
    _series_prediction_frame(dataset, fit_result.predictions).to_csv(output_path / "fit_predictions.csv", index=False)
    sensitivity_df.to_csv(output_path / "sensitivity.csv", index=False)
    profile_df.to_csv(output_path / "profile_likelihood.csv", index=False)
    posterior_df.to_csv(output_path / "posterior_summary.csv", index=False)

    _plot_fit(output_path / "fit_overview.png", dataset, fit_result)
    _plot_sensitivity(output_path / "sensitivity.png", sensitivity_df)
    _plot_profile(output_path / "profile_likelihood.png", profile_df)
    _plot_posterior(output_path / "posterior.png", sampler, posterior_samples, DEFAULT_FIT_SPECS)
    _plot_posterior_predictive(
        output_path / "posterior_predictive.png",
        dataset,
        posterior_samples,
        DEFAULT_FIT_SPECS,
        base_params,
    )

    acceptance_fraction = float(np.mean(sampler.acceptance_fraction))
    try:
        autocorr_time = sampler.get_autocorr_time(tol=0)
        autocorr_summary = ",".join(f"{value:.2f}" for value in autocorr_time)
    except Exception:
        autocorr_summary = "unavailable"

    summary_lines = [
        f"global_cost,{fit_result.global_cost:.6f}",
        f"local_cost,{fit_result.local_cost:.6f}",
        f"residual_norm,{fit_result.residual_norm:.6f}",
        f"posterior_samples,{posterior_samples.shape[0]}",
        f"mean_acceptance_fraction,{acceptance_fraction:.6f}",
        f"autocorr_time,{autocorr_summary}",
    ]
    (output_path / "study_summary.csv").write_text("\n".join(summary_lines) + "\n", encoding="utf-8")

    print("Synthetic study complete")
    print("Output directory:", output_path)
    print("Acceptance fraction:", f"{acceptance_fraction:.3f}")
    print("Posterior samples:", posterior_samples.shape[0])
    return output_path


if __name__ == "__main__":
    run_synthetic_study()
