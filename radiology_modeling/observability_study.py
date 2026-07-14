from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import emcee
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import differential_evolution, least_squares

from .fit import DEFAULT_FIT_SPECS, FitParameterSpec
from .model import ModelParameters, get_observable_series, replace_params, simulate_experiment


@dataclass(frozen=True)
class ObservableMeasurement:
    name: str
    coverage: float
    dose_gy: float
    treatment_day: int
    observable: str
    absolute_days: np.ndarray
    relative_days: np.ndarray
    clean_values: np.ndarray
    observed_values: np.ndarray
    noise_sigma: float


@dataclass(frozen=True)
class ObservabilityDataset:
    measurements: tuple[ObservableMeasurement, ...]
    true_params: ModelParameters
    seed: int


@dataclass(frozen=True)
class ObservabilityFitResult:
    fitted_values: dict[str, float]
    true_values: dict[str, float]
    local_cost: float
    residual_norm: float


@dataclass(frozen=True)
class ArmConfig:
    dose_gy: float
    coverage: float
    treatment_day: int | None = None


@dataclass(frozen=True)
class ScenarioConfig:
    name: str
    observables: tuple[str, ...]
    coverages: tuple[float, ...] = ()
    arms: tuple[ArmConfig, ...] = ()
    noise_sigma: float = 0.08
    relative_days: tuple[int, ...] = tuple(range(0, 31, 3))


DEFAULT_SCENARIOS: tuple[ScenarioConfig, ...] = (
    ScenarioConfig("total_2arm", ("total_volume",), (0.5, 1.0)),
    ScenarioConfig("total_plus_immune_2arm", ("total_volume", "immune_effect"), (0.5, 1.0)),
    ScenarioConfig("total_plus_lymph_2arm", ("total_volume", "lymphocyte_volume"), (0.5, 1.0)),
    ScenarioConfig("total_plus_trigger_2arm", ("total_volume", "triggering_density"), (0.5, 1.0)),
    ScenarioConfig("total_4arm", ("total_volume",), (0.25, 0.5, 0.75, 1.0)),
)

DEFAULT_DOSE_GRID_SCENARIOS: tuple[ScenarioConfig, ...] = (
    ScenarioConfig("dosegrid_total_2arm_d10", ("total_volume",), arms=(ArmConfig(10.0, 0.5), ArmConfig(10.0, 1.0))),
    ScenarioConfig("dosegrid_total_4arm_d10", ("total_volume",), arms=(ArmConfig(10.0, 0.25), ArmConfig(10.0, 0.5), ArmConfig(10.0, 0.75), ArmConfig(10.0, 1.0))),
    ScenarioConfig("dosegrid_total_fulldose_ladder", ("total_volume",), arms=(ArmConfig(4.0, 1.0), ArmConfig(10.0, 1.0), ArmConfig(16.0, 1.0))),
    ScenarioConfig("dosegrid_total_partialdose_ladder", ("total_volume",), arms=(ArmConfig(4.0, 0.5), ArmConfig(10.0, 0.5), ArmConfig(16.0, 0.5))),
    ScenarioConfig("dosegrid_total_dose_coverage_grid", ("total_volume",), arms=(ArmConfig(4.0, 0.5), ArmConfig(4.0, 1.0), ArmConfig(10.0, 0.5), ArmConfig(10.0, 1.0), ArmConfig(16.0, 0.5), ArmConfig(16.0, 1.0))),
    ScenarioConfig("dosegrid_total_plus_immune_grid", ("total_volume", "immune_effect"), arms=(ArmConfig(4.0, 0.5), ArmConfig(4.0, 1.0), ArmConfig(10.0, 0.5), ArmConfig(10.0, 1.0), ArmConfig(16.0, 0.5), ArmConfig(16.0, 1.0))),
)

DEFAULT_START_VOLUME_SCENARIOS: tuple[ScenarioConfig, ...] = (
    ScenarioConfig("startvol_partial_pair_d10", ("total_volume",), arms=(ArmConfig(10.0, 0.5, 10), ArmConfig(10.0, 0.5, 15))),
    ScenarioConfig("startvol_full_pair_d10", ("total_volume",), arms=(ArmConfig(10.0, 1.0, 10), ArmConfig(10.0, 1.0, 15))),
    ScenarioConfig("startvol_mixed_4arm_d10", ("total_volume",), arms=(ArmConfig(10.0, 0.5, 10), ArmConfig(10.0, 0.5, 15), ArmConfig(10.0, 1.0, 10), ArmConfig(10.0, 1.0, 15))),
    ScenarioConfig("startvol_mixed_4arm_plus_immune", ("total_volume", "immune_effect"), arms=(ArmConfig(10.0, 0.5, 10), ArmConfig(10.0, 0.5, 15), ArmConfig(10.0, 1.0, 10), ArmConfig(10.0, 1.0, 15))),
)


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


def _scenario_arms(scenario: ScenarioConfig, default_dose_gy: float, default_treatment_day: int) -> tuple[ArmConfig, ...]:
    if scenario.arms:
        return scenario.arms
    return tuple(ArmConfig(default_dose_gy, coverage, default_treatment_day) for coverage in scenario.coverages)

def generate_observability_dataset(
    true_params: ModelParameters,
    scenario: ScenarioConfig,
    *,
    seed: int = 31,
    num_days: int = 181,
    n_voxels: int = 100,
) -> ObservabilityDataset:
    rng = np.random.default_rng(seed)
    measurements: list[ObservableMeasurement] = []
    relative_days = np.asarray(scenario.relative_days, dtype=int)
    for arm in _scenario_arms(scenario, true_params.dose_gy, true_params.treatment_day):
        treatment_day = true_params.treatment_day if arm.treatment_day is None else arm.treatment_day
        arm_params = replace_params(true_params, dose_gy=arm.dose_gy, treatment_day=treatment_day)
        result = simulate_experiment(arm_params, coverage=arm.coverage, num_days=num_days, n_voxels=n_voxels)
        absolute_days = arm_params.treatment_day + relative_days
        for observable in scenario.observables:
            clean = get_observable_series(result, observable)[absolute_days]
            noise = np.exp(rng.normal(0.0, scenario.noise_sigma, size=clean.shape))
            noisy = clean * noise
            measurements.append(
                ObservableMeasurement(
                    name=f"{observable}_dose_{arm.dose_gy:.1f}_coverage_{arm.coverage:.2f}_tx_{treatment_day}",
                    coverage=arm.coverage,
                    dose_gy=arm.dose_gy,
                    treatment_day=treatment_day,
                    observable=observable,
                    absolute_days=absolute_days,
                    relative_days=relative_days,
                    clean_values=clean,
                    observed_values=noisy,
                    noise_sigma=scenario.noise_sigma,
                )
            )
    return ObservabilityDataset(measurements=tuple(measurements), true_params=true_params, seed=seed)


def _residual_vector(
    values: np.ndarray,
    dataset: ObservabilityDataset,
    base_params: ModelParameters,
    specs: tuple[FitParameterSpec, ...],
    num_days: int,
    n_voxels: int,
    prior_log_sigma: float | None,
) -> np.ndarray:
    try:
        params = _vector_to_params(base_params, specs, values)
        residuals = []
        for measurement in dataset.measurements:
            arm_params = replace_params(params, dose_gy=measurement.dose_gy)
            result = simulate_experiment(arm_params, coverage=measurement.coverage, num_days=num_days, n_voxels=n_voxels)
            prediction = get_observable_series(result, measurement.observable)[measurement.absolute_days]
            if not np.all(np.isfinite(prediction)) or np.any(prediction < 0.0):
                return np.full(sum(len(m.observed_values) for m in dataset.measurements), 1e6)
            residuals.append(np.log(prediction + 1e-12) - np.log(measurement.observed_values + 1e-12))
        combined = np.concatenate(residuals)
        priors = _prior_residuals(values, base_params, specs, prior_log_sigma)
        return np.concatenate([combined, priors]) if priors.size else combined
    except Exception:
        return np.full(sum(len(m.observed_values) for m in dataset.measurements), 1e6)


def fit_observability_dataset(
    dataset: ObservabilityDataset,
    *,
    base_params: ModelParameters | None = None,
    fit_specs: tuple[FitParameterSpec, ...] = DEFAULT_FIT_SPECS,
    num_days: int = 181,
    n_voxels: int = 100,
    seed: int = 41,
    prior_log_sigma: float | None = 0.35,
) -> ObservabilityFitResult:
    if base_params is None:
        base_params = ModelParameters()
    bounds = [(spec.lower, spec.upper) for spec in fit_specs]
    objective = lambda x: np.sum(_residual_vector(x, dataset, base_params, fit_specs, num_days, n_voxels, prior_log_sigma) ** 2)
    global_result = differential_evolution(
        objective,
        bounds=bounds,
        seed=seed,
        polish=False,
        updating="deferred",
        workers=1,
        maxiter=60,
        popsize=12,
    )
    local_result = least_squares(
        lambda x: _residual_vector(x, dataset, base_params, fit_specs, num_days, n_voxels, prior_log_sigma),
        x0=global_result.x,
        bounds=np.asarray(bounds, dtype=float).T,
        method="trf",
        max_nfev=220,
    )
    fitted_values = {spec.name: float(value) for spec, value in zip(fit_specs, local_result.x, strict=True)}
    true_values = {spec.name: float(getattr(dataset.true_params, spec.name)) for spec in fit_specs}
    return ObservabilityFitResult(
        fitted_values=fitted_values,
        true_values=true_values,
        local_cost=float(np.sum(local_result.fun**2)),
        residual_norm=float(np.linalg.norm(local_result.fun)),
    )


def sample_observability_posterior(
    dataset: ObservabilityDataset,
    fit_result: ObservabilityFitResult,
    *,
    base_params: ModelParameters | None = None,
    fit_specs: tuple[FitParameterSpec, ...] = DEFAULT_FIT_SPECS,
    num_days: int = 181,
    n_voxels: int = 100,
    prior_log_sigma: float | None = 0.35,
    n_walkers: int = 20,
    n_steps: int = 420,
    burn_in: int = 180,
    thin: int = 3,
    seed: int = 43,
) -> tuple[np.ndarray, emcee.EnsembleSampler]:
    if base_params is None:
        base_params = ModelParameters()
    bounds = np.asarray([(spec.lower, spec.upper) for spec in fit_specs], dtype=float)
    center = np.asarray([fit_result.fitted_values[spec.name] for spec in fit_specs], dtype=float)
    ndim = len(fit_specs)
    rng = np.random.default_rng(seed)
    initial = np.tile(center, (n_walkers, 1)) * np.exp(rng.normal(0.0, 0.025, size=(n_walkers, ndim)))
    initial = np.clip(initial, bounds[:, 0] + 1e-8, bounds[:, 1] - 1e-8)

    def log_prob(values: np.ndarray) -> float:
        if np.any(values <= bounds[:, 0]) or np.any(values >= bounds[:, 1]):
            return -np.inf
        residual = _residual_vector(values, dataset, base_params, fit_specs, num_days, n_voxels, prior_log_sigma)
        if not np.all(np.isfinite(residual)):
            return -np.inf
        return float(-0.5 * np.sum(residual**2))

    sampler = emcee.EnsembleSampler(n_walkers, ndim, log_prob)
    sampler.run_mcmc(initial, n_steps, progress=False)
    samples = sampler.get_chain(discard=burn_in, thin=thin, flat=True)
    return samples, sampler


def _posterior_summary_frame(samples: np.ndarray, fit_specs: tuple[FitParameterSpec, ...], true_params: ModelParameters, scenario_name: str) -> pd.DataFrame:
    rows = []
    for index, spec in enumerate(fit_specs):
        values = samples[:, index]
        truth = float(getattr(true_params, spec.name))
        q05 = float(np.quantile(values, 0.05))
        q95 = float(np.quantile(values, 0.95))
        rows.append(
            {
                "scenario": scenario_name,
                "parameter": spec.name,
                "true": truth,
                "mean": float(np.mean(values)),
                "std": float(np.std(values, ddof=1)),
                "q05": q05,
                "q50": float(np.quantile(values, 0.50)),
                "q95": q95,
                "width90": q95 - q05,
                "rel_width90": (q95 - q05) / truth,
            }
        )
    return pd.DataFrame(rows)


def _recovery_frame(fit_result: ObservabilityFitResult, scenario_name: str) -> pd.DataFrame:
    rows = []
    for parameter, truth in fit_result.true_values.items():
        fitted = fit_result.fitted_values[parameter]
        rows.append(
            {
                "scenario": scenario_name,
                "parameter": parameter,
                "true": truth,
                "fitted": fitted,
                "rel_error_pct": 100.0 * (fitted - truth) / truth,
            }
        )
    return pd.DataFrame(rows)


def _scenario_summary_frame(
    scenario_name: str,
    dataset: ObservabilityDataset,
    fit_result: ObservabilityFitResult,
    sampler: emcee.EnsembleSampler,
    samples: np.ndarray,
) -> pd.DataFrame:
    try:
        autocorr = sampler.get_autocorr_time(tol=0)
        autocorr_mean = float(np.mean(autocorr))
    except Exception:
        autocorr_mean = np.nan
    representative = {(m.dose_gy, m.coverage, m.treatment_day) for m in dataset.measurements}
    return pd.DataFrame(
        [
            {
                "scenario": scenario_name,
                "n_measurements": len(dataset.measurements),
                "n_points": int(sum(len(m.observed_values) for m in dataset.measurements)),
                "observables": ";".join(sorted({m.observable for m in dataset.measurements})),
                "arms": ";".join(f"{dose:.1f}Gy@{coverage:.2f}/tx{treatment_day}" for dose, coverage, treatment_day in sorted(representative)),
                "local_cost": fit_result.local_cost,
                "residual_norm": fit_result.residual_norm,
                "posterior_samples": int(samples.shape[0]),
                "mean_acceptance_fraction": float(np.mean(sampler.acceptance_fraction)),
                "mean_autocorr_time": autocorr_mean,
            }
        ]
    )


def _plot_relative_widths(output_path: Path, posterior_df: pd.DataFrame) -> None:
    pivot = posterior_df.pivot(index="parameter", columns="scenario", values="rel_width90")
    fig, ax = plt.subplots(figsize=(10, 4.8), constrained_layout=True)
    image = ax.imshow(pivot.values, aspect="auto", cmap="viridis")
    ax.set_xticks(np.arange(pivot.shape[1]), labels=pivot.columns, rotation=25, ha="right")
    ax.set_yticks(np.arange(pivot.shape[0]), labels=pivot.index)
    ax.set_title("Relative 90% posterior width by scenario")
    for row in range(pivot.shape[0]):
        for col in range(pivot.shape[1]):
            ax.text(col, row, f"{pivot.values[row, col]:.2f}", ha="center", va="center", color="white", fontsize=8)
    fig.colorbar(image, ax=ax, label="(q95-q05)/true")
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def _plot_recovery(output_path: Path, recovery_df: pd.DataFrame) -> None:
    pivot = recovery_df.pivot(index="parameter", columns="scenario", values="rel_error_pct")
    fig, ax = plt.subplots(figsize=(10, 4.8), constrained_layout=True)
    image = ax.imshow(pivot.values, aspect="auto", cmap="coolwarm", vmin=-25, vmax=25)
    ax.set_xticks(np.arange(pivot.shape[1]), labels=pivot.columns, rotation=25, ha="right")
    ax.set_yticks(np.arange(pivot.shape[0]), labels=pivot.index)
    ax.set_title("Point-estimate relative error by scenario [%]")
    for row in range(pivot.shape[0]):
        for col in range(pivot.shape[1]):
            ax.text(col, row, f"{pivot.values[row, col]:.1f}", ha="center", va="center", color="black", fontsize=8)
    fig.colorbar(image, ax=ax, label="relative error [%]")
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def _plot_measurement_examples(output_path: Path, datasets: dict[str, ObservabilityDataset]) -> None:
    scenario_names = list(datasets)
    fig, axes = plt.subplots(len(scenario_names), 1, figsize=(11, 3.2 * len(scenario_names)), constrained_layout=True)
    if len(scenario_names) == 1:
        axes = [axes]
    for axis, scenario_name in zip(axes, scenario_names, strict=True):
        dataset = datasets[scenario_name]
        for measurement in dataset.measurements:
            axis.scatter(
                measurement.relative_days,
                measurement.observed_values,
                s=18,
                alpha=0.65,
                label=f"{measurement.observable} @ {measurement.dose_gy:.1f}Gy/{measurement.coverage:.2f}/tx{measurement.treatment_day}",
            )
        axis.set_title(scenario_name)
        axis.set_xlabel("Days from treatment")
        axis.set_ylabel("Observed value")
        axis.grid(True, alpha=0.3)
        axis.legend(fontsize=7, ncol=2)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def _run_scenario_collection(
    scenarios: tuple[ScenarioConfig, ...],
    *,
    output_dir: str,
    summary_title: str,
) -> Path:
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
    datasets: dict[str, ObservabilityDataset] = {}
    recovery_frames = []
    posterior_frames = []
    scenario_frames = []

    for scenario_index, scenario in enumerate(scenarios):
        dataset = generate_observability_dataset(true_params, scenario, seed=31 + scenario_index)
        fit_result = fit_observability_dataset(dataset, base_params=base_params, seed=41 + scenario_index)
        samples, sampler = sample_observability_posterior(dataset, fit_result, base_params=base_params, seed=43 + scenario_index)
        datasets[scenario.name] = dataset
        recovery_frames.append(_recovery_frame(fit_result, scenario.name))
        posterior_frames.append(_posterior_summary_frame(samples, DEFAULT_FIT_SPECS, true_params, scenario.name))
        scenario_frames.append(_scenario_summary_frame(scenario.name, dataset, fit_result, sampler, samples))

    recovery_df = pd.concat(recovery_frames, ignore_index=True)
    posterior_df = pd.concat(posterior_frames, ignore_index=True)
    scenario_df = pd.concat(scenario_frames, ignore_index=True)
    ranking_df = (
        posterior_df.groupby("scenario", as_index=False)["rel_width90"]
        .mean()
        .rename(columns={"rel_width90": "mean_rel_width90"})
        .sort_values("mean_rel_width90")
    )

    scenario_df.to_csv(output_path / "scenario_summary.csv", index=False)
    recovery_df.to_csv(output_path / "parameter_recovery.csv", index=False)
    posterior_df.to_csv(output_path / "posterior_widths.csv", index=False)
    ranking_df.to_csv(output_path / "scenario_ranking.csv", index=False)

    _plot_relative_widths(output_path / "posterior_widths.png", posterior_df)
    _plot_recovery(output_path / "recovery.png", recovery_df)
    _plot_measurement_examples(output_path / "measurement_examples.png", datasets)

    best = ranking_df.iloc[0]
    print(summary_title)
    print("Output directory:", output_path)
    print("Best scenario:", best["scenario"], "mean_rel_width90=", f"{best['mean_rel_width90']:.3f}")
    return output_path


def run_observability_comparison(output_dir: str = "artifacts/observability_study") -> Path:
    return _run_scenario_collection(DEFAULT_SCENARIOS, output_dir=output_dir, summary_title="Observability comparison complete")


def run_dose_coverage_grid(output_dir: str = "artifacts/dose_coverage_grid") -> Path:
    return _run_scenario_collection(DEFAULT_DOSE_GRID_SCENARIOS, output_dir=output_dir, summary_title="Dose coverage grid complete")

def run_start_volume_grid(output_dir: str = "artifacts/start_volume_grid") -> Path:
    return _run_scenario_collection(DEFAULT_START_VOLUME_SCENARIOS, output_dir=output_dir, summary_title="Start volume grid complete")


if __name__ == "__main__":
    run_observability_comparison()
