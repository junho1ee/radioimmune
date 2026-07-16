from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from .fit import DEFAULT_FIT_SPECS, fit_synthetic_dataset, generate_synthetic_dataset
from .model import ModelParameters


def _format_param_table(true_params: dict[str, float], fitted_params: dict[str, float]) -> str:
    rows = ["parameter,true,fitted,rel_error_pct"]
    for name in true_params:
        truth = true_params[name]
        fitted = fitted_params[name]
        rel_error = 100.0 * (fitted - truth) / truth
        rows.append(f"{name},{truth:.6g},{fitted:.6g},{rel_error:.2f}")
    return "\n".join(rows)


def run_poc(output_dir: str = "artifacts") -> Path:
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
    dataset = generate_synthetic_dataset(true_params, noise_sigma=0.08, seed=7)
    fit_result = fit_synthetic_dataset(dataset, base_params=ModelParameters(), fit_specs=DEFAULT_FIT_SPECS, seed=11)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), constrained_layout=True)

    for series in dataset.series:
        prediction = fit_result.predictions[series.name]
        color = "tab:gray" if np.isclose(series.coverage, 0.5) else "black"
        label_prefix = "50%" if np.isclose(series.coverage, 0.5) else "100%"
        axes[0].plot(
            prediction.days - true_params.treatment_day,
            prediction.immune_effect,
            color=color,
            linewidth=2,
            label=f"{label_prefix} fitted",
        )
        axes[1].scatter(
            series.relative_days,
            series.observed_volume,
            color=color,
            s=28,
            label=f"{label_prefix} noisy obs",
        )
        axes[1].plot(
            series.relative_days,
            series.clean_volume,
            color=color,
            linestyle=":",
            linewidth=1.5,
            label=f"{label_prefix} true",
        )
        axes[1].plot(
            series.relative_days,
            prediction.total_volume[series.absolute_days],
            color=color,
            linewidth=2,
            label=f"{label_prefix} fitted",
        )

    axes[0].axhline(true_params.mu, color="tab:red", linestyle="--", linewidth=1.5, label="mu")
    axes[0].set_title("Synthetic immune effect")
    axes[0].set_xlabel("Days from treatment")
    axes[0].set_ylabel("Immune effect")
    axes[0].set_xlim(0, 40)
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(fontsize=8)

    axes[1].set_title("Synthetic total volume fit")
    axes[1].set_xlabel("Days from treatment")
    axes[1].set_ylabel("Tumor volume (T + D) [cc]")
    axes[1].set_xlim(0, 30)
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(fontsize=8, ncol=2)

    figure_path = output_path / "poc_synthetic_fit.png"
    fig.savefig(figure_path, dpi=180)
    plt.close(fig)

    table = _format_param_table(fit_result.true_values, fit_result.fitted_values)
    metrics = [
        f"global_cost={fit_result.global_cost:.6f}",
        f"local_cost={fit_result.local_cost:.6f}",
        f"residual_norm={fit_result.residual_norm:.6f}",
    ]
    summary_path = output_path / "poc_fit_summary.csv"
    summary_path.write_text("\n".join(metrics) + "\n" + table + "\n", encoding="utf-8")

    print("Synthetic POC complete")
    print("Saved figure:", figure_path)
    print("Saved summary:", summary_path)
    print(table)
    return figure_path


if __name__ == "__main__":
    run_poc()
