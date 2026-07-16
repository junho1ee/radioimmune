from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from radioimmune_modeling import (
    STANDARD_THREE_ARMS,
    ModelParameters,
    generate_synthetic_dataset,
    simulate,
)


ARM_COLORS = {
    "No radiation": "tab:blue",
    "10 Gy, 50%": "0.50",
    "10 Gy, 100%": "black",
}


def main(output_path: Path | None = None) -> None:
    params = ModelParameters(
        n_days=901,
        n_bins=100,
    )
    results = {arm.name: simulate(params, arm.coverage) for arm in STANDARD_THREE_ARMS}
    relative_days = np.arange(params.n_days) - params.treatment_day

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(12, 12),
        constrained_layout=True,
    )
    for arm in STANDARD_THREE_ARMS:
        result = results[arm.name]
        color = ARM_COLORS[arm.name]
        axes[0].plot(
            relative_days,
            result.immune_effect,
            color=color,
            linewidth=2,
            label=arm.name,
        )
        axes[1].plot(
            relative_days,
            result.total_volume,
            color=color,
            linewidth=2,
            label=arm.name,
        )

    axes[0].axhline(
        params.mu,
        color="tab:red",
        linestyle=":",
        linewidth=1.5,
        label=r"$\mu$",
    )
    axes[0].set(
        xlim=(0, 30),
        ylim=(0, 1),
        xlabel="Days from treatment",
        ylabel=r"Immune effect $Z_n$",
        title="Latent immune trajectories",
    )
    axes[1].set(
        xlim=(0, 30),
        ylim=(0, 2),
        xlabel="Days from treatment",
        ylabel=r"Total tumor volume $T_n+D_n$ [cc]",
        title="Noise-free tumor trajectories",
    )
    axes[0].legend()
    axes[1].legend()

    if output_path is None:
        output_path = Path(__file__).resolve().parents[1] / "fig4.png"
    fig.savefig(output_path)
    plt.close(fig)

    observation_days = np.arange(0, 31, 3)
    dataset = generate_synthetic_dataset(
        params,
        relative_days=observation_days,
        log_sigma=0.06,
        seed=29,
        arms=STANDARD_THREE_ARMS,
    )
    print(f"Saved {output_path}")
    print(f"Generated {len(dataset.arms)} synthetic treatment arms")


if __name__ == "__main__":
    main()
