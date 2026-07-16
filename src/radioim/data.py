"""Observation structures and dataset construction."""

from dataclasses import dataclass, replace
from typing import Sequence

import numpy as np
import pandas as pd

from .model import ModelParameters, simulate


@dataclass(frozen=True)
class TreatmentArm:
    """A named treatment condition with optional dose and timing overrides."""

    name: str
    coverage: float
    dose_gy: float | None = None
    treatment_day: int | None = None

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("Treatment arm name must be non-empty")
        if not 0.0 <= self.coverage <= 1.0:
            raise ValueError("coverage must lie in [0, 1]")
        if self.dose_gy is not None and (
            not np.isfinite(self.dose_gy) or self.dose_gy < 0.0
        ):
            raise ValueError("dose_gy must be finite and non-negative")
        if self.treatment_day is not None and (
            self.treatment_day < 0 or int(self.treatment_day) != self.treatment_day
        ):
            raise ValueError("treatment_day must be a non-negative integer")


STANDARD_THREE_ARMS = (
    TreatmentArm("No radiation", 0.0, dose_gy=0.0),
    TreatmentArm("10 Gy, 50%", 0.5, dose_gy=10.0),
    TreatmentArm("10 Gy, 100%", 1.0, dose_gy=10.0),
)


@dataclass(frozen=True)
class ArmObservation:
    """Measured volumes and resolved treatment settings for one arm."""

    name: str
    coverage: float
    dose_gy: float
    treatment_day: int
    relative_days: np.ndarray
    absolute_days: np.ndarray
    observed_volume: np.ndarray
    log_sigma: float | np.ndarray
    clean_volume: np.ndarray | None = None

    def __post_init__(self) -> None:
        if not self.name or not 0.0 <= self.coverage <= 1.0:
            raise ValueError("Arm observation requires a name and coverage in [0, 1]")
        if (
            not np.isfinite(self.dose_gy)
            or self.dose_gy < 0.0
            or self.treatment_day < 0
            or int(self.treatment_day) != self.treatment_day
        ):
            raise ValueError(
                "dose_gy must be finite and non-negative; treatment_day a non-negative integer"
            )
        relative = np.asarray(self.relative_days)
        absolute = np.asarray(self.absolute_days)
        observed = np.asarray(self.observed_volume, dtype=float)
        if relative.ndim != 1 or absolute.ndim != 1 or observed.ndim != 1:
            raise ValueError("Observation arrays must be one-dimensional")
        if not (len(relative) == len(absolute) == len(observed)) or len(observed) == 0:
            raise ValueError("Observation arrays must have the same non-zero length")
        if (
            not np.all(np.isfinite(relative))
            or not np.all(np.isfinite(absolute))
            or np.any(relative != np.floor(relative))
            or np.any(absolute != np.floor(absolute))
        ):
            raise ValueError("Observation days must be finite integers")
        if not np.array_equal(absolute, self.treatment_day + relative):
            raise ValueError("absolute_days must equal treatment_day + relative_days")
        if np.any(observed <= 0.0) or not np.all(np.isfinite(observed)):
            raise ValueError("observed_volume must be positive and finite")
        sigma = np.asarray(self.log_sigma, dtype=float)
        if sigma.ndim > 1 or (sigma.ndim == 1 and sigma.shape != observed.shape):
            raise ValueError("log_sigma must be scalar or match observed_volume")
        if np.any(sigma <= 0.0) or not np.all(np.isfinite(sigma)):
            raise ValueError("log_sigma must be positive and finite")
        if self.clean_volume is not None:
            clean = np.asarray(self.clean_volume, dtype=float)
            if (
                clean.shape != observed.shape
                or np.any(clean <= 0.0)
                or not np.all(np.isfinite(clean))
            ):
                raise ValueError(
                    "clean_volume must be positive, finite, and match observations"
                )


@dataclass(frozen=True)
class FittingDataset:
    """A collection of arms sharing biological model parameters."""

    arms: tuple[ArmObservation, ...]
    true_params: ModelParameters | None
    seed: int | None

    def __post_init__(self) -> None:
        if not self.arms:
            raise ValueError("FittingDataset requires at least one arm")
        if len({arm.name for arm in self.arms}) != len(self.arms):
            raise ValueError("Arm names must be unique")


def _resolved_parameters(base: ModelParameters, arm: TreatmentArm) -> ModelParameters:
    """Apply an arm's optional treatment overrides."""
    resolved = replace(
        base,
        dose_gy=base.dose_gy if arm.dose_gy is None else arm.dose_gy,
        treatment_day=base.treatment_day
        if arm.treatment_day is None
        else arm.treatment_day,
    )
    if not 0 <= resolved.treatment_day < resolved.n_days:
        raise ValueError("treatment_day must lie within the simulation range")
    return resolved


def generate_synthetic_dataset(
    p: ModelParameters,
    *,
    relative_days: np.ndarray,
    log_sigma: float | np.ndarray,
    seed: int,
    arms: Sequence[TreatmentArm] = STANDARD_THREE_ARMS,
) -> FittingDataset:
    """Generate multiplicative-noise observations for explicit treatment arms."""
    days = np.asarray(relative_days)
    if (
        days.ndim != 1
        or len(days) == 0
        or not np.all(np.isfinite(days))
        or np.any(days != np.floor(days))
    ):
        raise ValueError("relative_days must be a non-empty finite integer array")
    sigma = np.asarray(log_sigma, dtype=float)
    if (
        sigma.ndim > 1
        or (sigma.ndim == 1 and sigma.shape != days.shape)
        or np.any(sigma <= 0.0)
        or not np.all(np.isfinite(sigma))
    ):
        raise ValueError("log_sigma must be positive scalar or match relative_days")
    rng = np.random.default_rng(seed)
    observations: list[ArmObservation] = []
    for arm in arms:
        arm_params = _resolved_parameters(p, arm)
        absolute_days = arm_params.treatment_day + days
        if np.any(absolute_days < 0) or np.any(absolute_days >= arm_params.n_days):
            raise ValueError(f"Arm {arm.name!r} contains days outside simulation range")
        clean = simulate(arm_params, arm.coverage).total_volume[absolute_days]
        observed = clean * np.exp(rng.normal(0.0, sigma, size=clean.shape))
        observations.append(
            ArmObservation(
                arm.name,
                arm.coverage,
                arm_params.dose_gy,
                arm_params.treatment_day,
                days.copy(),
                absolute_days,
                observed,
                log_sigma,
                clean,
            )
        )
    return FittingDataset(tuple(observations), p, seed)


def dataset_from_frame(
    frame: pd.DataFrame, *, p: ModelParameters, default_log_sigma: float = 0.08
) -> FittingDataset:
    """Build a dataset from long-format volume data."""
    required = {"arm", "coverage", "day_from_treatment", "volume_cc"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")
    if default_log_sigma <= 0.0:
        raise ValueError("default_log_sigma must be positive")
    observations: list[ArmObservation] = []
    for arm_name, group in frame.groupby("arm", sort=False):
        group = group.sort_values("day_from_treatment")
        coverage_values = group["coverage"].unique()
        if len(coverage_values) != 1:
            raise ValueError(f"Arm {arm_name!r} has multiple coverage values")
        dose_values = (
            group["dose_gy"].dropna().unique() if "dose_gy" in group else np.array([])
        )
        day_values = (
            group["treatment_day"].dropna().unique()
            if "treatment_day" in group
            else np.array([])
        )
        if len(dose_values) > 1 or len(day_values) > 1:
            raise ValueError(f"Arm {arm_name!r} has multiple treatment settings")
        dose_gy = p.dose_gy if len(dose_values) == 0 else float(dose_values[0])
        raw_treatment_day = p.treatment_day if len(day_values) == 0 else day_values[0]
        raw_relative_days = group["day_from_treatment"].to_numpy(dtype=float)
        if (
            not np.isfinite(raw_treatment_day)
            or raw_treatment_day < 0
            or int(raw_treatment_day) != raw_treatment_day
            or not np.all(np.isfinite(raw_relative_days))
            or np.any(raw_relative_days != np.floor(raw_relative_days))
        ):
            raise ValueError(
                f"Arm {arm_name!r} contains invalid treatment or observation days"
            )
        treatment_day = int(raw_treatment_day)
        relative_days = raw_relative_days.astype(int)
        absolute_days = treatment_day + relative_days
        if np.any(absolute_days < 0) or np.any(absolute_days >= p.n_days):
            raise ValueError(f"Arm {arm_name!r} contains days outside simulation range")
        observed = group["volume_cc"].to_numpy(dtype=float)
        log_sigma = (
            group["log_sigma"].fillna(default_log_sigma).to_numpy(dtype=float)
            if "log_sigma" in group
            else np.full(len(group), default_log_sigma)
        )
        observations.append(
            ArmObservation(
                str(arm_name),
                float(coverage_values[0]),
                dose_gy,
                treatment_day,
                relative_days,
                absolute_days,
                observed,
                log_sigma,
            )
        )
    return FittingDataset(tuple(observations), None, None)
