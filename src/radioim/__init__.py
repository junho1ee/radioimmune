"""Reusable radio-immune simulation, fitting, and diagnostics tools."""

from .data import (
    STANDARD_THREE_ARMS,
    ArmObservation,
    FittingDataset,
    TreatmentArm,
    dataset_from_frame,
    generate_synthetic_dataset,
)
from .diagnostics import (
    local_uncertainty_summary,
    profile_parameter,
    repeated_noise_recovery,
)
from .model import (
    ModelParameters,
    SimulationResult,
    build_dose_schedule,
    clearance_weights,
    normal_pdf,
    simulate,
    simulate_dose_schedule,
    survival_fractions,
)
from .optimize import (
    DEFAULT_VOLUME_FIT_SPECS,
    FitParameterSpec,
    FitResult,
    chi_square,
    fit_dataset,
    log_bounds,
    parameter_recovery_table,
    params_from_log_vector,
    residual_vector,
)

__all__ = [
    "ArmObservation",
    "DEFAULT_VOLUME_FIT_SPECS",
    "FitParameterSpec",
    "FitResult",
    "FittingDataset",
    "ModelParameters",
    "STANDARD_THREE_ARMS",
    "SimulationResult",
    "TreatmentArm",
    "build_dose_schedule",
    "chi_square",
    "clearance_weights",
    "dataset_from_frame",
    "fit_dataset",
    "generate_synthetic_dataset",
    "local_uncertainty_summary",
    "log_bounds",
    "normal_pdf",
    "parameter_recovery_table",
    "params_from_log_vector",
    "profile_parameter",
    "repeated_noise_recovery",
    "residual_vector",
    "simulate",
    "simulate_dose_schedule",
    "survival_fractions",
]
