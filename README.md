# Radiology Modeling

A Python project for simulating the effects of radiotherapy and immune responses on tumor volume and fitting model parameters to observed data. It includes Figure 4 reproduction and parameter-recovery experiments using synthetic data.

## Features

- Daily radio-immune tumor model simulation
- Treatment-arm configuration by radiation coverage, dose, and treatment timing
- Synthetic observation generation and long-format data loading
- Parameter fitting with differential evolution followed by bounded least squares
- Figure 4 analysis and visualization in Jupyter

## Requirements

- Python 3.11 or 3.12
- [uv](https://docs.astral.sh/uv/)

## Initial Installation

Run the following command from the repository root:

```bash
uv sync
```

This command creates `.venv` in the project root and installs the core dependencies, Jupyter development tools, and the `src/radioim` package in editable mode. Changes under `src/radioim` are therefore available without reinstalling the package.

Verify the installation:

```bash
uv run python -c "import radioim; print(radioim.__file__)"
```

## Running the Notebook

Start JupyterLab from the repository root:

```bash
uv run jupyter lab
```

Open `fig4_fitting.ipynb` in JupyterLab. Restart the notebook kernel if the environment was installed or updated while Jupyter was already running.

## Basic Usage

```python
from radioim import ModelParameters, simulate

params = ModelParameters(
    n_days=61,
    n_bins=2,
    dose_gy=10.0,
    treatment_day=10,
)

result = simulate(params, coverage=1.0)
print(result.total_volume)
```

## Project Structure

```text
.
├── fig4_fitting.ipynb   # Figure 4 fitting and visualization notebook
├── src/radioim/         # Simulation, data, optimization, and diagnostics package
├── pyproject.toml       # Project metadata and dependency configuration
└── uv.lock              # Reproducible dependency lockfile
```

The distribution name is `radiology-modeling`, while the Python import name is `radioim`.

## License

This project is distributed under the MIT License. See [`LICENSE`](LICENSE) for details.
