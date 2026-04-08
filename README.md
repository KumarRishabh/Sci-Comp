# Sci-Comp

Scientific computation coursework and experiments, organized by artifact type and by problem.

## Repository layout

- `scripts/`: runnable Python scripts for each problem.
- `notebooks/`: Jupyter notebooks for interactive exploration.
- `figures/problem1/` through `figures/problem4/`: saved plots grouped by question.

## Current contents

- `scripts/problem1_finite_difference.py`
- `scripts/problem2_dissipation_dispersion.py`
- `scripts/problem3_hyperbolic_fd.py`
- `scripts/problem4_burgers.py`
- `scripts/project_paths.py`: shared path helpers for consistent figure output.
- `notebooks/problem3_hyperbolic_fd.ipynb`

## Running the scripts

Install the Python dependencies first:

```bash
python3 -m pip install -r requirements.txt
```

Run scripts from the repository root:

```bash
python3 scripts/problem1_finite_difference.py
python3 scripts/problem2_dissipation_dispersion.py
python3 scripts/problem3_hyperbolic_fd.py
python3 scripts/problem4_burgers.py
```

Each script now writes figures into its own folder under `figures/`.
