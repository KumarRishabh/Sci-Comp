from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
FIGURES_ROOT = PROJECT_ROOT / "figures"


def ensure_problem_figure_dir(problem_name: str) -> Path:
    figure_dir = FIGURES_ROOT / problem_name
    figure_dir.mkdir(parents=True, exist_ok=True)
    return figure_dir


def relative_to_root(path: Path) -> Path:
    return path.relative_to(PROJECT_ROOT)
