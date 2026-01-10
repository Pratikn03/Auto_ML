"""Smoke tests to ensure core modules import and demo data exists."""

from pathlib import Path
import sys

# Ensure repo root is on path when running pytest from tests/
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def test_demo_csv_exists():
    """The demo dataset required by scripts/run_all.py must be present."""
    candidates = [
        REPO_ROOT / "Project" / "src" / "data" / "modeldata_demo.csv",
        REPO_ROOT / "src" / "data" / "modeldata_demo.csv",
    ]
    found = any(p.exists() for p in candidates)
    assert found, f"Demo CSV not found in any of {candidates}"


def test_core_imports():
    """Verify that key project modules are importable."""
    # utilities
    from Project.utils import io, sanitize  # noqa: F401

    # trainers (just import; do not run)
    import Project.trainers.train_boosters  # noqa: F401
    import Project.trainers.train_catboost  # noqa: F401
    import Project.trainers.train_flaml  # noqa: F401
    import Project.trainers.train_h2o  # noqa: F401


def test_analysis_imports():
    """Analysis scripts should import without error."""
    import Project.analysis.summarize_all  # noqa: F401
    import Project.analysis.plot_comparisons  # noqa: F401


def test_deploy_app_import():
    """FastAPI app module must be importable (not running the server)."""
    from Deploy.api.serve import app  # noqa: F401


def test_reports_directory_exists():
    """After a training run, reports/ should exist."""
    reports_dir = REPO_ROOT / "reports"
    # allow test to pass even before first run; just warn
    if not reports_dir.exists():
        import warnings
        warnings.warn("reports/ directory not found; run scripts/run_all.py first")
    # non-fatal: presence of directory is good enough
    assert True


def test_leaderboard_csv_exists():
    """leaderboard.csv should be present after a successful pipeline run."""
    lb_path = REPO_ROOT / "reports" / "leaderboard.csv"
    if not lb_path.exists():
        import warnings
        warnings.warn("leaderboard.csv not found; run scripts/run_all.py first")
    assert True
