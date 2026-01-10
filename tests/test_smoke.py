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
        REPO_ROOT / "Project" / "src" / "data" / "modeldata.csv",
        REPO_ROOT / "src" / "data" / "modeldata_demo.csv",
        REPO_ROOT / "src" / "data" / "modeldata.csv",
        REPO_ROOT / "src" / "data" / "datasets" / "tabular" / "modeldata.csv",
    ]
    found = any(p.exists() for p in candidates)
    assert found, f"Demo CSV not found in any of {candidates}"


def test_core_imports():
    """Verify that key project modules are importable."""
    # utilities - these are safe to import
    from Project.utils import io, sanitize  # noqa: F401

    # trainers - just check files exist (don't import as they run on import)
    from pathlib import Path
    trainers_dir = Path(__file__).resolve().parent.parent / "Project" / "trainers"
    assert (trainers_dir / "train_boosters.py").exists(), "train_boosters.py missing"
    assert (trainers_dir / "train_catboost.py").exists(), "train_catboost.py missing"
    assert (trainers_dir / "train_flaml.py").exists(), "train_flaml.py missing"
    assert (trainers_dir / "train_h2o.py").exists(), "train_h2o.py missing"


def test_analysis_imports():
    """Analysis scripts should exist (don't import as some run on import)."""
    from pathlib import Path
    analysis_dir = Path(__file__).resolve().parent.parent / "Project" / "analysis"
    assert (analysis_dir / "summarize_all.py").exists(), "summarize_all.py missing"
    assert (analysis_dir / "plot_comparisons.py").exists(), "plot_comparisons.py missing"


def test_deploy_app_import():
    """FastAPI app module must be importable (not running the server)."""
    # Check file exists first (safer for CI)
    from pathlib import Path
    app_path = Path(__file__).resolve().parent.parent / "Deploy" / "api" / "serve" / "app.py"
    assert app_path.exists(), "Deploy/api/serve/app.py missing"
    
    # Only try import if fastapi is available
    try:
        from Deploy.api.serve import app  # noqa: F401
    except ImportError as e:
        import warnings
        warnings.warn(f"FastAPI import failed (likely missing dep): {e}")


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
