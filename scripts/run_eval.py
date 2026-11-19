"""Run baseline vs improved evaluation benchmark and save metrics to evaluation/results/.

python scripts/run_eval.py
"""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "backend"))

from sqlalchemy import create_engine  # noqa: E402
from sqlalchemy.orm import sessionmaker  # noqa: E402
from sqlalchemy.pool import StaticPool  # noqa: E402

from app.config import Settings  # noqa: E402
from app.database import Base, enable_sqlite_fks  # noqa: E402
from app.evaluation.runner import compare_evaluation_runs, run_evaluation_suite  # noqa: E402


def main() -> int:
    fixture = ROOT / "evaluation" / "datasets" / "fixtures" / "demo_web.json"
    settings = Settings(
        _env_file=None,
        llm_provider="fake",
        search_provider="fixture",
        search_fixture_path=str(fixture),
    )
    eng = create_engine(
        "sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool
    )
    enable_sqlite_fks(eng)
    Base.metadata.create_all(eng)
    Session = sessionmaker(bind=eng, autoflush=False, expire_on_commit=False)
    db = Session()
    try:
        base = run_evaluation_suite(db, settings, variant="baseline", dataset_name="demo-web")
        imp = run_evaluation_suite(db, settings, variant="improved", dataset_name="demo-web")
        db.commit()
        comp = compare_evaluation_runs(base, imp)
        out_dir = ROOT / "evaluation" / "results"
        out_dir.mkdir(parents=True, exist_ok=True)
        payload = {
            "baseline": base.metrics,
            "improved": imp.metrics,
            "comparison": comp["deltas"],
        }
        (out_dir / "latest.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(json.dumps(payload, indent=2))
    finally:
        db.close()
        eng.dispose()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
