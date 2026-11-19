from fastapi import APIRouter, Depends
from sqlalchemy import inspect, text
from sqlalchemy.orm import Session

from app.database import get_db

router = APIRouter(tags=["health"])
REQUIRED_TABLE = "research_sessions"


@router.get("/health")
def health(db: Session = Depends(get_db)) -> dict:
    """Liveness + database connectivity + schema (migration) check."""
    db_up = schema_ok = False
    try:
        db.execute(text("SELECT 1"))
        db_up = True
        schema_ok = REQUIRED_TABLE in inspect(db.get_bind()).get_table_names()
    except Exception:
        pass
    return {
        "status": "ok" if db_up and schema_ok else "degraded",
        "database": "up" if db_up else "down",
        "schema": "migrated" if schema_ok else "missing",
    }
