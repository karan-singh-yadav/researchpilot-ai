import os

os.environ["DATABASE_URL"] = "sqlite://"  # in-memory; must be set before app imports
os.environ["API_KEYS"] = ""
os.environ["CHECKPOINT_PATH"] = ":memory:"
os.environ["RUN_EXECUTOR"] = "inline"

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from app.config import get_settings
from app.database import Base, enable_sqlite_fks, get_db
from app.main import create_app
from app.security.rate_limit import limiter


@pytest.fixture()
def db_session():
    engine = create_engine(
        "sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool
    )
    import app.models  # noqa: F401

    enable_sqlite_fks(engine)
    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine, autoflush=False, expire_on_commit=False)
    s = Session()
    yield s
    s.close()
    engine.dispose()


@pytest.fixture()
def client(db_session):
    get_settings.cache_clear()
    limiter.reset()
    app = create_app()

    def _override():
        try:
            yield db_session
            db_session.commit()
        except Exception:
            db_session.rollback()
            raise

    app.dependency_overrides[get_db] = _override
    with TestClient(app, raise_server_exceptions=False) as c:
        yield c
    get_settings.cache_clear()
