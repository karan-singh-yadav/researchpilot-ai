def test_health_ok(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok", "database": "up", "schema": "migrated"}


def test_unknown_route_uses_error_envelope(client):
    r = client.get("/nope")
    assert r.status_code == 404
    assert r.json()["error"]["code"] == "http_error"


def test_validation_error_envelope(client):
    r = client.post("/api/research", json={"start": False, "query": "short"})
    assert r.status_code == 422
    body = r.json()["error"]
    assert body["code"] == "validation_error"
    assert body["details"][0]["field"].endswith("query")


def test_health_reports_missing_schema(client, db_session):
    from app.database import Base

    Base.metadata.drop_all(db_session.get_bind())
    body = client.get("/health").json()
    assert body == {"status": "degraded", "database": "up", "schema": "missing"}
