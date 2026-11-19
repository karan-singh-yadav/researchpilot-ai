Q = "Research the latest developments in agentic AI frameworks."


def test_create_get_status_list_delete(client):
    r = client.post("/api/research", json={"start": False, "query": Q, "depth": "quick"})
    assert r.status_code == 201
    body = r.json()
    rid = body["id"]
    assert body["status"] == "pending" and body["depth"] == "quick" and body["mode"] == "web"

    assert client.get(f"/api/research/{rid}").json()["query"] == Q
    assert client.get(f"/api/research/{rid}/status").json()["status"] == "pending"

    lst = client.get("/api/research").json()
    assert lst["total"] == 1 and lst["items"][0]["id"] == rid

    assert client.get("/api/research", params={"q": "agentic"}).json()["total"] == 1
    assert client.get("/api/research", params={"q": "zzzz"}).json()["total"] == 0

    assert client.delete(f"/api/research/{rid}").status_code == 204
    assert client.get(f"/api/research/{rid}").status_code == 404


def test_pagination_bounds(client):
    assert client.get("/api/research", params={"limit": 0}).status_code == 422
    assert client.get("/api/research", params={"limit": 101}).status_code == 422


def test_document_mode_requires_documents(client):
    r = client.post("/api/research", json={"start": False, "query": Q, "mode": "document"})
    assert r.status_code == 422
    assert r.json()["error"]["code"] == "documents_required"


def test_invalid_enum_rejected(client):
    assert (
        client.post(
            "/api/research", json={"start": False, "query": Q, "depth": "extreme"}
        ).status_code
        == 422
    )
