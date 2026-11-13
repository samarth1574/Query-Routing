def test_resolve_endpoint_resolves_and_stores_a_ticket(client):
    r = client.post("/api/v1/resolve", json={"text": "I forgot my passcode", "customer_ref": "CUST-A1"})
    assert r.status_code == 200
    d = r.json()
    assert d["status"] == "resolved" and d["agent"] == "account_agent"
    assert "decision" not in d
    t = client.get(f"/api/v1/tickets/{d['ticket_ref']}").json()
    assert t["status"] == "resolved"
    assert any(e["actor"] == "agent:account_agent" for e in t["events"])


def test_escalated_case_stays_open_for_a_human(client):
    d = client.post("/api/v1/resolve", json={"text": "I was scammed by a caller pretending to be the bank"}).json()
    assert d["status"] == "escalated" and d["escalation"]["reasons"]
    t = client.get(f"/api/v1/tickets/{d['ticket_ref']}").json()
    assert t["status"] == "open" and t["priority"] == "urgent"


def test_resolve_masks_pii(client):
    d = client.post("/api/v1/resolve", json={"text": "card 4539 1488 0343 6467 skimmed", "create_ticket": False}).json()
    assert "4539" not in d["text"] and d["ticket_ref"] is None


def test_resolve_batch(client):
    texts = ["my card is not working", "how long does a bank transfer take", "I lost my card"] * 4
    r = client.post("/api/v1/resolve/batch", json={"texts": texts, "workers": 4})
    assert r.status_code == 200
    s = r.json()["summary"]
    assert s["total"] == 12 and s["resolved"] + s["escalated"] == 12 and s["workers"] == 4
    assert [x["text"] for x in r.json()["results"]] == texts


def test_resolve_batch_validation(client):
    assert client.post("/api/v1/resolve/batch", json={"texts": ["  "]}).status_code == 422
    assert client.post("/api/v1/resolve/batch", json={"texts": ["a"], "workers": 99}).status_code == 422


def test_agents_and_knowledge_endpoints(client):
    d = client.get("/api/v1/agents").json()
    assert {a["category"] for a in d["agents"]} == {"payments", "cards", "loans", "account", "fraud"}
    assert any(t["name"] == "block_card" and t["side_effect"] for t in d["tools"])
    k = client.get("/api/v1/knowledge/search", params={"q": "pin blocked", "category": "cards"}).json()
    assert k["results"][0]["heading"].startswith("PIN")
    assert client.get("/api/v1/knowledge/search", params={"q": "pin", "category": "nope"}).status_code == 422


def test_assistant_page_and_metrics(client):
    assert client.get("/assistant").status_code == 200
    client.post("/api/v1/resolve", json={"text": "my card is not working", "create_ticket": False})
    assert "bankq_resolutions_total" in client.get("/metrics").text
