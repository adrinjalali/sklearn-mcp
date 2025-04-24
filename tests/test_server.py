"""Basic tests for the MCP server."""

from fastapi.testclient import TestClient

from ds_mcp.server import app


client = TestClient(app)


def test_health_endpoint():
    """Test the health endpoint returns proper status."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert "version" in data
