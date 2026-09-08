"""Unit tests for the standalone SuperAdmin Monitoring API microservice."""

import pytest
from fastapi.testclient import TestClient

from apps.api_admin.config import admin_settings
from apps.api_admin.main import app

client = TestClient(app)

AUTH_HEADERS = {
    "X-Admin-Api-Key": (
        admin_settings.super_admin_api_key or "mock_firebase_admin_token_unit_tests"
    )
}


def test_admin_health_check():
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["service"] == "api_admin"


def test_admin_auth_status():
    response = client.get("/api/admin/auth/status")
    assert response.status_code == 200
    data = response.json()
    assert "admin_auth_enabled" in data


def test_unauthenticated_request_fails_when_auth_enabled():
    if admin_settings.admin_auth_enabled:
        response = client.get("/api/admin/cloud-monitor")
        assert response.status_code == 401
    else:
        response = client.get("/api/admin/cloud-monitor")
        assert response.status_code == 200


def test_authenticated_cloud_monitor_endpoint():
    response = client.get("/api/admin/cloud-monitor", headers=AUTH_HEADERS)
    assert response.status_code == 200
    data = response.json()
    assert "database_provider" in data
    assert "cache_hit_ratio" in data
    assert data["database_status"].startswith("Healthy")


def test_authenticated_agent_performance_endpoint():
    response = client.get("/api/admin/agent-performance", headers=AUTH_HEADERS)
    assert response.status_code == 200
    data = response.json()
    assert "total_tokens_consumed" in data
    assert "toon_savings_percentage" in data


def test_authenticated_run_agent_evaluations_endpoint():
    payload = {"suite_name": "unit_test_benchmark"}
    response = client.post("/api/admin/agent-performance/evaluations/run", json=payload, headers=AUTH_HEADERS)
    assert response.status_code == 200
    data = response.json()
    assert data["suite_name"] == "unit_test_benchmark"
    assert data["status"] == "Completed"
    assert data["accuracy_score"] > 90.0


def test_authenticated_marketing_telemetry_endpoint():
    response = client.get("/api/admin/marketing-telemetry", headers=AUTH_HEADERS)
    assert response.status_code == 200
    data = response.json()
    assert "monthly_active_organizations" in data
    assert len(data["top_used_features"]) > 0


def test_authenticated_governance_overview_endpoint():
    response = client.get("/api/admin/governance", headers=AUTH_HEADERS)
    assert response.status_code == 200
    data = response.json()
    assert data["total_organizations"] > 0
    assert len(data["recent_organizations"]) > 0
