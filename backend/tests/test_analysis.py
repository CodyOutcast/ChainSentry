from fastapi.testclient import TestClient

from app.main import app


client = TestClient(app)


def test_unlimited_approval_to_flagged_contract_returns_critical_findings() -> None:
    response = client.post(
        "/api/v1/analyze",
        json={
            "chain_id": 1,
            "from_address": "0x1111111111111111111111111111111111111111",
            "to_address": "0xdead00000000000000000000000000000000beef",
            "method_name": "approve",
            "token_symbol": "USDC",
            "token_amount": 250,
            "approval_amount": 1000000000,
            "spender_address": "0xdead00000000000000000000000000000000beef",
            "contract_name": "Demo Drain Contract",
            "simulation_profile": "allowance_drain",
        },
    )

    assert response.status_code == 200
    payload = response.json()

    assert payload["overall_severity"] == "critical"
    assert payload["recommended_action"] == "reject"
    categories = {finding["category"] for finding in payload["findings"]}
    assert {"approval", "destination", "simulation"}.issubset(categories)
    assert any(finding["id"] == "simulation-allowance-grant" for finding in payload["findings"])


def test_standard_transfer_returns_low_risk() -> None:
    response = client.post(
        "/api/v1/analyze",
        json={
            "chain_id": 1,
            "from_address": "0x1111111111111111111111111111111111111111",
            "to_address": "0x3333333333333333333333333333333333333333",
            "method_name": "transfer",
            "token_symbol": "ETH",
            "token_amount": 0.15,
            "contract_name": "Friend Wallet",
            "simulation_profile": "none",
        },
    )

    assert response.status_code == 200
    payload = response.json()

    assert payload["overall_severity"] == "low"
    assert payload["recommended_action"] == "proceed"
    assert payload["findings"] == []


def test_large_approval_to_benign_router_returns_medium_risk() -> None:
    response = client.post(
        "/api/v1/analyze",
        json={
            "chain_id": 1,
            "from_address": "0x1111111111111111111111111111111111111111",
            "to_address": "0x5555555555555555555555555555555555555555",
            "method_name": "approve",
            "token_symbol": "USDC",
            "approval_amount": 50000,
            "spender_address": "0x5555555555555555555555555555555555555555",
            "contract_name": "Demo Router",
            "simulation_profile": "none",
        },
    )

    assert response.status_code == 200
    payload = response.json()

    assert payload["overall_severity"] == "medium"
    assert payload["recommended_action"] == "inspect_further"
    finding_ids = {finding["id"] for finding in payload["findings"]}
    assert "approval-large" in finding_ids
    assert "simulation-allowance-grant" in finding_ids


def test_operator_control_case_returns_critical_risk() -> None:
    response = client.post(
        "/api/v1/analyze",
        json={
            "chain_id": 1,
            "from_address": "0x1111111111111111111111111111111111111111",
            "to_address": "0x6666666666666666666666666666666666666666",
            "method_name": "setApprovalForAll",
            "spender_address": "0x6666666666666666666666666666666666666666",
            "contract_name": "Collection Manager",
            "simulation_profile": "allowance_drain",
        },
    )

    assert response.status_code == 200
    payload = response.json()

    assert payload["overall_severity"] == "critical"
    assert payload["recommended_action"] == "reject"
    finding_ids = {finding["id"] for finding in payload["findings"]}
    assert "simulation-operator-control" in finding_ids
    assert "simulation-unexpected-outflow" in finding_ids


def test_privilege_escalation_returns_high_risk() -> None:
    response = client.post(
        "/api/v1/analyze",
        json={
            "chain_id": 1,
            "from_address": "0x1111111111111111111111111111111111111111",
            "to_address": "0x4444444444444444444444444444444444444444",
            "method_name": "grantRole",
            "contract_name": "Shadow Vault",
            "simulation_profile": "privilege_escalation",
        },
    )

    assert response.status_code == 200
    payload = response.json()

    assert payload["overall_severity"] == "high"
    assert payload["recommended_action"] == "reject"
    assert any(finding["id"] == "simulation-privilege-escalation" for finding in payload["findings"])


def test_blank_required_address_returns_validation_error() -> None:
    response = client.post(
        "/api/v1/analyze",
        json={
            "chain_id": 1,
            "from_address": "",
            "to_address": "0x3333333333333333333333333333333333333333",
            "method_name": "transfer",
            "value_eth": 0,
            "simulation_profile": "none",
        },
    )

    assert response.status_code == 422