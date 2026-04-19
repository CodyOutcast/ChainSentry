from pathlib import Path

from app.ml.training.external_datasets import (
    load_eth_labels,
    load_etherscamdb,
    load_forta_labels,
    load_ptxphish,
    load_ptxphish_initial_addresses,
    summarize_available_external_datasets,
)


DATA_ROOT = Path(__file__).resolve().parents[1].parent / "data"


def test_forta_label_loader_reads_records() -> None:
    records = load_forta_labels(DATA_ROOT, chain_id=1)

    assert records
    assert any(record.source == "forta-phishing" for record in records)
    assert any("address" in record.payload or "contract_address" in record.payload for record in records)


def test_eth_labels_loader_filters_chain() -> None:
    records = load_eth_labels(DATA_ROOT, chain_id=1)

    assert records
    assert all(int(record.payload["chainId"]) == 1 for record in records[:100])


def test_ptxphish_loader_reads_xlsx_rows() -> None:
    records = load_ptxphish(DATA_ROOT)

    assert records
    first = records[0]
    assert first.source == "ptxphish"
    assert isinstance(first.payload, dict)
    assert first.payload


def test_ptxphish_initial_address_loader_extracts_family_rows() -> None:
    records = load_ptxphish_initial_addresses(DATA_ROOT)

    assert records
    first = records[0]
    assert first.source == "ptxphish-initial-address"
    assert {"family", "address", "tx_total", "address_type"}.issubset(first.payload)


def test_etherscamdb_loader_reads_yaml_rows() -> None:
    records = load_etherscamdb(DATA_ROOT)

    assert records
    assert records[0].source == "etherscamdb"
    assert "category" in records[0].payload or "name" in records[0].payload


def test_external_dataset_summary_reports_expected_files() -> None:
    summary = summarize_available_external_datasets(DATA_ROOT)

    assert summary["forta_labels"]["exists"] is True
    assert summary["ptxphish"]["exists"] is True
    assert summary["ptxphish_initial_addresses"]["exists"] is True
    assert summary["forta_malicious_contracts"]["exists"] is True
