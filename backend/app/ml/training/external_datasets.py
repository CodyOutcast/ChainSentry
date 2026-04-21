from __future__ import annotations

import csv
import json
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from xml.etree import ElementTree as ET

import yaml


DATA_ROOT = Path(__file__).resolve().parents[4] / "data"


@dataclass(frozen=True)
class ExternalDatasetRecord:
    source: str
    payload: dict[str, Any]


def load_forta_labels(
    data_root: Path = DATA_ROOT,
    *,
    chain_id: int = 1,
) -> list[ExternalDatasetRecord]:
    base = data_root / "forta-labelled-datasets" / "labels" / str(chain_id)
    records: list[ExternalDatasetRecord] = []
    records.extend(_load_csv_records(base / "phishing_scams.csv", "forta-phishing"))
    records.extend(_load_csv_records(base / "malicious_smart_contracts.csv", "forta-malicious-contract"))
    records.extend(_load_csv_records(base / "etherscan_malicious_labels.csv", "forta-etherscan-malicious"))
    return records


def load_eth_labels(
    data_root: Path = DATA_ROOT,
    *,
    chain_id: int | None = 1,
) -> list[ExternalDatasetRecord]:
    path = data_root / "eth-labels" / "data" / "csv" / "accounts.csv"
    records = _load_csv_records(path, "eth-labels-account")
    if chain_id is None:
        return records
    filtered: list[ExternalDatasetRecord] = []
    for record in records:
        row_chain_id = record.payload.get("chainId")
        if row_chain_id is None:
            continue
        if int(row_chain_id) == chain_id:
            filtered.append(record)
    return filtered


def load_ptxphish(
    data_root: Path = DATA_ROOT,
) -> list[ExternalDatasetRecord]:
    path = data_root / "PTXPhish" / "dataset" / "PTXPHISH.xlsx"
    return _load_xlsx_records(path, "ptxphish")


def load_ptxphish_initial_addresses(
    data_root: Path = DATA_ROOT,
) -> list[ExternalDatasetRecord]:
    path = data_root / "PTXPhish" / "dataset" / "InitialAddress.xlsx"
    rows = _read_xlsx_rows(path)
    if not rows:
        return []

    header_values = rows[0]
    records: list[ExternalDatasetRecord] = []
    for values in rows[1:]:
        for index in range(0, len(header_values), 3):
            family = header_values[index] if index < len(header_values) else ""
            if not family:
                continue
            address = values[index] if index < len(values) else ""
            tx_total = values[index + 1] if index + 1 < len(values) else ""
            address_type = values[index + 2] if index + 2 < len(values) else ""
            if not isinstance(address, str) or not address.startswith("0x"):
                continue
            records.append(
                ExternalDatasetRecord(
                    source="ptxphish-initial-address",
                    payload={
                        "family": family,
                        "address": address,
                        "tx_total": tx_total,
                        "address_type": address_type,
                    },
                )
            )
    return records


def load_etherscamdb(
    data_root: Path = DATA_ROOT,
) -> list[ExternalDatasetRecord]:
    path = data_root / "EtherScamDB" / "_data" / "scams.yaml"
    with path.open("r", encoding="utf-8") as file_handle:
        rows = yaml.safe_load(file_handle) or []
    return [ExternalDatasetRecord(source="etherscamdb", payload=dict(row)) for row in rows]


def load_forta_malicious_contracts(
    data_root: Path = DATA_ROOT,
    *,
    limit: int | None = None,
) -> list[ExternalDatasetRecord]:
    path = data_root / "forta-malicious-smart-contract-dataset" / "malicious_contract_training_dataset_final.parquet"
    return _load_parquet_records(path, "forta-malicious-smart-contract", limit=limit)


def load_raven(
    data_root: Path = DATA_ROOT,
    *,
    split: str = "evaluation",
    limit: int | None = None,
) -> list[ExternalDatasetRecord]:
    valid_splits = {"evaluation", "finetuning"}
    if split not in valid_splits:
        raise ValueError(f"Unsupported raven split: {split}")
    path = data_root / "raven-dataset" / "data" / f"{split}-00000-of-00001.parquet"
    return _load_parquet_records(path, f"raven-{split}", limit=limit)


def load_ethereum_fraud_by_activity(
    data_root: Path = DATA_ROOT,
    *,
    section: str = "labels",
    limit: int | None = None,
) -> list[ExternalDatasetRecord]:
    base = data_root / "ethereum_fraud_dataset_by_activity"
    if section == "labels":
        label_path = base / "gnn_dataset" / "labels" / "targets_global.parquet"
        return _load_parquet_records(label_path, "ethereum-fraud-labels", limit=limit)
    if section == "weekly_targets":
        target_path = base / "gnn_dataset" / "targets" / "week_targets.parquet"
        return _load_parquet_records(target_path, "ethereum-fraud-week-targets", limit=limit)
    if section == "monthly_targets":
        target_path = base / "gnn_dataset" / "targets" / "month_targets.parquet"
        return _load_parquet_records(target_path, "ethereum-fraud-month-targets", limit=limit)
    if section == "address_labels_balanced":
        path = base / "addr_labels_balanced.csv.zst"
        return _load_zstd_csv_records(path, "ethereum-fraud-address-labels-balanced", limit=limit)
    if section == "address_labels_big":
        path = base / "addr_labels_big.csv.zst"
        return _load_zstd_csv_records(path, "ethereum-fraud-address-labels-big", limit=limit)
    raise ValueError(f"Unsupported ethereum_fraud_dataset_by_activity section: {section}")


def summarize_available_external_datasets(data_root: Path = DATA_ROOT) -> dict[str, dict[str, Any]]:
    base = Path(data_root)
    paths = {
        "forta_labels": base / "forta-labelled-datasets" / "labels" / "1" / "phishing_scams.csv",
        "eth_labels": base / "eth-labels" / "data" / "csv" / "accounts.csv",
        "ptxphish": base / "PTXPhish" / "dataset" / "PTXPHISH.xlsx",
        "ptxphish_initial_addresses": base / "PTXPhish" / "dataset" / "InitialAddress.xlsx",
        "etherscamdb": base / "EtherScamDB" / "_data" / "scams.yaml",
        "forta_malicious_contracts": base / "forta-malicious-smart-contract-dataset" / "malicious_contract_training_dataset_final.parquet",
        "raven_evaluation": base / "raven-dataset" / "data" / "evaluation-00000-of-00001.parquet",
        "raven_finetuning": base / "raven-dataset" / "data" / "finetuning-00000-of-00001.parquet",
        "ethereum_fraud_labels": base / "ethereum_fraud_dataset_by_activity" / "gnn_dataset" / "labels" / "targets_global.parquet",
        "ethereum_fraud_addr_labels_big": base / "ethereum_fraud_dataset_by_activity" / "addr_labels_big.csv.zst",
    }
    summary: dict[str, dict[str, Any]] = {}
    for name, path in paths.items():
        summary[name] = {
            "path": str(path),
            "exists": path.exists(),
            "size_bytes": path.stat().st_size if path.exists() else None,
        }
    return summary


def _load_csv_records(path: Path, source: str) -> list[ExternalDatasetRecord]:
    with path.open("r", encoding="utf-8", newline="") as file_handle:
        reader = csv.DictReader(file_handle)
        return [ExternalDatasetRecord(source=source, payload=dict(row)) for row in reader]


def _load_zstd_csv_records(
    path: Path,
    source: str,
    *,
    limit: int | None = None,
) -> list[ExternalDatasetRecord]:
    try:
        import zstandard as zstd
    except Exception as exc:  # pragma: no cover - optional dependency
        raise RuntimeError("zstandard is required to load .zst CSV datasets.") from exc

    records: list[ExternalDatasetRecord] = []
    with path.open("rb") as file_handle:
        decompressor = zstd.ZstdDecompressor()
        with decompressor.stream_reader(file_handle) as reader:
            text_stream = reader.read().decode("utf-8")
    csv_reader = csv.DictReader(text_stream.splitlines())
    for index, row in enumerate(csv_reader):
        if limit is not None and index >= limit:
            break
        records.append(ExternalDatasetRecord(source=source, payload=dict(row)))
    return records


def _load_parquet_records(
    path: Path,
    source: str,
    *,
    limit: int | None = None,
) -> list[ExternalDatasetRecord]:
    try:
        import pyarrow.parquet as pq
    except Exception as exc:  # pragma: no cover - optional dependency
        raise RuntimeError("pyarrow is required to load parquet datasets.") from exc

    table = pq.read_table(path)
    if limit is not None:
        table = table.slice(0, limit)
    rows = table.to_pylist()
    return [ExternalDatasetRecord(source=source, payload=dict(row)) for row in rows]


def _load_xlsx_records(path: Path, source: str) -> list[ExternalDatasetRecord]:
    rows = _read_xlsx_rows(path)
    if not rows:
        return []

    header = [str(value or f"column_{index}") for index, value in enumerate(rows[0])]
    return [
        ExternalDatasetRecord(
            source=source,
            payload=dict(
                zip(
                    header,
                    (row + [""] * max(len(header) - len(row), 0))[: len(header)],
                    strict=True,
                )
            ),
        )
        for row in rows[1:]
    ]


def _read_xlsx_rows(path: Path) -> list[list[str]]:
    workbook_ns = {"main": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
    package_ns = {"rel": "http://schemas.openxmlformats.org/package/2006/relationships"}

    with zipfile.ZipFile(path) as workbook:
        shared_strings = _read_shared_strings(workbook, workbook_ns)
        workbook_root = ET.fromstring(workbook.read("xl/workbook.xml"))
        sheet = workbook_root.find("main:sheets/main:sheet", workbook_ns)
        if sheet is None:
            return []
        relationship_id = sheet.attrib.get("{http://schemas.openxmlformats.org/officeDocument/2006/relationships}id")
        relationships_root = ET.fromstring(workbook.read("xl/_rels/workbook.xml.rels"))
        target_path = None
        for relationship in relationships_root.findall("rel:Relationship", package_ns):
            if relationship.attrib.get("Id") == relationship_id:
                target_path = relationship.attrib.get("Target")
                break
        if target_path is None:
            raise RuntimeError("Could not resolve first worksheet in xlsx dataset.")
        sheet_xml = workbook.read(f"xl/{target_path}")

    sheet_root = ET.fromstring(sheet_xml)
    rows = []
    for row in sheet_root.findall("main:sheetData/main:row", workbook_ns):
        values = [_read_xlsx_cell(cell, shared_strings, workbook_ns) for cell in row.findall("main:c", workbook_ns)]
        rows.append(values)
    return rows


def _read_shared_strings(workbook: zipfile.ZipFile, workbook_ns: dict[str, str]) -> list[str]:
    try:
        shared_xml = workbook.read("xl/sharedStrings.xml")
    except KeyError:
        return []
    root = ET.fromstring(shared_xml)
    values: list[str] = []
    for item in root.findall("main:si", workbook_ns):
        text_nodes = item.findall(".//main:t", workbook_ns)
        values.append("".join(node.text or "" for node in text_nodes))
    return values


def _read_xlsx_cell(cell, shared_strings: list[str], workbook_ns: dict[str, str]) -> str:
    cell_type = cell.attrib.get("t")
    value_node = cell.find("main:v", workbook_ns)
    if value_node is None or value_node.text is None:
        return ""
    raw_value = value_node.text
    if cell_type == "s":
        return shared_strings[int(raw_value)]
    return raw_value
