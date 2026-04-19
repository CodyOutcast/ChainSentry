from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from app.ml.training.adaptors.base import DEFAULT_DATA_ROOT, DatasetSplitConfig, limit_items, normalize_address, split_items
from app.ml.training.external_datasets import load_etherscamdb
from app.ml.training.unified_sample import build_unified_training_sample
from app.models import TransactionRequest


@dataclass(frozen=True)
class EtherScamDbAdaptor:
    data_root: Path = DEFAULT_DATA_ROOT
    split_config: DatasetSplitConfig = DatasetSplitConfig()
    sample_weight: float = 0.9

    name: str = "etherscamdb"

    def build_samples(
        self,
        *,
        split: str,
        limit: int | None = None,
        seed: int = 17,
    ) -> list:
        records = load_etherscamdb(self.data_root)
        prepared = _prepare_records(records)
        selected = split_items(prepared, split=split, seed=seed, config=self.split_config)
        selected = limit_items(selected, limit)
        return [
            _build_destination_sample(record, index=index, sample_weight=self.sample_weight)
            for index, record in enumerate(selected)
        ]


def _prepare_records(records: list) -> list[dict[str, str]]:
    deduped: dict[str, dict[str, str]] = {}
    for record in records:
        addresses = record.payload.get("addresses") or []
        if not isinstance(addresses, list):
            continue
        for address in addresses:
            normalized = normalize_address(address)
            if normalized is None:
                continue
            deduped[normalized] = {
                "address": normalized,
                "category": str(record.payload.get("category") or ""),
                "subcategory": str(record.payload.get("subcategory") or ""),
                "name": str(record.payload.get("name") or ""),
                "url": str(record.payload.get("url") or ""),
            }
    return sorted(deduped.values(), key=lambda item: item["address"])


def _build_destination_sample(record: dict[str, str], *, index: int, sample_weight: float):
    request = TransactionRequest(
        chain_id=1,
        from_address=_wallet_address(index),
        to_address=record["address"],
        method_name="claimAirdrop",
        contract_name=record["subcategory"] or record["category"] or "Scam Destination",
        interaction_label="Claim campaign rewards",
        notes=record["url"] or record["name"],
        simulation_profile="none",
    )
    return build_unified_training_sample(
        dataset_name="etherscamdb",
        sample_id=f"etherscamdb-destination-{record['address']}",
        request=request,
        binary_targets={
            "destination": 1.0,
            "address_malicious": 1.0,
            "failure_aux": 0.0,
        },
        binary_target_mask={
            "destination": True,
            "address_malicious": True,
            "failure_aux": True,
        },
        sample_weight=sample_weight,
        metadata={
            "category": record["category"],
            "subcategory": record["subcategory"],
            "shell": "destination",
        },
    )


def _wallet_address(index: int) -> str:
    return f"0x{0x3000000000000000000000000000000000000000 + index:040x}"
