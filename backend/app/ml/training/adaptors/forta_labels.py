from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from app.ml.training.adaptors.base import DEFAULT_DATA_ROOT, DatasetSplitConfig, limit_items, normalize_address, split_items
from app.ml.training.external_datasets import ExternalDatasetRecord, load_forta_labels
from app.ml.training.unified_sample import build_unified_training_sample
from app.models import TransactionRequest


TOKENS = ("USDC", "USDT", "WETH", "DAI")


@dataclass(frozen=True)
class FortaLabelAdaptor:
    data_root: Path = DEFAULT_DATA_ROOT
    split_config: DatasetSplitConfig = DatasetSplitConfig()
    chain_id: int = 1
    approval_shell_ratio: float = 0.35
    sample_weight: float = 1.0

    name: str = "forta"

    def build_samples(
        self,
        *,
        split: str,
        limit: int | None = None,
        seed: int = 17,
    ) -> list:
        records = load_forta_labels(self.data_root, chain_id=self.chain_id)
        prepared = _prepare_records(records)
        selected = split_items(prepared, split=split, seed=seed, config=self.split_config)
        selected = limit_items(selected, limit)
        samples = []
        for index, record in enumerate(selected):
            samples.append(_build_destination_sample(record, index=index, sample_weight=self.sample_weight))
            if index % 100 < int(self.approval_shell_ratio * 100):
                samples.append(_build_approval_sample(record, index=index, sample_weight=self.sample_weight))
        return samples


def _prepare_records(records: list[ExternalDatasetRecord]) -> list[dict[str, str]]:
    deduped: dict[str, dict[str, str]] = {}
    for record in records:
        payload = record.payload
        address = normalize_address(
            payload.get("address")
            or payload.get("contract_address")
            or payload.get("banned_address")
        )
        if address is None:
            continue
        deduped[address] = {
            "address": address,
            "source": record.source,
            "label": str(
                payload.get("etherscan_labels")
                or payload.get("contract_tag")
                or payload.get("wallet_tag")
                or payload.get("source")
                or "forta-flagged"
            ),
            "notes": str(payload.get("notes") or ""),
        }
    return sorted(deduped.values(), key=lambda item: item["address"])


def _build_destination_sample(record: dict[str, str], *, index: int, sample_weight: float):
    request = TransactionRequest(
        chain_id=1,
        from_address=_wallet_address(index),
        to_address=record["address"],
        method_name="claimRewards",
        contract_name="Forta Flagged Destination",
        interaction_label="Claim campaign rewards",
        notes=record["notes"] or record["label"],
        simulation_profile="none",
    )
    return build_unified_training_sample(
        dataset_name="forta",
        sample_id=f"forta-destination-{record['address']}",
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
            "label_source": record["source"],
            "label": record["label"],
            "shell": "destination",
        },
    )


def _build_approval_sample(record: dict[str, str], *, index: int, sample_weight: float):
    token_symbol = TOKENS[index % len(TOKENS)]
    request = TransactionRequest(
        chain_id=1,
        from_address=_wallet_address(index + 50_000),
        to_address=record["address"],
        method_name="approve",
        token_symbol=token_symbol,
        token_amount=250.0 + float(index % 9),
        approval_amount=1_000_000_000,
        spender_address=record["address"],
        contract_name="Forta Flagged Spender",
        interaction_label="Approve campaign router",
        notes=record["label"],
        simulation_profile="allowance_drain",
    )
    return build_unified_training_sample(
        dataset_name="forta",
        sample_id=f"forta-approval-{record['address']}",
        request=request,
        binary_targets={
            "approval": 1.0,
            "address_malicious": 1.0,
            "failure_aux": 0.0,
        },
        binary_target_mask={
            "approval": True,
            "address_malicious": True,
            "failure_aux": True,
        },
        sample_weight=sample_weight,
        metadata={
            "label_source": record["source"],
            "label": record["label"],
            "shell": "approval",
        },
    )


def _wallet_address(index: int) -> str:
    return f"0x{0x1000000000000000000000000000000000000000 + index:040x}"
