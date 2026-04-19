from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from app.ml.training.adaptors.base import DEFAULT_DATA_ROOT, DatasetSplitConfig, limit_items, normalize_address, split_items, stable_float
from app.ml.training.external_datasets import load_raven
from app.ml.training.unified_sample import build_unified_training_sample
from app.models import TransactionRequest


@dataclass(frozen=True)
class RavenAdaptor:
    data_root: Path = DEFAULT_DATA_ROOT
    train_val_split_config: DatasetSplitConfig = DatasetSplitConfig(train_ratio=0.9, val_ratio=0.1)
    sample_weight: float = 0.75

    name: str = "raven"

    def build_samples(
        self,
        *,
        split: str,
        limit: int | None = None,
        seed: int = 17,
    ) -> list:
        if split == "test":
            records = _prepare_records(load_raven(self.data_root, split="evaluation", limit=limit))
            return [
                _build_failure_sample(record, index=index, sample_weight=self.sample_weight)
                for index, record in enumerate(records)
            ]

        records = _prepare_records(load_raven(self.data_root, split="finetuning"))
        selected = split_items(records, split=split, seed=seed, config=self.train_val_split_config)
        selected = limit_items(selected, limit)
        return [
            _build_failure_sample(record, index=index, sample_weight=self.sample_weight)
            for index, record in enumerate(selected)
        ]


def _prepare_records(records: list) -> list[dict[str, str]]:
    prepared = []
    for record in records:
        from_address = normalize_address(record.payload.get("from_address"))
        to_address = normalize_address(record.payload.get("to_address"))
        if from_address is None or to_address is None:
            continue
        prepared.append(
            {
                "hash": str(record.payload.get("hash") or ""),
                "from_address": from_address,
                "to_address": to_address,
                "tx_input": str(record.payload.get("tx_input") or ""),
                "value": str(record.payload.get("value") or "0"),
                "failure_reason": str(record.payload.get("failure_reason") or ""),
                "failure_message": str(record.payload.get("failure_message") or ""),
                "failure_invariant": str(record.payload.get("failure_invariant") or ""),
            }
        )
    return prepared


def _build_failure_sample(record: dict[str, str], *, index: int, sample_weight: float):
    request = TransactionRequest(
        chain_id=1,
        from_address=record["from_address"],
        to_address=record["to_address"],
        calldata=record["tx_input"],
        value_eth=stable_float(record["value"]) / 1e18,
        contract_name="RAVEN Failure Contract",
        interaction_label="Replay reverted transaction",
        notes=record["failure_message"] or record["failure_reason"] or record["failure_invariant"],
        simulation_profile="none",
    )
    return build_unified_training_sample(
        dataset_name="raven",
        sample_id=record["hash"] or f"raven-{index}",
        request=request,
        binary_targets={"failure_aux": 1.0},
        binary_target_mask={"failure_aux": True},
        sample_weight=sample_weight,
        metadata={
            "failure_reason": record["failure_reason"],
            "failure_message": record["failure_message"],
            "failure_invariant": record["failure_invariant"],
        },
    )
