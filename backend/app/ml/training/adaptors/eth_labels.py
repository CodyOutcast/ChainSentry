from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from app.ml.training.adaptors.base import DEFAULT_DATA_ROOT, DatasetSplitConfig, limit_items, normalize_address, split_items
from app.ml.training.external_datasets import load_eth_labels
from app.ml.training.unified_sample import build_unified_training_sample, severity_label_to_index
from app.models import TransactionRequest


BENIGN_LABEL_ALLOWLIST = {
    "0x-protocol",
    "aave",
    "balancer",
    "bancor",
    "maker-vault-owner",
    "pancakeswap",
    "pendle",
    "sushiswap",
    "synthetix",
    "the-graph",
    "yearn",
}
TOKENS = ("USDC", "USDT", "WETH", "DAI")


@dataclass(frozen=True)
class EthLabelsAdaptor:
    data_root: Path = DEFAULT_DATA_ROOT
    split_config: DatasetSplitConfig = DatasetSplitConfig()
    chain_id: int = 1
    sample_weight: float = 1.0

    name: str = "eth-labels"

    def build_samples(
        self,
        *,
        split: str,
        limit: int | None = None,
        seed: int = 17,
    ) -> list:
        records = load_eth_labels(self.data_root, chain_id=self.chain_id)
        prepared = _prepare_records(records)
        selected = split_items(prepared, split=split, seed=seed, config=self.split_config)
        selected = limit_items(selected, limit)
        samples = []
        for index, record in enumerate(selected):
            samples.append(_build_destination_sample(record, index=index, sample_weight=self.sample_weight))
            samples.append(_build_approval_sample(record, index=index, sample_weight=self.sample_weight))
            samples.append(_build_simulation_sample(record, index=index, sample_weight=self.sample_weight))
        return samples


def _prepare_records(records: list) -> list[dict[str, str]]:
    deduped: dict[str, dict[str, str]] = {}
    for record in records:
        label = str(record.payload.get("label") or "").strip().lower()
        if label not in BENIGN_LABEL_ALLOWLIST:
            continue
        address = normalize_address(record.payload.get("address"))
        if address is None:
            continue
        deduped[address] = {
            "address": address,
            "label": label,
            "name_tag": str(record.payload.get("nameTag") or ""),
        }
    return sorted(deduped.values(), key=lambda item: item["address"])


def _build_destination_sample(record: dict[str, str], *, index: int, sample_weight: float):
    request = TransactionRequest(
        chain_id=1,
        from_address=_wallet_address(index),
        to_address=record["address"],
        method_name="contractCall",
        contract_name=record["name_tag"] or record["label"].title(),
        interaction_label="Interact with a known protocol",
        notes=record["label"],
        simulation_profile="none",
    )
    return build_unified_training_sample(
        dataset_name="eth-labels",
        sample_id=f"eth-destination-{record['address']}",
        request=request,
        binary_targets={
            "destination": 0.0,
            "address_malicious": 0.0,
            "failure_aux": 0.0,
        },
        binary_target_mask={
            "destination": True,
            "address_malicious": True,
            "failure_aux": True,
        },
        multiclass_targets={"severity": severity_label_to_index("low")},
        multiclass_target_mask={"severity": True},
        sample_weight=sample_weight,
        metadata={"label": record["label"], "shell": "destination"},
    )


def _build_approval_sample(record: dict[str, str], *, index: int, sample_weight: float):
    token_symbol = TOKENS[index % len(TOKENS)]
    request = TransactionRequest(
        chain_id=1,
        from_address=_wallet_address(index + 25_000),
        to_address=record["address"],
        method_name="approve",
        token_symbol=token_symbol,
        token_amount=100.0 + float(index % 13),
        approval_amount=250.0 + float(index % 7),
        spender_address=record["address"],
        contract_name=record["name_tag"] or record["label"].title(),
        interaction_label="Approve exact protocol spend",
        notes=record["label"],
        simulation_profile="none",
    )
    return build_unified_training_sample(
        dataset_name="eth-labels",
        sample_id=f"eth-approval-{record['address']}",
        request=request,
        binary_targets={
            "approval": 0.0,
            "address_malicious": 0.0,
            "failure_aux": 0.0,
        },
        binary_target_mask={
            "approval": True,
            "address_malicious": True,
            "failure_aux": True,
        },
        multiclass_targets={"severity": severity_label_to_index("low")},
        multiclass_target_mask={"severity": True},
        sample_weight=sample_weight,
        metadata={"label": record["label"], "shell": "approval"},
    )


def _build_simulation_sample(record: dict[str, str], *, index: int, sample_weight: float):
    token_symbol = TOKENS[(index + 1) % len(TOKENS)]
    request = TransactionRequest(
        chain_id=1,
        from_address=_wallet_address(index + 55_000),
        to_address=record["address"],
        method_name="swapExactTokensForTokens",
        token_symbol=token_symbol,
        token_amount=55.0 + float(index % 19),
        spender_address=record["address"],
        contract_name=record["name_tag"] or record["label"].title(),
        interaction_label="Swap through a known router",
        notes=record["label"],
        simulation_profile="none",
    )
    return build_unified_training_sample(
        dataset_name="eth-labels",
        sample_id=f"eth-simulation-{record['address']}",
        request=request,
        binary_targets={
            "simulation": 0.0,
            "address_malicious": 0.0,
            "failure_aux": 0.0,
        },
        binary_target_mask={
            "simulation": True,
            "address_malicious": True,
            "failure_aux": True,
        },
        multiclass_targets={"severity": severity_label_to_index("low")},
        multiclass_target_mask={"severity": True},
        sample_weight=sample_weight,
        metadata={"label": record["label"], "shell": "simulation"},
    )


def _wallet_address(index: int) -> str:
    return f"0x{0x2000000000000000000000000000000000000000 + index:040x}"
