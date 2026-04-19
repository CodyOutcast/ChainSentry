from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from app.ml.training.adaptors.base import DEFAULT_DATA_ROOT, DatasetSplitConfig, limit_items, split_items
from app.ml.training.external_datasets import load_ptxphish_initial_addresses
from app.ml.training.unified_sample import build_unified_training_sample, severity_label_to_index
from app.models import TransactionRequest


TOKENS = ("USDC", "USDT", "WETH", "NFT")


@dataclass(frozen=True)
class PTXPhishAdaptor:
    data_root: Path = DEFAULT_DATA_ROOT
    split_config: DatasetSplitConfig = DatasetSplitConfig()
    sample_weight: float = 1.15

    name: str = "ptxphish"

    def build_samples(
        self,
        *,
        split: str,
        limit: int | None = None,
        seed: int = 17,
    ) -> list:
        records = load_ptxphish_initial_addresses(self.data_root)
        selected = split_items(records, split=split, seed=seed, config=self.split_config)
        selected = limit_items(selected, limit)
        samples = []
        for index, record in enumerate(selected):
            sample = _build_family_sample(record.payload, index=index, sample_weight=self.sample_weight)
            if sample is not None:
                samples.append(sample)
        return samples


def _build_family_sample(payload: dict[str, str], *, index: int, sample_weight: float):
    family = str(payload.get("family") or "").strip()
    address = str(payload.get("address") or "").lower()
    metadata = {
        "family": family,
        "tx_total": str(payload.get("tx_total") or ""),
        "address_type": str(payload.get("address_type") or ""),
    }
    if family == "Approve":
        return _build_approval_sample(
            address,
            family,
            index=index,
            method_name="approve",
            sample_weight=sample_weight,
            metadata=metadata,
        )
    if family == "permit":
        return _build_approval_sample(
            address,
            family,
            index=index,
            method_name="permit",
            sample_weight=sample_weight,
            metadata=metadata,
        )
    if family == "setApproveForAll":
        return _build_operator_sample(address, family, index=index, sample_weight=sample_weight, metadata=metadata)
    if family in {"Bulk transfer", "Proxy upgrade", "Free buy order", "Airdrop function", "Wallet function"}:
        return _build_simulation_sample(address, family, index=index, sample_weight=sample_weight, metadata=metadata)
    if family in {"Zero value transfer", "Fake token transfer", "Dust value transfer"}:
        return _build_destination_sample(address, family, index=index, sample_weight=sample_weight, metadata=metadata)
    return None


def _build_approval_sample(
    address: str,
    family: str,
    *,
    index: int,
    method_name: str,
    sample_weight: float,
    metadata: dict[str, str],
):
    token_symbol = TOKENS[index % 3]
    request = TransactionRequest(
        chain_id=1,
        from_address=_wallet_address(index),
        to_address=address,
        method_name=method_name,
        token_symbol=token_symbol,
        token_amount=150.0 + float(index % 11),
        approval_amount=1_000_000_000,
        spender_address=address,
        contract_name=f"PTXPhish {family}",
        interaction_label=f"PTXPhish {family}",
        notes=family,
        simulation_profile="allowance_drain",
    )
    return build_unified_training_sample(
        dataset_name="ptxphish",
        sample_id=f"ptxphish-{family.lower().replace(' ', '-')}-{address}",
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
        multiclass_targets={"severity": severity_label_to_index("high")},
        multiclass_target_mask={"severity": True},
        sample_weight=sample_weight,
        metadata={**metadata, "shell": "approval"},
    )


def _build_operator_sample(address: str, family: str, *, index: int, sample_weight: float, metadata: dict[str, str]):
    request = TransactionRequest(
        chain_id=1,
        from_address=_wallet_address(index + 20_000),
        to_address=address,
        method_name="setApprovalForAll",
        spender_address=address,
        contract_name="PTXPhish NFT Operator",
        interaction_label="Grant collection-wide operator",
        notes=family,
        simulation_profile="privilege_escalation",
    )
    return build_unified_training_sample(
        dataset_name="ptxphish",
        sample_id=f"ptxphish-operator-{address}",
        request=request,
        binary_targets={
            "approval": 1.0,
            "simulation": 1.0,
            "address_malicious": 1.0,
            "failure_aux": 0.0,
        },
        binary_target_mask={
            "approval": True,
            "simulation": True,
            "address_malicious": True,
            "failure_aux": True,
        },
        multiclass_targets={"severity": severity_label_to_index("high")},
        multiclass_target_mask={"severity": True},
        sample_weight=sample_weight,
        metadata={**metadata, "shell": "operator"},
    )


def _build_simulation_sample(
    address: str,
    family: str,
    *,
    index: int,
    sample_weight: float,
    metadata: dict[str, str],
):
    profile = "privilege_escalation" if family == "Proxy upgrade" else "unexpected_outflow"
    method_name = {
        "Bulk transfer": "bulkTransfer",
        "Proxy upgrade": "upgradeProxy",
        "Free buy order": "matchOrders",
        "Airdrop function": "claimAirdrop",
        "Wallet function": "walletFunction",
    }[family]
    value_eth = 0.15 if family in {"Airdrop function", "Wallet function"} else 0.0
    request = TransactionRequest(
        chain_id=1,
        from_address=_wallet_address(index + 40_000),
        to_address=address,
        method_name=method_name,
        value_eth=value_eth,
        token_symbol=TOKENS[(index + 1) % len(TOKENS)],
        token_amount=80.0 + float(index % 17),
        spender_address=address,
        contract_name=f"PTXPhish {family}",
        interaction_label=family,
        notes=family,
        simulation_profile=profile,
    )
    severity = "critical" if family in {"Proxy upgrade", "Airdrop function", "Wallet function"} else "high"
    return build_unified_training_sample(
        dataset_name="ptxphish",
        sample_id=f"ptxphish-simulation-{family.lower().replace(' ', '-')}-{address}",
        request=request,
        binary_targets={
            "simulation": 1.0,
            "address_malicious": 1.0,
            "failure_aux": 0.0,
        },
        binary_target_mask={
            "simulation": True,
            "address_malicious": True,
            "failure_aux": True,
        },
        multiclass_targets={"severity": severity_label_to_index(severity)},
        multiclass_target_mask={"severity": True},
        sample_weight=sample_weight,
        metadata={**metadata, "shell": "simulation"},
    )


def _build_destination_sample(
    address: str,
    family: str,
    *,
    index: int,
    sample_weight: float,
    metadata: dict[str, str],
):
    token_amount = 0.0 if family == "Zero value transfer" else 0.01 + float(index % 3) * 0.01
    request = TransactionRequest(
        chain_id=1,
        from_address=_wallet_address(index + 60_000),
        to_address=address,
        method_name="transfer",
        token_symbol="ETH",
        token_amount=token_amount,
        contract_name=f"PTXPhish {family}",
        interaction_label=family,
        notes=family,
        simulation_profile="none",
    )
    return build_unified_training_sample(
        dataset_name="ptxphish",
        sample_id=f"ptxphish-destination-{family.lower().replace(' ', '-')}-{address}",
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
        metadata={**metadata, "shell": "destination"},
    )


def _wallet_address(index: int) -> str:
    return f"0x{0x4000000000000000000000000000000000000000 + index:040x}"
