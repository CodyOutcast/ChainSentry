from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from random import Random
from typing import Protocol, TypeVar

from app.ml.training.unified_sample import UnifiedTrainingSample


ADDRESS_PATTERN = re.compile(r"^0x[a-fA-F0-9]{40}$")
DEFAULT_DATA_ROOT = Path(__file__).resolve().parents[5] / "data"
T = TypeVar("T")


class DatasetAdaptor(Protocol):
    name: str

    def build_samples(
        self,
        *,
        split: str,
        limit: int | None = None,
        seed: int = 17,
    ) -> list[UnifiedTrainingSample]:
        ...


@dataclass(frozen=True)
class DatasetSplitConfig:
    train_ratio: float = 0.8
    val_ratio: float = 0.1


def normalize_address(value: str | None) -> str | None:
    if value is None:
        return None
    stripped = value.strip()
    if not ADDRESS_PATTERN.fullmatch(stripped):
        return None
    return stripped.lower()


def split_items(
    items: list[T],
    *,
    split: str,
    seed: int,
    config: DatasetSplitConfig | None = None,
) -> list[T]:
    if split not in {"train", "val", "test"}:
        raise ValueError(f"Unsupported split: {split}")
    if not items:
        return []

    split_config = config or DatasetSplitConfig()
    shuffled = list(items)
    Random(seed).shuffle(shuffled)

    train_end = int(len(shuffled) * split_config.train_ratio)
    val_end = train_end + int(len(shuffled) * split_config.val_ratio)

    if split == "train":
        return shuffled[:train_end]
    if split == "val":
        return shuffled[train_end:val_end]
    return shuffled[val_end:]


def limit_items(items: list[T], limit: int | None) -> list[T]:
    if limit is None:
        return items
    return items[:limit]


def stable_int(value: str | int | float | None, default: int = 0) -> int:
    if value is None:
        return default
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return default


def stable_float(value: str | int | float | None, default: float = 0.0) -> float:
    if value is None:
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default
