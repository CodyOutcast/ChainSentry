from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from random import Random
from typing import Iterable

from torch.utils.data import DataLoader, Dataset, Sampler

from app.ml.training.adaptors import EthLabelsAdaptor, EtherScamDbAdaptor, FortaLabelAdaptor, PTXPhishAdaptor, RavenAdaptor
from app.ml.training.adaptors.base import DEFAULT_DATA_ROOT, DatasetAdaptor
from app.ml.training.unified_sample import UnifiedTrainingSample, summarize_target_coverage


def collate_training_samples(batch: list[UnifiedTrainingSample]) -> list[UnifiedTrainingSample]:
    return batch


@dataclass(frozen=True)
class SplitSampleSummary:
    split: str
    total_samples: int
    dataset_counts: dict[str, int]
    target_coverage: dict[str, object]


class MultiDatasetTrainingSet(Dataset):
    def __init__(self, samples: list[UnifiedTrainingSample]) -> None:
        self.samples = samples
        self.dataset_to_indices: dict[str, list[int]] = defaultdict(list)
        for index, sample in enumerate(samples):
            self.dataset_to_indices[sample.dataset_name].append(index)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> UnifiedTrainingSample:
        return self.samples[index]

    def summarize(self, split: str) -> SplitSampleSummary:
        dataset_counts = Counter(sample.dataset_name for sample in self.samples)
        return SplitSampleSummary(
            split=split,
            total_samples=len(self.samples),
            dataset_counts=dict(dataset_counts),
            target_coverage=summarize_target_coverage(self.samples),
        )


class WeightedDatasetSampler(Sampler[int]):
    def __init__(
        self,
        dataset: MultiDatasetTrainingSet,
        *,
        dataset_weights: dict[str, float],
        num_samples: int,
        seed: int = 17,
    ) -> None:
        self.dataset = dataset
        self.dataset_weights = {
            name: float(dataset_weights.get(name, 1.0))
            for name, indices in dataset.dataset_to_indices.items()
            if indices
        }
        self.num_samples = num_samples
        self.seed = seed
        self.epoch = 0

    def __iter__(self) -> Iterable[int]:
        rng = Random(self.seed + self.epoch)
        dataset_names = list(self.dataset_weights)
        weights = [self.dataset_weights[name] for name in dataset_names]
        for _ in range(self.num_samples):
            dataset_name = rng.choices(dataset_names, weights=weights, k=1)[0]
            yield rng.choice(self.dataset.dataset_to_indices[dataset_name])

    def __len__(self) -> int:
        return self.num_samples

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch


def build_default_adaptors():
    return {
        "forta": FortaLabelAdaptor(data_root=DEFAULT_DATA_ROOT),
        "eth-labels": EthLabelsAdaptor(data_root=DEFAULT_DATA_ROOT),
        "etherscamdb": EtherScamDbAdaptor(data_root=DEFAULT_DATA_ROOT),
        "ptxphish": PTXPhishAdaptor(data_root=DEFAULT_DATA_ROOT),
        "raven": RavenAdaptor(data_root=DEFAULT_DATA_ROOT),
    }


def build_split_dataset(
    *,
    split: str,
    adaptors: dict[str, DatasetAdaptor],
    limits: dict[str, int] | None = None,
    seed: int = 17,
) -> MultiDatasetTrainingSet:
    samples: list[UnifiedTrainingSample] = []
    for dataset_name, adaptor in adaptors.items():
        limit = limits.get(dataset_name) if limits else None
        samples.extend(adaptor.build_samples(split=split, limit=limit, seed=seed))
    return MultiDatasetTrainingSet(samples)


def build_data_loader(
    dataset: MultiDatasetTrainingSet,
    *,
    batch_size: int,
    shuffle: bool = False,
    sampler: Sampler[int] | None = None,
) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle if sampler is None else False,
        sampler=sampler,
        collate_fn=collate_training_samples,
    )
