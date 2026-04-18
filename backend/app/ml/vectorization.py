from __future__ import annotations

from dataclasses import dataclass

import torch

from app.ml.features import ScalarFeatureSet
from app.ml.graph_builder import TransactionGraph


UNKNOWN_TOKEN = "__unknown__"


@dataclass(frozen=True)
class FeatureVocabulary:
    numeric_keys: tuple[str, ...]
    boolean_keys: tuple[str, ...]
    categorical_keys: tuple[str, ...]
    node_types: tuple[str, ...]
    edge_types: tuple[str, ...]
    categorical_values: dict[str, tuple[str, ...]]

    def to_dict(self) -> dict[str, object]:
        return {
            "numeric_keys": list(self.numeric_keys),
            "boolean_keys": list(self.boolean_keys),
            "categorical_keys": list(self.categorical_keys),
            "node_types": list(self.node_types),
            "edge_types": list(self.edge_types),
            "categorical_values": {key: list(values) for key, values in self.categorical_values.items()},
        }

    @classmethod
    def from_dict(cls, payload: dict[str, object]) -> "FeatureVocabulary":
        raw_values = payload["categorical_values"]
        assert isinstance(raw_values, dict)
        categorical_values = {
            str(key): tuple(str(item) for item in values)
            for key, values in raw_values.items()
        }
        return cls(
            numeric_keys=tuple(str(item) for item in payload["numeric_keys"]),
            boolean_keys=tuple(str(item) for item in payload["boolean_keys"]),
            categorical_keys=tuple(str(item) for item in payload["categorical_keys"]),
            node_types=tuple(str(item) for item in payload["node_types"]),
            edge_types=tuple(str(item) for item in payload["edge_types"]),
            categorical_values=categorical_values,
        )


@dataclass(frozen=True)
class NormalizationStats:
    means: dict[str, float]
    stds: dict[str, float]

    def to_dict(self) -> dict[str, object]:
        return {"means": self.means, "stds": self.stds}

    @classmethod
    def from_dict(cls, payload: dict[str, object]) -> "NormalizationStats":
        raw_means = payload["means"]
        raw_stds = payload["stds"]
        assert isinstance(raw_means, dict)
        assert isinstance(raw_stds, dict)
        return cls(
            means={str(key): float(value) for key, value in raw_means.items()},
            stds={str(key): float(value) for key, value in raw_stds.items()},
        )


@dataclass(frozen=True)
class EncodedGraphSample:
    node_type_ids: torch.Tensor
    edge_index: torch.Tensor
    edge_type_ids: torch.Tensor
    anchor_index: int
    numeric: torch.Tensor
    boolean: torch.Tensor
    categorical: torch.Tensor


def build_feature_vocabulary(
    graphs: list[TransactionGraph],
    feature_sets: list[ScalarFeatureSet],
) -> FeatureVocabulary:
    if not feature_sets:
        raise ValueError("Cannot build a feature vocabulary without feature sets.")

    numeric_keys = tuple(sorted(feature_sets[0].numeric.keys()))
    boolean_keys = tuple(sorted(feature_sets[0].boolean.keys()))
    categorical_keys = tuple(sorted(feature_sets[0].categorical.keys()))

    node_types = tuple(sorted({node.node_type for graph in graphs for node in graph.nodes}))
    edge_types = tuple(sorted({edge.edge_type for graph in graphs for edge in graph.edges}))

    categorical_values: dict[str, tuple[str, ...]] = {}
    for key in categorical_keys:
        values = {UNKNOWN_TOKEN}
        for feature_set in feature_sets:
            values.add(feature_set.categorical.get(key, UNKNOWN_TOKEN))
        categorical_values[key] = tuple(sorted(values))

    return FeatureVocabulary(
        numeric_keys=numeric_keys,
        boolean_keys=boolean_keys,
        categorical_keys=categorical_keys,
        node_types=node_types,
        edge_types=edge_types,
        categorical_values=categorical_values,
    )


def fit_normalization(
    feature_sets: list[ScalarFeatureSet],
    numeric_keys: tuple[str, ...],
) -> NormalizationStats:
    if not feature_sets:
        raise ValueError("Cannot fit normalization statistics without feature sets.")

    means: dict[str, float] = {}
    stds: dict[str, float] = {}
    for key in numeric_keys:
        values = [float(feature_set.numeric.get(key, 0.0)) for feature_set in feature_sets]
        mean = sum(values) / len(values)
        variance = sum((value - mean) ** 2 for value in values) / max(len(values), 1)
        std = variance ** 0.5
        means[key] = mean
        stds[key] = std if std > 1e-6 else 1.0
    return NormalizationStats(means=means, stds=stds)


def encode_sample(
    graph: TransactionGraph,
    features: ScalarFeatureSet,
    vocabulary: FeatureVocabulary,
    normalization: NormalizationStats,
) -> EncodedGraphSample:
    node_type_to_id = {node_type: index for index, node_type in enumerate(vocabulary.node_types)}
    edge_type_to_id = {edge_type: index for index, edge_type in enumerate(vocabulary.edge_types)}
    categorical_id_maps = {
        key: {value: index for index, value in enumerate(values)}
        for key, values in vocabulary.categorical_values.items()
    }

    node_id_to_index = {node.node_id: index for index, node in enumerate(graph.nodes)}
    anchor_index = node_id_to_index[graph.anchor_node_id]

    node_type_ids = torch.tensor(
        [node_type_to_id[node.node_type] for node in graph.nodes],
        dtype=torch.long,
    )

    edge_pairs = [(node_id_to_index[edge.source_id], node_id_to_index[edge.target_id]) for edge in graph.edges]
    edge_index = torch.tensor(edge_pairs, dtype=torch.long) if edge_pairs else torch.empty((0, 2), dtype=torch.long)
    edge_type_ids = torch.tensor(
        [edge_type_to_id[edge.edge_type] for edge in graph.edges],
        dtype=torch.long,
    ) if graph.edges else torch.empty((0,), dtype=torch.long)

    numeric_values = [
        _normalize_numeric_value(features.numeric.get(key, 0.0), normalization.means[key], normalization.stds[key])
        for key in vocabulary.numeric_keys
    ]
    boolean_values = [1.0 if features.boolean.get(key, False) else 0.0 for key in vocabulary.boolean_keys]
    categorical_values = [
        categorical_id_maps[key].get(features.categorical.get(key, UNKNOWN_TOKEN), 0)
        for key in vocabulary.categorical_keys
    ]

    return EncodedGraphSample(
        node_type_ids=node_type_ids,
        edge_index=edge_index,
        edge_type_ids=edge_type_ids,
        anchor_index=anchor_index,
        numeric=torch.tensor(numeric_values, dtype=torch.float32),
        boolean=torch.tensor(boolean_values, dtype=torch.float32),
        categorical=torch.tensor(categorical_values, dtype=torch.long),
    )


def _normalize_numeric_value(value: float, mean: float, std: float) -> float:
    return (float(value) - mean) / std
