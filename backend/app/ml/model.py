from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Protocol

import torch
from torch import nn
from torch.nn import functional as F

from app.ml.features import ScalarFeatureSet
from app.ml.graph_builder import TransactionGraph
from app.ml.vectorization import FeatureVocabulary, NormalizationStats, encode_sample


SEVERITY_LABELS = ("low", "medium", "high", "critical")
MAIN_BINARY_HEADS = ("approval", "destination", "simulation")


@dataclass(frozen=True)
class GraphModelMetadata:
    architecture: str
    framework: str
    artifact_path: Path
    node_types: tuple[str, ...] = (
        "transaction",
        "address",
        "contract",
        "token",
        "effect",
    )
    edge_types: tuple[str, ...] = (
        "initiates",
        "targets",
        "approves",
        "requests_allowance_for",
        "transfers_value_to",
        "transfers_token_to",
        "routes_to",
        "grants_operator_to",
        "grants_privilege_to",
        "triggers_effect",
    )
    task_heads: tuple[str, ...] = ("approval", "destination", "simulation", "severity")
    auxiliary_binary_heads: tuple[str, ...] = ()


@dataclass(frozen=True)
class GraphPredictionScores:
    category_scores: dict[str, float] = field(default_factory=dict)
    severity_scores: dict[str, float] = field(default_factory=dict)


@dataclass(frozen=True)
class GraphForwardOutput:
    binary_logits: dict[str, torch.Tensor] = field(default_factory=dict)
    multiclass_logits: dict[str, torch.Tensor] = field(default_factory=dict)


class GraphRiskModel(Protocol):
    metadata: GraphModelMetadata

    def predict(self, graph: TransactionGraph, features: ScalarFeatureSet) -> GraphPredictionScores:
        ...


@dataclass(frozen=True)
class RelationAwareGraphModelConfig:
    hidden_dim: int = 64
    relation_layers: int = 2
    categorical_embedding_dim: int = 8
    feature_hidden_dim: int = 48
    head_hidden_dim: int = 64
    dropout: float = 0.15


class RelationAwareGraphModel(nn.Module):
    def __init__(
        self,
        *,
        num_node_types: int,
        num_edge_types: int,
        categorical_cardinalities: list[int],
        numeric_dim: int,
        boolean_dim: int,
        auxiliary_binary_heads: tuple[str, ...] = (),
        config: RelationAwareGraphModelConfig,
    ) -> None:
        super().__init__()
        self.config = config
        self.main_binary_heads = MAIN_BINARY_HEADS
        self.auxiliary_binary_heads = auxiliary_binary_heads
        self.node_embedding = nn.Embedding(num_node_types, config.hidden_dim)
        self.self_layers = nn.ModuleList(
            [nn.Linear(config.hidden_dim, config.hidden_dim) for _ in range(config.relation_layers)]
        )
        self.relation_layers = nn.ModuleList(
            [
                nn.ModuleList(
                    [nn.Linear(config.hidden_dim, config.hidden_dim, bias=False) for _ in range(num_edge_types)]
                )
                for _ in range(config.relation_layers)
            ]
        )
        self.categorical_embeddings = nn.ModuleList(
            [nn.Embedding(cardinality, config.categorical_embedding_dim) for cardinality in categorical_cardinalities]
        )
        feature_input_dim = numeric_dim + boolean_dim + len(categorical_cardinalities) * config.categorical_embedding_dim
        self.feature_encoder = nn.Sequential(
            nn.Linear(feature_input_dim, config.feature_hidden_dim),
            nn.ReLU(),
            nn.Dropout(config.dropout),
        )
        combined_dim = config.hidden_dim * 3 + config.feature_hidden_dim
        self.binary_heads = nn.ModuleDict(
            {
                head_name: _build_binary_head(combined_dim, config.head_hidden_dim, config.dropout)
                for head_name in self.main_binary_heads + self.auxiliary_binary_heads
            }
        )
        self.multiclass_heads = nn.ModuleDict(
            {
                "severity": _build_multiclass_head(
                    combined_dim,
                    config.head_hidden_dim,
                    len(SEVERITY_LABELS),
                    config.dropout,
                )
            }
        )

    def forward(self, sample) -> GraphForwardOutput:
        node_states = self.node_embedding(sample.node_type_ids)
        for layer_index in range(self.config.relation_layers):
            messages = torch.zeros_like(node_states)
            if sample.edge_type_ids.numel() > 0:
                for relation_index, relation_layer in enumerate(self.relation_layers[layer_index]):
                    relation_mask = sample.edge_type_ids == relation_index
                    if not torch.any(relation_mask):
                        continue
                    relation_edges = sample.edge_index[relation_mask]
                    source_indices = relation_edges[:, 0]
                    target_indices = relation_edges[:, 1]
                    transformed = relation_layer(node_states[source_indices])
                    messages.index_add_(0, target_indices, transformed)
            node_states = F.relu(self.self_layers[layer_index](node_states) + messages)

        anchor_state = node_states[sample.anchor_index]
        graph_mean = node_states.mean(dim=0)
        graph_max = node_states.max(dim=0).values

        feature_parts = [sample.numeric, sample.boolean]
        for index, embedding in enumerate(self.categorical_embeddings):
            feature_parts.append(embedding(sample.categorical[index].unsqueeze(0)).squeeze(0))
        feature_vector = torch.cat(feature_parts, dim=0)
        encoded_features = self.feature_encoder(feature_vector)

        combined = torch.cat((anchor_state, graph_mean, graph_max, encoded_features), dim=0)
        binary_logits = {
            head_name: head(combined).squeeze(-1)
            for head_name, head in self.binary_heads.items()
        }
        multiclass_logits = {
            head_name: head(combined)
            for head_name, head in self.multiclass_heads.items()
        }
        return GraphForwardOutput(binary_logits=binary_logits, multiclass_logits=multiclass_logits)


class LoadedGraphRiskModel:
    def __init__(
        self,
        *,
        metadata: GraphModelMetadata,
        vocabulary: FeatureVocabulary,
        normalization: NormalizationStats,
        config: RelationAwareGraphModelConfig,
        thresholds: dict[str, float],
        model: RelationAwareGraphModel,
        metrics: dict[str, object] | None = None,
    ) -> None:
        self.metadata = metadata
        self.vocabulary = vocabulary
        self.normalization = normalization
        self.config = config
        self.thresholds = thresholds
        self.model = model
        self.metrics = metrics or {}
        self.model.eval()

    @classmethod
    def load(cls, artifact_path: Path) -> "LoadedGraphRiskModel":
        artifact = torch.load(artifact_path, map_location="cpu")
        metadata = GraphModelMetadata(**artifact["metadata"])
        vocabulary = FeatureVocabulary.from_dict(artifact["vocabulary"])
        normalization = NormalizationStats.from_dict(artifact["normalization"])
        config = RelationAwareGraphModelConfig(**artifact["model_config"])
        model = RelationAwareGraphModel(
            num_node_types=len(vocabulary.node_types),
            num_edge_types=len(vocabulary.edge_types),
            categorical_cardinalities=[len(vocabulary.categorical_values[key]) for key in vocabulary.categorical_keys],
            numeric_dim=len(vocabulary.numeric_keys),
            boolean_dim=len(vocabulary.boolean_keys),
            auxiliary_binary_heads=tuple(metadata.auxiliary_binary_heads),
            config=config,
        )
        model.load_state_dict(artifact["state_dict"])
        model.eval()
        return cls(
            metadata=metadata,
            vocabulary=vocabulary,
            normalization=normalization,
            config=config,
            thresholds={str(key): float(value) for key, value in artifact["thresholds"].items()},
            model=model,
            metrics=artifact.get("metrics", {}),
        )

    def predict(self, graph: TransactionGraph, features: ScalarFeatureSet) -> GraphPredictionScores:
        encoded = encode_sample(graph, features, self.vocabulary, self.normalization)
        with torch.no_grad():
            outputs = self.model(encoded)
        category_scores = {}
        for label in MAIN_BINARY_HEADS:
            if label not in outputs.binary_logits:
                continue
            category_scores[label] = round(float(torch.sigmoid(outputs.binary_logits[label]).item()), 6)
        severity_logits = outputs.multiclass_logits["severity"]
        severity_values = torch.softmax(severity_logits, dim=0).tolist()
        return GraphPredictionScores(
            category_scores=category_scores,
            severity_scores={
                label: round(float(score), 6)
                for label, score in zip(SEVERITY_LABELS, severity_values, strict=True)
            },
        )


def save_graph_model_artifact(
    *,
    artifact_path: Path,
    metadata: GraphModelMetadata,
    vocabulary: FeatureVocabulary,
    normalization: NormalizationStats,
    model_config: RelationAwareGraphModelConfig,
    model_state: dict[str, torch.Tensor],
    thresholds: dict[str, float],
    metrics: dict[str, object],
) -> None:
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "metadata": {**asdict(metadata), "artifact_path": str(metadata.artifact_path)},
            "vocabulary": vocabulary.to_dict(),
            "normalization": normalization.to_dict(),
            "model_config": asdict(model_config),
            "state_dict": model_state,
            "thresholds": thresholds,
            "metrics": metrics,
        },
        artifact_path,
    )


def _build_binary_head(input_dim: int, hidden_dim: int, dropout: float) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(input_dim, hidden_dim),
        nn.ReLU(),
        nn.Dropout(dropout),
        nn.Linear(hidden_dim, 1),
    )


def _build_multiclass_head(input_dim: int, hidden_dim: int, output_dim: int, dropout: float) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(input_dim, hidden_dim),
        nn.ReLU(),
        nn.Dropout(dropout),
        nn.Linear(hidden_dim, output_dim),
    )
