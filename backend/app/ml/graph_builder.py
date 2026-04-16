from __future__ import annotations

from dataclasses import dataclass, field

from app.models import NormalizedTransaction, SimulationSummary, TransactionKind


GraphAttributeValue = str | int | float | bool | None


@dataclass(frozen=True)
class GraphNode:
    node_id: str
    node_type: str
    attributes: dict[str, GraphAttributeValue] = field(default_factory=dict)


@dataclass(frozen=True)
class GraphEdge:
    source_id: str
    target_id: str
    edge_type: str
    attributes: dict[str, GraphAttributeValue] = field(default_factory=dict)


@dataclass(frozen=True)
class TransactionGraph:
    anchor_node_id: str
    nodes: list[GraphNode]
    edges: list[GraphEdge]

    @property
    def node_count(self) -> int:
        return len(self.nodes)

    @property
    def edge_count(self) -> int:
        return len(self.edges)


def build_transaction_graph(
    transaction: NormalizedTransaction,
    simulation: SimulationSummary,
) -> TransactionGraph:
    nodes: dict[str, GraphNode] = {}
    edges: list[GraphEdge] = []

    anchor_node_id = _transaction_node_id(transaction)
    _add_node(
        nodes,
        GraphNode(
            node_id=anchor_node_id,
            node_type="transaction",
            attributes={
                "chain_id": transaction.chain_id,
                "transaction_kind": transaction.transaction_kind.value,
                "method_name": transaction.method_name.lower(),
                "selector": transaction.selector,
            },
        ),
    )

    initiator_id = f"address:{transaction.from_address}"
    _add_node(
        nodes,
        GraphNode(
            node_id=initiator_id,
            node_type="address",
            attributes={"role": "initiator", "chain_id": transaction.chain_id},
        ),
    )
    edges.append(GraphEdge(initiator_id, anchor_node_id, "initiates"))

    target_id = f"{_target_node_type(transaction)}:{transaction.to_address}"
    _add_node(
        nodes,
        GraphNode(
            node_id=target_id,
            node_type=_target_node_type(transaction),
            attributes={
                "role": "target",
                "chain_id": transaction.chain_id,
                "label": transaction.contract_name,
            },
        ),
    )
    edges.append(GraphEdge(anchor_node_id, target_id, "targets"))

    spender_id: str | None = None
    if transaction.spender_address:
        spender_node_type = _spender_node_type(transaction)
        spender_id = f"{spender_node_type}:{transaction.spender_address}"
        _add_node(
            nodes,
            GraphNode(
                node_id=spender_id,
                node_type=spender_node_type,
                attributes={"role": "spender", "chain_id": transaction.chain_id},
            ),
        )
        edges.append(GraphEdge(anchor_node_id, spender_id, _spender_edge_type(transaction)))

    if transaction.token_symbol:
        token_id = f"token:{transaction.chain_id}:{transaction.token_symbol.lower()}"
        _add_node(
            nodes,
            GraphNode(
                node_id=token_id,
                node_type="token",
                attributes={"symbol": transaction.token_symbol, "chain_id": transaction.chain_id},
            ),
        )
        token_edge_type = (
            "requests_allowance_for"
            if transaction.transaction_kind == TransactionKind.approval
            else "transfers_token_to"
        )
        token_amount = transaction.approval_amount if transaction.transaction_kind == TransactionKind.approval else transaction.token_amount
        edges.append(GraphEdge(anchor_node_id, token_id, token_edge_type, {"amount": token_amount or 0.0}))

    if transaction.value_eth > 0:
        native_value_node = f"token:{transaction.chain_id}:eth"
        _add_node(
            nodes,
            GraphNode(
                node_id=native_value_node,
                node_type="token",
                attributes={"symbol": "ETH", "chain_id": transaction.chain_id},
            ),
        )
        edges.append(
            GraphEdge(anchor_node_id, target_id, "transfers_value_to", {"amount": transaction.value_eth})
        )

    if transaction.transaction_kind == TransactionKind.swap and spender_id:
        edges.append(GraphEdge(anchor_node_id, spender_id, "routes_to"))

    for index, effect in enumerate(simulation.effects):
        effect_id = f"effect:{anchor_node_id}:{effect.effect_type.value}:{index}"
        _add_node(
            nodes,
            GraphNode(
                node_id=effect_id,
                node_type="effect",
                attributes={
                    "effect_type": effect.effect_type.value,
                    "summary": effect.summary,
                },
            ),
        )
        edges.append(GraphEdge(anchor_node_id, effect_id, "triggers_effect"))

    return TransactionGraph(anchor_node_id=anchor_node_id, nodes=list(nodes.values()), edges=edges)


def _add_node(nodes: dict[str, GraphNode], node: GraphNode) -> None:
    nodes[node.node_id] = node


def _transaction_node_id(transaction: NormalizedTransaction) -> str:
    method = transaction.method_name.lower()
    return f"transaction:{transaction.chain_id}:{method}:{transaction.from_address[-8:]}:{transaction.to_address[-8:]}"


def _target_node_type(transaction: NormalizedTransaction) -> str:
    if transaction.transaction_kind in {TransactionKind.transfer, TransactionKind.native_transfer}:
        return "address"
    return "contract"


def _spender_node_type(transaction: NormalizedTransaction) -> str:
    if transaction.transaction_kind in {TransactionKind.transfer, TransactionKind.native_transfer}:
        return "address"
    return "contract"


def _spender_edge_type(transaction: NormalizedTransaction) -> str:
    if transaction.transaction_kind == TransactionKind.approval and transaction.method_name.lower() == "setapprovalforall":
        return "grants_operator_to"
    if transaction.transaction_kind == TransactionKind.privilege:
        return "grants_privilege_to"
    return "approves"