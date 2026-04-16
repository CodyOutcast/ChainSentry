from __future__ import annotations

from app.models import NormalizedTransaction, TransactionKind, TransactionRequest


SELECTOR_METHODS = {
    "0x095ea7b3": "approve",
    "0xa22cb465": "setApprovalForAll",
    "0xa9059cbb": "transfer",
    "0x23b872dd": "transferFrom",
    "0x38ed1739": "swapExactTokensForTokens",
    "0x18cbafe5": "swapExactETHForTokens",
    "0x2f2ff15d": "grantRole",
}


def parse_transaction(request: TransactionRequest) -> NormalizedTransaction:
    selector = request.calldata[:10] if request.calldata else None
    inferred_method = SELECTOR_METHODS.get(selector, "") if selector else ""
    method_name = request.method_name or inferred_method or infer_method_name(request)
    normalized_method = method_name.strip()
    method_key = normalized_method.lower()
    transaction_kind = infer_transaction_kind(method_key, request.value_eth)
    summary = build_summary(request, transaction_kind, normalized_method)

    return NormalizedTransaction(
        chain_id=request.chain_id,
        transaction_kind=transaction_kind,
        from_address=request.from_address,
        to_address=request.to_address,
        spender_address=request.spender_address,
        contract_name=request.contract_name,
        method_name=normalized_method,
        selector=selector,
        value_eth=request.value_eth,
        token_symbol=request.token_symbol,
        token_amount=request.token_amount,
        approval_amount=request.approval_amount,
        interaction_label=request.interaction_label,
        summary=summary,
    )


def infer_method_name(request: TransactionRequest) -> str:
    if request.approval_amount is not None:
        return "approve"
    if request.value_eth > 0 and not request.calldata:
        return "nativeTransfer"
    return "contractCall"


def infer_transaction_kind(method_name: str, value_eth: float) -> TransactionKind:
    if method_name in {"approve", "setapprovalforall", "increaseallowance"}:
        return TransactionKind.approval
    if "grantrole" in method_name or "setoperator" in method_name or "authorize" in method_name:
        return TransactionKind.privilege
    if "swap" in method_name:
        return TransactionKind.swap
    if "transfer" in method_name and value_eth == 0:
        return TransactionKind.transfer
    if value_eth > 0 and method_name == "nativetransfer":
        return TransactionKind.native_transfer
    return TransactionKind.contract_call


def build_summary(request: TransactionRequest, kind: TransactionKind, method_name: str) -> str:
    target = request.contract_name or shorten(request.to_address)
    spender = request.spender_address or request.to_address
    token_symbol = request.token_symbol or "tokens"

    if kind == TransactionKind.approval and method_name.lower() == "setapprovalforall":
        return f"Grant operator-wide approval to {shorten(spender)} through {target}."

    if kind == TransactionKind.approval:
        amount = describe_amount(request.approval_amount, token_symbol)
        return f"Approve {shorten(spender)} to spend {amount}."

    if kind == TransactionKind.transfer:
        amount = describe_amount(request.token_amount, token_symbol)
        return f"Transfer {amount} to {target}."

    if kind == TransactionKind.native_transfer:
        return f"Transfer {round(request.value_eth, 6)} ETH to {target}."

    if kind == TransactionKind.swap:
        amount = describe_amount(request.token_amount, token_symbol)
        return f"Swap or route {amount} through {target}."

    if kind == TransactionKind.privilege:
        return f"Grant elevated permissions through {target}."

    return f"Call {method_name} on {target}."


def describe_amount(value: float | None, symbol: str) -> str:
    if value is None:
        return symbol
    if value >= 1_000_000_000:
        return f"unlimited {symbol}"
    display = int(value) if value.is_integer() else round(value, 4)
    return f"{display} {symbol}"


def shorten(address: str) -> str:
    return f"{address[:6]}...{address[-4:]}"
