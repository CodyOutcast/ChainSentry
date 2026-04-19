# Student 2 模型输入输出与 5 个案例验证

## 1. 这份文档回答什么问题

这份文档专门回答下面三个问题：

1. 当前项目的**完整输入输出**到底是什么。
2. GNN 模型本身输出什么，后端最终又返回什么。
3. 用 5 个模拟输入去跑当前训练好的模型后，系统能不能反映不同类型的问题。

本文档基于当前仓库里的已训练 artifact：

- `backend/app/ml/artifacts/graph-model.pt`
- `backend/app/ml/artifacts/graph-model-metrics.json`

验证方式不是手写推测，而是直接调用当前后端分析链路：

- `TransactionRequest`
- `parse_transaction(...)`
- `simulation_engine.simulate(...)`
- `get_predictor().predict(...)`
- `analyze_transaction(...)`

---

## 2. 先把三层输入输出说清楚

当前系统里一共有三层“输入输出”，不要混在一起。

### 2.1 外部 API 输入

前端或调用方真正提交给后端的是：

- `TransactionRequest`

核心字段包括：

- `chain_id`
- `from_address`
- `to_address`
- `method_name`
- `calldata`
- `value_eth`
- `token_symbol`
- `token_amount`
- `approval_amount`
- `spender_address`
- `contract_name`
- `interaction_label`
- `notes`
- `simulation_profile`

### 2.2 模型内部输入

后端不会把 `TransactionRequest` 直接塞进 GNN。

真正进入模型的是：

- `TransactionGraph`
- `ScalarFeatureSet`

也就是：

1. 先把请求解析成 `NormalizedTransaction`
2. 再跑 simulation
3. 再构图和抽特征
4. 然后才送进图模型

### 2.3 模型内部输出

当前模型主输出是：

- `approval` 风险分数
- `destination` 风险分数
- `simulation` 风险分数
- `severity` 严重度分布

注意：

- `severity` 不是第 4 类风险
- 它只是单独的严重度 head

当前训练里还有辅助 head：

- `address_malicious`
- `failure_aux`

但它们**只用于训练，不直接暴露给前端**。

### 2.4 后端最终输出

前端真正拿到的是：

- `AnalysisResponse`

字段包括：

- `normalized_transaction`
- `overall_severity`
- `recommended_action`
- `summary`
- `findings[]`
- `simulation`

所以一句话总结是：

**API 输入不是模型输入；模型输出也不是最终 API 输出。**

---

## 3. 一个完整输入输出样例

下面给一个**完整的端到端样例**。  
这个例子选的是“向已知可疑目标地址转账”，因为它的响应比较完整但不至于太长。

### 3.1 输入请求

```json
{
  "chain_id": 1,
  "from_address": "0x1111111111111111111111111111111111111111",
  "to_address": "0xdead00000000000000000000000000000000beef",
  "method_name": "transfer",
  "token_symbol": "USDT",
  "token_amount": 1200.0,
  "contract_name": "Flagged Recipient",
  "simulation_profile": "none"
}
```

### 3.2 模型内部输出

```json
{
  "category_scores": {
    "approval": 0.998039,
    "destination": 1.0,
    "simulation": 0.889261
  },
  "severity_scores": {
    "low": 0.016166,
    "medium": 0.0005,
    "high": 0.69169,
    "critical": 0.291643
  }
}
```

### 3.3 后端最终输出

```json
{
  "normalized_transaction": {
    "chain_id": 1,
    "transaction_kind": "transfer",
    "from_address": "0x1111111111111111111111111111111111111111",
    "to_address": "0xdead00000000000000000000000000000000beef",
    "spender_address": null,
    "contract_name": "Flagged Recipient",
    "method_name": "transfer",
    "selector": null,
    "value_eth": 0.0,
    "token_symbol": "USDT",
    "token_amount": 1200.0,
    "approval_amount": null,
    "interaction_label": null,
    "summary": "Transfer 1200 USDT to Flagged Recipient."
  },
  "overall_severity": "critical",
  "recommended_action": "reject",
  "summary": "ChainSentry found 1 risk signal. Highest severity: critical.",
  "findings": [
    {
      "id": "destination-destination",
      "category": "destination",
      "severity": "critical",
      "expected_action": "Transfer 1200 USDT to Flagged Recipient.",
      "risk_reason": "The destination is listed in ChainSentry's demo flagged-contract set as Demo Drain Contract.",
      "possible_impact": "This demo address represents a spender contract associated with post-approval token drain behavior.",
      "recommended_action": "Reject until the destination is independently verified and expected.",
      "evidence": [
        "Flagged role: destination",
        "Address: 0xdead00000000000000000000000000000000beef",
        "Dataset label: Demo Drain Contract",
        "Graph model destination score: 1.00",
        "Graph shape: 4 nodes, 3 edges",
        "Graph model severity profile: low:0.02, medium:0.00, high:0.69, critical:0.29"
      ]
    }
  ],
  "simulation": {
    "engine": "heuristic",
    "profile": "none",
    "triggered": false,
    "description": null,
    "effects": []
  }
}
```

### 3.4 这个例子说明了什么

- 模型内部确实给出了三类风险分数和严重度分布。
- 但最终不是把这些分数原样返回，而是由后端组装成 `findings` 和 `overall_severity`。
- 这个案例最终命中的是 `destination` 问题，不是 `approval` 或 `simulation`。

---

## 4. 五个模拟输入

下面 5 个案例都是直接用当前仓库里训练好的模型和后端逻辑跑出来的。

### 案例 1：普通转账基线

关注点：

- 应该是低风险
- 用来验证系统会不会把普通交易全部报红

输入：

```json
{
  "chain_id": 1,
  "from_address": "0x1111111111111111111111111111111111111111",
  "to_address": "0x3333333333333333333333333333333333333333",
  "method_name": "transfer",
  "token_symbol": "ETH",
  "token_amount": 0.15,
  "contract_name": "Friend Wallet",
  "simulation_profile": "none"
}
```

### 案例 2：大额授权，但目标地址本身不在 demo 黑名单

关注点：

- 应该主要体现 `approval` 风险
- 同时 simulation 会看到“可复用 allowance”

输入：

```json
{
  "chain_id": 1,
  "from_address": "0x1111111111111111111111111111111111111111",
  "to_address": "0x5555555555555555555555555555555555555555",
  "method_name": "approve",
  "token_symbol": "USDC",
  "approval_amount": 50000.0,
  "spender_address": "0x5555555555555555555555555555555555555555",
  "contract_name": "Demo Router",
  "simulation_profile": "none"
}
```

### 案例 3：向已知可疑地址转账

关注点：

- 应该主要体现 `destination` 风险

输入：

```json
{
  "chain_id": 1,
  "from_address": "0x1111111111111111111111111111111111111111",
  "to_address": "0xdead00000000000000000000000000000000beef",
  "method_name": "transfer",
  "token_symbol": "USDT",
  "token_amount": 1200.0,
  "contract_name": "Flagged Recipient",
  "simulation_profile": "none"
}
```

### 案例 4：`setApprovalForAll`，触发 operator control

关注点：

- 应该体现 broad operator control
- 同时出现 approval、destination、simulation 多种风险信号

输入：

```json
{
  "chain_id": 1,
  "from_address": "0x1111111111111111111111111111111111111111",
  "to_address": "0x6666666666666666666666666666666666666666",
  "method_name": "setApprovalForAll",
  "spender_address": "0x6666666666666666666666666666666666666666",
  "contract_name": "Collection Manager",
  "simulation_profile": "allowance_drain"
}
```

### 案例 5：`grantRole`，触发权限升级

关注点：

- 应该主要体现 `privilege escalation`
- 最终应落在 `simulation` 风险上

输入：

```json
{
  "chain_id": 1,
  "from_address": "0x1111111111111111111111111111111111111111",
  "to_address": "0x4444444444444444444444444444444444444444",
  "method_name": "grantRole",
  "contract_name": "Shadow Vault",
  "simulation_profile": "privilege_escalation"
}
```

---

## 5. 五个案例的实际输出对比

下表里的分数和结果都来自当前训练好的模型与后端链路。

| 案例 | 最终严重度 | 推荐动作 | approval 分数 | destination 分数 | simulation 分数 | 严重度 head 最大类 | 命中的 finding |
| --- | --- | --- | ---: | ---: | ---: | --- | --- |
| `clean_transfer` | `low` | `proceed` | 0.669220 | 1.000000 | 0.833042 | `low` | 无 |
| `large_approval_benign` | `medium` | `inspect_further` | 0.000004 | 0.564600 | 1.000000 | `low` | `approval-large`, `simulation-allowance-grant` |
| `flagged_transfer` | `critical` | `reject` | 0.998039 | 1.000000 | 0.889261 | `high` | `destination-destination` |
| `operator_control` | `critical` | `reject` | 1.000000 | 1.000000 | 1.000000 | `high` | `model-approval-signal`, `model-destination-signal`, `simulation-operator-control`, `simulation-unexpected-outflow` |
| `privilege_escalation` | `high` | `reject` | 0.995982 | 0.999996 | 1.000000 | `high` | `simulation-privilege-escalation` |

---

## 6. 这 5 个输出能不能反映不同问题

结论分两层说。

### 6.1 看最终后端输出：可以，且区分度是够的

从最终 `AnalysisResponse` 来看，这 5 个案例已经能区分出不同问题：

- 普通转账会落到 `low / proceed`
- 大额授权会落到 `medium / inspect_further`
- 已知可疑目标地址会落到 `critical / reject`
- `setApprovalForAll` 会落到多重风险叠加
- `grantRole` 会落到权限升级语义

也就是说，**最终用户看到的结果是能反映不同问题类型的**。

### 6.2 看模型内部原始分数：还不能直接单独当最终判断

这里有一个很重要的观察：

- 在 `clean_transfer` 这个低风险案例里，模型原始分数其实并不低：
  - `approval = 0.669220`
  - `destination = 1.000000`
  - `simulation = 0.833042`
- 但因为：
  - 严重度 head 的 dominant class 是 `low`
  - 又没有 heuristic finding 命中
  - 后处理规则也不会随便生成 user-facing finding
- 所以最终输出仍然是 `low`

这说明当前系统应该被准确描述为：

**hybrid model + rules + post-processing**

而不是：

**只看 GNN 三个风险分数就直接下结论**

---

## 7. 更准确的结论

如果你现在要向别人解释这个系统，最准确的说法是：

1. 外部输入是固定 schema 的 `TransactionRequest`。
2. 后端把它转成 `TransactionGraph + ScalarFeatureSet` 送进图模型。
3. 图模型输出 3 个风险分数和 1 个严重度分布。
4. 后端把模型结果和 heuristic / simulation 结果合并，最后生成 `AnalysisResponse`。
5. 因此，当前系统的“可解释输出”主要来自**模型 + 规则 + 后处理**的联合结果。

---

## 8. 当前这 5 个案例最值得记住的两个结论

### 结论 A

**最终 API 输出已经能反映不同问题。**

它能区分：

- 低风险普通转账
- 中风险大额授权
- 高风险/极高风险可疑目标地址交互
- operator control
- privilege escalation

### 结论 B

**模型内部三个风险 head 目前还没有好到可以脱离后处理单独使用。**

所以当前对外描述不应该说：

- “模型只输出 4 类风险置信度，系统直接按这个给结果”

更准确的说法应该是：

- “模型输出 3 个风险分数和 1 个严重度分布，后端再结合规则和 simulation 组装最终风险报告”

---

## 9. 推荐你后续怎么用这份文档

如果你是为了答辩或和队友统一口径，直接用下面这句话：

> 当前 ChainSentry 的外部输入是 `TransactionRequest`，模型内部输入是 `TransactionGraph + ScalarFeatureSet`。模型输出 3 个风险分数和 1 个严重度分布，但最终给前端的不是裸分数，而是后端融合 heuristic、simulation 和模型结果之后生成的 `AnalysisResponse`。从 5 个实际案例看，最终系统已经能区分普通转账、大额授权、可疑目标地址、operator control 和 privilege escalation 这几类不同问题；但模型内部风险分数还不能脱离后处理单独作为最终判断依据。

