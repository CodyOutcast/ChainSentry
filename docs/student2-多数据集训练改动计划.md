# Student 2 多数据集训练方案与落地记录

## 1. 目标

这份文档最初用于说明如何把 Student 2 的训练路径从 synthetic baseline 迁移到多数据集训练。  
当前仓库已经按这条路线完成了第一版落地实现，因此这里统一记录**现在实际采用的训练目标、输入输出边界和固定配置**。

当前训练路径改为：

- 从 `/data` 中选择**真正可用且与项目目标相关**的数据集。
- 为不同格式的数据集设计 **dataset adaptor + multi-dataset dataloader**。
- 让这些 adaptor 统一产出同一种 `UnifiedTrainingSample`。
- 用**同一个 backbone**进行训练，但允许按数据集挂接不同 supervision head。
- 保留当前 ChainSentry 的主任务定义：
  - 3 个风险 head：`approval` / `destination` / `simulation`
  - 1 个严重度 head：`severity`
  - 辅助 head 只用于训练，不直接暴露给前端

这里的“同一个 backbone”指的是：  
**不同数据集共享一套图编码器和标量特征编码器，不为每个数据集单独造一套 backbone。**

当前项目里这四层边界必须区分清楚：

- 外部 API 输入：`TransactionRequest`
- 模型内部输入：`TransactionGraph + ScalarFeatureSet`
- 模型主输出：`approval` / `destination` / `simulation` 三个风险分数，加一个 `severity` 分布
- 后端最终输出：`AnalysisResponse`

---

## 2. 当前问题

### 2.1 现状

上一阶段 synthetic baseline 的训练流程是：

1. synthetic request
2. parser / simulation / detector
3. graph builder / scalar features
4. graph model training
5. 后端把模型输出和 heuristic findings 组装为最终 `AnalysisResponse`

这条链路能证明“训练、推理、部署闭环”已经存在，但问题也很明显：

- 训练数据是 synthetic，不是真实链上样本。
- 标签来自当前 detector 的 pseudo-label，无法证明模型学到了真实世界风险。
- 训练集分布完全由我们手工设定，容易过拟合到 demo 逻辑。

### 2.2 为什么不能简单把 `/data` 全塞进去

`/data` 里的数据并不是同一种监督任务，至少分成四类：

- **地址标签型**：给出 address / contract 是否可疑。
- **交易钓鱼型**：给出 phishing transaction 或 benign transaction。
- **失败交易型**：给出 reverted tx 与 invariant/failure 信息。
- **地址行为型**：给出地址级 scam label 与时序/图行为。

它们在以下维度都不一致：

- 样本粒度不同：地址级 / 交易级 / 合约级 / 时序窗口级
- 标签空间不同：binary / multi-label / invariant class / weak label
- 可用字段不同：有的只有 address，有的有 from/to/input/value，有的还有时序和图边
- 数据质量不同：有些文件在当前仓库里只是 LFS pointer，并没有真实 payload

所以这里不应该做“一个通用 CSV reader”，而应该做：

- **每个数据集一个 adaptor**
- **所有 adaptor 输出统一样本对象**
- **训练时用 shared backbone + task-aware head**

---

## 3. 数据集分层与纳入策略

## 3.1 第一阶段纳入的数据集

这些数据集和当前 ChainSentry 任务最接近，而且当前仓库内可直接读取或较容易解析。

### A. `forta-labelled-datasets`

用途：

- `destination` 风险的正样本
- 可疑 spender / malicious contract 的正样本
- approval 场景中的可疑目标监督

特点：

- CSV，字段清晰
- 包含 `phishing_scams.csv`
- 包含 `malicious_smart_contracts.csv`
- 包含 `etherscan_malicious_labels.csv`

建议角色：

- 主要正样本来源
- address / contract 级 destination-risk supervision

### B. `eth-labels`

用途：

- 提供大规模地址元数据
- 可作为“已知协议地址 / 已知 benign 实体”的弱负样本来源

注意：

- 这个库本身不是恶意标签库，不能把所有地址都当 negative
- 应只挑选高置信 benign 标签，例如头部协议、router、vault、bridge、官方多签等

建议角色：

- weak negative pool
- metadata enrichment

### C. `EtherScamDB`

用途：

- 从 `scams.yaml` 中抽取诈骗地址与诈骗域名对应信息
- 增强 phishing / scam address 正样本

建议角色：

- destination-risk 正样本补充
- explanation provenance 补充来源

### D. `PTXPhish`

用途：

- 提供真实交易钓鱼样本与 benign 样本
- 非常适合对 `approval` / `simulation` 风险做交易级监督

特点：

- 粒度是交易型，不是单纯地址型
- 包含 benign 类
- 包含多个 phishing family：approve / permit / setApprovalForAll / payable / address poisoning 等

建议角色：

- transaction-risk 主要训练集
- approval-risk / simulation-risk 的真实监督来源

### E. `raven-dataset`

用途：

- failed transaction + failure message + invariant
- 不直接等于 ChainSentry 的三类风险，但可作为辅助任务

建议角色：

- auxiliary pretraining dataset
- 训练 backbone 学习 transaction failure / anomaly 语义

不建议：

- 直接把 `RAVEN` 的 `invariant_id` 强行映射为 `approval/destination/simulation`

---

## 3.2 第二阶段再考虑的数据集

### F. `ethereum_fraud_dataset_by_activity`

这个数据集理论上很有价值，但当前仓库里多个 `.parquet` 文件实际上仍是 **Git LFS pointer**，不是实体数据。

现状表现：

- `targets_global.parquet` 是文本指针
- `edges.parquet` 是文本指针
- `weekly.parquet` / `monthly.parquet` 也是文本指针

因此当前阶段不要把它纳入主训练计划。  
等真实 payload 拉齐后，再考虑：

- 地址级二分类预训练
- 时间窗口图编码
- 时序特征蒸馏到 ChainSentry backbone

### G. `forta-malicious-smart-contract-dataset`

这个数据集适合作为 contract-level 恶意分类辅助任务，但当前仓库状态需要先确认文件完整性和可读性。  
在没有稳定读取和字段映射前，不进入第一阶段训练主线。

---

## 4. 统一样本接口设计

## 4.1 新的统一样本对象

建议新增：

```python
@dataclass(frozen=True)
class UnifiedTrainingSample:
    dataset_name: str
    sample_id: str
    normalized_transaction: NormalizedTransaction
    simulation: SimulationSummary
    graph: TransactionGraph
    features: ScalarFeatureSet
    binary_targets: dict[str, float]
    binary_target_mask: dict[str, bool]
    multiclass_targets: dict[str, int]
    multiclass_target_mask: dict[str, bool]
    sample_weight: float
    metadata: dict[str, str | int | float | bool | None]
```

字段含义：

- `dataset_name`：样本来自哪个数据集
- `sample_id`：样本的稳定标识
- `normalized_transaction` / `simulation`：与当前推理路径一致的中间表示
- `graph`：统一图表示
- `features`：统一标量特征
- `binary_targets` / `multiclass_targets`：按 head 分开的标签字典
- `binary_target_mask` / `multiclass_target_mask`：哪些 head 对该样本生效
- `sample_weight`：平衡不同数据集大小与噪声
- `metadata`：保留原始来源、family、reason、label source

---

## 4.2 为什么必须有 `target_mask`

不同数据集不可能都提供同一套标签。

例如：

- Forta / EtherScamDB 地址标签主要监督 `destination` 与 `address_malicious`
- PTXPhish 主要监督 `approval` / `simulation` / `severity`
- Eth Labels 主要提供高置信 benign negative，监督主任务的 0 类样本
- RAVEN 只监督辅助 head `failure_aux`

所以每个样本都必须带：

- 哪些 target 可训练
- 哪些 target 要跳过 loss

否则就会出现：

- 拿没有 `approval` 标签的地址样本去训练 approval head
- 拿只有 failure message 的样本去训练 severity

这会直接污染训练。

---

## 5. Dataset Adaptor 设计

## 5.1 Adaptor 接口

建议新增统一 adaptor 协议：

```python
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
```

每个 adaptor 负责：

1. 读取自己的原始文件
2. 过滤无效记录
3. 做必要字段清洗
4. 映射到统一样本对象
5. 生成图和特征
6. 产出 target 与 target mask

---

## 5.2 第一阶段需要的 adaptor

建议新增：

- `FortaLabelAdaptor`
- `EthLabelsAdaptor`
- `EtherScamDbAdaptor`
- `PTXPhishAdaptor`
- `RavenAdaptor`

建议目录：

```text
backend/app/ml/training/
  unified_sample.py
  multi_dataset.py
  adaptors/
    base.py
    forta_labels.py
    eth_labels.py
    etherscamdb.py
    ptxphish.py
    raven.py
```

---

## 5.3 各 adaptor 的映射策略

### Forta / EtherScamDB

样本粒度：

- 地址级 / 合约级

映射方式：

- 构造一个**最小交易图**，把被标记地址放在 `to_address` 或 `spender_address`
- transaction anchor 可以是 synthetic shell，但**标签不再 synthetic**

用途：

- 监督 `destination`
- 在 approval-like shell 下补充 malicious spender 正样本

输出：

- `targets = {"destination": 1}`
- `target_mask = {"destination": True, "approval": False, "simulation": False, "severity": False}`

### Eth Labels

样本粒度：

- 地址级

映射方式：

- 只挑高置信 benign 标签作为 negative
- 同样构造 minimal graph

输出：

- `targets = {"destination": 0}`
- 只训练 `destination`

### PTXPhish

样本粒度：

- 当前仓库本地可直接使用的是 **family + 初始地址表**，不是完整原始交易表

映射方式：

- 从 family 恢复当前项目支持的交易壳语义
- 生成 transaction-like minimal graph
- 对 approve / permit / setApprovalForAll 等 family 直接映射到 `approval`
- 对 payable / hidden-outflow / privilege-like family 映射到 `simulation`
- 对 address poisoning 相关 family 映射到 `destination`
- `setApprovalForAll` 这类 family 可以同时监督 `approval` + `simulation`

输出建议：

- `approval`
- `simulation`
- `severity`

注意：

- 这里不是恢复完整真实链上原始交易，而是基于本地 family/address 表构造最小合法训练壳
- 如果后续拿到完整 PTXPhish 明细交易表，再升级为更强的 transaction reconstruction

### RAVEN

样本粒度：

- 失败交易级

映射方式：

- 用 `from_address / to_address / tx_input / gas_limit / gas_used / failure_message / failure_invariant`
  构图
- 增加 `effect` 或 `failure` 节点
- 不直接训练 ChainSentry 最终 head，而是训练辅助 head

输出建议：

- `failure_aux`

不直接参与：

- `severity`
- `approval`
- `destination`

---

## 6. 统一图构造策略

## 6.1 不同数据集不能各自发明图 schema

当前 repo 已经有统一的 transaction-centered graph：

- node types:
  - `transaction`
  - `address`
  - `contract`
  - `token`
  - `effect`
- edge types:
  - `initiates`
  - `targets`
  - `approves`
  - `requests_allowance_for`
  - `transfers_value_to`
  - `transfers_token_to`
  - `routes_to`
  - `grants_operator_to`
  - `grants_privilege_to`
  - `triggers_effect`

这个 schema 应继续保留，避免训练和推理图结构脱节。

---

## 6.2 对“字段不完整”的数据集怎么处理

原则：

- 不完整不等于不能用
- 但要构造**最小合法图**

例如地址标签型样本可构造：

- 1 个 transaction node
- 1 个 initiator address node
- 1 个 target contract/address node
- 可选的 spender node
- 可选的 token node

这样做的好处：

- 所有样本仍然走同一个 backbone
- 推理期 backbone 不需要换
- 数据集之间只是信息密度不同，不是 schema 完全不同

---

## 7. Shared Backbone + Multi-Head 训练方案

## 7.1 为什么不能只保留一个统一输出头

如果强行让所有数据集都训练同一套主任务：

- 3 个风险 head：`approval` / `destination` / `simulation`
- 1 个严重度 head：`severity`

会出现两个问题：

- 很多数据集没有这些标签
- 一些数据集标签语义和最终任务并不等价

因此推荐：

- **共享 backbone**
- **按任务挂多个 head**

---

## 7.2 推荐 head 设计

### 主任务 head

- `approval_head`
- `destination_head`
- `simulation_head`
- `severity_head`

其中前 3 个是风险 head，`severity_head` 是严重度 head。  
这些 head 的输出属于**模型内部输出**，后端仍需再组装成最终 API 响应。

### 辅助任务 head

- `address_malicious_head`
- `failure_aux_head`

这些 head 用来吸收其他数据集的监督信号，但不直接暴露给前端。

---

## 7.3 训练时的 loss 计算

每个 batch 中：

- backbone 总是跑
- 只有 `target_mask=True` 的 head 才参与 loss

示意：

```python
total_loss = 0.0
for head_name, enabled in sample.target_mask.items():
    if not enabled:
        continue
    total_loss += loss_fns[head_name](predictions[head_name], sample.targets[head_name])
```

如果同一 batch 里混不同数据集：

- 用 `collate_fn` 返回样本列表
- 逐样本 forward + 累积 loss

---

## 8. Multi-Dataset DataLoader 设计

## 8.1 不建议先做复杂 batched graph training

当前 `RelationAwareGraphModel.forward(...)` 还是单样本编码风格。  
如果一开始就做真正的 batched heterogeneous graph，会把这次重构范围拉太大。

因此第一阶段建议：

- dataloader 层面完成多数据集采样
- `collate_fn` 返回样本列表
- 训练循环逐样本编码并累计 loss

也就是：

- **先解决真实数据训练**
- **暂不强推图 batch 化**

---

## 8.2 推荐 dataloader 结构

建议新增：

```python
class MultiDatasetTrainingSet(Dataset):
    ...

class WeightedDatasetSampler(Sampler[int]):
    ...
```

设计要点：

- 每个 adaptor 先产出 `list[UnifiedTrainingSample]`
- 合并到一个统一 dataset 容器
- 每个样本带 `dataset_name`
- sampler 按数据集权重采样，而不是按原始数量硬拼

---

## 8.3 为什么必须做 weighted sampling

不同数据集大小差别极大：

- `eth-labels` 很大
- `EtherScamDB` 较小
- `RAVEN` 中等
- `PTXPhish` 中等

如果直接拼接：

- 大数据集会淹没小数据集
- shared backbone 会主要学到 easy shortcut

当前已落地的是**按数据集权重的 sampler**，默认混合比例与代码保持一致：

- Forta: `0.24`
- PTXPhish: `0.22`
- Eth Labels: `0.22`
- EtherScamDB: `0.12`
- RAVEN: `0.20`

这样做的目的不是让大数据集完全主导，而是让：

- Forta / PTXPhish 继续承担主任务监督
- Eth Labels 提供足够多的高置信 benign 负样本
- RAVEN 保持辅助 failure 语义
- EtherScamDB 作为 destination 正样本补充

---

## 8.4 当前已定版的模型与训练配置

当前仓库中已经固定下来的第一版配置如下：

- backbone：`RelationAwareGraphModel`
- relation layers：`2`
- hidden dim：`64`
- categorical embedding dim：`8`
- feature hidden dim：`48`
- head hidden dim：`64`
- dropout：`0.15`
- optimizer：`AdamW`
- learning rate：`3e-4`
- weight decay：`1e-4`
- batch size：`16`
- gradient clip norm：`1.0`

当前 loss 设计：

- `approval` / `destination` / `simulation` / `address_malicious` / `failure_aux`：masked `BCEWithLogitsLoss`
- `severity`：masked `CrossEntropyLoss`
- binary heads 使用按训练集统计得到的 `pos_weight`
- `severity` 使用按类别频次反比构造的 class weight

当前 loss 权重：

- `approval`: `1.0`
- `destination`: `1.0`
- `simulation`: `1.0`
- `address_malicious`: `0.5`
- `failure_aux`: `0.35`
- `severity`: `0.75`

当前 threshold 选择方式：

- 在 validation split 上搜索 `0.30` 到 `0.65`
- 以每个主风险 head 的 F1 最优阈值作为最终阈值
- 辅助 head 固定使用 `0.5`

---

## 9. 数据清洗与标签策略

## 9.1 正负样本定义

### 正样本

- Forta phishing / malicious contract / malicious label
- EtherScamDB 中能抽取到 address 的 scam entries
- PTXPhish 中非 benign 样本

### 负样本

- PTXPhish benign
- Eth Labels 中高置信 benign 地址

### 不直接做负样本

- Forta 未出现地址
- Eth Labels 全量地址
- EtherScamDB 未标注地址记录

原因：

- “没有被标成恶意”不等于“就是安全”

---

## 9.2 Severity 标签不要随便硬造

`severity` 目前是 ChainSentry 最终输出的一部分，但它不是第 4 类风险。  
真实数据集中大多数并没有 severity 标注，所以第一阶段建议：

- 只在 PTXPhish 和高置信 benign shell 上训练 `severity`
- Forta / EtherScamDB 默认不直接监督 severity

否则 severity 会再次退化为人工伪造标签。

---

## 10. 分阶段实施计划

## 10.1 Phase 1: 最小可行版本

目标：

- 去掉 synthetic 作为主训练集
- 跑通真实数据多源训练

范围：

- Forta
- Eth Labels
- EtherScamDB
- PTXPhish
- RAVEN

交付：

- adaptor 基础设施
- unified sample
- multi-dataset dataloader
- shared backbone + multi-head training loop
- 新训练文档

---

## 10.2 Phase 2: 对齐最终 ChainSentry 任务

目标：

- 让 backbone 学到的数据分布更贴近推理期输入

动作：

- 增强 PTXPhish adaptor，把更多 phishing family 映射到 approval/simulation
- 从 Forta / EtherScamDB 中构造 approval shell / destination shell
- 为 explanation 增加 provenance metadata

---

## 10.3 Phase 3: 扩展数据源

前提：

- LFS 数据真正可读
- 文件完整

动作：

- 纳入 `ethereum_fraud_dataset_by_activity`
- 尝试纳入 `forta-malicious-smart-contract-dataset`
- 引入时序辅助任务或地址行为预训练

---

## 11. 代码改动清单

## 11.1 新增文件

已新增：

```text
backend/app/ml/training/unified_sample.py
backend/app/ml/training/multi_dataset.py
backend/app/ml/training/adaptors/base.py
backend/app/ml/training/adaptors/forta_labels.py
backend/app/ml/training/adaptors/eth_labels.py
backend/app/ml/training/adaptors/etherscamdb.py
backend/app/ml/training/adaptors/ptxphish.py
backend/app/ml/training/adaptors/raven.py
backend/app/ml/training/train_multidataset_model.py
```

## 11.2 已修改的文件

- `backend/app/ml/training/external_datasets.py`
  - 保留 raw loader
  - 不再承担标签映射逻辑

- `backend/app/ml/training/__init__.py`
  - 导出新的训练入口

- `backend/app/ml/model.py`
  - 从当前主任务 3 风险 head + 1 severity head 扩成 shared backbone + optional auxiliary heads

- `backend/app/ml/inference.py`
  - 推理只保留主任务 head，不暴露辅助 head

- `backend/tests/`
  - 为 adaptor、新 dataloader、multi-task loss 增加测试

---

## 12. 验收标准

完成这轮改造后，至少要满足：

1. 不依赖 synthetic dataset 也能完成训练。
2. 至少 4 个真实数据源能通过 adaptor 进入统一训练流。
3. 所有样本都能映射成统一 `TransactionGraph + ScalarFeatureSet`。
4. 训练代码支持 shared backbone + target mask。
5. 推理接口不改，仍然兼容当前前端。
6. 文档能明确说明每个数据集训练了哪些 head，哪些没有训练。

---

## 13. 明确不做的事

这一轮不建议同时做：

- 把所有 `/data` 数据集一次性纳入
- 强行把所有数据集映射成同一个标签空间
- 一开始就重写成真正 batched heterogeneous graph engine
- 用无标签地址默认当负样本
- 继续把 severity 大量伪造出来

这些做法会让训练规模变大，但不会让结果更可信。

---

## 14. 推荐结论

这次改造的正确方向不是：

- “把 synthetic 换成一个更大的 synthetic”
- “把所有数据直接 concat”

而是：

- 用 **dataset adaptor** 处理格式差异
- 用 **unified sample** 保证 shared backbone 一致输入
- 用 **multi-dataset dataloader** 控制采样比例
- 用 **shared backbone + multi-head + target mask** 消化不同监督任务

这样改完之后，Student 2 的训练部分才算真正从“课程演示级伪数据训练”升级到“基于真实公开数据源的多源训练框架”。
