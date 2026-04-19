# ChainSentry Student 2 中文使用说明

这份文档面向 Student 2 和项目维护者。

如果你想看的是老师或普通演示用户该怎么使用系统，请直接看 `docs/用户使用说明.md`。

## 1. 当前状态

当前仓库已经不是“只剩 handoff 的骨架版”，而是一个完成了 Student 2 集成的可运行版本。Student 2 在不破坏 Student 1 既有前后端接口的前提下，补齐了以下内容：

- 基于当前 `TransactionRequest -> AnalysisResponse` 接口的图模型训练与推理链路
- 多数据集 adaptor、统一样本格式和 weighted dataloader
- 模型评测与 metrics 导出
- SwanLab 训练指标记录
- 后端自动加载图模型并在失败时回退到 Student 1 的 heuristic fallback
- 保持 `POST /api/v1/analyze` 的输入输出字段不变

当前实现采用的是：

- `PyTorch`：训练和推理
- 自定义 transaction graph 表达：图结构准备
- `scikit-learn`：评测指标与数据切分

## 2. 接口兼容说明

前端仍然使用原来的接口：

- `POST /api/v1/analyze`

请求字段和返回字段不需要修改，Student 1 写好的这些文件仍然兼容：

- `frontend/src/api/client.ts`
- `frontend/src/types.ts`
- `backend/app/models.py`

也就是说，前端不用改请求结构，就可以继续调用 Student 2 的模型版后端。

## 3. 你现在还需要做什么

如果你是 Student 2，当前不需要再去补新的外部 API 或再发明一套新模型。你真正需要做的是：

- 能复现训练命令
- 能解释当前 metrics
- 能跑通前后端演示
- 能说明当前范围为什么是收缩后的课程原型
- 能清楚说明当前限制

## 4. 环境准备

在项目根目录执行：

```bash
conda create -y -p .conda-envs/chainsentry python=3.12
conda run -p .conda-envs/chainsentry python -m pip install -r backend/requirements.txt
cd frontend
npm install
cd ..
```

如果你已经有现成的 Python 环境，也可以把下面命令里的 `.conda-envs/chainsentry/bin/python` 替换成你自己的解释器；但当前仓库默认按 `chainsentry` conda 环境记录和复现。

## 5. 训练图模型

在项目根目录执行：

```bash
SWANLAB_API_KEY=your_key_here \
PYTHONPATH=backend .conda-envs/chainsentry/bin/python -m app.ml.training.train_multidataset_model \
  --artifact-path backend/app/ml/artifacts/graph-model.pt \
  --metrics-path backend/app/ml/artifacts/graph-model-metrics.json \
  --epochs 18 \
  --size-profile standard \
  --enable-swanlab \
  --swanlab-project chainsentry
```

训练完成后会生成：

- `backend/app/ml/artifacts/graph-model.pt`
- `backend/app/ml/artifacts/graph-model-metrics.json`

说明：

- 当前主训练入口已经不是 synthetic 主训练集，而是多数据集训练：
  - `forta-labelled-datasets`
  - `eth-labels`
  - `EtherScamDB`
  - `PTXPhish`
  - `raven-dataset`
- 这些数据源并不共享同一种标签空间，因此系统先用 adaptor 把它们统一映射成 `TransactionGraph + ScalarFeatureSet`，再用 target mask 训练多头模型。
- 这仍然是**弱监督 + shell transaction projection** 路线，不应被表述为生产级真实链上风控训练。

当前一次完整训练后，metrics 会写入 `backend/app/ml/artifacts/graph-model-metrics.json`，同时在 SwanLab 中记录：

- 每 epoch 总 loss
- 各 head 的 loss
- 各风险 head 的 precision / recall / F1 / ROC-AUC / AP
- severity accuracy / macro F1 / per-class precision / recall / F1
- train / val / test 各数据集样本量

因此当前应以**同一次训练生成的 metrics JSON 和 SwanLab 看板**为准，而不是再引用旧 synthetic baseline 数字。

## 6. 启动后端

默认情况下，后端会优先使用图模型。

在项目根目录执行：

```bash
CHAIN_SENTRY_PREDICTOR_BACKEND=graph-model \
PYTHONPATH=backend .conda-envs/chainsentry/bin/python -m uvicorn app.main:app --reload
```

如果你想强制切回 Student 1 的启发式版本：

```bash
CHAIN_SENTRY_PREDICTOR_BACKEND=heuristic-fallback \
PYTHONPATH=backend .conda-envs/chainsentry/bin/python -m uvicorn app.main:app --reload
```

后端地址：

- `http://localhost:8000`
- 健康检查：`http://localhost:8000/health`

## 7. 启动前端

```bash
cd frontend
cp .env.example .env
npm run dev
```

前端地址：

- `http://localhost:5173`

## 8. 自动训练机制

如果后端以 `graph-model` 模式启动，但本地还没有模型 artifact，系统会自动执行一次快速训练并生成 artifact，然后继续提供服务。

这样做的目的是：

- 降低第一次运行的门槛
- 保证演示时不需要手动处理 artifact 缺失问题
- 同时保留可复现的显式训练命令

## 9. 当前模型的工作方式

当前模型流程如下：

1. 前端把交易请求发给后端。
2. 后端先走 Student 1 的 parser，把交易规范化。
3. 后端构建 transaction-centered graph。
4. 后端提取 scalar features。
5. 图模型输出：
   - 三个风险分数：`approval` / `destination` / `simulation`
   - 一个严重度分布：`low` / `medium` / `high` / `critical`
   - 可选辅助 head：`address_malicious` / `failure_aux`，但它们只用于训练
6. 系统把模型输出和 Student 1 的 heuristic findings 合并。
7. 后端再组装成最终的 risk card 响应结构。

这意味着当前系统是一个更稳的 **hybrid model + rules** 方案，而不是把 Student 1 的逻辑全部删掉。

## 10. 输出结果怎么理解

返回格式保持不变，仍然包含：

- `normalized_transaction`
- `overall_severity`
- `recommended_action`
- `summary`
- `findings`
- `simulation`

区别是：

- 现在 `findings[].evidence` 中会补充图模型分数和图结构信息
- 当 heuristic 没有命中、但模型分数高时，系统也可以生成 model-backed finding

## 11. 验证命令

后端测试：

```bash
PYTHONPATH=backend .conda-envs/chainsentry/bin/python -m pytest backend/tests -q
```

前端构建：

```bash
cd frontend
npm run build
```

## 12. 当前限制

当前版本是课程项目标准下的完整 Student 2 闭环，但仍有边界：

- 训练数据来自公开标签源 + shell transaction projection，不是端到端真实链上交易标注集
- `severity` 主要由 PTXPhish 和高置信 benign shell 监督，不代表全量风险语义都已被充分标注
- simulation 仍以 Student 1 的 baseline heuristic engine 为主，没有完全切到 Foundry trace
- 这个图模型更适合课程演示和结构化评测，不是生产级安全模型

## 13. 现在不需要做的事

当前版本下，Student 2 不需要再额外做这些事情才能完成课程原型：

- 不需要申请外部风控网站 API
- 不需要接真实链上情报平台才能演示
- 不需要把系统扩成 full static analysis 平台
- 不需要把 heuristic 全部删除

## 14. 答辩时建议怎么说

可以这样概括：

> Student 1 完成了前后端系统、交易解析和 baseline 检测；Student 2 在保持接口不变的情况下，补齐了图模型训练、artifact 导出、后端推理接入、评测和中文使用文档，使整个项目从启发式 demo 升级为带训练模型的可演示原型。
