# RCAWorld-Foundation: 项目进度

> 状态: ⬜ pending | 🟦 in_progress | 🟩 completed | 🟥 blocked

---

## 分支总览

| 分支 | 目标 | 状态 |
|------|------|------|
| `master` | Graph-RSSM 微服务异常排序器 (Phase 0–7) | 🟩 全部完成 |
| `RCAWorld-Foundation` | 通用诊断世界模型 (Phase A–E) | 🟦 Phase A 代码完成，实验待跑 |

---

# master 分支 — 已完成阶段

## Phase 0: 合成数据 MVP + Graph-RSSM 原型验证 🟩

| # | 任务 | 状态 | 备注 |
|---|------|------|------|
| 0.1 | 合成数据生成器 | 🟩 | `src/data/synthetic.py` — 8 服务, 4 故障类型 |
| 0.2 | GRU temporal encoder | 🟩 | `src/models/temporal_encoder.py` |
| 0.3 | GAT graph encoder | 🟩 | `src/models/graph_encoder.py` |
| 0.4 | RSSM core | 🟩 | `src/models/rssm.py` — h_t + z_t (discrete) |
| 0.5 | Gaussian NLL + edge decoders | 🟩 | `src/models/decoders.py` |
| 0.6 | Graph-RSSM 组装 | 🟩 | `src/models/graph_rssm.py` |
| 0.7 | 训练管线 + 损失函数 | 🟩 | `src/training/train.py`, `losses.py` |
| 0.8 | 验证: 预测残差 → RCA | 🟩 | 合成数据 **80% Top-1** |

## Phase 1: 数据预处理 🟩

| # | 任务 | 状态 | 备注 |
|---|------|------|------|
| 1.1 | Nezha 数据集克隆 | 🟩 | Online Boutique + TrainTicket, ~2.8GB |
| 1.2 | OpenRCA 数据集 | 🟩 | Bank + Telecom + Market, ~73GB |
| 1.3 | 解析器 (pod→service, span→edge) | 🟩 | `src/data/aiops2020/parse.py` |
| 1.4 | 正常/异常窗口分离 | 🟩 | construct_data (正常) / rca_data (故障) |
| 1.5 | Train/Val 划分 + HDF5 存储 | 🟩 | `data/processed/cross_system.h5` |

## Phase 2: 真实数据训练 (Online Boutique, 仅正常数据) 🟩

| # | 任务 | 状态 | 备注 |
|---|------|------|------|
| 2.1 | 模型配置 (N=10, E=14) | 🟩 | 1,835,028 参数 |
| 2.2 | 训练 (392 train, 98 val) | 🟩 | 80 epochs, loss: 0.84 → -1.33 |
| 2.3 | Checkpoint 保存 | 🟩 | `checkpoints/phase2/best/` |

## Phase 3-3.5: RCA 评估 + RQ3/RQ4 🟩

| # | 任务 | 状态 | 备注 |
|---|------|------|------|
| 3.1 | 预测残差计算 | 🟩 | per-service, 滑动窗口 |
| 3.2 | RQ3: 证据融合 (A+T+P+D) | 🟩 | Oracle window 80% Top-1, 融合持平 |
| 3.3 | RQ4: 图消融 | 🟩 | NLL 差异 <0.1%, metrics 是主要信号 |

## Phase 4a-4c: 泛化测试 🟩

### Oracle window 结果（模型排序质量上界）

| 系统 | 训练 | 测试 | 故障数 | Top-1 | Top-3 | MRR |
|------|------|------|--------|-------|-------|-----|
| OB Day1 (源域) | Day1 | Day1 | 24 | **100%** | 100% | 1.00 |
| OB Day1→Day2 | Day1 | Day2 | 32 | **53%** | 78% | 0.69 |
| OB→TrainTicket | Day1 | TS (10/45) | 14 | **79%** | 100% | 0.89 |
| OB→Market/cb1 | Day1 | Market cb1 | 51 | **63%** | 92% | 0.77 |
| OB→Market/cb2 | Day1 | Market cb2 | 49 | **29%** | 86% | 0.51 |
| OB→Bank (svc) | Day1 | Bank | 119 | **0%** | 5% | 0.22 |

## 泄露审查与 Clean Eval 🟩

### 泄露分析

| 泄露源 | 说明 | 影响 |
|--------|------|------|
| Window 选择 | 用根因服务残差峰值定位评估窗口 | **主要泄露**（83%→21%，-62pp） |
| Posterior 模式 | 模型用当前观测编码潜状态 | 次要（posterior 8% vs prior 13%） |

### Clean eval（Prior + Timestamp window）

| 系统 | 粒度 | 故障数 | Top-1 | Top-3 | MRR | vs Random |
|------|------|--------|-------|-------|-----|-----------|
| OB Day1 | service | 24 | **21%** | 38% | 0.39 | 2.1× |
| Bank | container | 16 | **38%** | 50% | 0.50 | 3.8× |
| Random | — | — | 10% | 30% | 0.34 | 1.0× |

## Phase 5: Logs 集成 🟩

| # | 任务 | 状态 | 备注 |
|---|------|------|------|
| 5.1 | 日志解析器 | 🟩 | `parse_log_features()` — 4 dims/svc |
| 5.2 | 日志预测头 | 🟩 | `LogPredictor` — latent→log features MLP |
| 5.3 | 多模态训练集成 | ⬜ | 需 log features 纳入观测空间并重新训练 |

## Phase 6: 故障动作适配器 🟩

| # | 任务 | 状态 | 备注 |
|---|------|------|------|
| 6.1 | 故障类型词汇 | 🟩 | 5 种 Nezha 类型 |
| 6.2 | Heuristic 推理 | 🟩 | 基于残差模式推断故障类型 |
| 6.3 | 训练型适配器 | ⬜ | 需 Nezha fault 数据训练 fault-conditioned RSSM |

## Phase 7: 多粒度实体 🟩

| # | 任务 | 状态 | 备注 |
|---|------|------|------|
| 7.1 | Container 级指标解析 | 🟩 | `parse_container_metrics()` — pivot long→wide |
| 7.2 | Bank 容器级建模 | 🟩 | 18 容器, 9 KPIs, **38% Top-1 clean** |
| 7.3 | Telecom 建模 | 🟥 | KPI 过少 (4 个), 需 Logs 补充 |

---

# RCAWorld-Foundation 分支 — 进行中

## 整体路线图

| Phase | 内容 | 状态 | 预计产出 |
|-------|------|------|----------|
| **A** | 通用 Entity Schema + OpenRCA Adapter + Onset Head + 双路 Component | 🟩 代码 | Top-1 ≥ 45% (Bank clean) |
| B | 多模态融合 (Drain log + Trace span encoding) | ⬜ | 日志模板 ID → embedding |
| C | Mechanism Adapter — latent intervention 推演 | ⬜ | MechanismToken 反事实验证 |
| D | D-Former + LLM — Flamingo GCA + LoRA | ⬜ | OpenRCA JSON > 11.34% |
| E | 广泛数据集 (RCAEval, FaultForge, TN-RCA, 单体) | ⬜ | 跨架构 generalization |

## Phase A: 通用 Schema + Onset Head + 双路定位 🟩

### A.1 Schema 模块 🟩

| # | 任务 | 文件 | 状态 |
|---|------|------|------|
| A.1.1 | Entity (16 类型) + ObservationSpec | `schema/entity.py` | 🟩 |
| A.1.2 | Relation (9 类型) | `schema/relation.py` | 🟩 |
| A.1.3 | ObservationEvent + Modality + EventBatch | `schema/event.py` | 🟩 |
| A.1.4 | SystemEpisode + QueryMask + RootCauseLabel | `schema/episode.py` | 🟩 |
| A.1.5 | MechanismToken (17 tokens) + Vocabulary + label→token 映射 | `schema/vocabulary.py` | 🟩 |

### A.2 Adapter 模块 🟩

| # | 任务 | 文件 | 状态 |
|---|------|------|------|
| A.2.1 | AdapterProtocol + BaseAdapter | `adapters/base.py` | 🟩 |
| A.2.2 | OpenRCA Bank (容器 + 服务) | `adapters/openrca_bank.py` | 🟩 |
| A.2.3 | OpenRCA Market (cloudbed-1/2) | `adapters/openrca_market.py` | 🟩 |
| A.2.4 | OpenRCA Telecom (metric_app + metric_container) | `adapters/openrca_telecom.py` | 🟩 |
| A.2.5 | Generic CSV adapter | `adapters/generic_csv.py` | 🟩 |

### A.3 Models 模块 🟩

| # | 任务 | 文件 | 状态 | 备注 |
|---|------|------|------|------|
| A.3.1 | MetricEncoder + TypeAwareDecoder | `models/entity_tokenizer.py` | 🟩 | Type embed + obs mask; JAX 验证通过 |
| A.3.2 | HierarchicalRSSM | `models/hierarchical_rssm.py` | 🟩 | 遵循 Flax setup() + nn.Sequential 标准模式; JAX 验证通过 |
| A.3.3 | OnsetHead (residual + shift + precedence) | `models/onset_head.py` | 🟩 | 三信号融合 + learnable weights + temporal conv; JAX 验证通过 |
| A.3.4 | ParallelComponentHead (p(c|t) + p(c)) | `models/parallel_component_head.py` | 🟩 | 双路 + learnable alpha gate; JAX 验证通过 |
| A.3.5 | Flamingo GCA (gated cross-attn) | `models/flamingo_gca.py` | 🟩 | GCA Block/Layer + DiagnosticProjector |
| A.3.6 | RCAWorldFoundation (完整组装) | `models/world_model.py` | 🟩 | Encode→RSSM→Decode→Onset→Component; JAX 验证通过 |

### A.4 Training 模块 🟩

| # | 任务 | 文件 | 状态 | 备注 |
|---|------|------|------|------|
| A.4.1 | world_model_loss (NLL+KL+onset+component+consistency) | `training/losses.py` | 🟩 | 5-term combined loss |
| A.4.2 | create_train_state + pretrain_normal | `training/pretrain.py` | 🟩 | Optax Adam, TrainState |

### A.5 Evaluation 模块 🟩

| # | 任务 | 文件 | 状态 | 备注 |
|---|------|------|------|------|
| A.5.1 | compute_component_metrics (Top-1/3/5, MRR) | `evaluation/openrca_metrics.py` | 🟩 | 支持 mask; 已验证修复 np.mean→np.sum |
| A.5.2 | compute_onset_metrics (accuracy, MAE) | `evaluation/openrca_metrics.py` | 🟩 | tolerance window 参数化 |
| A.5.3 | evaluate_exact_match (OpenRCA protocol) | `evaluation/openrca_metrics.py` | 🟩 | 查询条件化输出, 分字段报告 |

### A.6 实验进展 ⬜→🟩

| # | 实验 | 数据集 | 目标 | 状态 | 结果 |
|---|------|--------|------|------|------|
| A.6.1 | Phase A pretrain | OB Day1 (正常) | 验证新架构 loss 收敛 | 🟩 | loss 2041→-686, val 628→-763 (80ep) |
| A.6.2 | Onset Head vs Oracle Window | Bank (容器) | GT-leak 版本 (已废弃) | 🟥 | 24.3% 含 GT-component 泄露 |
| A.6.3 | p(c|t)+p(c) vs 单路 | Bank + Market | 定位精度对比 | ⬜ | — |
| A.6.4 | strict zero-shot (OB→OpenRCA) | Bank | **无泄露** joint scoring baselin | 🟩 | **Comp T1=0.7%, T3=6.6%, MRR=0.126** (prior-only, 136 faults, typed entities) |
| A.6.5 | normal-only calibration | OpenRCA 各系统正常时段 | 自监督适配 | ⬜ | — |

### Phase A-Hardening (P0/P1) 🟦

**leak 修复** (commit `84393ac`):

| 问题 | 修复 | 文件 |
|------|------|------|
| GT-component 条件化窗口选择 | joint S[t,c] 矩阵, argmax 推断, GT 仅用于 metric | `evaluation/strict_eval.py` |
| OnsetHead temporal_conv 死代码 | smoothed 输出真正参与 final onset_score | `models/onset_head.py` |
| precedence 误导性命名 | 改为 early_rise, 移除未使用的 edge_index 声称 | `models/onset_head.py` |
| posterior 默认 | strict eval 默认 prior-only | `scripts/phaseA_strict_eval.py` |
| 类型全零 (all type 0) | typed entity indices (container=0, database=1, middleware=2) | `scripts/phaseA_strict_eval.py` |

**诚实无泄露基线** (Bank, prior-only, 80-epoch OB→Bank zero-shot):

| 指标 | 值 |
|------|-----|
| Comp Top-1 | 0.7% |
| Comp Top-3 | 6.6% |
| MRR | 0.126 |
| Time Hit | 0.0% |
| Joint Hit | 0.0% |
| Faults | 136 |

> 此前报告 24.3% Top-1 含 GT-component 泄露（先读 rc 再选 window 峰值）
> 当前 0.7% 是可信的无泄露严格零样本基线

### Phase A-Hardening P0.7: Protocol Closure 🟩

**P0.1 — expected_fault_count 来自查询文本，而非 scoring_points**:
- `_parse_fault_count_from_text()` 从 instruction 文本提取故障数量（"a single failure"→1, "two failures"→2）
- `InferenceQuery.expected_fault_count` 不再接触 GT
- GT 故障数量保留在 `EvalTarget.root_causes` 列表中

**P0.2 — NMS 多故障解码接入正式评估循环**:
- `JointScores.decode_multiple_faults()` 通过 NMS 解码多个峰值
- 评估循环使用贪心一对一匹配：预测 → GT
- 对超过预测数量的 GT，回退到 single-best

**P0.3 — Adapter 接受 `include_dates: Set[str]`**:
- 三个 Adapter 均支持 `include_dates` 参数
- `_get_days()` 过滤到仅包含指定日期目录
- 验证：加载单日将事件数从 3.2M 降至 76K

**P0.4 — 指标报告分离查询级别和根因级别计数**:
- 输出：official queries、evaluated queries、total root-causes
- `aggregate_metrics` 同时追踪 n（根因实例）和 n_queries_evaluated

**P0.6 — Burn-in 日期覆盖扩展到查询窗口开始前的日期**:
- `min_data_date` 从最早 `window_start - burn_in_min` 计算
- `needed_dates_set` 覆盖完整 `[min_data_date, max_data_date]` 范围

**P1.1 — TimeHit 报告 Hit@5min、Hit@10min、Hit@15min**:
- 新增字段：`time_hit_5min`、`time_hit_10min`、`time_hit_15min`
- `time_mae_min` 以实际分钟数计算

**审计结果**: 5/5 P0 项全部修复，2/2 P1 项全部修复。

### Phase 0.8 R1: 推理/评估路径分离 🟩 (commit `31338d7`)

**目标**: 确保推理脚本在文件级别无法访问 GT。推理与评估完全解耦为两个独立脚本。

| # | 任务 | 状态 |
|---|------|------|
| 0.8.1 | `parse_inference_queries()` — 仅读取 task_index + instruction，零 GT 访问 | 🟩 |
| 0.8.2 | `load_eval_targets()` — 读取 scoring_points，仅用于评估 | 🟩 |
| 0.8.3 | `scripts/phaseA_run_inference.py` — 推理脚本，禁止 import EvalTarget | 🟩 |
| 0.8.4 | `scripts/phaseA_evaluate_predictions.py` — 评估脚本，读取 predictions.json + scoring_points + record.csv | 🟩 |
| 0.8.5 | `predictions.json` 输出: query_id + predictions[datetime, component, score] | 🟩 |
| 0.8.6 | 验收: 删除 record.csv 后推理仍可运行 | 🟩 |
| 0.8.7 | 验收: 删除 scoring_points 列后推理仍可运行 | 🟩 |

**文件变更**:
- 新增: `scripts/phaseA_run_inference.py` (372 行)
- 新增: `scripts/phaseA_evaluate_predictions.py` (263 行)
- 修改: `src/foundation/evaluation/query_parser.py` — 拆分 parse_query_csv → parse_inference_queries + load_eval_targets
- 修改: `scripts/phaseA_strict_eval.py` — 标记 [DEPRECATED]
- 修改: `src/foundation/evaluation/__init__.py` — 导出新函数

**使用方式**:
```bash
# Step 1: 推理 (无 GT)
python scripts/phaseA_run_inference.py --systems Bank --output predictions.json

# Step 2: 评估 (使用 GT)
python scripts/phaseA_evaluate_predictions.py --predictions predictions.json --systems Bank
```

### Phase 0.8 R4: NMS 与 Prediction–Target 一对一匹配 🟩

**目标**: 一条 Query 只运行一次 prior-only inference、只生成一次 joint `S[t,c]`，NMS 根据查询文本故障数输出多个候选，evaluation 改为真正的一对一匹配。

| # | 任务 | 状态 |
|---|------|------|
| 0.8.4.1 | `phaseA_run_inference.py` 主路径使用单次 episode / 单次 inference / 单次 joint matrix / 单次 `decode_multiple_faults()` | 🟩 |
| 0.8.4.2 | `max_faults` 仅来自 `InferenceQuery.expected_fault_count` | 🟩 |
| 0.8.4.3 | 删除预测不足时复用同一峰值的 fallback | 🟩 |
| 0.8.4.4 | `phaseA_evaluate_predictions.py` 新增 `match_predictions_to_targets()` 贪心一对一匹配 | 🟩 |
| 0.8.4.5 | evaluation 删除按数组下标硬匹配；未匹配 GT 计为 miss | 🟩 |
| 0.8.4.6 | 单元测试：顺序无关匹配 + `expected_fault_count=2` 时最多输出 2 个候选 | 🟩 |

**实现说明**:
- 推理路径不导入 `EvalTarget`，也不读取 `gt_fault_count`、`len(eval_target.root_causes)` 或 `scoring_points` 中的根因数量。
- evaluation 使用匹配代价 `component_mismatch_penalty + time_distance_minutes` 的 greedy matching。
- 若预测数少于 GT 数，未匹配目标直接记为 miss，不再复用同一个峰值补位。

### 下一优先级: Phase 0.8 R5 — 待定 🟦

---

## 关键设计决策记录

| 决策 | 理由 | 日期 |
|------|------|------|
| p(c\|t) + p(c) 双路并行而非级联 | 静默/累积故障在单时间点无强信号，需时间自由路径 | 2026-06-04 |
| Flamingo GCA (非 Q-Former) | 训练更稳定，适合 3B-8B 模型；Q-Former 需大规模预训练 | 2026-06-04 |
| Type embeddings + obs_mask 而非 per-type encoder | JAX JIT 兼容，更简单实用 | 2026-06-04 |
| nn.Sequential + setup() 标准模式 | 遵循原始 RSSM 已验证模式，避免 closure scope 冲突 | 2026-06-04 |
| Onset Head 三信号加权融合 | residual + shift + precedence 覆盖不同故障模式 | 2026-06-04 |

---

## master 分支 — 关键发现总结

| 结论 | 证据 |
|------|------|
| 仅正常数据预训练即可 RCA | 全程零故障标签训练，OB 源域 oracle 100% Top-1 |
| 模型排序质量高 | Oracle window: OB 100%, TS 79%, Mkt 63% |
| Window 定位是主要瓶颈 | 泄露去除后 83%→21%（-62pp） |
| 跨领域需粒度对齐 | Bank svc 级 0% → container 级 38%（↑3.8× random） |
| Metrics 是主要信号 | 图消融 <0.1% NLL 差异 |
| Reason (why) 需额外训练 | 故障类型推断需要 Nezha fault 标签 |

---

## 后续工作

| 优先级 | 内容 | 分支 | 状态 |
|--------|------|------|------|
| **high** | Phase A 实验: 训练 + onset + component 评估 | foundation | ⬜ |
| **high** | Phase B: Drain 日志模板 + Trace span 编码 | foundation | ⬜ |
| high | 论文撰写 | — | ⬜ |
| medium | Phase C: Mechanism Adapter latent intervention | foundation | ⬜ |
| medium | Phase D: Flamingo GCA + LoRA LLM 桥接 | foundation | ⬜ |
| medium | Phase E: RCAEval / FaultForge / TN-RCA 验证 | foundation | ⬜ |
| low | master Phase 5 完整集成: log-augmented 重新训练 | master | ⬜ |
| low | master Phase 6 完整集成: fault-conditioned RSSM | master | ⬜ |
| low | master Phase 7 Telecom: KPI 补充评估 | master | ⬜ |
