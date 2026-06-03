# MicroDynamics-RCA: 项目进度

> 状态: ⬜ pending | 🟦 in_progress | 🟩 completed | 🟥 blocked

---

## Phase 0: 合成数据 MVP + Graph-RSSM 原型验证

| # | 任务 | 状态 | 备注 |
|---|------|------|------|
| 0.1 | 合成数据生成器 (N 服务 metrics + 动态调用图) | ⬜ | `src/data/synthetic.py` |
| 0.2 | GRU temporal encoder (共享参数) | ⬜ | `src/models/temporal_encoder.py` |
| 0.3 | GAT graph encoder (动态边) | ⬜ | `src/models/graph_encoder.py` |
| 0.4 | RSSM core (prior + posterior + transition, Flax) | ⬜ | `src/models/rssm.py` |
| 0.5 | Gaussian NLL decoder + edge decoder | ⬜ | `src/models/decoders.py` |
| 0.6 | Graph-RSSM 组装 | ⬜ | `src/models/graph_rssm.py` |
| 0.7 | 训练管线 (仅正常窗口, ELBO + rollout loss) | ⬜ | `src/training/train.py` |
| 0.8 | 损失函数 (Gaussian NLL + Edge + KL + Mask + Rollout) | ⬜ | `src/training/losses.py` |
| 0.9 | 验证: 预测残差区分正常/故障 | ⬜ | `notebooks/phase0_validation.ipynb` |

---

## Phase 1: AIOps 2020 数据处理

| # | 任务 | 状态 | 备注 |
|---|------|------|------|
| 1.1 | 数据集下载 | ⬜ | 清华云 / Google Drive (~16GB) |
| 1.2 | 业务指标解析 | ⬜ | `src/data/aiops2020/parse_metrics.py` |
| 1.3 | 平台指标解析 | ⬜ | 同上 |
| 1.4 | 调用链解析 (span → 动态调用图) | ⬜ | `src/data/aiops2020/parse_traces.py` |
| 1.5 | 故障标签解析 (故障整理.csv) | ⬜ | `src/data/aiops2020/parse_faults.py` |
| 1.6 | 正常/异常窗口分离 (按故障标签) | ⬜ | `src/data/aiops2020/split.py` |
| 1.7 | 时间对齐 & JAX Dataset 构建 | ⬜ | `src/data/aiops2020/dataset.py` |
| 1.8 | 10s 窗口聚合 + 标准化 | ⬜ | 同上 |
| 1.9 | Train/Val/Test 按 episode 划分 | ⬜ | 防止时间泄漏 |

---

## Phase 2: Graph-RSSM 训练（仅正常数据）

| # | 任务 | 状态 | 备注 |
|---|------|------|------|
| 2.1 | 配置管理 (Hydra) | ⬜ | `configs/` |
| 2.2 | 训练入口 (Optax + Orbax checkpoint) | ⬜ | `src/training/train.py` |
| 2.3 | 超参搜索 (latent dim, stochastic dim, rollout H) | ⬜ | |
| 2.4 | RQ1 验证: Metrics NLL, Edge F1, Multi-step drift | ⬜ | |
| 2.5 | TensorBoard/WandB 日志 | ⬜ | |

---

## Phase 3: RCA 评估

| # | 任务 | 状态 | 备注 |
|---|------|------|------|
| 3.1 | 预测残差计算 (per-service, per-window) | ⬜ | `src/rca/residual.py` |
| 3.2 | 首次偏离时间检测 | ⬜ | 同上 |
| 3.3 | 传播链提取 (沿调用图) | ⬜ | `src/rca/propagation.py` |
| 3.4 | 修复干预推演 (潜态替换 + rollout) | ⬜ | `src/rca/intervention.py` |
| 3.5 | 证据融合打分 (A+T+P+R-U) | ⬜ | `src/rca/evidence_fusion.py` |
| 3.6 | 基线实现: Z-score | ⬜ | `src/evaluation/baselines/` |
| 3.7 | 基线实现: EWMA | ⬜ | |
| 3.8 | 基线实现: LSTM-AE | ⬜ | |
| 3.9 | 基线实现: RUN-style (Granger + PageRank) | ⬜ | |
| 3.10 | 评估指标 (AC@K, MRR, MTTD, AURC) | ⬜ | `src/evaluation/metrics.py` |
| 3.11 | RQ2: 预测残差提升 RCA? | ⬜ | |
| 3.12 | RQ3: 修复干预区分根因 vs 症状? (消融) | ⬜ | |
| 3.13 | RQ4: 动态图有用? (MLP/GRU/GRU+静态/Graph-RSSM) | ⬜ | |

---

## Phase 4: 论文产出

| # | 任务 | 状态 | 备注 |
|---|------|------|------|
| 4.1 | 结果表格 + 消融图 | ⬜ | |
| 4.2 | Case study: 故障 latent space 传播可视化 | ⬜ | |
| 4.3 | 复现包 (Docker + 固定种子) | ⬜ | |

---

## Phase 5: Logs 扩展（后续）

| # | 任务 | 状态 | 备注 |
|---|------|------|------|
| 5.1 | Drain 日志解析器 | ⬜ | |
| 5.2 | 日志模板预测头 | ⬜ | |
| 5.3 | 三模态世界模型训练 | ⬜ | |

---

## Phase 6: 故障传播适配器（后续）

| # | 任务 | 状态 | 备注 |
|---|------|------|------|
| 6.1 | 故障动作 encoding | ⬜ | |
| 6.2 | p(z_{t+1} | z_t, G_t, w_t, a_t) | ⬜ | |
| 6.3 | 故障类型匹配 (最小扰动解释故障) | ⬜ | |

---

## Phase 7: 多粒度实体（后续）

| # | 任务 | 状态 | 备注 |
|---|------|------|------|
| 7.1 | Pod/容器节点扩展 | ⬜ | |
| 7.2 | 层次化图构建 | ⬜ | |
| 7.3 | 跨层根因定位 | ⬜ | |

---

## Phase 8: NoiseLab 集成（后续）

| # | 任务 | 状态 | 备注 |
|---|------|------|------|
| 8.1 | 世界模型 JSON 证据输出 | ⬜ | |
| 8.2 | NoiseLab 证据融合 | ⬜ | |
| 8.3 | LLM 解释生成 | ⬜ | |
