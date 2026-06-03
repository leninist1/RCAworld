# RCAWorld: 项目进度

> 状态: ⬜ pending | 🟦 in_progress | 🟩 completed | 🟥 blocked

---

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
| 0.9 | JIT 编译 & GPU 训练 | 🟩 | CUDA RTX 3090, ~250 steps/sec |

---

## Phase 1: 数据预处理 🟩

| # | 任务 | 状态 | 备注 |
|---|------|------|------|
| 1.1 | Nezha 数据集克隆 | 🟩 | Online Boutique + TrainTicket, ~2.8GB |
| 1.2 | OpenRCA 数据集 | 🟩 | Bank + Telecom + Market, ~73GB (已预先下载) |
| 1.3 | 解析器 (pod→service, span→edge) | 🟩 | `src/data/aiops2020/parse.py` |
| 1.4 | 正常/异常窗口分离 | 🟩 | construct_data (正常) / rca_data (故障) |
| 1.5 | 时间对齐 + 归一化 | 🟩 | 60s 窗口, 正常窗口上计算 stats |
| 1.6 | Train/Val/Test 划分 | 🟩 | 按 episode 划分, 防止时间泄漏 |
| 1.7 | HDF5 存储 | 🟩 | `data/processed/cross_system.h5` (8.3MB) |

---

## Phase 2: 真实数据训练 (Online Boutique, 仅正常数据) 🟩

| # | 任务 | 状态 | 备注 |
|---|------|------|------|
| 2.1 | 模型配置 (N=10, E=14) | 🟩 | 1,835,028 参数 |
| 2.2 | 训练 (392 train, 98 val windows) | 🟩 | 80 epochs, loss: 0.84 → -1.33 |
| 2.3 | RQ1 验证: 正常动力学学习 | 🟩 | Val NLL 收敛, 各服务预测质量不均衡 |
| 2.4 | Checkpoint 保存 | 🟩 | `checkpoints/phase2/best/` (Orbax) |
| 2.5 | 预测残差基准 | 🟩 | 正常窗口 baseline: μ, σ established |

---

## Phase 3: RCA 评估 🟩

| # | 任务 | 状态 | 备注 |
|---|------|------|------|
| 3.1 | 预测残差计算 (per-service, per-window) | 🟩 | 批量计算, 滑动窗口 |
| 3.2 | 首次偏离时间检测 | 🟩 | μ+3σ 阈值法 |
| 3.3 | 传播链提取 | 🟩 | 沿 edge_index 追踪 |
| 3.4 | 证据融合 (A+T+P+D) | 🟩 | 残差 80% Top-1, 融合持平 |
| 3.5 | 评估指标 (Top-K, MRR, AvgRank) | 🟩 | 全部实现 |

---

## Phase 3.5: RQ3 + RQ4 🟩

| # | 任务 | 状态 | 备注 |
|---|------|------|------|
| 3.5.1 | RQ3: 证据融合 (residual + temporal + topology + latent dev) | 🟩 | 残差已足够强 (80%), 附加信号未带来额外提升 |
| 3.5.2 | RQ4: 图消融 (dynamic / static / no graph) | 🟩 | NLL 差异 <0.1%, metrics 是主要信号源 |

---

## Phase 4a: 时间泛化 (Day1 → Day2) 🟩

| 系统 | 训练 | 测试 | 故障数 | Top-1 | Top-3 | MRR |
|------|------|------|--------|-------|-------|-----|
| Online Boutique | Day1 (正常) | Day1 (故障) | 24 | **100%** | 100% | 1.000 |
| Online Boutique | Day1 (正常) | Day2 (故障) | 32 | **53%** | 78% | 0.686 |

> 泛化差距: +47% — 时间特异性存在但世界模型显著优于随机

---

## Phase 4b: 跨系统迁移 (OB → TrainTicket) 🟩

| 系统 | 训练 | 测试 | 故障数 | Top-1 | Top-3 | MRR |
|------|------|------|--------|-------|-------|-----|
| TrainTicket (10/45 svcs) | OB Day1 | TS 01-29 | 14 | **79%** | 100% | 0.893 |

> 10 个 TS 服务通过角色映射到 OB 模型槽位，零样本迁移

---

## Phase 4c: 跨领域泛化 (OpenRCA) 🟩

| 系统 | 领域 | 故障数 | Component Top-1 | C+T Top-1 | Top-3 | MRR |
|------|------|--------|-----------------|-----------|-------|-----|
| Market/cloudbed-1 | 微服务 | 51 | **63%** | **61%** | 92% | 0.771 |
| Market/cloudbed-2 | 微服务 | 49 | **29%** | **24%** | 86% | 0.507 |
| Bank | 银行 | 119 | **0%** | 0% | 5% | 0.221 |
| Telecom | 电信 | — | — | — | — | — |

> **Component**: 定位异常服务 (where)
> **C+T**: Component × Time 联合命中 (where + when)
> **Reason**: 需故障分类器，未实现（第二阶段）

---

## 关键发现总结

| 结论 | 证据 |
|------|------|
| 仅正常数据预训练即可 RCA | 全程零故障标签训练，OB 源域 100% Top-1 |
| 微服务领域内泛化强 | OB→TrainTicket 79%, OB→Market 63% |
| 跨领域不迁移 | Bank 0% (随机水平) — 定义了适用边界 |
| 时间泛化有限 | Day1→Day2 53% — 动力学有时间特异性 |
| Metrics 是主要信号 | 图消融 <0.1% NLL 差异 |
| 残差排序已足够强 | 证据融合持平于纯残差 (80%) |
| 修复干预待深化 | RQ3 简化版未带来增益，完整 rollout 待实现 |

---

## 后续工作

| Phase | 内容 | 优先级 | 状态 |
|-------|------|--------|------|
| Phase 5 | Logs 集成 | medium | ⬜ |
| Phase 6 | 故障传播适配器 (Reason 分类) | high | ⬜ |
| Phase 7 | 多粒度实体 (Pod/容器) | medium | ⬜ |
| Phase 8 | NoiseLab 集成 | medium | ⬜ |
| — | 论文撰写 | high | ⬜ |
| — | 基线对比 (DynaCausal, CHASE, RUN) | high | ⬜ |
