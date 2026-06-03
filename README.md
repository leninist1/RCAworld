# RCAWorld: 基于系统动力学世界模型的微服务根因分析

> 正常情况下，微服务系统的状态演化具有稳定规律。
> 根因是最早破坏这一规律，并且能够解释后续异常传播的实体。
> 将该实体恢复到正常状态后，世界模型推演出的下游异常应显著减弱。

---

## 实验结果总览

| 实验 | 训练 | 测试系统 | 测试故障数 | Component Top-1 | C+T Top-1 | Top-3 | MRR | AvgRank |
|------|------|----------|-----------|-----------------|-----------|-------|-----|---------|
| Phase 0 (合成) | Synth Normal | Synth Fault | 5 | **80%** | — | — | — | 1.20 |
| Source (分布内) | OB Day1 | OB Day1 | 24 | **100%** | — | 100% | 1.000 | 1.00 |
| OB→TrainTicket | OB Day1 | TrainTicket | 14 | **79%** | — | 100% | 0.893 | 1.21 |
| OB→Market/cb1 | OB Day1 | Market cloudbed-1 | 51 | **63%** | **61%** | 92% | 0.771 | 1.67 |
| OB Day1→Day2 | OB Day1 | OB Day2 | 32 | **53%** | — | 78% | 0.686 | 2.66 |
| OB→Market/cb2 | OB Day1 | Market cloudbed-2 | 49 | **29%** | **24%** | 86% | 0.507 | 2.78 |
| OB→Bank | OB Day1 | Bank (银行) | 119 | **0%** | 0% | 5% | 0.221 | 4.84 |
| OB→Telecom | OB Day1 | Telecom (电信) | — | — | — | — | — | — |

> **C+T = Component + Time 联合命中**  
> **OB = Online Boutique**  
> 所有结果均**仅用正常数据预训练，零 RCA 标签训练**

---

## 核心发现

1. **领域内泛化强**：在微服务领域（OB→TrainTicket 79%, OB→Market 63%）世界模型学到可迁移的系统动力学规律
2. **跨领域失效**：非微服务系统（Bank 0%, Telecom N/A）完全无法迁移 — 定义了方法的适用边界
3. **时间泛化有限**：同一系统不同日期（Day1→Day2 53%）表明动力学具有时间特异性
4. **仅正常数据训练**：全程未使用任何故障标签，RCA 推理阶段零额外训练
5. **Component + Time 可定位**：同时回答 where (63%) + when (61%)；Reason (why) 需要故障分类器，属于第二阶段工作

---

## 数据来源

| 数据集 | 系统 | 用途 | 数据量 |
|--------|------|------|--------|
| [Nezha](https://github.com/IntelligentDDS/Nezha) (FSE 2023) | Online Boutique, TrainTicket | 训练 + 评估 | ~2.8GB |
| [OpenRCA](https://github.com/microsoft/OpenRCA) (ICLR 2025) | Market, Bank, Telecom | 跨领域评估 | ~73GB |

**训练数据**: Nezha `construct_data/` (fault-free 正常阶段) — Online Boutique Day1 (2022-08-22)
**评估数据**: Nezha `rca_data/` (故障阶段) + OpenRCA 全量 — 均未参与训练

---

## 模型架构: Graph-RSSM

```
Metrics [T, N, D_node] + Trace 边特征 [T, E, D_edge]
              │
    ┌─────────┴──────────┐
    │ Node Encoder (MLP)  │  per-timestep 特征编码
    │ GAT Graph Encoder   │  2-layer 服务间消息传递
    │ Workload Context    │  总 QPS / 成功率 / 活跃服务数
    └─────────┬──────────┘
              ▼
    ┌─────────────────────┐
    │  RSSM Latent Dynamics │  h_t: 确定性状态 [N, 256]
    │                       │  z_t: 随机离散潜态 [N, 32, 32]
    │  p(z_{t+1} | z_t, G_t, w_t)
    └─────────┬───────────┘
              │
    ┌─────────┼───────────┐
    │ Metrics Decoder      │  μ, σ per service → Gaussian NLL
    │ Edge Decoder         │  调用次数 / 错误率 / 延迟
    └─────────────────────┘

训练目标: L = L_metric (Gaussian NLL) + λ_e·L_edge (MSE) + λ_k·L_kl
参数量: 1,835,028 | 框架: JAX + Flax | 优化器: Optax AdamW
```

---

## RCA 推理流程（零额外训练）

```
Step 1: 故障前窗口 → 世界模型推演"正常情况下应该发生什么"
        z_t0 = E(O_{t0-11:t0})  →  F_θ → Ô_{t0+1:t0+H}

Step 2: 计算每服务预测残差
        A_t(v) = -log p_θ(x_t(v) | z_<t)

Step 3: 提取传播链
        首次偏离时间 + 沿依赖图传播一致性

Step 4: 证据融合打分（可选：修复干预推演）
        S(v) = α·A(v) + β·T(v) + γ·P(v) + δ·R(v) - η·U(v)
        异常程度 + 时间先行 + 拓扑一致 + 修复增益 - 不确定性

Step 5: 输出根因排序（Component + Time / Component only）
```

---

## 与现有工作的关键差异

| 维度 | DynaCausal (2025) | CHASE (2024) | RUN (2024, AAAI) | **本方案** |
|------|-------------------|-------------|-------------------|-----------|
| 训练范式 | 判别式（故障标签） | 判别式 | 判别式 | **生成式（仅正常数据）** |
| 学习目标 | Pairwise ranking | Hypergraph causal | Granger + PageRank | **正常动力学 ELBO** |
| 排序模型 | 学习的 ranker | 学习的 ranker | PageRank | **无（透明证据融合）** |
| 未来预测 | 无 | 无 | 时序预测 | **多步 rollout** |
| 修复干预 | 无 | 无 | 无 | **潜态替换 + rollout** |
| 跨系统泛化 | 未验证 | 未验证 | 未验证 | **验证通过 (79%/63%)** |

---

## 项目结构

```
RCAWorld/
├── src/
│   ├── data/
│   │   ├── synthetic.py                # Phase 0: 合成数据
│   │   └── aiops2020/
│   │       ├── download.py             # 数据集下载
│   │       └── parse.py                # Nezha/OpenRCA 解析器
│   ├── models/
│   │   ├── temporal_encoder.py         # GRU temporal encoder
│   │   ├── graph_encoder.py            # GAT graph encoder
│   │   ├── rssm.py                     # RSSM 核心
│   │   ├── decoders.py                 # Gaussian NLL + edge decoder
│   │   └── graph_rssm.py               # 完整 Graph-RSSM
│   ├── training/
│   │   ├── train.py                    # 训练管线
│   │   └── losses.py                   # 损失函数
│   ├── rca/
│   └── evaluation/
├── scripts/
│   ├── phase0_validate.py              # Phase 0 验证
│   ├── phase2_train.py                 # Phase 2 训练 (OB)
│   ├── phase3_evaluate.py              # Phase 3 RCA 评估
│   ├── phase35_evaluate.py             # RQ3 (证据融合) + RQ4 (图消融)
│   ├── phase4_cross_system.py          # 时间泛化 (Day1→Day2)
│   ├── phase4c_openrca.py              # 跨领域泛化 (OpenRCA)
│   └── build_dataset.py                # 数据构建管线
├── checkpoints/phase2/best/            # 训练好的模型 (~40MB)
├── data/processed/                     # 预处理 HDF5 数据
├── README.md                           # 本文档
├── STATUS.md                           # 进度跟踪
├── requirements.txt
└── models_artifacts.tar.gz             # 模型 + 数据归档 (41MB)
```

---

## 技术栈

| 组件 | 选择 |
|------|------|
| ML Framework | JAX 0.10 + Flax 0.12 |
| Optimizer | Optax (AdamW + warmup cosine) |
| Checkpointing | Orbax |
| 评估基准 | Nezha (FSE 2023), OpenRCA (ICLR 2025) |
| 计算 | NVIDIA RTX 3090 (24GB) |

---

## 后续扩展路径

| Phase | 内容 | 说明 |
|-------|------|------|
| Phase 5 | Logs 集成 | Drain 日志解析 → 日志模板预测头 |
| Phase 6 | 故障传播适配器 | 故障动作 encoding → 区分 Reason (why) |
| Phase 7 | 多粒度实体 | Pod/容器/数据库 → 层次化根因定位 |
| Phase 8 | NoiseLab 集成 | 世界模型输出 JSON 证据 → LLM 解释生成 |

---

## 参考

- DynaCausal (arXiv:2510.22613)
- CHASE (arXiv:2406.19711)
- RUN (arXiv:2402.01140, AAAI 2024)
- IDI (OpenReview, 2025)
- DreamerV3 (arXiv:2301.04104)
- Nezha (FSE 2023)
- OpenRCA (ICLR 2025)
