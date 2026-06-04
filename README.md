# RCAWorld: 基于系统动力学世界模型的微服务根因分析

> 正常情况下，微服务系统的状态演化具有稳定规律。
> 根因是最早破坏这一规律，并且能够解释后续异常传播的实体。
> 将该实体恢复到正常状态后，世界模型推演出的下游异常应显著减弱。

---

## 实验结果总览

### 窗口已知（oracle window，模型排序能力上界）

| 实验 | 训练 | 测试系统 | 故障数 | Top-1 | Top-3 | MRR |
|------|------|----------|--------|-------|-------|-----|
| Phase 0 (合成) | Synth Normal | Synth Fault | 5 | **80%** | — | — |
| OB Day1 (源域) | OB Day1 | OB Day1 | 24 | **100%** | 100% | 1.00 |
| OB→TrainTicket | OB Day1 | TrainTicket (10/45) | 14 | **79%** | 100% | 0.89 |
| OB→Market/cb1 | OB Day1 | Market cloudbed-1 | 51 | **63%** | 92% | 0.77 |
| OB→Market/cb2 | OB Day1 | Market cloudbed-2 | 49 | **29%** | 86% | 0.51 |
| OB Day1→Day2 | OB Day1 | OB Day2 | 32 | **53%** | 78% | 0.69 |

### 窗口未知（clean eval，不含 oracle leak）

| 实验 | 模式 | 粒度 | 故障数 | Top-1 | Top-3 | MRR | vs Random |
|------|------|------|--------|-------|-------|-----|-----------|
| OB Day1 | Prior + Timestamp | service | 24 | **21%** | 38% | 0.39 | **2.1×** |
| Bank | Prior + Timestamp | container | 16 | **38%** | 50% | 0.50 | **3.8×** |
| Random | — | — | — | 10% | 30% | 0.34 | 1.0× |

> **窗口已知**: 用根因服务的残差峰值定位窗口（oracle leak）→ 测量模型排序质量的上界  
> **窗口未知 (clean)**: 用 record.csv 时间戳定位窗口（SRE 报告时间），无 oracle → 真实场景评估  
> **OB = Online Boutique** | 所有训练仅用正常数据，零 RCA 标签

---

## 核心发现

1. **模型学到了可迁移的动力学**：oracle window 内排序精度 63-100%（微服务领域），窗口正确时根因排序极强
2. **窗口定位是主要瓶颈**：无 oracle 时 Top-1 从 83%→21%（-62pp），异常检测精度决定端到端性能
3. **跨领域需粒度对齐**：Bank 服务级 0%，但容器级 38%（↑3.8×random）— 实体粒度匹配是关键
4. **仅正常数据训练**：全程未使用任何故障标签，RCA 推理阶段零额外训练
5. **Reason (why) 不可用**：区分故障类型需要故障标签训练，属于第二阶段工作
6. **图贡献有限**：RQ4 消融 NLL <0.1% 差异 — metrics 是主要信号源

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

---

## 参考

- DynaCausal (arXiv:2510.22613)
- CHASE (arXiv:2406.19711)
- RUN (arXiv:2402.01140, AAAI 2024)
- IDI (OpenReview, 2025)
- DreamerV3 (arXiv:2301.04104)
- Nezha (FSE 2023)
- OpenRCA (ICLR 2025)
