# RCAWorld-Foundation: 面向异构软件系统的查询条件化诊断世界模型

> RCAWorld 将 RCA 建模为一个查询条件化的系统动力学解释问题：
> 根因是最早破坏正常演化规律、能够解释后续传播轨迹，
> 并在反事实修复后使异常显著减弱的实体—时间—机理三元组。

---

## 分支说明

| 分支 | 定位 |
|------|------|
| `master` | Graph-RSSM 驱动的微服务异常排序器（已完成 Phase 0–7） |
| **`RCAWorld-Foundation`** | 通用诊断世界模型 — 异构实体、Onset 学习、p(c|t)+p(c) 双路、Flamingo GCA LLM 桥接 |

---

## RCAWorld-Foundation 架构

```
Telemetry Events (metrics / logs / traces / alerts)
                     │
     ┌───────────────┴───────────────┐
     │  EntityTokenizer              │
     │  MetricEncoder (typed) +      │
     │  type embeddings + obs_mask   │
     └───────────────┬───────────────┘
                     ▼
     ┌───────────────────────────────┐
     │  HierarchicalRSSM             │
     │  h_t, z_t ~ q(z|h,o)         │
     │  Type-modulated transition    │
     └───────────────┬───────────────┘
                     │
     ┌───────────────┼───────────────┐
     │  TypeAwareDecoder             │
     │  pred_mu, pred_logsigma       │
     │         ↓                     │
     │  ┌─────────────────────┐      │
     │  │  OnsetHead           │      │
     │  │  residual + shift    │      │
     │  │  + precedence fusion │      │
     │  └─────────┬───────────┘      │
     │            ▼                  │
     │  ┌─────────────────────┐      │
     │  │ ParallelComponentHead│      │
     │  │ p(c|t) + p(c) dual  │      │
     │  │ with learnable α    │      │
     │  └─────────┬───────────┘      │
     └────────────┼──────────────────┘
                  ▼
     ┌───────────────────────────────┐
     │  D-Former / Flamingo GCA      │  ← Phase D (LLM integration)
     │  Diagnostic tokens → LLM      │
     └───────────────────────────────┘
```

---

## 实验结果总览（master 分支）

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

---

## 核心发现（master）

1. **模型学到了可迁移的动力学** — oracle window 内排序 63-100%（微服务领域）
2. **窗口定位是主要瓶颈** — 无 oracle 时 Top-1 从 83%→21%（-62pp）
3. **跨领域需粒度对齐** — Bank 服务级 0%，容器级 38%（↑3.8×random）
4. **仅正常数据训练** — 全程未使用任何故障标签
5. **图贡献有限** — RQ4 消融 NLL <0.1% 差异

---

## Foundation 路线图

| Phase | 内容 | 状态 |
|-------|------|------|
| **Phase A** | 通用 Entity Schema、OpenRCA 三系统 Adapter、Onset Head、p(c|t)+p(c) 双路、Flamingo GCA | 🟩 |
| Phase B | 多模态融合 (Drain 日志、Trace span 编码) | ⬜ |
| Phase C | Mechanism Adapter — reason 机理推演 | ⬜ |
| Phase D | D-Former + LLM — 诊断 Token 桥接 + LoRA | ⬜ |
| Phase E | 广泛数据集验证 (RCAEval, FaultForge, TN-RCA) | ⬜ |

### Phase A 验收目标

| 任务 | 目标 | 状态 |
|------|------|------|
| 通用 Entity/Relation/Event Schema | Bank 容器、Market 服务、Telecom 混合实体无损表示 | 🟩 |
| OpenRCA 三系统 Adapter | 正常时段数据可训练，故障时段数据可评估 | 🟩 |
| Onset Head 替换窗口选择 | Bank clean eval Top-1 >= 45%（当前38%） | ⬜ 待实验 |
| time-component 联合准确率 | 不低于当前 clean eval | ⬜ 待实验 |
| strict zero-shot (OB→OpenRCA) | 报告 baseline 供 Phase B 对比 | ⬜ 待实验 |

---

## 项目结构

```
RCAWorld/
├── src/
│   ├── data/                          # master 分支 — Nezha/OpenRCA 解析器
│   ├── models/                        # master 分支 — Graph-RSSM (baseline)
│   ├── training/                      # master 分支 — 训练管线
│   ├── rca/                           # master 分支 — RCA 推理
│   ├── evaluation/                    # master 分支 — 评估
│   │
│   └── foundation/                    # ★ RCAWorld-Foundation (新架构)
│       ├── schema/
│       │   ├── entity.py              # Entity (16 类型), ObservationSpec
│       │   ├── relation.py            # Relation (9 类型)
│       │   ├── event.py               # ObservationEvent, Modality, EventBatch
│       │   ├── episode.py             # SystemEpisode, QueryMask, RootCauseLabel
│       │   └── vocabulary.py          # MechanismToken (17 tokens), label mapping
│       │
│       ├── adapters/
│       │   ├── base.py                # AdapterProtocol, BaseAdapter
│       │   ├── openrca_bank.py        # Bank 数据集适配器 (容器+服务)
│       │   ├── openrca_market.py      # Market 数据集适配器
│       │   ├── openrca_telecom.py     # Telecom 数据集适配器
│       │   └── generic_csv.py         # 通用 CSV 适配器
│       │
│       ├── models/
│       │   ├── entity_tokenizer.py    # MetricEncoder, TypeAwareDecoder
│       │   ├── hierarchical_rssm.py   # Type-modulated RSSM
│       │   ├── onset_head.py          # 三信号 onset 检测
│       │   ├── parallel_component_head.py  # p(c|t) + p(c) 双路 + alpha 融合
│       │   ├── flamingo_gca.py        # Flamingo 门控交叉注意力 + DiagnosticProjector
│       │   └── world_model.py         # RCAWorldFoundation 完整组装
│       │
│       ├── training/
│       │   ├── losses.py              # NLL+KL+onset+component+consistency
│       │   └── pretrain.py            # Pretrain + finetune 管线
│       │
│       └── evaluation/
│           └── openrca_metrics.py     # Top-K, MRR, onset accuracy, exact match
│
├── scripts/                           # master 分支 — 各 Phase 评估脚本
├── checkpoints/                       # 训练好的模型
├── data/processed/                    # 预处理 HDF5 数据
├── README.md                          # 本文档
├── STATUS.md                          # 进度跟踪
└── requirements.txt
```

---

## 关键设计决策

| 决策 | 说明 |
|------|------|
| **p(c\|t) + p(c) 双路并行** | 时间条件注意力 + 时间自由残差聚合，可学习 alpha 融合 — 同时处理突发故障和累积故障 |
| **Flamingo GCA** | 门控交叉注意力（alpha=0 初始化），比 BLIP-2 Q-Former 更轻量 — 适合 3B-8B LLM |
| **Onset Head** | 预测残差 + 潜态偏移 + 传播先行性三信号融合，替换固定 oracle 窗口 |
| **Type embeddings + obs_mask** | 异构实体编码：所有实体 pad 到 max_obs_dim，类型嵌入区分语义，mask 处理缺失特征 |
| **nn.Sequential + setup()** | 遵循 Flax 标准模式，已验证与 JAX 0.10 / Flax 0.12 兼容 |

---

## 技术栈

| 组件 | 选择 |
|------|------|
| ML Framework | JAX + Flax |
| 基线模型 (master) | Graph-RSSM (1.8M params) |
| 新架构 (foundation) | RCAWorldFoundation (HierarchicalRSSM + Onset + Component heads) |
| 评估基准 | Nezha (FSE 2023), OpenRCA (ICLR 2025) |

---

## 参考

- 原始 RCAWorld: [leninist1/RCAworld](https://github.com/leninist1/RCAworld)
- OpenRCA: [microsoft/OpenRCA](https://github.com/microsoft/OpenRCA) (ICLR 2025)
- Flamingo: [Alayrac et al., 2022](https://arxiv.org/abs/2204.14198)
- Nezha: [FSE 2023](https://github.com/IntelligentDDS/Nezha)
