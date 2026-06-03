# MicroDynamics-RCA: 基于系统动力学世界模型的微服务根因分析

> 正常情况下，微服务系统的状态演化具有稳定规律。
> 根因是最早破坏这一规律，并且能够解释后续异常传播的实体。
> 将该实体恢复到正常状态后，世界模型推演出的下游异常应显著减弱。

---

## 一、研究边界（第一版）

| 维度 | 范围 |
|------|------|
| 粒度 | service-level（暂不细分 Pod/容器/API/代码行） |
| 模态 | Metrics + Traces（Logs 后续扩展） |
| 训练数据 | AIOps Challenge 2020 数据集（正常窗口预训练） |
| 泛化 | 系统内建模（不强求跨系统零样本迁移） |
| 排序器 | 不训练 LTR/分类器，使用透明证据融合 |

---

## 二、核心理念

### 从判别式 RCA 到生成式 RCA

现有 SOTA（DynaCausal, CHASE, RUN, CausalRCA 等）的训练范式：

> 输入故障窗口 → 学习判别式排序器 → 输出根因排名

本方案的训练范式：

> 仅输入正常窗口 → 学习系统状态转移规律（世界模型）→ RCA 阶段零额外训练

**不依赖故障标签训练排序器，不需要在 benchmark 上拟合 LTR。**

### 与现有工作的关键差异

| 维度 | DynaCausal (2025) | CHASE (2024) | RUN (2024, AAAI) | **本方案** |
|------|-------------------|-------------|-------------------|-----------|
| 训练范式 | 判别式（故障标签） | 判别式 | 判别式 | **生成式（仅正常数据）** |
| 学习目标 | Pairwise ranking loss | Hypergraph causal | Granger + PageRank | **正常动力学 ELBO + rollout** |
| 排序模型 | 学习的 ranker | 学习的 ranker | PageRank | **无（透明证据融合）** |
| 未来预测 | 无 | 无 | 时序预测 | **多步 rollout 预测** |
| 修复干预 | 无 | 无 | 无 | **潜态替换 + rollout 推演** |

---

## 三、系统状态表示

时间窗口：**10 秒**，历史上下文：**12 窗口（2 分钟）**，预测目标：**未来 1/3/6 窗口（10s/30s/60s）**

```
G_t = (V, E_t)
```

### 节点特征 (per-service, d_n 维)

| 类型 | 示例 |
|------|------|
| 资源状态 | CPU、内存、网络收发量 |
| 请求状态 | QPS、成功/失败请求数 |
| 延迟状态 | p50、p90、p99 latency |
| 容器状态 | 重启次数、运行实例数 |
| Trace 聚合 | span 数量、异常 span 数、平均 duration |

### 边特征 (per-edge, d_e 维)

| 特征 | 含义 |
|------|------|
| call_count | 当前窗口调用次数 |
| error_rate | 调用失败比例 |
| latency_p50/p95 | 调用延迟 |
| timeout_count | 超时次数 |

### 外部上下文

显式输入 workload（QPS、活跃用户、实例数），避免将正常流量上涨误判为故障。

---

## 四、模型架构：Graph-RSSM

```
Metrics + 动态 Trace 调用图
              │
    ┌─────────┴──────────┐
    │ Temporal Encoder    │  GRU (共享参数, per-service)
    │ Graph Encoder       │  2-layer GAT 消息传递
    │ Workload Context    │  QPS/CDP 显式输入
    └─────────┬──────────┘
              ▼
    ┌─────────────────────┐
    │  RSSM Latent Space   │  h_t (确定性) + z_t (随机性)
    │  Latent Dynamics     │  p(z_{t+1} | z_{≤t}, G_t, w_t)
    └─────────┬───────────┘
              │
    ┌─────────┼───────────┐
    │ Future Metrics Decoder   │  μ, σ per service (Gaussian NLL)
    │ Future Edge Decoder      │  edge existence + latency/error
    │ Masked Reconstruction    │  随机遮挡 → 从邻域恢复
    └─────────────────────────┘
```

- **Temporal Encoder**：GRU，所有服务共享参数
- **Graph Encoder**：2-layer GAT，建模服务间故障传播
- **RSSM Core**：确定性状态 h_t + 随机离散状态 z_t，概率性状态转移
- **Decoder**：防止潜空间退化，保留可解释的系统信息

---

## 五、训练目标（仅正常数据）

```
L = λ1·L_metric + λ2·L_edge + λ3·L_latent + λ4·L_mask + λ5·L_rollout
```

| 损失项 | 公式 | 作用 |
|--------|------|------|
| L_metric | Σ [(x-μ)²/(2σ²) + log σ] | 预测未来指标分布 (Gaussian NLL) |
| L_edge | BCE + MAE | 预测未来调用边及特征 |
| L_latent | KL(z_post ∥ z_prior) | 约束潜空间平滑可预测 |
| L_mask | 遮挡→邻域恢复 | 跨服务信息补偿 |
| L_rollout | 连续 3/6/12 步滚动预测 | 减少长期漂移 |

---

## 六、RCA 推理流程（5 步，零额外训练）

### Step 1: 推演"正常情况下应该发生什么"
```
z_t0 = E(O_{t0-11:t0})
Ô_{t0+1:t0+H} = F_θ(z_t0, G_{t0:t0+H}, w_{t0:t0+H})
```

### Step 2: 计算每服务预测残差
```
A_t(v) = -log p_θ(x_t^(v) | z_<t)
```

### Step 3: 提取传播链
```
首次显著偏离时间 + 异常持续时间 + 沿调用图传播一致性
```

### Step 4: 修复干预推演
```
z̃_t(v) ~ p_θ(z | normal)         # 从正常分布采样替换异常潜态
z̃_{t+1:t+H} = F_θ(z̃, G, w)       # 滚动推演下游 1-3 步
R(v) = Δ downstream anomaly       # 量化修复增益
```

### Step 5: 证据融合打分
```
S(v) = α·A(v) + β·T(v) + γ·P(v) + δ·R(v) - η·U(v)

A = 异常程度     T = 时间先行性     P = 拓扑传播一致性
R = 修复增益     U = 模型不确定性
```

权重不在 benchmark 上训练。可在自行生成数据上设定，或在开发集上调节。

---

## 七、数据来源

| 用途 | 数据集 | 说明 |
|------|--------|------|
| 主训练数据 | AIOps Challenge 2020 | 业务指标+平台指标+调用链+故障标签，~16GB |
| 后续扩展 | RCAEval / OpenRCA | evaluation-only，不用于训练排序器 |
| 后续扩展 | OTel Astronomy Shop | 自行采集正常轨迹，补充训练数据多样性 |

---

## 八、实验设计（四个 Research Question）

### RQ1: 模型是否真正学会正常动力学？
- Metrics NLL, MAE/RMSE, Edge F1, Multi-step drift (10s/30s/60s)

### RQ2: 预测残差能否提升 RCA？
- 对比：Z-score, EWMA, BARO, RUN, Graph-RSSM residual (逐步加入 onset/topology)
- 指标：AC@1, AC@3, Avg@5, MRR

### RQ3: 修复干预是否能区分根因和症状？
- 消融：无干预 / 归零 / 均值替换 / 正常潜态采样 / 采样+rollout

### RQ4: 动态图是否真的有用？
- 消融：MLP / GRU / GRU+静态图 / GRU+动态图 / Graph-RSSM

---

## 九、项目结构

```
mace/
├── src/
│   ├── data/
│   │   ├── synthetic.py              # Phase 0: 合成数据生成
│   │   └── aiops2020/
│   │       ├── download.py           # 数据集下载
│   │       ├── parse_metrics.py      # 业务指标 & 平台指标解析
│   │       ├── parse_traces.py       # 调用链解析 & 动态调用图重建
│   │       ├── parse_faults.py       # 故障标签解析
│   │       ├── split.py              # 按故障标签分离正常/异常窗口
│   │       └── dataset.py            # 时间对齐 & JAX Dataset
│   ├── models/
│   │   ├── temporal_encoder.py       # GRU (共享参数)
│   │   ├── graph_encoder.py          # GAT
│   │   ├── rssm.py                   # RSSM core (prior+posterior+transition)
│   │   ├── decoders.py               # Gaussian NLL + edge decoder
│   │   └── graph_rssm.py             # 完整 Graph-RSSM 组装
│   ├── training/
│   │   ├── train.py                  # 仅正常数据训练入口
│   │   └── losses.py                 # ELBO + rollout + mask 损失
│   ├── rca/
│   │   ├── residual.py               # 预测残差 + 首次偏离时间
│   │   ├── propagation.py            # 传播链提取
│   │   ├── intervention.py           # 修复干预推演
│   │   └── evidence_fusion.py        # 证据融合打分
│   └── evaluation/
│       ├── baselines/                # Z-score, EWMA, LSTM-AE, PC, RUN
│       └── metrics.py                # AC@K, MRR, MTTD, AURC
├── configs/                          # Hydra/OmegaConf 配置
├── notebooks/                        # EDA + 可视化
├── scripts/                          # 编排脚本
├── requirements.txt                  # jax, flax, optax, orbax, ...
├── README.md                         # 本文档
└── STATUS.md                         # 项目进度
```

---

## 十、实施计划

| Phase | 内容 | 时间 |
|-------|------|------|
| **Phase 0** | 合成数据 MVP + Graph-RSSM 原型验证 | Week 1 |
| **Phase 1** | AIOps 2020 数据处理（分离正常/故障窗口，构建动态图） | Week 1-2 |
| **Phase 2** | 在正常窗口上训练 Graph-RSSM | Week 2-4 |
| **Phase 3** | RCA 评估：残差+传播+修复推演+基线对比 | Week 4-5 |
| **Phase 4** | 论文撰写：消融实验+可视化+复现包 | Week 5-6 |

---

## 十一、后续扩展路径

### Phase 5: Logs 集成
- Drain 解析器 → 日志模板预测头 → 三模态完整世界模型

### Phase 6: 故障传播适配器
- 故障动作 encoding → p(z_{t+1} | z_t, G_t, w_t, a_t) → 匹配故障类型

### Phase 7: 多粒度实体
- Pod/容器/数据库/业务实体 → 层次化图 → 跨层根因定位

### Phase 8: NoiseLab 集成
- 世界模型输出 JSON 证据 → NoiseLab 证据筛选与融合 → LLM 解释生成

---

## 技术栈

| 组件 | 选择 |
|------|------|
| ML Framework | JAX + Flax |
| Optimizer | Optax |
| Checkpointing | Orbax |
| Config | Hydra |
| Logging | TensorBoard / WandB |
| Dataset | AIOps Challenge 2020 |

---

## 参考

- DynaCausal (arXiv:2510.22613)
- CHASE (arXiv:2406.19711)
- RUN (arXiv:2402.01140, AAAI 2024)
- IDI (OpenReview, 2025)
- DreamerV3 (arXiv:2301.04104)
- RCAEval (arXiv:2412.17015)
- OpenRCA (ICLR 2025)
