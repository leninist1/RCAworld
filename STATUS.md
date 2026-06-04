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

---

## Phase 1: 数据预处理 🟩

| # | 任务 | 状态 | 备注 |
|---|------|------|------|
| 1.1 | Nezha 数据集克隆 | 🟩 | Online Boutique + TrainTicket, ~2.8GB |
| 1.2 | OpenRCA 数据集 | 🟩 | Bank + Telecom + Market, ~73GB |
| 1.3 | 解析器 (pod→service, span→edge) | 🟩 | `src/data/aiops2020/parse.py` |
| 1.4 | 正常/异常窗口分离 | 🟩 | construct_data (正常) / rca_data (故障) |
| 1.5 | Train/Val 划分 + HDF5 存储 | 🟩 | `data/processed/cross_system.h5` |

---

## Phase 2: 真实数据训练 (Online Boutique, 仅正常数据) 🟩

| # | 任务 | 状态 | 备注 |
|---|------|------|------|
| 2.1 | 模型配置 (N=10, E=14) | 🟩 | 1,835,028 参数 |
| 2.2 | 训练 (392 train, 98 val) | 🟩 | 80 epochs, loss: 0.84 → -1.33 |
| 2.3 | Checkpoint 保存 | 🟩 | `checkpoints/phase2/best/` |

---

## Phase 3-3.5: RCA 评估 + RQ3/RQ4 🟩

| # | 任务 | 状态 | 备注 |
|---|------|------|------|
| 3.1 | 预测残差计算 | 🟩 | per-service, 滑动窗口 |
| 3.2 | RQ3: 证据融合 (A+T+P+D) | 🟩 | 残差 80% Top-1 (oracle window), 融合持平 |
| 3.3 | RQ4: 图消融 | 🟩 | NLL 差异 <0.1%, metrics 是主要信号 |

---

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

> Oracle window = 用根因服务残差峰值定位窗口 → 测量模型排序质量上界  
> 所有结果仅用正常数据预训练，零 RCA 标签训练

---

## 泄露审查与 Clean Eval 🟩

发现两个 oracle leak，修复后重新评估：

### 泄露分析

| 泄露源 | 说明 | 影响 |
|--------|------|------|
| Window 选择 | 用根因服务的残差峰值定位评估窗口 | **主要泄露**（83%→21%，-62pp） |
| Posterior 模式 | 模型用当前观测编码潜状态 | 次要（posterior 8% vs prior 13%） |

### Clean eval（Prior + Timestamp window）

| 系统 | 粒度 | 故障数 | Top-1 | Top-3 | MRR | vs Random |
|------|------|--------|-------|-------|-----|-----------|
| OB Day1 | service | 24 | **21%** | 38% | 0.39 | 2.1× |
| Bank | container | 16 | **38%** | 50% | 0.50 | 3.8× |
| Random | — | — | 10% | 30% | 0.34 | 1.0× |

> Timestamp window = 用 record.csv 时间戳定位窗口（SRE 报告时间）

**核心结论**: 模型排序质量高（上界 83-100%），瓶颈在**窗口定位精度**而非残差质量。

---

## Phase 5: Logs 集成 🟩

| # | 任务 | 状态 | 备注 |
|---|------|------|------|
| 5.1 | 日志解析器 | 🟩 | `parse_log_features()` — 4 dims/svc (count/error/warn/severity) |
| 5.2 | 日志预测头 | 🟩 | `LogPredictor` — latent→log features MLP |
| 5.3 | 多模态训练集成 | ⬜ | 需要将 log features 纳入模型观测空间并重新训练 |

---

## Phase 6: 故障动作适配器 🟩

| # | 任务 | 状态 | 备注 |
|---|------|------|------|
| 6.1 | 故障类型词汇 | 🟩 | 5 种 Nezha 类型 (cpu_contention/network_delay/...) |
| 6.2 | Heuristic 推理 | 🟩 | 基于残差模式推断故障类型 |
| 6.3 | 训练型适配器 | ⬜ | 需要 Nezha fault 数据训练 fault-conditioned RSSM |

---

## Phase 7: 多粒度实体 🟩

| # | 任务 | 状态 | 备注 |
|---|------|------|------|
| 7.1 | Container 级指标解析 | 🟩 | `parse_container_metrics()` — pivot long→wide format |
| 7.2 | Bank 容器级建模 | 🟩 | 18 容器, 9 KPIs, **38% Top-1 clean** |
| 7.3 | Telecom 建模 | 🟥 | KPI 过少 (4 个), 需要 Logs 补充 |

---

## 关键发现总结

| 结论 | 证据 |
|------|------|
| 仅正常数据预训练即可 RCA | 全程零故障标签训练，OB 源域 oracle 100% Top-1 |
| 模型排序质量高 | Oracle window: OB 100%, TS 79%, Mkt 63% |
| **Window 定位是主要瓶颈** | 泄露去除后 83%→21%（-62pp），异常检测精度决定端到端性能 |
| 跨领域需粒度对齐 | Bank svc 级 0% → container 级 38%（↑3.8× random） |
| Metrics 是主要信号 | 图消融 <0.1% NLL 差异 |
| Reason (why) 需额外训练 | 故障类型推断需要 Nezha fault 标签 |

---

## 后续工作

| 优先级 | 内容 | 状态 |
|--------|------|------|
| high | 论文撰写 | ⬜ |
| high | 基线对比 (DynaCausal, CHASE, RUN) — 需在同一窗口内比较 | ⬜ |
| medium | Phase 5 完整集成: log-augmented 重新训练 | ⬜ |
| medium | Phase 6 完整集成: Nezha fault 数据训练 fault-conditioned RSSM | ⬜ |
| medium | Phase 7 Telecom: 补充 KPI 数据或日志后重新评估 | ⬜ |
| low | 改进异常检测精度以缩小 window 定位差距 | ⬜ |
