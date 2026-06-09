# Phase 0.9.2.3b — Bank Canonicalized KPI Residual Scale Audit Report

## 1. Executive Summary

- **Total raw records**: 12,218,094
- **Accepted by `bank_safe_v1`**: 163,849 (1.34%)
- **Dropped by `bank_safe_v1`**: 2,593,354 (21.23%)
- **Unmapped (no regex match)**: 9,460,891 (77.43%)
- **Canonical slots**: 4
- **Slots with scale-risk flags**: 4

Slots with highest flag counts:

- **cpu**: 29 flagged raw KPIs (of 29 total)
- **mem**: 29 flagged raw KPIs (of 29 total)
- **net_tx**: 10 flagged raw KPIs (of 28 total)
- **net_rx**: 5 flagged raw KPIs (of 24 total)

## 2. Per-Slot Summary

### 2.1. Slot `cpu`

- **Total rows**: 138,365
- **Raw KPI names**: 29
- **Unique components**: 18
- **Flagged KPIs**: 29

- **WARNING**: Possible 0-1 vs 0-100 ratio/percent mix detected in this slot.

- **Median scale mismatch ratio**: 395.9x (max: `OSLinux-CPU_CPU_CPUCpuUtil` = 25.73, min: `Container-DOCKER_CONTAINER_bbc780df1fce--bcou-role-st-uat-statefulset-0--bcou--UATWKR18_CpuPercent` = 0.065)

| Raw KPI Name | n | min | p01 | p05 | p50 | p95 | p99 | max | mean | std | IQR | MAD | z_ratio | neg_ratio | nfin | flags |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `Container-DOCKER_CONTAINER_01eeda2c9f0b--bcou-trace-st-uat-statefulset-1--bcou--UATWKR18_CpuPercent` | 50 | 0 | 0 | 0 | 0 | 0 | 0.1581 | 0.31 | 0.0062 | 0.0434 | 0 | 0 | 0.980 | 0.000 | 50 | MEDIAN_SCALE_MISMATCH, POSSIBLE_RATIO_PERCENT_MIX |
| `Container-DOCKER_CONTAINER_0f6f3aa7920c--bcou-role-st-uat-statefulset-0--bcou--UATWKR18_CpuPercent` | 60 | 0 | 0 | 0 | 0 | 0.676 | 5.054 | 9.91 | 0.2322 | 1.29 | 0 | 0 | 0.900 | 0.000 | 60 | MEDIAN_SCALE_MISMATCH, POSSIBLE_RATIO_PERCENT_MIX |
| `Container-DOCKER_CONTAINER_1bc4fc80d241--bcou-trace-st-uat-statefulset-1--bcou--UATWKR18_CpuPercent` | 204 | 0 | 0 | 0 | 0.07 | 1.155 | 3.272 | 6.12 | 0.331 | 0.681 | 0.4425 | 0.07 | 0.309 | 0.000 | 204 | MEDIAN_SCALE_MISMATCH, POSSIBLE_RATIO_PERCENT_MIX |
| `Container-DOCKER_CONTAINER_23bdcf67c7e3--bcou-role-st-uat-statefulset-0--bcou--UATWKR18_CpuPercent` | 74 | 0.04 | 0.04 | 0.05 | 0.27 | 1.15 | 1.737 | 2 | 0.4241 | 0.4175 | 0.5775 | 0.22 | 0.000 | 0.000 | 74 | MEDIAN_SCALE_MISMATCH, POSSIBLE_RATIO_PERCENT_MIX |
| `Container-DOCKER_CONTAINER_2c2336e2994f--bcou-role-st-uat-statefulset-1--bcou--UATWKR06_CpuPercent` | 541 | 0.05 | 0.06 | 0.08 | 0.36 | 1.06 | 1.424 | 1.85 | 0.4387 | 0.3304 | 0.43 | 0.2 | 0.000 | 0.000 | 541 | MEDIAN_SCALE_MISMATCH, POSSIBLE_RATIO_PERCENT_MIX |
| `Container-DOCKER_CONTAINER_2d16f5b2e830--bcou-trace-st-uat-statefulset-1--bcou--UATWKR18_CpuPercent` | 233 | 0 | 0 | 0 | 0.26 | 1.646 | 2.007 | 10.3 | 0.5685 | 0.841 | 0.82 | 0.2 | 0.064 | 0.000 | 233 | MEDIAN_SCALE_MISMATCH, POSSIBLE_RATIO_PERCENT_MIX |
| `Container-DOCKER_CONTAINER_2fa7eaebac26--bcou-role-st-uat-statefulset-1--bcou--UATWKR02_CpuPercent` | 120 | 0 | 0 | 0 | 0 | 0 | 0 | 14.28 | 0.119 | 1.298 | 0 | 0 | 0.992 | 0.000 | 120 | SPARSE_SIGNAL, MEDIAN_SCALE_MISMATCH, POSSIBLE_RATIO_PERCENT_MIX |
| `Container-DOCKER_CONTAINER_350771a68ac2--bcou-trace-st-uat-statefulset-1--bcou--UATWKR18_CpuPercent` | 140 | 0 | 0 | 0 | 0 | 0.3005 | 0.4144 | 0.45 | 0.05043 | 0.09568 | 0.07 | 0 | 0.657 | 0.000 | 140 | MEDIAN_SCALE_MISMATCH, POSSIBLE_RATIO_PERCENT_MIX |
| `Container-DOCKER_CONTAINER_3bf443a64876--bcou-role-st-uat-statefulset-1--bcou--UATWKR03_CpuPercent` | 946 | 0 | 0 | 0 | 0 | 23.64 | 42.91 | 64.61 | 3.767 | 8.924 | 1.755 | 0 | 0.740 | 0.000 | 946 | MEDIAN_SCALE_MISMATCH, POSSIBLE_RATIO_PERCENT_MIX |
| `Container-DOCKER_CONTAINER_464dc801314b--bcou-trace-st-uat-statefulset-0--bcou--UATWKR04_CpuPercent` | 248 | 0 | 0.08 | 0.09 | 0.62 | 2.362 | 3.037 | 4.16 | 0.8381 | 0.7811 | 1.06 | 0.5 | 0.004 | 0.000 | 248 | MEDIAN_SCALE_MISMATCH, POSSIBLE_RATIO_PERCENT_MIX |
| `Container-DOCKER_CONTAINER_69b53a78b2eb--bcou-role-st-uat-statefulset-1--bcou--UATWKR06_CpuPercent` | 296 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1.000 | 0.000 | 296 | CONSTANT_OR_NEAR_CONSTANT_SIGNAL, SPARSE_SIGNAL, MEDIAN_SCALE_MISMATCH, POSSIBLE_RATIO_PERCENT_MIX |
| `Container-DOCKER_CONTAINER_6d83a96887c3--bcou-role-st-uat-statefulset-0--bcou--UATWKR18_CpuPercent` | 58 | 0 | 0 | 0 | 0.435 | 1.276 | 1.753 | 2.26 | 0.5438 | 0.4644 | 0.6475 | 0.355 | 0.155 | 0.000 | 58 | MEDIAN_SCALE_MISMATCH, POSSIBLE_RATIO_PERCENT_MIX |
| `Container-DOCKER_CONTAINER_76d31070b844--bcou-role-st-uat-statefulset-1--bcou--UATWKR06_CpuPercent` | 246 | 0 | 0 | 0 | 0 | 0 | 0.429 | 1.18 | 0.01138 | 0.1043 | 0 | 0 | 0.988 | 0.000 | 246 | MEDIAN_SCALE_MISMATCH, POSSIBLE_RATIO_PERCENT_MIX |
| `Container-DOCKER_CONTAINER_7b4b80f345e0--bcou-role-st-uat-statefulset-0--bcou--UATWKR04_CpuPercent` | 411 | 0 | 0 | 0 | 0 | 0.91 | 1.274 | 3.08 | 0.2167 | 0.356 | 0.31 | 0 | 0.516 | 0.000 | 411 | MEDIAN_SCALE_MISMATCH, POSSIBLE_RATIO_PERCENT_MIX |
| `Container-DOCKER_CONTAINER_89f0c1e5346c--bcou-trace-st-uat-statefulset-0--bcou--UATWKR04_CpuPercent` | 345 | 0 | 0 | 0 | 0 | 0.786 | 1.186 | 3.53 | 0.1538 | 0.3169 | 0.14 | 0 | 0.603 | 0.000 | 345 | MEDIAN_SCALE_MISMATCH, POSSIBLE_RATIO_PERCENT_MIX |
| `Container-DOCKER_CONTAINER_94eca4f96efe--bcou-role-st-uat-statefulset-1--bcou--UATWKR04_CpuPercent` | 974 | 0 | 0 | 0 | 0 | 1.67 | 2.601 | 12.35 | 0.2644 | 0.8104 | 0.1 | 0 | 0.725 | 0.000 | 974 | MEDIAN_SCALE_MISMATCH, POSSIBLE_RATIO_PERCENT_MIX |
| `Container-DOCKER_CONTAINER_9e8c309d0aab--bcou-trace-st-uat-statefulset-0--bcou--UATWKR04_CpuPercent` | 399 | 0 | 0 | 0 | 0 | 1.814 | 2.598 | 9.68 | 0.4429 | 0.7974 | 0.73 | 0 | 0.516 | 0.000 | 399 | MEDIAN_SCALE_MISMATCH, POSSIBLE_RATIO_PERCENT_MIX |
| `Container-DOCKER_CONTAINER_ac3ba8476104--bcou-trace-st-uat-statefulset-1--bcou--UATWKR18_CpuPercent` | 4 | 0 | 0 | 0 | 0.135 | 10.21 | 11.61 | 11.96 | 3.058 | 5.141 | 3.193 | 0.135 | 0.500 | 0.000 | 4 | MEDIAN_SCALE_MISMATCH, POSSIBLE_RATIO_PERCENT_MIX |
| `Container-DOCKER_CONTAINER_b2fc383d2438--bcou-trace-st-uat-statefulset-1--bcou--UATWKR18_CpuPercent` | 55 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1.000 | 0.000 | 55 | CONSTANT_OR_NEAR_CONSTANT_SIGNAL, SPARSE_SIGNAL, MEDIAN_SCALE_MISMATCH, POSSIBLE_RATIO_PERCENT_MIX |
| `Container-DOCKER_CONTAINER_b30097144a13--bcou-trace-st-uat-statefulset-0--bcou--UATWKR04_CpuPercent` | 591 | 0 | 0 | 0 | 0.12 | 2.03 | 3.03 | 10.81 | 0.4516 | 0.8086 | 0.33 | 0.12 | 0.176 | 0.000 | 591 | MEDIAN_SCALE_MISMATCH, POSSIBLE_RATIO_PERCENT_MIX |
| `Container-DOCKER_CONTAINER_b6760337dc49--bcou-role-st-uat-statefulset-0--bcou--UATWKR18_CpuPercent` | 214 | 0 | 0 | 0 | 0.205 | 1.477 | 8.981 | 10.32 | 0.5801 | 1.25 | 0.72 | 0.205 | 0.121 | 0.000 | 214 | MEDIAN_SCALE_MISMATCH, POSSIBLE_RATIO_PERCENT_MIX |
| `Container-DOCKER_CONTAINER_bbc780df1fce--bcou-role-st-uat-statefulset-0--bcou--UATWKR18_CpuPercent` | 206 | 0 | 0 | 0 | 0.065 | 1.07 | 1.413 | 3.53 | 0.2478 | 0.4377 | 0.3375 | 0.065 | 0.398 | 0.000 | 206 | MEDIAN_SCALE_MISMATCH, POSSIBLE_RATIO_PERCENT_MIX |
| `Container-DOCKER_CONTAINER_c67422614c81--bcou-role-st-uat-statefulset-1--bcou--UATWKR02_CpuPercent` | 138 | 0 | 0 | 0 | 0 | 0.6815 | 2.988 | 21.18 | 0.2638 | 1.831 | 0 | 0 | 0.877 | 0.000 | 138 | MEDIAN_SCALE_MISMATCH, POSSIBLE_RATIO_PERCENT_MIX |
| `Container-DOCKER_CONTAINER_cb2bbb5e3f90--bcou-role-st-uat-statefulset-1--bcou--UATWKR02_CpuPercent` | 342 | 0 | 0 | 0 | 0.65 | 5.099 | 6.977 | 92.27 | 1.514 | 5.355 | 1.738 | 0.65 | 0.342 | 0.000 | 342 | EXTREME_OUTLIER_TAIL, MEDIAN_SCALE_MISMATCH, POSSIBLE_RATIO_PERCENT_MIX |
| `Container-DOCKER_CONTAINER_cd2b4a29291e--bcou-role-st-uat-statefulset-1--bcou--UATWKR04_CpuPercent` | 652 | 0 | 0 | 0 | 0 | 0 | 0.3547 | 10.49 | 0.02529 | 0.4165 | 0 | 0 | 0.975 | 0.000 | 652 | EXTREME_OUTLIER_TAIL, MEDIAN_SCALE_MISMATCH, POSSIBLE_RATIO_PERCENT_MIX |
| `Container-DOCKER_CONTAINER_d27123361435--bcou-trace-st-uat-statefulset-1--bcou--UATWKR18_CpuPercent` | 156 | 0.05 | 0.05 | 0.06 | 0.425 | 1.29 | 1.543 | 1.71 | 0.5045 | 0.3975 | 0.66 | 0.32 | 0.000 | 0.000 | 156 | MEDIAN_SCALE_MISMATCH, POSSIBLE_RATIO_PERCENT_MIX |
| `Container-DOCKER_CONTAINER_ef6cb138bb10--bcou-role-st-uat-statefulset-0--bcou--UATWKR18_CpuPercent` | 73 | 0 | 0 | 0 | 0 | 0.578 | 1.038 | 1.29 | 0.1123 | 0.245 | 0 | 0 | 0.753 | 0.000 | 73 | MEDIAN_SCALE_MISMATCH, POSSIBLE_RATIO_PERCENT_MIX |
| `Container-DOCKER_CONTAINER_f861998c398e--bcou-role-st-uat-statefulset-0--bcou--UATWKR18_CpuPercent` | 1 | 0.44 | 0.44 | 0.44 | 0.44 | 0.44 | 0.44 | 0.44 | 0.44 | 0 | 0 | 0 | 0.000 | 0.000 | 1 | MEDIAN_SCALE_MISMATCH, POSSIBLE_RATIO_PERCENT_MIX |
| `OSLinux-CPU_CPU_CPUCpuUtil` | 130588 | 0.2119 | 0.3284 | 0.4734 | 25.73 | 28.48 | 45.69 | 100 | 19.94 | 11.71 | 21.04 | 1.14 | 0.000 | 0.000 | 130588 | MEDIAN_SCALE_MISMATCH, POSSIBLE_RATIO_PERCENT_MIX |

### 2.2. Slot `mem`

- **Total rows**: 15,195
- **Raw KPI names**: 29
- **Unique components**: 8
- **Flagged KPIs**: 29

- **WARNING**: Possible 0-1 vs 0-100 ratio/percent mix detected in this slot.

| Raw KPI Name | n | min | p01 | p05 | p50 | p95 | p99 | max | mean | std | IQR | MAD | z_ratio | neg_ratio | nfin | flags |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `Container-DOCKER_CONTAINER_01eeda2c9f0b--bcou-trace-st-uat-statefulset-1--bcou--UATWKR18_MemPercent` | 50 | 0 | 0 | 0 | 0 | 0 | 29.13 | 57.12 | 1.142 | 7.997 | 0 | 0 | 0.980 | 0.000 | 50 | POSSIBLE_RATIO_PERCENT_MIX |
| `Container-DOCKER_CONTAINER_0f6f3aa7920c--bcou-role-st-uat-statefulset-0--bcou--UATWKR18_MemPercent` | 60 | 0 | 0 | 0 | 0 | 57.2 | 57.57 | 58.03 | 5.721 | 17.17 | 0 | 0 | 0.900 | 0.000 | 60 | POSSIBLE_RATIO_PERCENT_MIX |
| `Container-DOCKER_CONTAINER_1bc4fc80d241--bcou-trace-st-uat-statefulset-1--bcou--UATWKR18_MemPercent` | 204 | 0 | 0 | 0 | 75.45 | 95.76 | 95.8 | 95.84 | 56.37 | 39.02 | 82.31 | 20.17 | 0.309 | 0.000 | 204 | POSSIBLE_RATIO_PERCENT_MIX |
| `Container-DOCKER_CONTAINER_23bdcf67c7e3--bcou-role-st-uat-statefulset-0--bcou--UATWKR18_MemPercent` | 75 | 57.73 | 57.73 | 57.73 | 57.9 | 60.27 | 60.32 | 60.33 | 58.89 | 1.224 | 2.49 | 0.17 | 0.000 | 0.000 | 75 | POSSIBLE_RATIO_PERCENT_MIX |
| `Container-DOCKER_CONTAINER_2c2336e2994f--bcou-role-st-uat-statefulset-1--bcou--UATWKR06_MemPercent` | 552 | 57.4 | 57.44 | 57.58 | 60.15 | 60.31 | 60.31 | 60.43 | 60.01 | 0.6344 | 0.12 | 0.04 | 0.000 | 0.000 | 552 | POSSIBLE_RATIO_PERCENT_MIX |
| `Container-DOCKER_CONTAINER_2d16f5b2e830--bcou-trace-st-uat-statefulset-1--bcou--UATWKR18_MemPercent` | 233 | 0 | 0 | 0 | 75.76 | 94.16 | 94.22 | 94.24 | 67.63 | 21.81 | 16.82 | 16.55 | 0.064 | 0.000 | 233 | POSSIBLE_RATIO_PERCENT_MIX |
| `Container-DOCKER_CONTAINER_2fa7eaebac26--bcou-role-st-uat-statefulset-1--bcou--UATWKR02_MemPercent` | 120 | 0 | 0 | 0 | 0 | 0 | 0 | 57.31 | 0.4776 | 5.21 | 0 | 0 | 0.992 | 0.000 | 120 | SPARSE_SIGNAL, POSSIBLE_RATIO_PERCENT_MIX |
| `Container-DOCKER_CONTAINER_350771a68ac2--bcou-trace-st-uat-statefulset-1--bcou--UATWKR18_MemPercent` | 140 | 0 | 0 | 0 | 0 | 57.09 | 57.11 | 57.11 | 19.57 | 27.09 | 57.06 | 0 | 0.657 | 0.000 | 140 | POSSIBLE_RATIO_PERCENT_MIX |
| `Container-DOCKER_CONTAINER_3bf443a64876--bcou-role-st-uat-statefulset-1--bcou--UATWKR03_MemPercent` | 588 | 0 | 0 | 0 | 0 | 71.38 | 71.38 | 71.38 | 29.75 | 35.08 | 70.9 | 0 | 0.582 | 0.000 | 588 | POSSIBLE_RATIO_PERCENT_MIX |
| `Container-DOCKER_CONTAINER_464dc801314b--bcou-trace-st-uat-statefulset-0--bcou--UATWKR04_MemPercent` | 253 | 0 | 72.77 | 72.77 | 73.22 | 73.33 | 73.33 | 73.33 | 72.77 | 4.59 | 0.51 | 0.11 | 0.004 | 0.000 | 253 | POSSIBLE_RATIO_PERCENT_MIX |
| `Container-DOCKER_CONTAINER_69b53a78b2eb--bcou-role-st-uat-statefulset-1--bcou--UATWKR06_MemPercent` | 297 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1.000 | 0.000 | 297 | CONSTANT_OR_NEAR_CONSTANT_SIGNAL, SPARSE_SIGNAL, POSSIBLE_RATIO_PERCENT_MIX |
| `Container-DOCKER_CONTAINER_6d83a96887c3--bcou-role-st-uat-statefulset-0--bcou--UATWKR18_MemPercent` | 58 | 0 | 0 | 0 | 59.22 | 59.72 | 61.05 | 62.81 | 50.12 | 21.49 | 1.13 | 0.485 | 0.155 | 0.000 | 58 | POSSIBLE_RATIO_PERCENT_MIX |
| `Container-DOCKER_CONTAINER_76d31070b844--bcou-role-st-uat-statefulset-1--bcou--UATWKR06_MemPercent` | 278 | 0 | 0 | 0 | 0 | 0 | 13.66 | 59.38 | 0.6407 | 6.134 | 0 | 0 | 0.989 | 0.000 | 278 | POSSIBLE_RATIO_PERCENT_MIX |
| `Container-DOCKER_CONTAINER_7b4b80f345e0--bcou-role-st-uat-statefulset-0--bcou--UATWKR04_MemPercent` | 412 | 0 | 0 | 0 | 0 | 58.36 | 58.36 | 58.65 | 28.27 | 29.11 | 58.35 | 0 | 0.515 | 0.000 | 412 | POSSIBLE_RATIO_PERCENT_MIX |
| `Container-DOCKER_CONTAINER_89f0c1e5346c--bcou-trace-st-uat-statefulset-0--bcou--UATWKR04_MemPercent` | 303 | 0 | 0 | 0 | 0 | 60.43 | 60.44 | 60.44 | 21.1 | 28.77 | 60.26 | 0 | 0.650 | 0.000 | 303 | POSSIBLE_RATIO_PERCENT_MIX |
| `Container-DOCKER_CONTAINER_94eca4f96efe--bcou-role-st-uat-statefulset-1--bcou--UATWKR04_MemPercent` | 974 | 0 | 0 | 0 | 0 | 72.39 | 74.35 | 76.34 | 19.61 | 31.89 | 69.51 | 0 | 0.725 | 0.000 | 974 | POSSIBLE_RATIO_PERCENT_MIX |
| `Container-DOCKER_CONTAINER_9e8c309d0aab--bcou-trace-st-uat-statefulset-0--bcou--UATWKR04_MemPercent` | 423 | 0 | 0 | 0 | 58.06 | 76.63 | 76.67 | 78.99 | 35.56 | 35.15 | 74.91 | 18.61 | 0.489 | 0.000 | 423 | POSSIBLE_RATIO_PERCENT_MIX |
| `Container-DOCKER_CONTAINER_ac3ba8476104--bcou-trace-st-uat-statefulset-1--bcou--UATWKR18_MemPercent` | 4 | 0 | 0 | 0 | 28.55 | 88.43 | 92.85 | 93.96 | 37.76 | 39.95 | 66.31 | 28.55 | 0.500 | 0.000 | 4 | POSSIBLE_RATIO_PERCENT_MIX |
| `Container-DOCKER_CONTAINER_b2fc383d2438--bcou-trace-st-uat-statefulset-1--bcou--UATWKR18_MemPercent` | 56 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1.000 | 0.000 | 56 | CONSTANT_OR_NEAR_CONSTANT_SIGNAL, SPARSE_SIGNAL, POSSIBLE_RATIO_PERCENT_MIX |
| `Container-DOCKER_CONTAINER_b30097144a13--bcou-trace-st-uat-statefulset-0--bcou--UATWKR04_MemPercent` | 592 | 0 | 0 | 0 | 70.69 | 71.14 | 71.14 | 71.62 | 55.18 | 26.15 | 12.52 | 0.43 | 0.177 | 0.000 | 592 | POSSIBLE_RATIO_PERCENT_MIX |
| `Container-DOCKER_CONTAINER_b6760337dc49--bcou-role-st-uat-statefulset-0--bcou--UATWKR18_MemPercent` | 165 | 0 | 0 | 0 | 58.48 | 81.58 | 81.84 | 81.89 | 57.28 | 17.94 | 1.65 | 0.87 | 0.073 | 0.000 | 165 | POSSIBLE_RATIO_PERCENT_MIX |
| `Container-DOCKER_CONTAINER_bbc780df1fce--bcou-role-st-uat-statefulset-0--bcou--UATWKR18_MemPercent` | 206 | 0 | 0 | 0 | 57.83 | 77.41 | 77.58 | 77.58 | 40.79 | 33.91 | 77.3 | 19.56 | 0.398 | 0.000 | 206 | POSSIBLE_RATIO_PERCENT_MIX |
| `Container-DOCKER_CONTAINER_c67422614c81--bcou-role-st-uat-statefulset-1--bcou--UATWKR02_MemPercent` | 138 | 0 | 0 | 0 | 0 | 57.48 | 57.48 | 57.48 | 7.081 | 18.89 | 0 | 0 | 0.877 | 0.000 | 138 | POSSIBLE_RATIO_PERCENT_MIX |
| `Container-DOCKER_CONTAINER_cb2bbb5e3f90--bcou-role-st-uat-statefulset-1--bcou--UATWKR02_MemPercent` | 342 | 0 | 0 | 0 | 59.18 | 83.39 | 84.99 | 86.02 | 40.41 | 29.68 | 59.26 | 0.17 | 0.342 | 0.000 | 342 | POSSIBLE_RATIO_PERCENT_MIX |
| `Container-DOCKER_CONTAINER_cd2b4a29291e--bcou-role-st-uat-statefulset-1--bcou--UATWKR04_MemPercent` | 652 | 0 | 0 | 0 | 0 | 0 | 57.67 | 58.55 | 1.414 | 8.918 | 0 | 0 | 0.975 | 0.000 | 652 | POSSIBLE_RATIO_PERCENT_MIX |
| `Container-DOCKER_CONTAINER_d27123361435--bcou-trace-st-uat-statefulset-1--bcou--UATWKR18_MemPercent` | 156 | 57.2 | 57.29 | 57.65 | 58.2 | 58.89 | 58.89 | 58.89 | 58.33 | 0.4637 | 0.97 | 0.37 | 0.000 | 0.000 | 156 | POSSIBLE_RATIO_PERCENT_MIX |
| `Container-DOCKER_CONTAINER_ef6cb138bb10--bcou-role-st-uat-statefulset-0--bcou--UATWKR18_MemPercent` | 74 | 0 | 0 | 0 | 0 | 59.99 | 59.99 | 59.99 | 14.47 | 25.52 | 0 | 0 | 0.757 | 0.000 | 74 | POSSIBLE_RATIO_PERCENT_MIX |
| `Container-DOCKER_CONTAINER_f861998c398e--bcou-role-st-uat-statefulset-0--bcou--UATWKR18_MemPercent` | 1 | 58.8 | 58.8 | 58.8 | 58.8 | 58.8 | 58.8 | 58.8 | 58.8 | 0 | 0 | 0 | 0.000 | 0.000 | 1 | POSSIBLE_RATIO_PERCENT_MIX |
| `Tomcat-MEMORY_7441-MEMORY_JVMMemoryUsedPercent` | 7789 | 7 | 22 | 24 | 37 | 51 | 75 | 89 | 37.14 | 9.56 | 11 | 6 | 0.000 | 0.000 | 7789 | POSSIBLE_RATIO_PERCENT_MIX |

### 2.3. Slot `net_rx`

- **Total rows**: 2,616
- **Raw KPI names**: 24
- **Unique components**: 4
- **Flagged KPIs**: 5

| Raw KPI Name | n | min | p01 | p05 | p50 | p95 | p99 | max | mean | std | IQR | MAD | z_ratio | neg_ratio | nfin | flags |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `Container-DOCKER_CONTAINER_0f6f3aa7920c--bcou-role-st-uat-statefulset-0--bcou--UATWKR18_NetworkRxBytes` | 5 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1.000 | 0.000 | 5 | — |
| `Container-DOCKER_CONTAINER_1bc4fc80d241--bcou-trace-st-uat-statefulset-1--bcou--UATWKR18_NetworkRxBytes` | 87 | 9.026 | 9.026 | 9.1 | 12.9 | 14.75 | 14.96 | 15.1 | 12.32 | 1.992 | 3.788 | 1.332 | 0.000 | 0.000 | 87 | — |
| `Container-DOCKER_CONTAINER_23bdcf67c7e3--bcou-role-st-uat-statefulset-0--bcou--UATWKR18_NetworkRxBytes` | 69 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1.000 | 0.000 | 69 | CONSTANT_OR_NEAR_CONSTANT_SIGNAL, SPARSE_SIGNAL |
| `Container-DOCKER_CONTAINER_2c2336e2994f--bcou-role-st-uat-statefulset-1--bcou--UATWKR06_NetworkRxBytes` | 559 | 9.645 | 9.753 | 9.884 | 13.44 | 14.65 | 14.98 | 15.19 | 12.98 | 1.527 | 1.875 | 0.8532 | 0.000 | 0.000 | 559 | — |
| `Container-DOCKER_CONTAINER_2d16f5b2e830--bcou-trace-st-uat-statefulset-1--bcou--UATWKR18_NetworkRxBytes` | 145 | 9.026 | 9.132 | 9.645 | 13.24 | 14.99 | 15.2 | 15.38 | 12.48 | 2.022 | 4.269 | 1.333 | 0.000 | 0.000 | 145 | — |
| `Container-DOCKER_CONTAINER_2fa7eaebac26--bcou-role-st-uat-statefulset-1--bcou--UATWKR02_NetworkRxBytes` | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1.000 | 0.000 | 1 | — |
| `Container-DOCKER_CONTAINER_350771a68ac2--bcou-trace-st-uat-statefulset-1--bcou--UATWKR18_NetworkRxBytes` | 29 | 9.026 | 9.026 | 9.026 | 9.586 | 13.94 | 14.07 | 14.12 | 10.54 | 1.761 | 2.388 | 0.5605 | 0.000 | 0.000 | 29 | — |
| `Container-DOCKER_CONTAINER_3bf443a64876--bcou-role-st-uat-statefulset-1--bcou--UATWKR03_NetworkRxBytes` | 154 | 9.645 | 9.645 | 9.753 | 13.44 | 14.84 | 15.27 | 15.92 | 12.78 | 1.799 | 2.634 | 0.8901 | 0.000 | 0.000 | 154 | — |
| `Container-DOCKER_CONTAINER_464dc801314b--bcou-trace-st-uat-statefulset-0--bcou--UATWKR04_NetworkRxBytes` | 181 | 9.026 | 9.026 | 9.142 | 13.54 | 15.04 | 15.36 | 15.44 | 12.78 | 2.029 | 2.946 | 1.105 | 0.000 | 0.000 | 181 | — |
| `Container-DOCKER_CONTAINER_6d83a96887c3--bcou-role-st-uat-statefulset-0--bcou--UATWKR18_NetworkRxBytes` | 26 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1.000 | 0.000 | 26 | CONSTANT_OR_NEAR_CONSTANT_SIGNAL, SPARSE_SIGNAL |
| `Container-DOCKER_CONTAINER_76d31070b844--bcou-role-st-uat-statefulset-1--bcou--UATWKR06_NetworkRxBytes` | 1 | 13.9 | 13.9 | 13.9 | 13.9 | 13.9 | 13.9 | 13.9 | 13.9 | 0 | 0 | 0 | 0.000 | 0.000 | 1 | — |
| `Container-DOCKER_CONTAINER_7b4b80f345e0--bcou-role-st-uat-statefulset-0--bcou--UATWKR04_NetworkRxBytes` | 150 | 9.026 | 9.026 | 9.026 | 12.29 | 14.13 | 14.25 | 14.52 | 11.69 | 1.964 | 4.091 | 1.588 | 0.000 | 0.000 | 150 | — |
| `Container-DOCKER_CONTAINER_89f0c1e5346c--bcou-trace-st-uat-statefulset-0--bcou--UATWKR04_NetworkRxBytes` | 94 | 9.026 | 9.026 | 9.1 | 11.76 | 14.5 | 14.75 | 14.78 | 11.63 | 2.104 | 4.327 | 2.335 | 0.000 | 0.000 | 94 | — |
| `Container-DOCKER_CONTAINER_94eca4f96efe--bcou-role-st-uat-statefulset-1--bcou--UATWKR04_NetworkRxBytes` | 171 | 9.026 | 9.026 | 9.026 | 13.05 | 14.69 | 14.81 | 14.92 | 12.36 | 2.064 | 4.457 | 1.246 | 0.000 | 0.000 | 171 | — |
| `Container-DOCKER_CONTAINER_9e8c309d0aab--bcou-trace-st-uat-statefulset-0--bcou--UATWKR04_NetworkRxBytes` | 171 | 9.645 | 9.674 | 9.808 | 13.87 | 14.99 | 15.43 | 15.57 | 13.15 | 1.75 | 1.992 | 0.7588 | 0.000 | 0.000 | 171 | — |
| `Container-DOCKER_CONTAINER_b30097144a13--bcou-trace-st-uat-statefulset-0--bcou--UATWKR04_NetworkRxBytes` | 316 | 9.026 | 9.026 | 9.026 | 11.22 | 14.84 | 15.2 | 15.48 | 11.37 | 2.141 | 4.055 | 1.937 | 0.000 | 0.000 | 316 | — |
| `Container-DOCKER_CONTAINER_b6760337dc49--bcou-role-st-uat-statefulset-0--bcou--UATWKR18_NetworkRxBytes` | 103 | 9.026 | 9.026 | 9.1 | 12.66 | 14.51 | 14.81 | 14.86 | 12.06 | 2.056 | 4.254 | 1.617 | 0.000 | 0.000 | 103 | — |
| `Container-DOCKER_CONTAINER_bbc780df1fce--bcou-role-st-uat-statefulset-0--bcou--UATWKR18_NetworkRxBytes` | 89 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1.000 | 0.000 | 89 | CONSTANT_OR_NEAR_CONSTANT_SIGNAL, SPARSE_SIGNAL |
| `Container-DOCKER_CONTAINER_c67422614c81--bcou-role-st-uat-statefulset-1--bcou--UATWKR02_NetworkRxBytes` | 10 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1.000 | 0.000 | 10 | CONSTANT_OR_NEAR_CONSTANT_SIGNAL, SPARSE_SIGNAL |
| `Container-DOCKER_CONTAINER_cb2bbb5e3f90--bcou-role-st-uat-statefulset-1--bcou--UATWKR02_NetworkRxBytes` | 142 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1.000 | 0.000 | 142 | CONSTANT_OR_NEAR_CONSTANT_SIGNAL, SPARSE_SIGNAL |
| `Container-DOCKER_CONTAINER_cd2b4a29291e--bcou-role-st-uat-statefulset-1--bcou--UATWKR04_NetworkRxBytes` | 10 | 10.86 | 10.87 | 10.9 | 11.95 | 13.25 | 13.33 | 13.34 | 11.96 | 0.8162 | 1.076 | 0.6267 | 0.000 | 0.000 | 10 | — |
| `Container-DOCKER_CONTAINER_d27123361435--bcou-trace-st-uat-statefulset-1--bcou--UATWKR18_NetworkRxBytes` | 90 | 9.026 | 9.026 | 9.1 | 13.55 | 15.02 | 15.19 | 15.23 | 12.95 | 1.849 | 1.733 | 0.9118 | 0.000 | 0.000 | 90 | — |
| `Container-DOCKER_CONTAINER_ef6cb138bb10--bcou-role-st-uat-statefulset-0--bcou--UATWKR18_NetworkRxBytes` | 12 | 9.645 | 9.657 | 9.705 | 12.65 | 13.79 | 14.01 | 14.07 | 12.15 | 1.454 | 2.393 | 0.7816 | 0.000 | 0.000 | 12 | — |
| `Container-DOCKER_CONTAINER_f861998c398e--bcou-role-st-uat-statefulset-0--bcou--UATWKR18_NetworkRxBytes` | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1.000 | 0.000 | 1 | — |

### 2.4. Slot `net_tx`

- **Total rows**: 7,673
- **Raw KPI names**: 28
- **Unique components**: 4
- **Flagged KPIs**: 10

| Raw KPI Name | n | min | p01 | p05 | p50 | p95 | p99 | max | mean | std | IQR | MAD | z_ratio | neg_ratio | nfin | flags |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `Container-DOCKER_CONTAINER_01eeda2c9f0b--bcou-trace-st-uat-statefulset-1--bcou--UATWKR18_NetworkTxBytes` | 50 | 0 | 0 | 0 | 0 | 0 | 5.927 | 11.62 | 0.2324 | 1.627 | 0 | 0 | 0.980 | 0.000 | 50 | — |
| `Container-DOCKER_CONTAINER_0f6f3aa7920c--bcou-role-st-uat-statefulset-0--bcou--UATWKR18_NetworkTxBytes` | 60 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1.000 | 0.000 | 60 | CONSTANT_OR_NEAR_CONSTANT_SIGNAL, SPARSE_SIGNAL |
| `Container-DOCKER_CONTAINER_1bc4fc80d241--bcou-trace-st-uat-statefulset-1--bcou--UATWKR18_NetworkTxBytes` | 204 | 0 | 0 | 0 | 7.834 | 12.71 | 12.91 | 13.36 | 7.101 | 5.049 | 11.71 | 4.062 | 0.309 | 0.000 | 204 | — |
| `Container-DOCKER_CONTAINER_23bdcf67c7e3--bcou-role-st-uat-statefulset-0--bcou--UATWKR18_NetworkTxBytes` | 113 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1.000 | 0.000 | 113 | CONSTANT_OR_NEAR_CONSTANT_SIGNAL, SPARSE_SIGNAL |
| `Container-DOCKER_CONTAINER_2c2336e2994f--bcou-role-st-uat-statefulset-1--bcou--UATWKR06_NetworkTxBytes` | 563 | 7.599 | 7.734 | 7.883 | 11.77 | 13.01 | 13.33 | 13.65 | 11.36 | 1.554 | 1.789 | 0.7687 | 0.000 | 0.000 | 563 | — |
| `Container-DOCKER_CONTAINER_2d16f5b2e830--bcou-trace-st-uat-statefulset-1--bcou--UATWKR18_NetworkTxBytes` | 234 | 0 | 0 | 0 | 11.48 | 12.89 | 13.16 | 14.25 | 9.976 | 3.268 | 4.423 | 1.274 | 0.064 | 0.000 | 234 | — |
| `Container-DOCKER_CONTAINER_2fa7eaebac26--bcou-role-st-uat-statefulset-1--bcou--UATWKR02_NetworkTxBytes` | 120 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1.000 | 0.000 | 120 | CONSTANT_OR_NEAR_CONSTANT_SIGNAL, SPARSE_SIGNAL |
| `Container-DOCKER_CONTAINER_350771a68ac2--bcou-trace-st-uat-statefulset-1--bcou--UATWKR18_NetworkTxBytes` | 139 | 0 | 0 | 0 | 0 | 10.84 | 11.59 | 12.56 | 2.971 | 4.267 | 7.342 | 0 | 0.662 | 0.000 | 139 | — |
| `Container-DOCKER_CONTAINER_3bf443a64876--bcou-role-st-uat-statefulset-1--bcou--UATWKR03_NetworkTxBytes` | 601 | 0 | 0 | 0 | 0 | 12.7 | 13.09 | 13.49 | 4.481 | 5.488 | 11.22 | 0 | 0.589 | 0.000 | 601 | — |
| `Container-DOCKER_CONTAINER_464dc801314b--bcou-trace-st-uat-statefulset-0--bcou--UATWKR04_NetworkTxBytes` | 284 | 0 | 7.06 | 7.06 | 11.56 | 13.02 | 13.2 | 13.37 | 10.87 | 2.095 | 2.408 | 1.044 | 0.004 | 0.000 | 284 | — |
| `Container-DOCKER_CONTAINER_69b53a78b2eb--bcou-role-st-uat-statefulset-1--bcou--UATWKR06_NetworkTxBytes` | 352 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1.000 | 0.000 | 352 | CONSTANT_OR_NEAR_CONSTANT_SIGNAL, SPARSE_SIGNAL |
| `Container-DOCKER_CONTAINER_6d83a96887c3--bcou-role-st-uat-statefulset-0--bcou--UATWKR18_NetworkTxBytes` | 58 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1.000 | 0.000 | 58 | CONSTANT_OR_NEAR_CONSTANT_SIGNAL, SPARSE_SIGNAL |
| `Container-DOCKER_CONTAINER_76d31070b844--bcou-role-st-uat-statefulset-1--bcou--UATWKR06_NetworkTxBytes` | 318 | 0 | 0 | 0 | 0 | 0 | 0 | 13.19 | 0.08101 | 1.019 | 0 | 0 | 0.994 | 0.000 | 318 | SPARSE_SIGNAL |
| `Container-DOCKER_CONTAINER_7b4b80f345e0--bcou-role-st-uat-statefulset-0--bcou--UATWKR04_NetworkTxBytes` | 450 | 0 | 0 | 0 | 7.06 | 12.02 | 12.4 | 12.79 | 5.009 | 5.101 | 10.62 | 5.338 | 0.489 | 0.000 | 450 | — |
| `Container-DOCKER_CONTAINER_89f0c1e5346c--bcou-trace-st-uat-statefulset-0--bcou--UATWKR04_NetworkTxBytes` | 297 | 0 | 0 | 0 | 0 | 12.03 | 12.6 | 13.51 | 3.56 | 4.929 | 7.75 | 0 | 0.643 | 0.000 | 297 | — |
| `Container-DOCKER_CONTAINER_94eca4f96efe--bcou-role-st-uat-statefulset-1--bcou--UATWKR04_NetworkTxBytes` | 832 | 0 | 0 | 0 | 0 | 12.62 | 13.01 | 13.16 | 3.417 | 5.112 | 7.829 | 0 | 0.679 | 0.000 | 832 | — |
| `Container-DOCKER_CONTAINER_9e8c309d0aab--bcou-trace-st-uat-statefulset-0--bcou--UATWKR04_NetworkTxBytes` | 598 | 0 | 0 | 0 | 7.87 | 12.97 | 13.28 | 13.51 | 6.27 | 5.759 | 12.18 | 5.049 | 0.443 | 0.000 | 598 | — |
| `Container-DOCKER_CONTAINER_ac3ba8476104--bcou-trace-st-uat-statefulset-1--bcou--UATWKR18_NetworkTxBytes` | 4 | 0 | 0 | 0 | 3.201 | 10.28 | 10.83 | 10.97 | 4.342 | 4.632 | 7.543 | 3.201 | 0.500 | 0.000 | 4 | — |
| `Container-DOCKER_CONTAINER_b2fc383d2438--bcou-trace-st-uat-statefulset-1--bcou--UATWKR18_NetworkTxBytes` | 55 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1.000 | 0.000 | 55 | CONSTANT_OR_NEAR_CONSTANT_SIGNAL, SPARSE_SIGNAL |
| `Container-DOCKER_CONTAINER_b30097144a13--bcou-trace-st-uat-statefulset-0--bcou--UATWKR04_NetworkTxBytes` | 598 | 0 | 0 | 0 | 7.59 | 12.78 | 13.24 | 13.64 | 7.82 | 4.178 | 4.085 | 2.922 | 0.182 | 0.000 | 598 | — |
| `Container-DOCKER_CONTAINER_b6760337dc49--bcou-role-st-uat-statefulset-0--bcou--UATWKR18_NetworkTxBytes` | 176 | 0 | 0 | 0 | 10.63 | 12.8 | 13.03 | 13.22 | 9.501 | 3.434 | 4.408 | 2.015 | 0.080 | 0.000 | 176 | — |
| `Container-DOCKER_CONTAINER_bbc780df1fce--bcou-role-st-uat-statefulset-0--bcou--UATWKR18_NetworkTxBytes` | 205 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1.000 | 0.000 | 205 | CONSTANT_OR_NEAR_CONSTANT_SIGNAL, SPARSE_SIGNAL |
| `Container-DOCKER_CONTAINER_c67422614c81--bcou-role-st-uat-statefulset-1--bcou--UATWKR02_NetworkTxBytes` | 137 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1.000 | 0.000 | 137 | CONSTANT_OR_NEAR_CONSTANT_SIGNAL, SPARSE_SIGNAL |
| `Container-DOCKER_CONTAINER_cb2bbb5e3f90--bcou-role-st-uat-statefulset-1--bcou--UATWKR02_NetworkTxBytes` | 343 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1.000 | 0.000 | 343 | CONSTANT_OR_NEAR_CONSTANT_SIGNAL, SPARSE_SIGNAL |
| `Container-DOCKER_CONTAINER_cd2b4a29291e--bcou-role-st-uat-statefulset-1--bcou--UATWKR04_NetworkTxBytes` | 652 | 0 | 0 | 0 | 0 | 0 | 10.64 | 11.93 | 0.2506 | 1.595 | 0 | 0 | 0.975 | 0.000 | 652 | — |
| `Container-DOCKER_CONTAINER_d27123361435--bcou-trace-st-uat-statefulset-1--bcou--UATWKR18_NetworkTxBytes` | 156 | 7.06 | 7.06 | 7.304 | 11.7 | 13.01 | 13.31 | 13.4 | 11.16 | 1.795 | 1.579 | 0.8196 | 0.000 | 0.000 | 156 | — |
| `Container-DOCKER_CONTAINER_ef6cb138bb10--bcou-role-st-uat-statefulset-0--bcou--UATWKR18_NetworkTxBytes` | 73 | 0 | 0 | 0 | 0 | 11.99 | 12.37 | 12.43 | 2.509 | 4.477 | 0 | 0 | 0.753 | 0.000 | 73 | — |
| `Container-DOCKER_CONTAINER_f861998c398e--bcou-role-st-uat-statefulset-0--bcou--UATWKR18_NetworkTxBytes` | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1.000 | 0.000 | 1 | — |

## 3. Flagged Raw KPIs

| # | Slot | Raw KPI Name | Flags | n | p01 | p50 | p95 | p99 | max |
|---|---|---|---|---|---|---|---|---|---|
| 1 | `cpu` | `Container-DOCKER_CONTAINER_01eeda2c9f0b--bcou-trace-st-uat-statefulset-1--bcou--UATWKR18_CpuPercent` | MEDIAN_SCALE_MISMATCH, POSSIBLE_RATIO_PERCENT_MIX | 50 | 0 | 0 | 0 | 0.1581 | 0.31 |
| 2 | `cpu` | `Container-DOCKER_CONTAINER_0f6f3aa7920c--bcou-role-st-uat-statefulset-0--bcou--UATWKR18_CpuPercent` | MEDIAN_SCALE_MISMATCH, POSSIBLE_RATIO_PERCENT_MIX | 60 | 0 | 0 | 0.676 | 5.054 | 9.91 |
| 3 | `cpu` | `Container-DOCKER_CONTAINER_1bc4fc80d241--bcou-trace-st-uat-statefulset-1--bcou--UATWKR18_CpuPercent` | MEDIAN_SCALE_MISMATCH, POSSIBLE_RATIO_PERCENT_MIX | 204 | 0 | 0.07 | 1.155 | 3.272 | 6.12 |
| 4 | `cpu` | `Container-DOCKER_CONTAINER_23bdcf67c7e3--bcou-role-st-uat-statefulset-0--bcou--UATWKR18_CpuPercent` | MEDIAN_SCALE_MISMATCH, POSSIBLE_RATIO_PERCENT_MIX | 74 | 0.04 | 0.27 | 1.15 | 1.737 | 2 |
| 5 | `cpu` | `Container-DOCKER_CONTAINER_2c2336e2994f--bcou-role-st-uat-statefulset-1--bcou--UATWKR06_CpuPercent` | MEDIAN_SCALE_MISMATCH, POSSIBLE_RATIO_PERCENT_MIX | 541 | 0.06 | 0.36 | 1.06 | 1.424 | 1.85 |
| 6 | `cpu` | `Container-DOCKER_CONTAINER_2d16f5b2e830--bcou-trace-st-uat-statefulset-1--bcou--UATWKR18_CpuPercent` | MEDIAN_SCALE_MISMATCH, POSSIBLE_RATIO_PERCENT_MIX | 233 | 0 | 0.26 | 1.646 | 2.007 | 10.3 |
| 7 | `cpu` | `Container-DOCKER_CONTAINER_2fa7eaebac26--bcou-role-st-uat-statefulset-1--bcou--UATWKR02_CpuPercent` | SPARSE_SIGNAL, MEDIAN_SCALE_MISMATCH, POSSIBLE_RATIO_PERCENT_MIX | 120 | 0 | 0 | 0 | 0 | 14.28 |
| 8 | `cpu` | `Container-DOCKER_CONTAINER_350771a68ac2--bcou-trace-st-uat-statefulset-1--bcou--UATWKR18_CpuPercent` | MEDIAN_SCALE_MISMATCH, POSSIBLE_RATIO_PERCENT_MIX | 140 | 0 | 0 | 0.3005 | 0.4144 | 0.45 |
| 9 | `cpu` | `Container-DOCKER_CONTAINER_3bf443a64876--bcou-role-st-uat-statefulset-1--bcou--UATWKR03_CpuPercent` | MEDIAN_SCALE_MISMATCH, POSSIBLE_RATIO_PERCENT_MIX | 946 | 0 | 0 | 23.64 | 42.91 | 64.61 |
| 10 | `cpu` | `Container-DOCKER_CONTAINER_464dc801314b--bcou-trace-st-uat-statefulset-0--bcou--UATWKR04_CpuPercent` | MEDIAN_SCALE_MISMATCH, POSSIBLE_RATIO_PERCENT_MIX | 248 | 0.08 | 0.62 | 2.362 | 3.037 | 4.16 |
| 11 | `cpu` | `Container-DOCKER_CONTAINER_69b53a78b2eb--bcou-role-st-uat-statefulset-1--bcou--UATWKR06_CpuPercent` | CONSTANT_OR_NEAR_CONSTANT_SIGNAL, SPARSE_SIGNAL, MEDIAN_SCALE_MISMATCH, POSSIBLE_RATIO_PERCENT_MIX | 296 | 0 | 0 | 0 | 0 | 0 |
| 12 | `cpu` | `Container-DOCKER_CONTAINER_6d83a96887c3--bcou-role-st-uat-statefulset-0--bcou--UATWKR18_CpuPercent` | MEDIAN_SCALE_MISMATCH, POSSIBLE_RATIO_PERCENT_MIX | 58 | 0 | 0.435 | 1.276 | 1.753 | 2.26 |
| 13 | `cpu` | `Container-DOCKER_CONTAINER_76d31070b844--bcou-role-st-uat-statefulset-1--bcou--UATWKR06_CpuPercent` | MEDIAN_SCALE_MISMATCH, POSSIBLE_RATIO_PERCENT_MIX | 246 | 0 | 0 | 0 | 0.429 | 1.18 |
| 14 | `cpu` | `Container-DOCKER_CONTAINER_7b4b80f345e0--bcou-role-st-uat-statefulset-0--bcou--UATWKR04_CpuPercent` | MEDIAN_SCALE_MISMATCH, POSSIBLE_RATIO_PERCENT_MIX | 411 | 0 | 0 | 0.91 | 1.274 | 3.08 |
| 15 | `cpu` | `Container-DOCKER_CONTAINER_89f0c1e5346c--bcou-trace-st-uat-statefulset-0--bcou--UATWKR04_CpuPercent` | MEDIAN_SCALE_MISMATCH, POSSIBLE_RATIO_PERCENT_MIX | 345 | 0 | 0 | 0.786 | 1.186 | 3.53 |
| 16 | `cpu` | `Container-DOCKER_CONTAINER_94eca4f96efe--bcou-role-st-uat-statefulset-1--bcou--UATWKR04_CpuPercent` | MEDIAN_SCALE_MISMATCH, POSSIBLE_RATIO_PERCENT_MIX | 974 | 0 | 0 | 1.67 | 2.601 | 12.35 |
| 17 | `cpu` | `Container-DOCKER_CONTAINER_9e8c309d0aab--bcou-trace-st-uat-statefulset-0--bcou--UATWKR04_CpuPercent` | MEDIAN_SCALE_MISMATCH, POSSIBLE_RATIO_PERCENT_MIX | 399 | 0 | 0 | 1.814 | 2.598 | 9.68 |
| 18 | `cpu` | `Container-DOCKER_CONTAINER_ac3ba8476104--bcou-trace-st-uat-statefulset-1--bcou--UATWKR18_CpuPercent` | MEDIAN_SCALE_MISMATCH, POSSIBLE_RATIO_PERCENT_MIX | 4 | 0 | 0.135 | 10.21 | 11.61 | 11.96 |
| 19 | `cpu` | `Container-DOCKER_CONTAINER_b2fc383d2438--bcou-trace-st-uat-statefulset-1--bcou--UATWKR18_CpuPercent` | CONSTANT_OR_NEAR_CONSTANT_SIGNAL, SPARSE_SIGNAL, MEDIAN_SCALE_MISMATCH, POSSIBLE_RATIO_PERCENT_MIX | 55 | 0 | 0 | 0 | 0 | 0 |
| 20 | `cpu` | `Container-DOCKER_CONTAINER_b30097144a13--bcou-trace-st-uat-statefulset-0--bcou--UATWKR04_CpuPercent` | MEDIAN_SCALE_MISMATCH, POSSIBLE_RATIO_PERCENT_MIX | 591 | 0 | 0.12 | 2.03 | 3.03 | 10.81 |
| 21 | `cpu` | `Container-DOCKER_CONTAINER_b6760337dc49--bcou-role-st-uat-statefulset-0--bcou--UATWKR18_CpuPercent` | MEDIAN_SCALE_MISMATCH, POSSIBLE_RATIO_PERCENT_MIX | 214 | 0 | 0.205 | 1.477 | 8.981 | 10.32 |
| 22 | `cpu` | `Container-DOCKER_CONTAINER_bbc780df1fce--bcou-role-st-uat-statefulset-0--bcou--UATWKR18_CpuPercent` | MEDIAN_SCALE_MISMATCH, POSSIBLE_RATIO_PERCENT_MIX | 206 | 0 | 0.065 | 1.07 | 1.413 | 3.53 |
| 23 | `cpu` | `Container-DOCKER_CONTAINER_c67422614c81--bcou-role-st-uat-statefulset-1--bcou--UATWKR02_CpuPercent` | MEDIAN_SCALE_MISMATCH, POSSIBLE_RATIO_PERCENT_MIX | 138 | 0 | 0 | 0.6815 | 2.988 | 21.18 |
| 24 | `cpu` | `Container-DOCKER_CONTAINER_cb2bbb5e3f90--bcou-role-st-uat-statefulset-1--bcou--UATWKR02_CpuPercent` | EXTREME_OUTLIER_TAIL, MEDIAN_SCALE_MISMATCH, POSSIBLE_RATIO_PERCENT_MIX | 342 | 0 | 0.65 | 5.099 | 6.977 | 92.27 |
| 25 | `cpu` | `Container-DOCKER_CONTAINER_cd2b4a29291e--bcou-role-st-uat-statefulset-1--bcou--UATWKR04_CpuPercent` | EXTREME_OUTLIER_TAIL, MEDIAN_SCALE_MISMATCH, POSSIBLE_RATIO_PERCENT_MIX | 652 | 0 | 0 | 0 | 0.3547 | 10.49 |
| 26 | `cpu` | `Container-DOCKER_CONTAINER_d27123361435--bcou-trace-st-uat-statefulset-1--bcou--UATWKR18_CpuPercent` | MEDIAN_SCALE_MISMATCH, POSSIBLE_RATIO_PERCENT_MIX | 156 | 0.05 | 0.425 | 1.29 | 1.543 | 1.71 |
| 27 | `cpu` | `Container-DOCKER_CONTAINER_ef6cb138bb10--bcou-role-st-uat-statefulset-0--bcou--UATWKR18_CpuPercent` | MEDIAN_SCALE_MISMATCH, POSSIBLE_RATIO_PERCENT_MIX | 73 | 0 | 0 | 0.578 | 1.038 | 1.29 |
| 28 | `cpu` | `Container-DOCKER_CONTAINER_f861998c398e--bcou-role-st-uat-statefulset-0--bcou--UATWKR18_CpuPercent` | MEDIAN_SCALE_MISMATCH, POSSIBLE_RATIO_PERCENT_MIX | 1 | 0.44 | 0.44 | 0.44 | 0.44 | 0.44 |
| 29 | `cpu` | `OSLinux-CPU_CPU_CPUCpuUtil` | MEDIAN_SCALE_MISMATCH, POSSIBLE_RATIO_PERCENT_MIX | 130588 | 0.3284 | 25.73 | 28.48 | 45.69 | 100 |
| 30 | `mem` | `Container-DOCKER_CONTAINER_01eeda2c9f0b--bcou-trace-st-uat-statefulset-1--bcou--UATWKR18_MemPercent` | POSSIBLE_RATIO_PERCENT_MIX | 50 | 0 | 0 | 0 | 29.13 | 57.12 |
| 31 | `mem` | `Container-DOCKER_CONTAINER_0f6f3aa7920c--bcou-role-st-uat-statefulset-0--bcou--UATWKR18_MemPercent` | POSSIBLE_RATIO_PERCENT_MIX | 60 | 0 | 0 | 57.2 | 57.57 | 58.03 |
| 32 | `mem` | `Container-DOCKER_CONTAINER_1bc4fc80d241--bcou-trace-st-uat-statefulset-1--bcou--UATWKR18_MemPercent` | POSSIBLE_RATIO_PERCENT_MIX | 204 | 0 | 75.45 | 95.76 | 95.8 | 95.84 |
| 33 | `mem` | `Container-DOCKER_CONTAINER_23bdcf67c7e3--bcou-role-st-uat-statefulset-0--bcou--UATWKR18_MemPercent` | POSSIBLE_RATIO_PERCENT_MIX | 75 | 57.73 | 57.9 | 60.27 | 60.32 | 60.33 |
| 34 | `mem` | `Container-DOCKER_CONTAINER_2c2336e2994f--bcou-role-st-uat-statefulset-1--bcou--UATWKR06_MemPercent` | POSSIBLE_RATIO_PERCENT_MIX | 552 | 57.44 | 60.15 | 60.31 | 60.31 | 60.43 |
| 35 | `mem` | `Container-DOCKER_CONTAINER_2d16f5b2e830--bcou-trace-st-uat-statefulset-1--bcou--UATWKR18_MemPercent` | POSSIBLE_RATIO_PERCENT_MIX | 233 | 0 | 75.76 | 94.16 | 94.22 | 94.24 |
| 36 | `mem` | `Container-DOCKER_CONTAINER_2fa7eaebac26--bcou-role-st-uat-statefulset-1--bcou--UATWKR02_MemPercent` | SPARSE_SIGNAL, POSSIBLE_RATIO_PERCENT_MIX | 120 | 0 | 0 | 0 | 0 | 57.31 |
| 37 | `mem` | `Container-DOCKER_CONTAINER_350771a68ac2--bcou-trace-st-uat-statefulset-1--bcou--UATWKR18_MemPercent` | POSSIBLE_RATIO_PERCENT_MIX | 140 | 0 | 0 | 57.09 | 57.11 | 57.11 |
| 38 | `mem` | `Container-DOCKER_CONTAINER_3bf443a64876--bcou-role-st-uat-statefulset-1--bcou--UATWKR03_MemPercent` | POSSIBLE_RATIO_PERCENT_MIX | 588 | 0 | 0 | 71.38 | 71.38 | 71.38 |
| 39 | `mem` | `Container-DOCKER_CONTAINER_464dc801314b--bcou-trace-st-uat-statefulset-0--bcou--UATWKR04_MemPercent` | POSSIBLE_RATIO_PERCENT_MIX | 253 | 72.77 | 73.22 | 73.33 | 73.33 | 73.33 |
| 40 | `mem` | `Container-DOCKER_CONTAINER_69b53a78b2eb--bcou-role-st-uat-statefulset-1--bcou--UATWKR06_MemPercent` | CONSTANT_OR_NEAR_CONSTANT_SIGNAL, SPARSE_SIGNAL, POSSIBLE_RATIO_PERCENT_MIX | 297 | 0 | 0 | 0 | 0 | 0 |
| 41 | `mem` | `Container-DOCKER_CONTAINER_6d83a96887c3--bcou-role-st-uat-statefulset-0--bcou--UATWKR18_MemPercent` | POSSIBLE_RATIO_PERCENT_MIX | 58 | 0 | 59.22 | 59.72 | 61.05 | 62.81 |
| 42 | `mem` | `Container-DOCKER_CONTAINER_76d31070b844--bcou-role-st-uat-statefulset-1--bcou--UATWKR06_MemPercent` | POSSIBLE_RATIO_PERCENT_MIX | 278 | 0 | 0 | 0 | 13.66 | 59.38 |
| 43 | `mem` | `Container-DOCKER_CONTAINER_7b4b80f345e0--bcou-role-st-uat-statefulset-0--bcou--UATWKR04_MemPercent` | POSSIBLE_RATIO_PERCENT_MIX | 412 | 0 | 0 | 58.36 | 58.36 | 58.65 |
| 44 | `mem` | `Container-DOCKER_CONTAINER_89f0c1e5346c--bcou-trace-st-uat-statefulset-0--bcou--UATWKR04_MemPercent` | POSSIBLE_RATIO_PERCENT_MIX | 303 | 0 | 0 | 60.43 | 60.44 | 60.44 |
| 45 | `mem` | `Container-DOCKER_CONTAINER_94eca4f96efe--bcou-role-st-uat-statefulset-1--bcou--UATWKR04_MemPercent` | POSSIBLE_RATIO_PERCENT_MIX | 974 | 0 | 0 | 72.39 | 74.35 | 76.34 |
| 46 | `mem` | `Container-DOCKER_CONTAINER_9e8c309d0aab--bcou-trace-st-uat-statefulset-0--bcou--UATWKR04_MemPercent` | POSSIBLE_RATIO_PERCENT_MIX | 423 | 0 | 58.06 | 76.63 | 76.67 | 78.99 |
| 47 | `mem` | `Container-DOCKER_CONTAINER_ac3ba8476104--bcou-trace-st-uat-statefulset-1--bcou--UATWKR18_MemPercent` | POSSIBLE_RATIO_PERCENT_MIX | 4 | 0 | 28.55 | 88.43 | 92.85 | 93.96 |
| 48 | `mem` | `Container-DOCKER_CONTAINER_b2fc383d2438--bcou-trace-st-uat-statefulset-1--bcou--UATWKR18_MemPercent` | CONSTANT_OR_NEAR_CONSTANT_SIGNAL, SPARSE_SIGNAL, POSSIBLE_RATIO_PERCENT_MIX | 56 | 0 | 0 | 0 | 0 | 0 |
| 49 | `mem` | `Container-DOCKER_CONTAINER_b30097144a13--bcou-trace-st-uat-statefulset-0--bcou--UATWKR04_MemPercent` | POSSIBLE_RATIO_PERCENT_MIX | 592 | 0 | 70.69 | 71.14 | 71.14 | 71.62 |
| 50 | `mem` | `Container-DOCKER_CONTAINER_b6760337dc49--bcou-role-st-uat-statefulset-0--bcou--UATWKR18_MemPercent` | POSSIBLE_RATIO_PERCENT_MIX | 165 | 0 | 58.48 | 81.58 | 81.84 | 81.89 |
| 51 | `mem` | `Container-DOCKER_CONTAINER_bbc780df1fce--bcou-role-st-uat-statefulset-0--bcou--UATWKR18_MemPercent` | POSSIBLE_RATIO_PERCENT_MIX | 206 | 0 | 57.83 | 77.41 | 77.58 | 77.58 |
| 52 | `mem` | `Container-DOCKER_CONTAINER_c67422614c81--bcou-role-st-uat-statefulset-1--bcou--UATWKR02_MemPercent` | POSSIBLE_RATIO_PERCENT_MIX | 138 | 0 | 0 | 57.48 | 57.48 | 57.48 |
| 53 | `mem` | `Container-DOCKER_CONTAINER_cb2bbb5e3f90--bcou-role-st-uat-statefulset-1--bcou--UATWKR02_MemPercent` | POSSIBLE_RATIO_PERCENT_MIX | 342 | 0 | 59.18 | 83.39 | 84.99 | 86.02 |
| 54 | `mem` | `Container-DOCKER_CONTAINER_cd2b4a29291e--bcou-role-st-uat-statefulset-1--bcou--UATWKR04_MemPercent` | POSSIBLE_RATIO_PERCENT_MIX | 652 | 0 | 0 | 0 | 57.67 | 58.55 |
| 55 | `mem` | `Container-DOCKER_CONTAINER_d27123361435--bcou-trace-st-uat-statefulset-1--bcou--UATWKR18_MemPercent` | POSSIBLE_RATIO_PERCENT_MIX | 156 | 57.29 | 58.2 | 58.89 | 58.89 | 58.89 |
| 56 | `mem` | `Container-DOCKER_CONTAINER_ef6cb138bb10--bcou-role-st-uat-statefulset-0--bcou--UATWKR18_MemPercent` | POSSIBLE_RATIO_PERCENT_MIX | 74 | 0 | 0 | 59.99 | 59.99 | 59.99 |
| 57 | `mem` | `Container-DOCKER_CONTAINER_f861998c398e--bcou-role-st-uat-statefulset-0--bcou--UATWKR18_MemPercent` | POSSIBLE_RATIO_PERCENT_MIX | 1 | 58.8 | 58.8 | 58.8 | 58.8 | 58.8 |
| 58 | `mem` | `Tomcat-MEMORY_7441-MEMORY_JVMMemoryUsedPercent` | POSSIBLE_RATIO_PERCENT_MIX | 7789 | 22 | 37 | 51 | 75 | 89 |
| 59 | `net_rx` | `Container-DOCKER_CONTAINER_23bdcf67c7e3--bcou-role-st-uat-statefulset-0--bcou--UATWKR18_NetworkRxBytes` | CONSTANT_OR_NEAR_CONSTANT_SIGNAL, SPARSE_SIGNAL | 69 | 0 | 0 | 0 | 0 | 0 |
| 60 | `net_rx` | `Container-DOCKER_CONTAINER_6d83a96887c3--bcou-role-st-uat-statefulset-0--bcou--UATWKR18_NetworkRxBytes` | CONSTANT_OR_NEAR_CONSTANT_SIGNAL, SPARSE_SIGNAL | 26 | 0 | 0 | 0 | 0 | 0 |
| 61 | `net_rx` | `Container-DOCKER_CONTAINER_bbc780df1fce--bcou-role-st-uat-statefulset-0--bcou--UATWKR18_NetworkRxBytes` | CONSTANT_OR_NEAR_CONSTANT_SIGNAL, SPARSE_SIGNAL | 89 | 0 | 0 | 0 | 0 | 0 |
| 62 | `net_rx` | `Container-DOCKER_CONTAINER_c67422614c81--bcou-role-st-uat-statefulset-1--bcou--UATWKR02_NetworkRxBytes` | CONSTANT_OR_NEAR_CONSTANT_SIGNAL, SPARSE_SIGNAL | 10 | 0 | 0 | 0 | 0 | 0 |
| 63 | `net_rx` | `Container-DOCKER_CONTAINER_cb2bbb5e3f90--bcou-role-st-uat-statefulset-1--bcou--UATWKR02_NetworkRxBytes` | CONSTANT_OR_NEAR_CONSTANT_SIGNAL, SPARSE_SIGNAL | 142 | 0 | 0 | 0 | 0 | 0 |
| 64 | `net_tx` | `Container-DOCKER_CONTAINER_0f6f3aa7920c--bcou-role-st-uat-statefulset-0--bcou--UATWKR18_NetworkTxBytes` | CONSTANT_OR_NEAR_CONSTANT_SIGNAL, SPARSE_SIGNAL | 60 | 0 | 0 | 0 | 0 | 0 |
| 65 | `net_tx` | `Container-DOCKER_CONTAINER_23bdcf67c7e3--bcou-role-st-uat-statefulset-0--bcou--UATWKR18_NetworkTxBytes` | CONSTANT_OR_NEAR_CONSTANT_SIGNAL, SPARSE_SIGNAL | 113 | 0 | 0 | 0 | 0 | 0 |
| 66 | `net_tx` | `Container-DOCKER_CONTAINER_2fa7eaebac26--bcou-role-st-uat-statefulset-1--bcou--UATWKR02_NetworkTxBytes` | CONSTANT_OR_NEAR_CONSTANT_SIGNAL, SPARSE_SIGNAL | 120 | 0 | 0 | 0 | 0 | 0 |
| 67 | `net_tx` | `Container-DOCKER_CONTAINER_69b53a78b2eb--bcou-role-st-uat-statefulset-1--bcou--UATWKR06_NetworkTxBytes` | CONSTANT_OR_NEAR_CONSTANT_SIGNAL, SPARSE_SIGNAL | 352 | 0 | 0 | 0 | 0 | 0 |
| 68 | `net_tx` | `Container-DOCKER_CONTAINER_6d83a96887c3--bcou-role-st-uat-statefulset-0--bcou--UATWKR18_NetworkTxBytes` | CONSTANT_OR_NEAR_CONSTANT_SIGNAL, SPARSE_SIGNAL | 58 | 0 | 0 | 0 | 0 | 0 |
| 69 | `net_tx` | `Container-DOCKER_CONTAINER_76d31070b844--bcou-role-st-uat-statefulset-1--bcou--UATWKR06_NetworkTxBytes` | SPARSE_SIGNAL | 318 | 0 | 0 | 0 | 0 | 13.19 |
| 70 | `net_tx` | `Container-DOCKER_CONTAINER_b2fc383d2438--bcou-trace-st-uat-statefulset-1--bcou--UATWKR18_NetworkTxBytes` | CONSTANT_OR_NEAR_CONSTANT_SIGNAL, SPARSE_SIGNAL | 55 | 0 | 0 | 0 | 0 | 0 |
| 71 | `net_tx` | `Container-DOCKER_CONTAINER_bbc780df1fce--bcou-role-st-uat-statefulset-0--bcou--UATWKR18_NetworkTxBytes` | CONSTANT_OR_NEAR_CONSTANT_SIGNAL, SPARSE_SIGNAL | 205 | 0 | 0 | 0 | 0 | 0 |
| 72 | `net_tx` | `Container-DOCKER_CONTAINER_c67422614c81--bcou-role-st-uat-statefulset-1--bcou--UATWKR02_NetworkTxBytes` | CONSTANT_OR_NEAR_CONSTANT_SIGNAL, SPARSE_SIGNAL | 137 | 0 | 0 | 0 | 0 | 0 |
| 73 | `net_tx` | `Container-DOCKER_CONTAINER_cb2bbb5e3f90--bcou-role-st-uat-statefulset-1--bcou--UATWKR02_NetworkTxBytes` | CONSTANT_OR_NEAR_CONSTANT_SIGNAL, SPARSE_SIGNAL | 343 | 0 | 0 | 0 | 0 | 0 |

## 4. Diagnostic Notes

### Slot `cpu`

- Slot-wide distribution: min=0, p50=25.68, p95=28.43, max=100

- **Container-DOCKER_CONTAINER_01eeda2c9f0b--bcou-trace-st-uat-statefulset-1--bcou--UATWKR18_CpuPercent** (50 rows, 1 components): p50=0, p95=0, range=[0, 0.31]
- **Container-DOCKER_CONTAINER_0f6f3aa7920c--bcou-role-st-uat-statefulset-0--bcou--UATWKR18_CpuPercent** (60 rows, 1 components): p50=0, p95=0.676, range=[0, 9.91]
- **Container-DOCKER_CONTAINER_1bc4fc80d241--bcou-trace-st-uat-statefulset-1--bcou--UATWKR18_CpuPercent** (204 rows, 1 components): p50=0.07, p95=1.155, range=[0, 6.12]
- **Container-DOCKER_CONTAINER_23bdcf67c7e3--bcou-role-st-uat-statefulset-0--bcou--UATWKR18_CpuPercent** (74 rows, 1 components): p50=0.27, p95=1.15, range=[0.04, 2]
- **Container-DOCKER_CONTAINER_2c2336e2994f--bcou-role-st-uat-statefulset-1--bcou--UATWKR06_CpuPercent** (541 rows, 1 components): p50=0.36, p95=1.06, range=[0.05, 1.85]
- **Container-DOCKER_CONTAINER_2d16f5b2e830--bcou-trace-st-uat-statefulset-1--bcou--UATWKR18_CpuPercent** (233 rows, 1 components): p50=0.26, p95=1.646, range=[0, 10.3]
- **Container-DOCKER_CONTAINER_2fa7eaebac26--bcou-role-st-uat-statefulset-1--bcou--UATWKR02_CpuPercent** (120 rows, 1 components): p50=0, p95=0, range=[0, 14.28]
- **Container-DOCKER_CONTAINER_350771a68ac2--bcou-trace-st-uat-statefulset-1--bcou--UATWKR18_CpuPercent** (140 rows, 1 components): p50=0, p95=0.3005, range=[0, 0.45]
- **Container-DOCKER_CONTAINER_3bf443a64876--bcou-role-st-uat-statefulset-1--bcou--UATWKR03_CpuPercent** (946 rows, 1 components): p50=0, p95=23.64, range=[0, 64.61]
- **Container-DOCKER_CONTAINER_464dc801314b--bcou-trace-st-uat-statefulset-0--bcou--UATWKR04_CpuPercent** (248 rows, 1 components): p50=0.62, p95=2.362, range=[0, 4.16]
- **Container-DOCKER_CONTAINER_69b53a78b2eb--bcou-role-st-uat-statefulset-1--bcou--UATWKR06_CpuPercent** (296 rows, 1 components): p50=0, p95=0, range=[0, 0]
- **Container-DOCKER_CONTAINER_6d83a96887c3--bcou-role-st-uat-statefulset-0--bcou--UATWKR18_CpuPercent** (58 rows, 1 components): p50=0.435, p95=1.276, range=[0, 2.26]
- **Container-DOCKER_CONTAINER_76d31070b844--bcou-role-st-uat-statefulset-1--bcou--UATWKR06_CpuPercent** (246 rows, 1 components): p50=0, p95=0, range=[0, 1.18]
- **Container-DOCKER_CONTAINER_7b4b80f345e0--bcou-role-st-uat-statefulset-0--bcou--UATWKR04_CpuPercent** (411 rows, 1 components): p50=0, p95=0.91, range=[0, 3.08]
- **Container-DOCKER_CONTAINER_89f0c1e5346c--bcou-trace-st-uat-statefulset-0--bcou--UATWKR04_CpuPercent** (345 rows, 1 components): p50=0, p95=0.786, range=[0, 3.53]
- **Container-DOCKER_CONTAINER_94eca4f96efe--bcou-role-st-uat-statefulset-1--bcou--UATWKR04_CpuPercent** (974 rows, 1 components): p50=0, p95=1.67, range=[0, 12.35]
- **Container-DOCKER_CONTAINER_9e8c309d0aab--bcou-trace-st-uat-statefulset-0--bcou--UATWKR04_CpuPercent** (399 rows, 1 components): p50=0, p95=1.814, range=[0, 9.68]
- **Container-DOCKER_CONTAINER_ac3ba8476104--bcou-trace-st-uat-statefulset-1--bcou--UATWKR18_CpuPercent** (4 rows, 1 components): p50=0.135, p95=10.21, range=[0, 11.96]
- **Container-DOCKER_CONTAINER_b2fc383d2438--bcou-trace-st-uat-statefulset-1--bcou--UATWKR18_CpuPercent** (55 rows, 1 components): p50=0, p95=0, range=[0, 0]
- **Container-DOCKER_CONTAINER_b30097144a13--bcou-trace-st-uat-statefulset-0--bcou--UATWKR04_CpuPercent** (591 rows, 1 components): p50=0.12, p95=2.03, range=[0, 10.81]
- **Container-DOCKER_CONTAINER_b6760337dc49--bcou-role-st-uat-statefulset-0--bcou--UATWKR18_CpuPercent** (214 rows, 1 components): p50=0.205, p95=1.477, range=[0, 10.32]
- **Container-DOCKER_CONTAINER_bbc780df1fce--bcou-role-st-uat-statefulset-0--bcou--UATWKR18_CpuPercent** (206 rows, 1 components): p50=0.065, p95=1.07, range=[0, 3.53]
- **Container-DOCKER_CONTAINER_c67422614c81--bcou-role-st-uat-statefulset-1--bcou--UATWKR02_CpuPercent** (138 rows, 1 components): p50=0, p95=0.6815, range=[0, 21.18]
- **Container-DOCKER_CONTAINER_cb2bbb5e3f90--bcou-role-st-uat-statefulset-1--bcou--UATWKR02_CpuPercent** (342 rows, 1 components): p50=0.65, p95=5.099, range=[0, 92.27]
- **Container-DOCKER_CONTAINER_cd2b4a29291e--bcou-role-st-uat-statefulset-1--bcou--UATWKR04_CpuPercent** (652 rows, 1 components): p50=0, p95=0, range=[0, 10.49]
- **Container-DOCKER_CONTAINER_d27123361435--bcou-trace-st-uat-statefulset-1--bcou--UATWKR18_CpuPercent** (156 rows, 1 components): p50=0.425, p95=1.29, range=[0.05, 1.71]
- **Container-DOCKER_CONTAINER_ef6cb138bb10--bcou-role-st-uat-statefulset-0--bcou--UATWKR18_CpuPercent** (73 rows, 1 components): p50=0, p95=0.578, range=[0, 1.29]
- **Container-DOCKER_CONTAINER_f861998c398e--bcou-role-st-uat-statefulset-0--bcou--UATWKR18_CpuPercent** (1 rows, 1 components): p50=0.44, p95=0.44, range=[0.44, 0.44]
- **OSLinux-CPU_CPU_CPUCpuUtil** (130588 rows, 14 components): p50=25.73, p95=28.48, range=[0.2119, 100]

### Slot `mem`

- Slot-wide distribution: min=0, p50=36, p95=73, max=95.84

- **Container-DOCKER_CONTAINER_01eeda2c9f0b--bcou-trace-st-uat-statefulset-1--bcou--UATWKR18_MemPercent** (50 rows, 1 components): p50=0, p95=0, range=[0, 57.12]
- **Container-DOCKER_CONTAINER_0f6f3aa7920c--bcou-role-st-uat-statefulset-0--bcou--UATWKR18_MemPercent** (60 rows, 1 components): p50=0, p95=57.2, range=[0, 58.03]
- **Container-DOCKER_CONTAINER_1bc4fc80d241--bcou-trace-st-uat-statefulset-1--bcou--UATWKR18_MemPercent** (204 rows, 1 components): p50=75.45, p95=95.76, range=[0, 95.84]
- **Container-DOCKER_CONTAINER_23bdcf67c7e3--bcou-role-st-uat-statefulset-0--bcou--UATWKR18_MemPercent** (75 rows, 1 components): p50=57.9, p95=60.27, range=[57.73, 60.33]
- **Container-DOCKER_CONTAINER_2c2336e2994f--bcou-role-st-uat-statefulset-1--bcou--UATWKR06_MemPercent** (552 rows, 1 components): p50=60.15, p95=60.31, range=[57.4, 60.43]
- **Container-DOCKER_CONTAINER_2d16f5b2e830--bcou-trace-st-uat-statefulset-1--bcou--UATWKR18_MemPercent** (233 rows, 1 components): p50=75.76, p95=94.16, range=[0, 94.24]
- **Container-DOCKER_CONTAINER_2fa7eaebac26--bcou-role-st-uat-statefulset-1--bcou--UATWKR02_MemPercent** (120 rows, 1 components): p50=0, p95=0, range=[0, 57.31]
- **Container-DOCKER_CONTAINER_350771a68ac2--bcou-trace-st-uat-statefulset-1--bcou--UATWKR18_MemPercent** (140 rows, 1 components): p50=0, p95=57.09, range=[0, 57.11]
- **Container-DOCKER_CONTAINER_3bf443a64876--bcou-role-st-uat-statefulset-1--bcou--UATWKR03_MemPercent** (588 rows, 1 components): p50=0, p95=71.38, range=[0, 71.38]
- **Container-DOCKER_CONTAINER_464dc801314b--bcou-trace-st-uat-statefulset-0--bcou--UATWKR04_MemPercent** (253 rows, 1 components): p50=73.22, p95=73.33, range=[0, 73.33]
- **Container-DOCKER_CONTAINER_69b53a78b2eb--bcou-role-st-uat-statefulset-1--bcou--UATWKR06_MemPercent** (297 rows, 1 components): p50=0, p95=0, range=[0, 0]
- **Container-DOCKER_CONTAINER_6d83a96887c3--bcou-role-st-uat-statefulset-0--bcou--UATWKR18_MemPercent** (58 rows, 1 components): p50=59.22, p95=59.72, range=[0, 62.81]
- **Container-DOCKER_CONTAINER_76d31070b844--bcou-role-st-uat-statefulset-1--bcou--UATWKR06_MemPercent** (278 rows, 1 components): p50=0, p95=0, range=[0, 59.38]
- **Container-DOCKER_CONTAINER_7b4b80f345e0--bcou-role-st-uat-statefulset-0--bcou--UATWKR04_MemPercent** (412 rows, 1 components): p50=0, p95=58.36, range=[0, 58.65]
- **Container-DOCKER_CONTAINER_89f0c1e5346c--bcou-trace-st-uat-statefulset-0--bcou--UATWKR04_MemPercent** (303 rows, 1 components): p50=0, p95=60.43, range=[0, 60.44]
- **Container-DOCKER_CONTAINER_94eca4f96efe--bcou-role-st-uat-statefulset-1--bcou--UATWKR04_MemPercent** (974 rows, 1 components): p50=0, p95=72.39, range=[0, 76.34]
- **Container-DOCKER_CONTAINER_9e8c309d0aab--bcou-trace-st-uat-statefulset-0--bcou--UATWKR04_MemPercent** (423 rows, 1 components): p50=58.06, p95=76.63, range=[0, 78.99]
- **Container-DOCKER_CONTAINER_ac3ba8476104--bcou-trace-st-uat-statefulset-1--bcou--UATWKR18_MemPercent** (4 rows, 1 components): p50=28.55, p95=88.43, range=[0, 93.96]
- **Container-DOCKER_CONTAINER_b2fc383d2438--bcou-trace-st-uat-statefulset-1--bcou--UATWKR18_MemPercent** (56 rows, 1 components): p50=0, p95=0, range=[0, 0]
- **Container-DOCKER_CONTAINER_b30097144a13--bcou-trace-st-uat-statefulset-0--bcou--UATWKR04_MemPercent** (592 rows, 1 components): p50=70.69, p95=71.14, range=[0, 71.62]
- **Container-DOCKER_CONTAINER_b6760337dc49--bcou-role-st-uat-statefulset-0--bcou--UATWKR18_MemPercent** (165 rows, 1 components): p50=58.48, p95=81.58, range=[0, 81.89]
- **Container-DOCKER_CONTAINER_bbc780df1fce--bcou-role-st-uat-statefulset-0--bcou--UATWKR18_MemPercent** (206 rows, 1 components): p50=57.83, p95=77.41, range=[0, 77.58]
- **Container-DOCKER_CONTAINER_c67422614c81--bcou-role-st-uat-statefulset-1--bcou--UATWKR02_MemPercent** (138 rows, 1 components): p50=0, p95=57.48, range=[0, 57.48]
- **Container-DOCKER_CONTAINER_cb2bbb5e3f90--bcou-role-st-uat-statefulset-1--bcou--UATWKR02_MemPercent** (342 rows, 1 components): p50=59.18, p95=83.39, range=[0, 86.02]
- **Container-DOCKER_CONTAINER_cd2b4a29291e--bcou-role-st-uat-statefulset-1--bcou--UATWKR04_MemPercent** (652 rows, 1 components): p50=0, p95=0, range=[0, 58.55]
- **Container-DOCKER_CONTAINER_d27123361435--bcou-trace-st-uat-statefulset-1--bcou--UATWKR18_MemPercent** (156 rows, 1 components): p50=58.2, p95=58.89, range=[57.2, 58.89]
- **Container-DOCKER_CONTAINER_ef6cb138bb10--bcou-role-st-uat-statefulset-0--bcou--UATWKR18_MemPercent** (74 rows, 1 components): p50=0, p95=59.99, range=[0, 59.99]
- **Container-DOCKER_CONTAINER_f861998c398e--bcou-role-st-uat-statefulset-0--bcou--UATWKR18_MemPercent** (1 rows, 1 components): p50=58.8, p95=58.8, range=[58.8, 58.8]
- **Tomcat-MEMORY_7441-MEMORY_JVMMemoryUsedPercent** (7789 rows, 4 components): p50=37, p95=51, range=[7, 89]

### Slot `net_rx`

- Slot-wide distribution: min=0, p50=12.58, p95=14.74, max=15.92

- **Container-DOCKER_CONTAINER_0f6f3aa7920c--bcou-role-st-uat-statefulset-0--bcou--UATWKR18_NetworkRxBytes** (5 rows, 1 components): p50=0, p95=0, range=[0, 0]
- **Container-DOCKER_CONTAINER_1bc4fc80d241--bcou-trace-st-uat-statefulset-1--bcou--UATWKR18_NetworkRxBytes** (87 rows, 1 components): p50=12.9, p95=14.75, range=[9.026, 15.1]
- **Container-DOCKER_CONTAINER_23bdcf67c7e3--bcou-role-st-uat-statefulset-0--bcou--UATWKR18_NetworkRxBytes** (69 rows, 1 components): p50=0, p95=0, range=[0, 0]
- **Container-DOCKER_CONTAINER_2c2336e2994f--bcou-role-st-uat-statefulset-1--bcou--UATWKR06_NetworkRxBytes** (559 rows, 1 components): p50=13.44, p95=14.65, range=[9.645, 15.19]
- **Container-DOCKER_CONTAINER_2d16f5b2e830--bcou-trace-st-uat-statefulset-1--bcou--UATWKR18_NetworkRxBytes** (145 rows, 1 components): p50=13.24, p95=14.99, range=[9.026, 15.38]
- **Container-DOCKER_CONTAINER_2fa7eaebac26--bcou-role-st-uat-statefulset-1--bcou--UATWKR02_NetworkRxBytes** (1 rows, 1 components): p50=0, p95=0, range=[0, 0]
- **Container-DOCKER_CONTAINER_350771a68ac2--bcou-trace-st-uat-statefulset-1--bcou--UATWKR18_NetworkRxBytes** (29 rows, 1 components): p50=9.586, p95=13.94, range=[9.026, 14.12]
- **Container-DOCKER_CONTAINER_3bf443a64876--bcou-role-st-uat-statefulset-1--bcou--UATWKR03_NetworkRxBytes** (154 rows, 1 components): p50=13.44, p95=14.84, range=[9.645, 15.92]
- **Container-DOCKER_CONTAINER_464dc801314b--bcou-trace-st-uat-statefulset-0--bcou--UATWKR04_NetworkRxBytes** (181 rows, 1 components): p50=13.54, p95=15.04, range=[9.026, 15.44]
- **Container-DOCKER_CONTAINER_6d83a96887c3--bcou-role-st-uat-statefulset-0--bcou--UATWKR18_NetworkRxBytes** (26 rows, 1 components): p50=0, p95=0, range=[0, 0]
- **Container-DOCKER_CONTAINER_76d31070b844--bcou-role-st-uat-statefulset-1--bcou--UATWKR06_NetworkRxBytes** (1 rows, 1 components): p50=13.9, p95=13.9, range=[13.9, 13.9]
- **Container-DOCKER_CONTAINER_7b4b80f345e0--bcou-role-st-uat-statefulset-0--bcou--UATWKR04_NetworkRxBytes** (150 rows, 1 components): p50=12.29, p95=14.13, range=[9.026, 14.52]
- **Container-DOCKER_CONTAINER_89f0c1e5346c--bcou-trace-st-uat-statefulset-0--bcou--UATWKR04_NetworkRxBytes** (94 rows, 1 components): p50=11.76, p95=14.5, range=[9.026, 14.78]
- **Container-DOCKER_CONTAINER_94eca4f96efe--bcou-role-st-uat-statefulset-1--bcou--UATWKR04_NetworkRxBytes** (171 rows, 1 components): p50=13.05, p95=14.69, range=[9.026, 14.92]
- **Container-DOCKER_CONTAINER_9e8c309d0aab--bcou-trace-st-uat-statefulset-0--bcou--UATWKR04_NetworkRxBytes** (171 rows, 1 components): p50=13.87, p95=14.99, range=[9.645, 15.57]
- **Container-DOCKER_CONTAINER_b30097144a13--bcou-trace-st-uat-statefulset-0--bcou--UATWKR04_NetworkRxBytes** (316 rows, 1 components): p50=11.22, p95=14.84, range=[9.026, 15.48]
- **Container-DOCKER_CONTAINER_b6760337dc49--bcou-role-st-uat-statefulset-0--bcou--UATWKR18_NetworkRxBytes** (103 rows, 1 components): p50=12.66, p95=14.51, range=[9.026, 14.86]
- **Container-DOCKER_CONTAINER_bbc780df1fce--bcou-role-st-uat-statefulset-0--bcou--UATWKR18_NetworkRxBytes** (89 rows, 1 components): p50=0, p95=0, range=[0, 0]
- **Container-DOCKER_CONTAINER_c67422614c81--bcou-role-st-uat-statefulset-1--bcou--UATWKR02_NetworkRxBytes** (10 rows, 1 components): p50=0, p95=0, range=[0, 0]
- **Container-DOCKER_CONTAINER_cb2bbb5e3f90--bcou-role-st-uat-statefulset-1--bcou--UATWKR02_NetworkRxBytes** (142 rows, 1 components): p50=0, p95=0, range=[0, 0]
- **Container-DOCKER_CONTAINER_cd2b4a29291e--bcou-role-st-uat-statefulset-1--bcou--UATWKR04_NetworkRxBytes** (10 rows, 1 components): p50=11.95, p95=13.25, range=[10.86, 13.34]
- **Container-DOCKER_CONTAINER_d27123361435--bcou-trace-st-uat-statefulset-1--bcou--UATWKR18_NetworkRxBytes** (90 rows, 1 components): p50=13.55, p95=15.02, range=[9.026, 15.23]
- **Container-DOCKER_CONTAINER_ef6cb138bb10--bcou-role-st-uat-statefulset-0--bcou--UATWKR18_NetworkRxBytes** (12 rows, 1 components): p50=12.65, p95=13.79, range=[9.645, 14.07]
- **Container-DOCKER_CONTAINER_f861998c398e--bcou-role-st-uat-statefulset-0--bcou--UATWKR18_NetworkRxBytes** (1 rows, 1 components): p50=0, p95=0, range=[0, 0]

### Slot `net_tx`

- Slot-wide distribution: min=0, p50=0, p95=12.7, max=14.25

- **Container-DOCKER_CONTAINER_01eeda2c9f0b--bcou-trace-st-uat-statefulset-1--bcou--UATWKR18_NetworkTxBytes** (50 rows, 1 components): p50=0, p95=0, range=[0, 11.62]
- **Container-DOCKER_CONTAINER_0f6f3aa7920c--bcou-role-st-uat-statefulset-0--bcou--UATWKR18_NetworkTxBytes** (60 rows, 1 components): p50=0, p95=0, range=[0, 0]
- **Container-DOCKER_CONTAINER_1bc4fc80d241--bcou-trace-st-uat-statefulset-1--bcou--UATWKR18_NetworkTxBytes** (204 rows, 1 components): p50=7.834, p95=12.71, range=[0, 13.36]
- **Container-DOCKER_CONTAINER_23bdcf67c7e3--bcou-role-st-uat-statefulset-0--bcou--UATWKR18_NetworkTxBytes** (113 rows, 1 components): p50=0, p95=0, range=[0, 0]
- **Container-DOCKER_CONTAINER_2c2336e2994f--bcou-role-st-uat-statefulset-1--bcou--UATWKR06_NetworkTxBytes** (563 rows, 1 components): p50=11.77, p95=13.01, range=[7.599, 13.65]
- **Container-DOCKER_CONTAINER_2d16f5b2e830--bcou-trace-st-uat-statefulset-1--bcou--UATWKR18_NetworkTxBytes** (234 rows, 1 components): p50=11.48, p95=12.89, range=[0, 14.25]
- **Container-DOCKER_CONTAINER_2fa7eaebac26--bcou-role-st-uat-statefulset-1--bcou--UATWKR02_NetworkTxBytes** (120 rows, 1 components): p50=0, p95=0, range=[0, 0]
- **Container-DOCKER_CONTAINER_350771a68ac2--bcou-trace-st-uat-statefulset-1--bcou--UATWKR18_NetworkTxBytes** (139 rows, 1 components): p50=0, p95=10.84, range=[0, 12.56]
- **Container-DOCKER_CONTAINER_3bf443a64876--bcou-role-st-uat-statefulset-1--bcou--UATWKR03_NetworkTxBytes** (601 rows, 1 components): p50=0, p95=12.7, range=[0, 13.49]
- **Container-DOCKER_CONTAINER_464dc801314b--bcou-trace-st-uat-statefulset-0--bcou--UATWKR04_NetworkTxBytes** (284 rows, 1 components): p50=11.56, p95=13.02, range=[0, 13.37]
- **Container-DOCKER_CONTAINER_69b53a78b2eb--bcou-role-st-uat-statefulset-1--bcou--UATWKR06_NetworkTxBytes** (352 rows, 1 components): p50=0, p95=0, range=[0, 0]
- **Container-DOCKER_CONTAINER_6d83a96887c3--bcou-role-st-uat-statefulset-0--bcou--UATWKR18_NetworkTxBytes** (58 rows, 1 components): p50=0, p95=0, range=[0, 0]
- **Container-DOCKER_CONTAINER_76d31070b844--bcou-role-st-uat-statefulset-1--bcou--UATWKR06_NetworkTxBytes** (318 rows, 1 components): p50=0, p95=0, range=[0, 13.19]
- **Container-DOCKER_CONTAINER_7b4b80f345e0--bcou-role-st-uat-statefulset-0--bcou--UATWKR04_NetworkTxBytes** (450 rows, 1 components): p50=7.06, p95=12.02, range=[0, 12.79]
- **Container-DOCKER_CONTAINER_89f0c1e5346c--bcou-trace-st-uat-statefulset-0--bcou--UATWKR04_NetworkTxBytes** (297 rows, 1 components): p50=0, p95=12.03, range=[0, 13.51]
- **Container-DOCKER_CONTAINER_94eca4f96efe--bcou-role-st-uat-statefulset-1--bcou--UATWKR04_NetworkTxBytes** (832 rows, 1 components): p50=0, p95=12.62, range=[0, 13.16]
- **Container-DOCKER_CONTAINER_9e8c309d0aab--bcou-trace-st-uat-statefulset-0--bcou--UATWKR04_NetworkTxBytes** (598 rows, 1 components): p50=7.87, p95=12.97, range=[0, 13.51]
- **Container-DOCKER_CONTAINER_ac3ba8476104--bcou-trace-st-uat-statefulset-1--bcou--UATWKR18_NetworkTxBytes** (4 rows, 1 components): p50=3.201, p95=10.28, range=[0, 10.97]
- **Container-DOCKER_CONTAINER_b2fc383d2438--bcou-trace-st-uat-statefulset-1--bcou--UATWKR18_NetworkTxBytes** (55 rows, 1 components): p50=0, p95=0, range=[0, 0]
- **Container-DOCKER_CONTAINER_b30097144a13--bcou-trace-st-uat-statefulset-0--bcou--UATWKR04_NetworkTxBytes** (598 rows, 1 components): p50=7.59, p95=12.78, range=[0, 13.64]
- **Container-DOCKER_CONTAINER_b6760337dc49--bcou-role-st-uat-statefulset-0--bcou--UATWKR18_NetworkTxBytes** (176 rows, 1 components): p50=10.63, p95=12.8, range=[0, 13.22]
- **Container-DOCKER_CONTAINER_bbc780df1fce--bcou-role-st-uat-statefulset-0--bcou--UATWKR18_NetworkTxBytes** (205 rows, 1 components): p50=0, p95=0, range=[0, 0]
- **Container-DOCKER_CONTAINER_c67422614c81--bcou-role-st-uat-statefulset-1--bcou--UATWKR02_NetworkTxBytes** (137 rows, 1 components): p50=0, p95=0, range=[0, 0]
- **Container-DOCKER_CONTAINER_cb2bbb5e3f90--bcou-role-st-uat-statefulset-1--bcou--UATWKR02_NetworkTxBytes** (343 rows, 1 components): p50=0, p95=0, range=[0, 0]
- **Container-DOCKER_CONTAINER_cd2b4a29291e--bcou-role-st-uat-statefulset-1--bcou--UATWKR04_NetworkTxBytes** (652 rows, 1 components): p50=0, p95=0, range=[0, 11.93]
- **Container-DOCKER_CONTAINER_d27123361435--bcou-trace-st-uat-statefulset-1--bcou--UATWKR18_NetworkTxBytes** (156 rows, 1 components): p50=11.7, p95=13.01, range=[7.06, 13.4]
- **Container-DOCKER_CONTAINER_ef6cb138bb10--bcou-role-st-uat-statefulset-0--bcou--UATWKR18_NetworkTxBytes** (73 rows, 1 components): p50=0, p95=11.99, range=[0, 12.43]
- **Container-DOCKER_CONTAINER_f861998c398e--bcou-role-st-uat-statefulset-0--bcou--UATWKR18_NetworkTxBytes** (1 rows, 1 components): p50=0, p95=0, range=[0, 0]

### Unit / Scale Mix Observations

The following observations are based on numerical evidence and KPI naming conventions visible in the raw data. They do NOT automatically trigger data modifications.

**Slot `cpu`** (29 raw KPIs):
  - `Container-DOCKER_CONTAINER_01eeda2c9f0b--bcou-trace-st-uat-statefulset-1--bcou--UATWKR18_CpuPercent`: range [0.0, 0.3, p50=0.0] — container CPU percent
  - `Container-DOCKER_CONTAINER_0f6f3aa7920c--bcou-role-st-uat-statefulset-0--bcou--UATWKR18_CpuPercent`: range [0.0, 9.9, p50=0.0] — container CPU percent
  - `Container-DOCKER_CONTAINER_1bc4fc80d241--bcou-trace-st-uat-statefulset-1--bcou--UATWKR18_CpuPercent`: range [0.0, 6.1, p50=0.1] — container CPU percent
  - `Container-DOCKER_CONTAINER_23bdcf67c7e3--bcou-role-st-uat-statefulset-0--bcou--UATWKR18_CpuPercent`: range [0.0, 2.0, p50=0.3] — container CPU percent
  - `Container-DOCKER_CONTAINER_2c2336e2994f--bcou-role-st-uat-statefulset-1--bcou--UATWKR06_CpuPercent`: range [0.1, 1.9, p50=0.4] — container CPU percent
  - `Container-DOCKER_CONTAINER_2d16f5b2e830--bcou-trace-st-uat-statefulset-1--bcou--UATWKR18_CpuPercent`: range [0.0, 10.3, p50=0.3] — container CPU percent
  - `Container-DOCKER_CONTAINER_2fa7eaebac26--bcou-role-st-uat-statefulset-1--bcou--UATWKR02_CpuPercent`: range [0.0, 14.3, p50=0.0] — container CPU percent
  - `Container-DOCKER_CONTAINER_350771a68ac2--bcou-trace-st-uat-statefulset-1--bcou--UATWKR18_CpuPercent`: range [0.0, 0.5, p50=0.0] — container CPU percent
  - `Container-DOCKER_CONTAINER_3bf443a64876--bcou-role-st-uat-statefulset-1--bcou--UATWKR03_CpuPercent`: range [0.0, 64.6, p50=0.0] — container CPU percent
  - `Container-DOCKER_CONTAINER_464dc801314b--bcou-trace-st-uat-statefulset-0--bcou--UATWKR04_CpuPercent`: range [0.0, 4.2, p50=0.6] — container CPU percent
  - `Container-DOCKER_CONTAINER_69b53a78b2eb--bcou-role-st-uat-statefulset-1--bcou--UATWKR06_CpuPercent`: range [0.0, 0.0, p50=0.0] — container CPU percent
  - `Container-DOCKER_CONTAINER_6d83a96887c3--bcou-role-st-uat-statefulset-0--bcou--UATWKR18_CpuPercent`: range [0.0, 2.3, p50=0.4] — container CPU percent
  - `Container-DOCKER_CONTAINER_76d31070b844--bcou-role-st-uat-statefulset-1--bcou--UATWKR06_CpuPercent`: range [0.0, 1.2, p50=0.0] — container CPU percent
  - `Container-DOCKER_CONTAINER_7b4b80f345e0--bcou-role-st-uat-statefulset-0--bcou--UATWKR04_CpuPercent`: range [0.0, 3.1, p50=0.0] — container CPU percent
  - `Container-DOCKER_CONTAINER_89f0c1e5346c--bcou-trace-st-uat-statefulset-0--bcou--UATWKR04_CpuPercent`: range [0.0, 3.5, p50=0.0] — container CPU percent
  - `Container-DOCKER_CONTAINER_94eca4f96efe--bcou-role-st-uat-statefulset-1--bcou--UATWKR04_CpuPercent`: range [0.0, 12.3, p50=0.0] — container CPU percent
  - `Container-DOCKER_CONTAINER_9e8c309d0aab--bcou-trace-st-uat-statefulset-0--bcou--UATWKR04_CpuPercent`: range [0.0, 9.7, p50=0.0] — container CPU percent
  - `Container-DOCKER_CONTAINER_ac3ba8476104--bcou-trace-st-uat-statefulset-1--bcou--UATWKR18_CpuPercent`: range [0.0, 12.0, p50=0.1] — container CPU percent
  - `Container-DOCKER_CONTAINER_b2fc383d2438--bcou-trace-st-uat-statefulset-1--bcou--UATWKR18_CpuPercent`: range [0.0, 0.0, p50=0.0] — container CPU percent
  - `Container-DOCKER_CONTAINER_b30097144a13--bcou-trace-st-uat-statefulset-0--bcou--UATWKR04_CpuPercent`: range [0.0, 10.8, p50=0.1] — container CPU percent
  - `Container-DOCKER_CONTAINER_b6760337dc49--bcou-role-st-uat-statefulset-0--bcou--UATWKR18_CpuPercent`: range [0.0, 10.3, p50=0.2] — container CPU percent
  - `Container-DOCKER_CONTAINER_bbc780df1fce--bcou-role-st-uat-statefulset-0--bcou--UATWKR18_CpuPercent`: range [0.0, 3.5, p50=0.1] — container CPU percent
  - `Container-DOCKER_CONTAINER_c67422614c81--bcou-role-st-uat-statefulset-1--bcou--UATWKR02_CpuPercent`: range [0.0, 21.2, p50=0.0] — container CPU percent
  - `Container-DOCKER_CONTAINER_cb2bbb5e3f90--bcou-role-st-uat-statefulset-1--bcou--UATWKR02_CpuPercent`: range [0.0, 92.3, p50=0.7] — container CPU percent
  - `Container-DOCKER_CONTAINER_cd2b4a29291e--bcou-role-st-uat-statefulset-1--bcou--UATWKR04_CpuPercent`: range [0.0, 10.5, p50=0.0] — container CPU percent
  - `Container-DOCKER_CONTAINER_d27123361435--bcou-trace-st-uat-statefulset-1--bcou--UATWKR18_CpuPercent`: range [0.1, 1.7, p50=0.4] — container CPU percent
  - `Container-DOCKER_CONTAINER_ef6cb138bb10--bcou-role-st-uat-statefulset-0--bcou--UATWKR18_CpuPercent`: range [0.0, 1.3, p50=0.0] — container CPU percent
  - `Container-DOCKER_CONTAINER_f861998c398e--bcou-role-st-uat-statefulset-0--bcou--UATWKR18_CpuPercent`: range [0.4, 0.4, p50=0.4] — container CPU percent
  - `OSLinux-CPU_CPU_CPUCpuUtil`: range [0.2, 100.0, p50=25.7] — looks like 0-100% CPU utilization
  - Assessment: All KPIs appear to be 0-100% scale. Low risk.

**Slot `mem`** (29 raw KPIs):
  - `Container-DOCKER_CONTAINER_01eeda2c9f0b--bcou-trace-st-uat-statefulset-1--bcou--UATWKR18_MemPercent`: range [0.0, 57.1, p50=0.0]
  - `Container-DOCKER_CONTAINER_0f6f3aa7920c--bcou-role-st-uat-statefulset-0--bcou--UATWKR18_MemPercent`: range [0.0, 58.0, p50=0.0]
  - `Container-DOCKER_CONTAINER_1bc4fc80d241--bcou-trace-st-uat-statefulset-1--bcou--UATWKR18_MemPercent`: range [0.0, 95.8, p50=75.5]
  - `Container-DOCKER_CONTAINER_23bdcf67c7e3--bcou-role-st-uat-statefulset-0--bcou--UATWKR18_MemPercent`: range [57.7, 60.3, p50=57.9]
  - `Container-DOCKER_CONTAINER_2c2336e2994f--bcou-role-st-uat-statefulset-1--bcou--UATWKR06_MemPercent`: range [57.4, 60.4, p50=60.1]
  - `Container-DOCKER_CONTAINER_2d16f5b2e830--bcou-trace-st-uat-statefulset-1--bcou--UATWKR18_MemPercent`: range [0.0, 94.2, p50=75.8]
  - `Container-DOCKER_CONTAINER_2fa7eaebac26--bcou-role-st-uat-statefulset-1--bcou--UATWKR02_MemPercent`: range [0.0, 57.3, p50=0.0]
  - `Container-DOCKER_CONTAINER_350771a68ac2--bcou-trace-st-uat-statefulset-1--bcou--UATWKR18_MemPercent`: range [0.0, 57.1, p50=0.0]
  - `Container-DOCKER_CONTAINER_3bf443a64876--bcou-role-st-uat-statefulset-1--bcou--UATWKR03_MemPercent`: range [0.0, 71.4, p50=0.0]
  - `Container-DOCKER_CONTAINER_464dc801314b--bcou-trace-st-uat-statefulset-0--bcou--UATWKR04_MemPercent`: range [0.0, 73.3, p50=73.2]
  - `Container-DOCKER_CONTAINER_69b53a78b2eb--bcou-role-st-uat-statefulset-1--bcou--UATWKR06_MemPercent`: range [0.0, 0.0, p50=0.0]
  - `Container-DOCKER_CONTAINER_6d83a96887c3--bcou-role-st-uat-statefulset-0--bcou--UATWKR18_MemPercent`: range [0.0, 62.8, p50=59.2]
  - `Container-DOCKER_CONTAINER_76d31070b844--bcou-role-st-uat-statefulset-1--bcou--UATWKR06_MemPercent`: range [0.0, 59.4, p50=0.0]
  - `Container-DOCKER_CONTAINER_7b4b80f345e0--bcou-role-st-uat-statefulset-0--bcou--UATWKR04_MemPercent`: range [0.0, 58.6, p50=0.0]
  - `Container-DOCKER_CONTAINER_89f0c1e5346c--bcou-trace-st-uat-statefulset-0--bcou--UATWKR04_MemPercent`: range [0.0, 60.4, p50=0.0]
  - `Container-DOCKER_CONTAINER_94eca4f96efe--bcou-role-st-uat-statefulset-1--bcou--UATWKR04_MemPercent`: range [0.0, 76.3, p50=0.0]
  - `Container-DOCKER_CONTAINER_9e8c309d0aab--bcou-trace-st-uat-statefulset-0--bcou--UATWKR04_MemPercent`: range [0.0, 79.0, p50=58.1]
  - `Container-DOCKER_CONTAINER_ac3ba8476104--bcou-trace-st-uat-statefulset-1--bcou--UATWKR18_MemPercent`: range [0.0, 94.0, p50=28.5]
  - `Container-DOCKER_CONTAINER_b2fc383d2438--bcou-trace-st-uat-statefulset-1--bcou--UATWKR18_MemPercent`: range [0.0, 0.0, p50=0.0]
  - `Container-DOCKER_CONTAINER_b30097144a13--bcou-trace-st-uat-statefulset-0--bcou--UATWKR04_MemPercent`: range [0.0, 71.6, p50=70.7]
  - `Container-DOCKER_CONTAINER_b6760337dc49--bcou-role-st-uat-statefulset-0--bcou--UATWKR18_MemPercent`: range [0.0, 81.9, p50=58.5]
  - `Container-DOCKER_CONTAINER_bbc780df1fce--bcou-role-st-uat-statefulset-0--bcou--UATWKR18_MemPercent`: range [0.0, 77.6, p50=57.8]
  - `Container-DOCKER_CONTAINER_c67422614c81--bcou-role-st-uat-statefulset-1--bcou--UATWKR02_MemPercent`: range [0.0, 57.5, p50=0.0]
  - `Container-DOCKER_CONTAINER_cb2bbb5e3f90--bcou-role-st-uat-statefulset-1--bcou--UATWKR02_MemPercent`: range [0.0, 86.0, p50=59.2]
  - `Container-DOCKER_CONTAINER_cd2b4a29291e--bcou-role-st-uat-statefulset-1--bcou--UATWKR04_MemPercent`: range [0.0, 58.5, p50=0.0]
  - `Container-DOCKER_CONTAINER_d27123361435--bcou-trace-st-uat-statefulset-1--bcou--UATWKR18_MemPercent`: range [57.2, 58.9, p50=58.2]
  - `Container-DOCKER_CONTAINER_ef6cb138bb10--bcou-role-st-uat-statefulset-0--bcou--UATWKR18_MemPercent`: range [0.0, 60.0, p50=0.0]
  - `Container-DOCKER_CONTAINER_f861998c398e--bcou-role-st-uat-statefulset-0--bcou--UATWKR18_MemPercent`: range [58.8, 58.8, p50=58.8]
  - `Tomcat-MEMORY_7441-MEMORY_JVMMemoryUsedPercent`: range [7.0, 89.0, p50=37.0]

**Slot `net_rx`** (24 raw KPIs, log1p-transformed):
  - `Container-DOCKER_CONTAINER_0f6f3aa7920c--bcou-role-st-uat-statefulset-0--bcou--UATWKR18_NetworkRxBytes`: 5 rows, raw value range noted, transformed: p50=0.000, p95=0.000, max=0.000
  - `Container-DOCKER_CONTAINER_1bc4fc80d241--bcou-trace-st-uat-statefulset-1--bcou--UATWKR18_NetworkRxBytes`: 87 rows, raw value range noted, transformed: p50=12.904, p95=14.748, max=15.103
  - `Container-DOCKER_CONTAINER_23bdcf67c7e3--bcou-role-st-uat-statefulset-0--bcou--UATWKR18_NetworkRxBytes`: 69 rows, raw value range noted, transformed: p50=0.000, p95=0.000, max=0.000
  - `Container-DOCKER_CONTAINER_2c2336e2994f--bcou-role-st-uat-statefulset-1--bcou--UATWKR06_NetworkRxBytes`: 559 rows, raw value range noted, transformed: p50=13.438, p95=14.652, max=15.186
  - `Container-DOCKER_CONTAINER_2d16f5b2e830--bcou-trace-st-uat-statefulset-1--bcou--UATWKR18_NetworkRxBytes`: 145 rows, raw value range noted, transformed: p50=13.236, p95=14.988, max=15.378
  - `Container-DOCKER_CONTAINER_2fa7eaebac26--bcou-role-st-uat-statefulset-1--bcou--UATWKR02_NetworkRxBytes`: 1 rows, raw value range noted, transformed: p50=0.000, p95=0.000, max=0.000
  - `Container-DOCKER_CONTAINER_350771a68ac2--bcou-trace-st-uat-statefulset-1--bcou--UATWKR18_NetworkRxBytes`: 29 rows, raw value range noted, transformed: p50=9.586, p95=13.941, max=14.117
  - `Container-DOCKER_CONTAINER_3bf443a64876--bcou-role-st-uat-statefulset-1--bcou--UATWKR03_NetworkRxBytes`: 154 rows, raw value range noted, transformed: p50=13.442, p95=14.839, max=15.921
  - `Container-DOCKER_CONTAINER_464dc801314b--bcou-trace-st-uat-statefulset-0--bcou--UATWKR04_NetworkRxBytes`: 181 rows, raw value range noted, transformed: p50=13.540, p95=15.036, max=15.441
  - `Container-DOCKER_CONTAINER_6d83a96887c3--bcou-role-st-uat-statefulset-0--bcou--UATWKR18_NetworkRxBytes`: 26 rows, raw value range noted, transformed: p50=0.000, p95=0.000, max=0.000
  - `Container-DOCKER_CONTAINER_76d31070b844--bcou-role-st-uat-statefulset-1--bcou--UATWKR06_NetworkRxBytes`: 1 rows, raw value range noted, transformed: p50=13.897, p95=13.897, max=13.897
  - `Container-DOCKER_CONTAINER_7b4b80f345e0--bcou-role-st-uat-statefulset-0--bcou--UATWKR04_NetworkRxBytes`: 150 rows, raw value range noted, transformed: p50=12.293, p95=14.132, max=14.525
  - `Container-DOCKER_CONTAINER_89f0c1e5346c--bcou-trace-st-uat-statefulset-0--bcou--UATWKR04_NetworkRxBytes`: 94 rows, raw value range noted, transformed: p50=11.765, p95=14.500, max=14.775
  - `Container-DOCKER_CONTAINER_94eca4f96efe--bcou-role-st-uat-statefulset-1--bcou--UATWKR04_NetworkRxBytes`: 171 rows, raw value range noted, transformed: p50=13.050, p95=14.693, max=14.920
  - `Container-DOCKER_CONTAINER_9e8c309d0aab--bcou-trace-st-uat-statefulset-0--bcou--UATWKR04_NetworkRxBytes`: 171 rows, raw value range noted, transformed: p50=13.872, p95=14.995, max=15.567
  - `Container-DOCKER_CONTAINER_b30097144a13--bcou-trace-st-uat-statefulset-0--bcou--UATWKR04_NetworkRxBytes`: 316 rows, raw value range noted, transformed: p50=11.216, p95=14.839, max=15.477
  - `Container-DOCKER_CONTAINER_b6760337dc49--bcou-role-st-uat-statefulset-0--bcou--UATWKR18_NetworkRxBytes`: 103 rows, raw value range noted, transformed: p50=12.661, p95=14.510, max=14.865
  - `Container-DOCKER_CONTAINER_bbc780df1fce--bcou-role-st-uat-statefulset-0--bcou--UATWKR18_NetworkRxBytes`: 89 rows, raw value range noted, transformed: p50=0.000, p95=0.000, max=0.000
  - `Container-DOCKER_CONTAINER_c67422614c81--bcou-role-st-uat-statefulset-1--bcou--UATWKR02_NetworkRxBytes`: 10 rows, raw value range noted, transformed: p50=0.000, p95=0.000, max=0.000
  - `Container-DOCKER_CONTAINER_cb2bbb5e3f90--bcou-role-st-uat-statefulset-1--bcou--UATWKR02_NetworkRxBytes`: 142 rows, raw value range noted, transformed: p50=0.000, p95=0.000, max=0.000
  - `Container-DOCKER_CONTAINER_cd2b4a29291e--bcou-role-st-uat-statefulset-1--bcou--UATWKR04_NetworkRxBytes`: 10 rows, raw value range noted, transformed: p50=11.955, p95=13.255, max=13.345
  - `Container-DOCKER_CONTAINER_d27123361435--bcou-trace-st-uat-statefulset-1--bcou--UATWKR18_NetworkRxBytes`: 90 rows, raw value range noted, transformed: p50=13.545, p95=15.023, max=15.232
  - `Container-DOCKER_CONTAINER_ef6cb138bb10--bcou-role-st-uat-statefulset-0--bcou--UATWKR18_NetworkRxBytes`: 12 rows, raw value range noted, transformed: p50=12.653, p95=13.793, max=14.066
  - `Container-DOCKER_CONTAINER_f861998c398e--bcou-role-st-uat-statefulset-0--bcou--UATWKR18_NetworkRxBytes`: 1 rows, raw value range noted, transformed: p50=0.000, p95=0.000, max=0.000
  - Assessment: All receive byte counts are log1p-transformed. Scale is consistent across KPIs in this slot.

**Slot `net_tx`** (28 raw KPIs, log1p-transformed):
  - `Container-DOCKER_CONTAINER_01eeda2c9f0b--bcou-trace-st-uat-statefulset-1--bcou--UATWKR18_NetworkTxBytes`: 50 rows, raw value range noted, transformed: p50=0.000, p95=0.000, max=11.622
  - `Container-DOCKER_CONTAINER_0f6f3aa7920c--bcou-role-st-uat-statefulset-0--bcou--UATWKR18_NetworkTxBytes`: 60 rows, raw value range noted, transformed: p50=0.000, p95=0.000, max=0.000
  - `Container-DOCKER_CONTAINER_1bc4fc80d241--bcou-trace-st-uat-statefulset-1--bcou--UATWKR18_NetworkTxBytes`: 204 rows, raw value range noted, transformed: p50=7.834, p95=12.705, max=13.364
  - `Container-DOCKER_CONTAINER_23bdcf67c7e3--bcou-role-st-uat-statefulset-0--bcou--UATWKR18_NetworkTxBytes`: 113 rows, raw value range noted, transformed: p50=0.000, p95=0.000, max=0.000
  - `Container-DOCKER_CONTAINER_2c2336e2994f--bcou-role-st-uat-statefulset-1--bcou--UATWKR06_NetworkTxBytes`: 563 rows, raw value range noted, transformed: p50=11.768, p95=13.007, max=13.653
  - `Container-DOCKER_CONTAINER_2d16f5b2e830--bcou-trace-st-uat-statefulset-1--bcou--UATWKR18_NetworkTxBytes`: 234 rows, raw value range noted, transformed: p50=11.478, p95=12.888, max=14.248
  - `Container-DOCKER_CONTAINER_2fa7eaebac26--bcou-role-st-uat-statefulset-1--bcou--UATWKR02_NetworkTxBytes`: 120 rows, raw value range noted, transformed: p50=0.000, p95=0.000, max=0.000
  - `Container-DOCKER_CONTAINER_350771a68ac2--bcou-trace-st-uat-statefulset-1--bcou--UATWKR18_NetworkTxBytes`: 139 rows, raw value range noted, transformed: p50=0.000, p95=10.842, max=12.558
  - `Container-DOCKER_CONTAINER_3bf443a64876--bcou-role-st-uat-statefulset-1--bcou--UATWKR03_NetworkTxBytes`: 601 rows, raw value range noted, transformed: p50=0.000, p95=12.701, max=13.487
  - `Container-DOCKER_CONTAINER_464dc801314b--bcou-trace-st-uat-statefulset-0--bcou--UATWKR04_NetworkTxBytes`: 284 rows, raw value range noted, transformed: p50=11.558, p95=13.021, max=13.374
  - `Container-DOCKER_CONTAINER_69b53a78b2eb--bcou-role-st-uat-statefulset-1--bcou--UATWKR06_NetworkTxBytes`: 352 rows, raw value range noted, transformed: p50=0.000, p95=0.000, max=0.000
  - `Container-DOCKER_CONTAINER_6d83a96887c3--bcou-role-st-uat-statefulset-0--bcou--UATWKR18_NetworkTxBytes`: 58 rows, raw value range noted, transformed: p50=0.000, p95=0.000, max=0.000
  - `Container-DOCKER_CONTAINER_76d31070b844--bcou-role-st-uat-statefulset-1--bcou--UATWKR06_NetworkTxBytes`: 318 rows, raw value range noted, transformed: p50=0.000, p95=0.000, max=13.190
  - `Container-DOCKER_CONTAINER_7b4b80f345e0--bcou-role-st-uat-statefulset-0--bcou--UATWKR04_NetworkTxBytes`: 450 rows, raw value range noted, transformed: p50=7.060, p95=12.024, max=12.793
  - `Container-DOCKER_CONTAINER_89f0c1e5346c--bcou-trace-st-uat-statefulset-0--bcou--UATWKR04_NetworkTxBytes`: 297 rows, raw value range noted, transformed: p50=0.000, p95=12.028, max=13.507
  - `Container-DOCKER_CONTAINER_94eca4f96efe--bcou-role-st-uat-statefulset-1--bcou--UATWKR04_NetworkTxBytes`: 832 rows, raw value range noted, transformed: p50=0.000, p95=12.618, max=13.162
  - `Container-DOCKER_CONTAINER_9e8c309d0aab--bcou-trace-st-uat-statefulset-0--bcou--UATWKR04_NetworkTxBytes`: 598 rows, raw value range noted, transformed: p50=7.870, p95=12.972, max=13.510
  - `Container-DOCKER_CONTAINER_ac3ba8476104--bcou-trace-st-uat-statefulset-1--bcou--UATWKR18_NetworkTxBytes`: 4 rows, raw value range noted, transformed: p50=3.201, p95=10.281, max=10.966
  - `Container-DOCKER_CONTAINER_b2fc383d2438--bcou-trace-st-uat-statefulset-1--bcou--UATWKR18_NetworkTxBytes`: 55 rows, raw value range noted, transformed: p50=0.000, p95=0.000, max=0.000
  - `Container-DOCKER_CONTAINER_b30097144a13--bcou-trace-st-uat-statefulset-0--bcou--UATWKR04_NetworkTxBytes`: 598 rows, raw value range noted, transformed: p50=7.590, p95=12.784, max=13.641
  - `Container-DOCKER_CONTAINER_b6760337dc49--bcou-role-st-uat-statefulset-0--bcou--UATWKR18_NetworkTxBytes`: 176 rows, raw value range noted, transformed: p50=10.634, p95=12.796, max=13.225
  - `Container-DOCKER_CONTAINER_bbc780df1fce--bcou-role-st-uat-statefulset-0--bcou--UATWKR18_NetworkTxBytes`: 205 rows, raw value range noted, transformed: p50=0.000, p95=0.000, max=0.000
  - `Container-DOCKER_CONTAINER_c67422614c81--bcou-role-st-uat-statefulset-1--bcou--UATWKR02_NetworkTxBytes`: 137 rows, raw value range noted, transformed: p50=0.000, p95=0.000, max=0.000
  - `Container-DOCKER_CONTAINER_cb2bbb5e3f90--bcou-role-st-uat-statefulset-1--bcou--UATWKR02_NetworkTxBytes`: 343 rows, raw value range noted, transformed: p50=0.000, p95=0.000, max=0.000
  - `Container-DOCKER_CONTAINER_cd2b4a29291e--bcou-role-st-uat-statefulset-1--bcou--UATWKR04_NetworkTxBytes`: 652 rows, raw value range noted, transformed: p50=0.000, p95=0.000, max=11.935
  - `Container-DOCKER_CONTAINER_d27123361435--bcou-trace-st-uat-statefulset-1--bcou--UATWKR18_NetworkTxBytes`: 156 rows, raw value range noted, transformed: p50=11.703, p95=13.008, max=13.397
  - `Container-DOCKER_CONTAINER_ef6cb138bb10--bcou-role-st-uat-statefulset-0--bcou--UATWKR18_NetworkTxBytes`: 73 rows, raw value range noted, transformed: p50=0.000, p95=11.986, max=12.433
  - `Container-DOCKER_CONTAINER_f861998c398e--bcou-role-st-uat-statefulset-0--bcou--UATWKR18_NetworkTxBytes`: 1 rows, raw value range noted, transformed: p50=0.000, p95=0.000, max=0.000
  - Assessment: All transmit byte counts are log1p-transformed. Scale is consistent across KPIs in this slot.

## 5. Recommendations for Phase 0.9.2.3c


### Slots requiring attention

- **`cpu`** (29 flagged KPIs):
  - Relevant flag types: CONSTANT_OR_NEAR_CONSTANT_SIGNAL, EXTREME_OUTLIER_TAIL, MEDIAN_SCALE_MISMATCH, POSSIBLE_RATIO_PERCENT_MIX, SPARSE_SIGNAL
  - **Action**: Consider per-KPI unit normalization before slot-level modeling
  - **Action**: Separate 0-1 and 0-100 KPIs into sub-slots or apply explicit rescaling
  - KPIs to review: Container-DOCKER_CONTAINER_01eeda2c9f0b--bcou-trace-st-uat-statefulset-1--bcou--UATWKR18_CpuPercent, Container-DOCKER_CONTAINER_0f6f3aa7920c--bcou-role-st-uat-statefulset-0--bcou--UATWKR18_CpuPercent, Container-DOCKER_CONTAINER_1bc4fc80d241--bcou-trace-st-uat-statefulset-1--bcou--UATWKR18_CpuPercent, Container-DOCKER_CONTAINER_23bdcf67c7e3--bcou-role-st-uat-statefulset-0--bcou--UATWKR18_CpuPercent, Container-DOCKER_CONTAINER_2c2336e2994f--bcou-role-st-uat-statefulset-1--bcou--UATWKR06_CpuPercent

- **`mem`** (29 flagged KPIs):
  - Relevant flag types: CONSTANT_OR_NEAR_CONSTANT_SIGNAL, POSSIBLE_RATIO_PERCENT_MIX, SPARSE_SIGNAL
  - **Action**: Separate 0-1 and 0-100 KPIs into sub-slots or apply explicit rescaling
  - KPIs to review: Container-DOCKER_CONTAINER_01eeda2c9f0b--bcou-trace-st-uat-statefulset-1--bcou--UATWKR18_MemPercent, Container-DOCKER_CONTAINER_0f6f3aa7920c--bcou-role-st-uat-statefulset-0--bcou--UATWKR18_MemPercent, Container-DOCKER_CONTAINER_1bc4fc80d241--bcou-trace-st-uat-statefulset-1--bcou--UATWKR18_MemPercent, Container-DOCKER_CONTAINER_23bdcf67c7e3--bcou-role-st-uat-statefulset-0--bcou--UATWKR18_MemPercent, Container-DOCKER_CONTAINER_2c2336e2994f--bcou-role-st-uat-statefulset-1--bcou--UATWKR06_MemPercent

- **`net_rx`** (5 flagged KPIs):
  - Relevant flag types: CONSTANT_OR_NEAR_CONSTANT_SIGNAL, SPARSE_SIGNAL
  - KPIs to review: Container-DOCKER_CONTAINER_23bdcf67c7e3--bcou-role-st-uat-statefulset-0--bcou--UATWKR18_NetworkRxBytes, Container-DOCKER_CONTAINER_6d83a96887c3--bcou-role-st-uat-statefulset-0--bcou--UATWKR18_NetworkRxBytes, Container-DOCKER_CONTAINER_bbc780df1fce--bcou-role-st-uat-statefulset-0--bcou--UATWKR18_NetworkRxBytes, Container-DOCKER_CONTAINER_c67422614c81--bcou-role-st-uat-statefulset-1--bcou--UATWKR02_NetworkRxBytes, Container-DOCKER_CONTAINER_cb2bbb5e3f90--bcou-role-st-uat-statefulset-1--bcou--UATWKR02_NetworkRxBytes

- **`net_tx`** (10 flagged KPIs):
  - Relevant flag types: CONSTANT_OR_NEAR_CONSTANT_SIGNAL, SPARSE_SIGNAL
  - KPIs to review: Container-DOCKER_CONTAINER_0f6f3aa7920c--bcou-role-st-uat-statefulset-0--bcou--UATWKR18_NetworkTxBytes, Container-DOCKER_CONTAINER_23bdcf67c7e3--bcou-role-st-uat-statefulset-0--bcou--UATWKR18_NetworkTxBytes, Container-DOCKER_CONTAINER_2fa7eaebac26--bcou-role-st-uat-statefulset-1--bcou--UATWKR02_NetworkTxBytes, Container-DOCKER_CONTAINER_69b53a78b2eb--bcou-role-st-uat-statefulset-1--bcou--UATWKR06_NetworkTxBytes, Container-DOCKER_CONTAINER_6d83a96887c3--bcou-role-st-uat-statefulset-0--bcou--UATWKR18_NetworkTxBytes

### Summary of actions

| Category | Slots / KPIs | Recommended Action |
|---|---|---|
| Ready to model | None | Proceed directly |
| Scale mismatch | `cpu` | Per-KPI robust scaling or unit normalization |
| Ratio/percent mix | `cpu` | Sub-slot separation + explicit ratio→percentage rescaling |
| Ratio/percent mix | `mem` | Sub-slot separation + explicit ratio→percentage rescaling |
| Dropped KPIs (by bank_safe_v1) | disk_io, mysql_io, threads, sessions, fgc, mem_usage, jvm_mem, jvm_cpu | Retain drop; insufficient semantic alignment with OB slots |