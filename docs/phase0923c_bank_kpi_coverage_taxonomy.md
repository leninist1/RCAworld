# Phase 0.9.2.3c — Bank KPI Coverage Taxonomy Audit Report

## 1. Executive Summary

- **Total raw records**: 12,218,094
- **Unique KPI names**: 497

### Acceptance Status Breakdown

| Status | Records | Record % | Unique KPIs |
|---|---|---|---|
| accepted | 163,849 | 1.34% | 110 |
| dropped | 2,593,354 | 21.23% | 112 |
| unmapped | 9,460,891 | 77.43% | 275 |

### Source Family Breakdown

| Source Family | Records | Record % |
|---|---|---|
| OSLinux | 7,305,870 | 59.80% |
| MySQL | 2,667,304 | 21.83% |
| Tomcat | 1,087,288 | 8.90% |
| Redis | 779,572 | 6.38% |
| JVM | 341,648 | 2.80% |
| Container | 36,412 | 0.30% |

### Semantic Family Breakdown

| Semantic Family | Records | Record % |
|---|---|---|
| database_io | 2,177,489 | 17.82% |
| filesystem | 1,843,131 | 15.09% |
| cpu | 1,709,235 | 13.99% |
| disk | 1,434,201 | 11.74% |
| memory | 1,042,747 | 8.53% |
| network | 790,672 | 6.47% |
| session | 639,242 | 5.23% |
| request | 502,572 | 4.11% |
| cache | 449,396 | 3.68% |
| process | 380,590 | 3.11% |
| unknown | 319,781 | 2.62% |
| connection | 224,687 | 1.84% |
| uptime | 198,564 | 1.63% |
| thread | 165,430 | 1.35% |
| monitoring | 126,439 | 1.03% |
| latency | 86,877 | 0.71% |
| swap | 81,241 | 0.66% |
| system | 32,280 | 0.26% |
| time_sync | 13,520 | 0.11% |

### Top KPI Coverage Concentration

- **Top 10 KPIs**: 13.0% of all records
- **Top 25 KPIs**: 26.6%
- **Top 50 KPIs**: 41.9%
- **Top 100 KPIs**: 63.6%

## 2. Top Unmapped KPIs

(Top 50 by row count)

| # | Raw KPI Name | Rows | Row% | Cum% | Source | Semantic | Kind | Unit | Action | Confidence |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | `OSLinux-OSLinux_PROCESS_PROCESS_PROCNoZombies` | 160,491 | 1.31% | 1.3% | OSLinux | unknown | unknown | unknown | manual_review | low |
| 2 | `OSLinux-OSLinux_PROCESS_zabbix_PROCPPCount` | 160,489 | 1.31% | 2.6% | OSLinux | process | counter | count | requires_new_slot | medium |
| 3 | `OSLinux-OSLinux_ZABBIX_Host_Uptime` | 160,460 | 1.31% | 3.9% | OSLinux | uptime | duration | milliseconds | manual_review | low |
| 4 | `OSLinux-OSLinux_NETWORK_NETWORK_TCP-CLOSE-WAIT` | 160,049 | 1.31% | 5.3% | OSLinux | network | unknown | unknown | requires_new_slot | medium |
| 5 | `OSLinux-OSLinux_NETWORK_NETWORK_TCP-FIN-WAIT` | 159,678 | 1.31% | 6.6% | OSLinux | network | unknown | unknown | requires_new_slot | medium |
| 6 | `OSLinux-CPU_CPU_CPULoad` | 158,529 | 1.30% | 7.9% | OSLinux | cpu | gauge | unknown | manual_review | low |
| 7 | `OSLinux-CPU_CPU_CPUidleutil` | 158,529 | 1.30% | 9.2% | OSLinux | cpu | ratio | percent | map_safe_v2 | high |
| 8 | `OSLinux-CPU_CPU_CPUWio` | 158,523 | 1.30% | 10.4% | OSLinux | cpu | unknown | unknown | manual_review | low |
| 9 | `OSLinux-CPU_CPU_CPUSysTime` | 158,515 | 1.30% | 11.7% | OSLinux | cpu | duration | milliseconds | manual_review | low |
| 10 | `OSLinux-CPU_CPU_CPUUserTime` | 158,512 | 1.30% | 13.0% | OSLinux | cpu | duration | milliseconds | manual_review | low |
| 11 | `OSLinux-OSLinux_PROCESS_zabbix-zabbix_agentd_PROCPPCPUPerc` | 158,483 | 1.30% | 14.3% | OSLinux | cpu | ratio | percent | map_safe_v2 | high |
| 12 | `OSLinux-OSLinux_MEMORY_MEMORY_MEMUsedMemPerc` | 135,715 | 1.11% | 15.5% | OSLinux | memory | ratio | percent | map_safe_v2 | high |
| 13 | `OSLinux-OSLinux_MEMORY_MEMORY_NoCacheMemPerc` | 130,345 | 1.07% | 17.6% | OSLinux | memory | ratio | percent | map_safe_v2 | high |
| 14 | `OSLinux-OSLinux_MEMORY_MEMORY_UserMem` | 121,488 | 0.99% | 18.6% | OSLinux | memory | gauge | unknown | manual_review | low |
| 15 | `OSLinux-OSLinux_PROCESS_PROCESS_PROCPPMem` | 119,082 | 0.97% | 19.6% | OSLinux | process | unknown | unknown | requires_new_slot | medium |
| 16 | `OSLinux-OSLinux_MEMORY_MEMORY_CacheMem` | 118,963 | 0.97% | 20.5% | OSLinux | memory | gauge | unknown | manual_review | low |
| 17 | `OSLinux-OSLinux_MEMORY_MEMORY_MEMFreeMem` | 118,949 | 0.97% | 21.5% | OSLinux | memory | gauge | unknown | manual_review | low |
| 18 | `OSLinux-OSLinux_FILE_-tmp-zabbix_agentd.log_FileSizeMB` | 114,436 | 0.94% | 22.4% | OSLinux | monitoring | gauge | unknown | manual_review | low |
| 19 | `OSLinux-OSLinux_NETWORK_NETWORK_TotalTcpConnNum` | 107,316 | 0.88% | 23.3% | OSLinux | network | gauge | count | requires_new_slot | medium |
| 20 | `OSLinux-OSLinux_SWAP_SWAP_So` | 79,669 | 0.65% | 24.0% | OSLinux | unknown | unknown | unknown | manual_review | low |
| 21 | `OSLinux-OSLinux_SWAP_SWAP_Si` | 79,621 | 0.65% | 24.6% | OSLinux | unknown | unknown | unknown | manual_review | low |
| 22 | `OSLinux-OSLinux_PROCESS_PROCESS_PROCPPMemPerc` | 79,241 | 0.65% | 25.3% | OSLinux | process | ratio | percent | requires_new_slot | medium |
| 23 | `OSLinux-CPU_CPU-0_SingleCpuidle` | 78,710 | 0.64% | 25.9% | OSLinux | cpu | unknown | unknown | manual_review | low |
| 24 | `OSLinux-CPU_CPU-2_SingleCpuidle` | 78,689 | 0.64% | 26.6% | OSLinux | cpu | unknown | unknown | manual_review | low |
| 25 | `OSLinux-OSLinux_SWAP_SWAP_SWPTotSwapUsedPercent` | 78,563 | 0.64% | 32.4% | OSLinux | swap | ratio | percent | requires_new_slot | medium |
| 26 | `OSLinux-OSLinux_FILESYSTEM_-tmp_FSCapacity` | 72,439 | 0.59% | 35.5% | OSLinux | filesystem | gauge | unknown | requires_new_slot | medium |
| 27 | `OSLinux-OSLinux_FILESYSTEM_-tmp_FSInodeUsedPercent` | 72,375 | 0.59% | 36.1% | OSLinux | filesystem | ratio | percent | requires_new_slot | medium |
| 28 | `OSLinux-OSLinux_FILESYSTEM_-_FSCapacity` | 72,360 | 0.59% | 36.7% | OSLinux | filesystem | gauge | unknown | requires_new_slot | medium |
| 29 | `OSLinux-OSLinux_FILESYSTEM_-home_FSCapacity` | 72,306 | 0.59% | 37.3% | OSLinux | filesystem | gauge | unknown | requires_new_slot | medium |
| 30 | `OSLinux-OSLinux_FILESYSTEM_-boot_FSCapacity` | 72,302 | 0.59% | 37.9% | OSLinux | filesystem | gauge | unknown | requires_new_slot | medium |
| 31 | `OSLinux-OSLinux_FILESYSTEM_-cmbc_admin_FSInodeUsedPercent` | 72,291 | 0.59% | 38.5% | OSLinux | filesystem | ratio | percent | requires_new_slot | medium |
| 32 | `OSLinux-OSLinux_FILESYSTEM_-_FSInodeUsedPercent` | 72,289 | 0.59% | 39.1% | OSLinux | filesystem | ratio | percent | requires_new_slot | medium |
| 33 | `OSLinux-OSLinux_FILESYSTEM_-cmbc_admin_FSCapacity` | 72,289 | 0.59% | 39.6% | OSLinux | filesystem | gauge | unknown | requires_new_slot | medium |
| 34 | `OSLinux-OSLinux_FILESYSTEM_-home_FSInodeUsedPercent` | 72,289 | 0.59% | 40.2% | OSLinux | filesystem | ratio | percent | requires_new_slot | medium |
| 35 | `OSLinux-OSLinux_FILESYSTEM_-boot_FSInodeUsedPercent` | 72,286 | 0.59% | 40.8% | OSLinux | filesystem | ratio | percent | requires_new_slot | medium |
| 36 | `OSLinux-CPU_CPU-2_SingleCpuUtil` | 65,146 | 0.53% | 43.6% | OSLinux | cpu | ratio | percent | map_safe_v2 | high |
| 37 | `OSLinux-CPU_CPU-0_SingleCpuUtil` | 65,130 | 0.53% | 44.1% | OSLinux | cpu | ratio | percent | map_safe_v2 | high |
| 38 | `OSLinux-OSLinux_FILESYSTEM_-tmp_FSUsedSpace` | 59,243 | 0.48% | 44.6% | OSLinux | filesystem | unknown | unknown | requires_new_slot | medium |
| 39 | `OSLinux-OSLinux_FILESYSTEM_-_FSAvailableSpace` | 59,225 | 0.48% | 45.1% | OSLinux | filesystem | gauge | unknown | requires_new_slot | medium |
| 40 | `OSLinux-OSLinux_FILESYSTEM_-tmp_FSAvailableSpace` | 59,206 | 0.48% | 45.6% | OSLinux | filesystem | gauge | unknown | requires_new_slot | medium |
| 41 | `OSLinux-OSLinux_FILESYSTEM_-home_FSUsedSpace` | 59,125 | 0.48% | 46.1% | OSLinux | filesystem | unknown | unknown | requires_new_slot | medium |
| 42 | `OSLinux-OSLinux_FILESYSTEM_-boot_FSAvailableSpace` | 59,050 | 0.48% | 46.5% | OSLinux | filesystem | gauge | unknown | requires_new_slot | medium |
| 43 | `OSLinux-OSLinux_FILESYSTEM_-cmbc_admin_FSAvailableSpace` | 59,027 | 0.48% | 47.0% | OSLinux | filesystem | gauge | unknown | requires_new_slot | medium |
| 44 | `OSLinux-OSLinux_FILESYSTEM_-_FSUsedSpace` | 59,022 | 0.48% | 47.5% | OSLinux | filesystem | unknown | unknown | requires_new_slot | medium |
| 45 | `OSLinux-OSLinux_FILESYSTEM_-boot_FSUsedSpace` | 59,018 | 0.48% | 48.0% | OSLinux | filesystem | unknown | unknown | requires_new_slot | medium |
| 46 | `OSLinux-OSLinux_FILESYSTEM_-cmbc_admin_FSUsedSpace` | 59,016 | 0.48% | 48.5% | OSLinux | filesystem | unknown | unknown | requires_new_slot | medium |
| 47 | `OSLinux-OSLinux_FILESYSTEM_-home_FSAvailableSpace` | 59,006 | 0.48% | 49.0% | OSLinux | filesystem | gauge | unknown | requires_new_slot | medium |
| 48 | `Tomcat-Threads_7441-"http-nio-8003"_CurrentThreadCountThreadInfo` | 56,907 | 0.47% | 49.4% | Tomcat | request | counter | count | requires_new_slot | medium |
| 49 | `Tomcat-Requests_7441-"http-nio-8003"_MaxTimeRequestInfo` | 56,906 | 0.47% | 49.9% | Tomcat | request | duration | milliseconds | requires_new_slot | medium |
| 50 | `Tomcat-Requests_7441-"http-nio-8003"_RequestCountRequestInfo` | 56,906 | 0.47% | 50.4% | Tomcat | request | counter | count | requires_new_slot | medium |

## 3. High-Confidence `bank_safe_v2` Candidates

### Candidate KPIs

| # | Raw KPI Name | Target Slot | Entity | Kind | Unit | Transform | New Rows | New Cov% | Reason |
|---|---|---|---|---|---|---|---|---|---|
| 1 | `OSLinux-CPU_CPU_CPUidleutil` | `cpu` | host | ratio | percent | unknown | 158,529 | 1.297% | Maps to existing slot 'cpu'; semantic=cpu, unit=percent, source=OSLinux |
| 2 | `OSLinux-OSLinux_PROCESS_zabbix-zabbix_agentd_PROCPPCPUPerc` | `cpu` | host | ratio | percent | identity | 158,483 | 1.297% | Maps to existing slot 'cpu'; semantic=cpu, unit=percent, source=OSLinux |
| 3 | `OSLinux-OSLinux_MEMORY_MEMORY_MEMUsedMemPerc` | `mem` | host | ratio | percent | identity | 135,715 | 1.111% | Maps to existing slot 'mem'; semantic=memory, unit=percent, source=OSLinux |
| 4 | `OSLinux-CPU_CPU_CPUCpuUtil` | `cpu` | host | ratio | percent | unknown | 130,588 | 1.069% | Maps to existing slot 'cpu'; semantic=cpu, unit=percent, source=OSLinux |
| 5 | `OSLinux-OSLinux_MEMORY_MEMORY_NoCacheMemPerc` | `mem` | host | ratio | percent | identity | 130,345 | 1.067% | Maps to existing slot 'mem'; semantic=memory, unit=percent, source=OSLinux |
| 6 | `OSLinux-CPU_CPU-2_SingleCpuUtil` | `cpu` | host | ratio | percent | unknown | 65,146 | 0.533% | Maps to existing slot 'cpu'; semantic=cpu, unit=percent, source=OSLinux |
| 7 | `OSLinux-CPU_CPU-0_SingleCpuUtil` | `cpu` | host | ratio | percent | unknown | 65,130 | 0.533% | Maps to existing slot 'cpu'; semantic=cpu, unit=percent, source=OSLinux |
| 8 | `OSLinux-CPU_CPU-3_SingleCpuUtil` | `cpu` | host | ratio | percent | unknown | 46,614 | 0.382% | Maps to existing slot 'cpu'; semantic=cpu, unit=percent, source=OSLinux |
| 9 | `OSLinux-CPU_CPU-1_SingleCpuUtil` | `cpu` | host | ratio | percent | unknown | 46,573 | 0.381% | Maps to existing slot 'cpu'; semantic=cpu, unit=percent, source=OSLinux |
| 10 | `Tomcat-MEMORY_7441-MEMORY_JVMMemoryUsedPercent` | `mem` | application_server | ratio | percent | identity | 7,789 | 0.064% | Maps to existing slot 'mem'; semantic=memory, unit=percent, source=Tomcat |
| 11 | `Container-DOCKER_CONTAINER_94eca4f96efe--bcou-role-st-uat-st` | `cpu` | container | ratio | percent | identity | 974 | 0.008% | Maps to existing slot 'cpu'; semantic=cpu, unit=percent, source=Container |
| 12 | `Container-DOCKER_CONTAINER_94eca4f96efe--bcou-role-st-uat-st` | `mem` | container | ratio | percent | identity | 974 | 0.008% | Maps to existing slot 'mem'; semantic=memory, unit=percent, source=Container |
| 13 | `Container-DOCKER_CONTAINER_3bf443a64876--bcou-role-st-uat-st` | `cpu` | container | ratio | percent | identity | 946 | 0.008% | Maps to existing slot 'cpu'; semantic=cpu, unit=percent, source=Container |
| 14 | `Container-DOCKER_CONTAINER_94eca4f96efe--bcou-role-st-uat-st` | `net_tx` | container | unknown | bytes | log1p | 832 | 0.007% | Maps to existing slot 'net_tx'; semantic=network, unit=bytes, source=Container |
| 15 | `Container-DOCKER_CONTAINER_cd2b4a29291e--bcou-role-st-uat-st` | `net_tx` | container | unknown | bytes | log1p | 652 | 0.005% | Maps to existing slot 'net_tx'; semantic=network, unit=bytes, source=Container |
| 16 | `Container-DOCKER_CONTAINER_cd2b4a29291e--bcou-role-st-uat-st` | `mem` | container | ratio | percent | identity | 652 | 0.005% | Maps to existing slot 'mem'; semantic=memory, unit=percent, source=Container |
| 17 | `Container-DOCKER_CONTAINER_cd2b4a29291e--bcou-role-st-uat-st` | `cpu` | container | ratio | percent | identity | 652 | 0.005% | Maps to existing slot 'cpu'; semantic=cpu, unit=percent, source=Container |
| 18 | `Container-DOCKER_CONTAINER_3bf443a64876--bcou-role-st-uat-st` | `net_tx` | container | unknown | bytes | log1p | 601 | 0.005% | Maps to existing slot 'net_tx'; semantic=network, unit=bytes, source=Container |
| 19 | `Container-DOCKER_CONTAINER_b30097144a13--bcou-trace-st-uat-s` | `net_tx` | container | unknown | bytes | log1p | 598 | 0.005% | Maps to existing slot 'net_tx'; semantic=network, unit=bytes, source=Container |
| 20 | `Container-DOCKER_CONTAINER_9e8c309d0aab--bcou-trace-st-uat-s` | `net_tx` | container | unknown | bytes | log1p | 598 | 0.005% | Maps to existing slot 'net_tx'; semantic=network, unit=bytes, source=Container |
| 21 | `Container-DOCKER_CONTAINER_b30097144a13--bcou-trace-st-uat-s` | `mem` | container | ratio | percent | identity | 592 | 0.005% | Maps to existing slot 'mem'; semantic=memory, unit=percent, source=Container |
| 22 | `Container-DOCKER_CONTAINER_b30097144a13--bcou-trace-st-uat-s` | `cpu` | container | ratio | percent | identity | 591 | 0.005% | Maps to existing slot 'cpu'; semantic=cpu, unit=percent, source=Container |
| 23 | `Container-DOCKER_CONTAINER_3bf443a64876--bcou-role-st-uat-st` | `mem` | container | ratio | percent | identity | 588 | 0.005% | Maps to existing slot 'mem'; semantic=memory, unit=percent, source=Container |
| 24 | `Container-DOCKER_CONTAINER_2c2336e2994f--bcou-role-st-uat-st` | `net_tx` | container | unknown | bytes | log1p | 563 | 0.005% | Maps to existing slot 'net_tx'; semantic=network, unit=bytes, source=Container |
| 25 | `Container-DOCKER_CONTAINER_2c2336e2994f--bcou-role-st-uat-st` | `net_rx` | container | unknown | bytes | log1p | 559 | 0.005% | Maps to existing slot 'net_rx'; semantic=network, unit=bytes, source=Container |
| 26 | `Container-DOCKER_CONTAINER_2c2336e2994f--bcou-role-st-uat-st` | `mem` | container | ratio | percent | identity | 552 | 0.005% | Maps to existing slot 'mem'; semantic=memory, unit=percent, source=Container |
| 27 | `Container-DOCKER_CONTAINER_2c2336e2994f--bcou-role-st-uat-st` | `cpu` | container | ratio | percent | identity | 541 | 0.004% | Maps to existing slot 'cpu'; semantic=cpu, unit=percent, source=Container |
| 28 | `Container-DOCKER_CONTAINER_7b4b80f345e0--bcou-role-st-uat-st` | `net_tx` | container | unknown | bytes | log1p | 450 | 0.004% | Maps to existing slot 'net_tx'; semantic=network, unit=bytes, source=Container |
| 29 | `Container-DOCKER_CONTAINER_9e8c309d0aab--bcou-trace-st-uat-s` | `mem` | container | ratio | percent | identity | 423 | 0.003% | Maps to existing slot 'mem'; semantic=memory, unit=percent, source=Container |
| 30 | `Container-DOCKER_CONTAINER_7b4b80f345e0--bcou-role-st-uat-st` | `mem` | container | ratio | percent | identity | 412 | 0.003% | Maps to existing slot 'mem'; semantic=memory, unit=percent, source=Container |
| 31 | `Container-DOCKER_CONTAINER_7b4b80f345e0--bcou-role-st-uat-st` | `cpu` | container | ratio | percent | identity | 411 | 0.003% | Maps to existing slot 'cpu'; semantic=cpu, unit=percent, source=Container |
| 32 | `Container-DOCKER_CONTAINER_9e8c309d0aab--bcou-trace-st-uat-s` | `cpu` | container | ratio | percent | identity | 399 | 0.003% | Maps to existing slot 'cpu'; semantic=cpu, unit=percent, source=Container |
| 33 | `Container-DOCKER_CONTAINER_69b53a78b2eb--bcou-role-st-uat-st` | `net_tx` | container | unknown | bytes | log1p | 352 | 0.003% | Maps to existing slot 'net_tx'; semantic=network, unit=bytes, source=Container |
| 34 | `Container-DOCKER_CONTAINER_89f0c1e5346c--bcou-trace-st-uat-s` | `cpu` | container | ratio | percent | identity | 345 | 0.003% | Maps to existing slot 'cpu'; semantic=cpu, unit=percent, source=Container |
| 35 | `Container-DOCKER_CONTAINER_cb2bbb5e3f90--bcou-role-st-uat-st` | `net_tx` | container | unknown | bytes | log1p | 343 | 0.003% | Maps to existing slot 'net_tx'; semantic=network, unit=bytes, source=Container |
| 36 | `Container-DOCKER_CONTAINER_cb2bbb5e3f90--bcou-role-st-uat-st` | `mem` | container | ratio | percent | identity | 342 | 0.003% | Maps to existing slot 'mem'; semantic=memory, unit=percent, source=Container |
| 37 | `Container-DOCKER_CONTAINER_cb2bbb5e3f90--bcou-role-st-uat-st` | `cpu` | container | ratio | percent | identity | 342 | 0.003% | Maps to existing slot 'cpu'; semantic=cpu, unit=percent, source=Container |
| 38 | `Container-DOCKER_CONTAINER_76d31070b844--bcou-role-st-uat-st` | `net_tx` | container | unknown | bytes | log1p | 318 | 0.003% | Maps to existing slot 'net_tx'; semantic=network, unit=bytes, source=Container |
| 39 | `Container-DOCKER_CONTAINER_b30097144a13--bcou-trace-st-uat-s` | `net_rx` | container | unknown | bytes | log1p | 316 | 0.003% | Maps to existing slot 'net_rx'; semantic=network, unit=bytes, source=Container |
| 40 | `Container-DOCKER_CONTAINER_89f0c1e5346c--bcou-trace-st-uat-s` | `mem` | container | ratio | percent | identity | 303 | 0.002% | Maps to existing slot 'mem'; semantic=memory, unit=percent, source=Container |
| 41 | `Container-DOCKER_CONTAINER_89f0c1e5346c--bcou-trace-st-uat-s` | `net_tx` | container | unknown | bytes | log1p | 297 | 0.002% | Maps to existing slot 'net_tx'; semantic=network, unit=bytes, source=Container |
| 42 | `Container-DOCKER_CONTAINER_69b53a78b2eb--bcou-role-st-uat-st` | `mem` | container | ratio | percent | identity | 297 | 0.002% | Maps to existing slot 'mem'; semantic=memory, unit=percent, source=Container |
| 43 | `Container-DOCKER_CONTAINER_69b53a78b2eb--bcou-role-st-uat-st` | `cpu` | container | ratio | percent | identity | 296 | 0.002% | Maps to existing slot 'cpu'; semantic=cpu, unit=percent, source=Container |
| 44 | `Container-DOCKER_CONTAINER_464dc801314b--bcou-trace-st-uat-s` | `net_tx` | container | unknown | bytes | log1p | 284 | 0.002% | Maps to existing slot 'net_tx'; semantic=network, unit=bytes, source=Container |
| 45 | `Container-DOCKER_CONTAINER_76d31070b844--bcou-role-st-uat-st` | `mem` | container | ratio | percent | identity | 278 | 0.002% | Maps to existing slot 'mem'; semantic=memory, unit=percent, source=Container |
| 46 | `Container-DOCKER_CONTAINER_464dc801314b--bcou-trace-st-uat-s` | `mem` | container | ratio | percent | identity | 253 | 0.002% | Maps to existing slot 'mem'; semantic=memory, unit=percent, source=Container |
| 47 | `Container-DOCKER_CONTAINER_464dc801314b--bcou-trace-st-uat-s` | `cpu` | container | ratio | percent | identity | 248 | 0.002% | Maps to existing slot 'cpu'; semantic=cpu, unit=percent, source=Container |
| 48 | `Container-DOCKER_CONTAINER_76d31070b844--bcou-role-st-uat-st` | `cpu` | container | ratio | percent | identity | 246 | 0.002% | Maps to existing slot 'cpu'; semantic=cpu, unit=percent, source=Container |
| 49 | `Container-DOCKER_CONTAINER_2d16f5b2e830--bcou-trace-st-uat-s` | `net_tx` | container | unknown | bytes | log1p | 234 | 0.002% | Maps to existing slot 'net_tx'; semantic=network, unit=bytes, source=Container |
| 50 | `Container-DOCKER_CONTAINER_2d16f5b2e830--bcou-trace-st-uat-s` | `cpu` | container | ratio | percent | identity | 233 | 0.002% | Maps to existing slot 'cpu'; semantic=cpu, unit=percent, source=Container |
| 51 | `Container-DOCKER_CONTAINER_2d16f5b2e830--bcou-trace-st-uat-s` | `mem` | container | ratio | percent | identity | 233 | 0.002% | Maps to existing slot 'mem'; semantic=memory, unit=percent, source=Container |
| 52 | `Container-DOCKER_CONTAINER_b6760337dc49--bcou-role-st-uat-st` | `cpu` | container | ratio | percent | identity | 214 | 0.002% | Maps to existing slot 'cpu'; semantic=cpu, unit=percent, source=Container |
| 53 | `Container-DOCKER_CONTAINER_bbc780df1fce--bcou-role-st-uat-st` | `mem` | container | ratio | percent | identity | 206 | 0.002% | Maps to existing slot 'mem'; semantic=memory, unit=percent, source=Container |
| 54 | `Container-DOCKER_CONTAINER_bbc780df1fce--bcou-role-st-uat-st` | `cpu` | container | ratio | percent | identity | 206 | 0.002% | Maps to existing slot 'cpu'; semantic=cpu, unit=percent, source=Container |
| 55 | `Container-DOCKER_CONTAINER_bbc780df1fce--bcou-role-st-uat-st` | `net_tx` | container | unknown | bytes | log1p | 205 | 0.002% | Maps to existing slot 'net_tx'; semantic=network, unit=bytes, source=Container |
| 56 | `Container-DOCKER_CONTAINER_1bc4fc80d241--bcou-trace-st-uat-s` | `net_tx` | container | unknown | bytes | log1p | 204 | 0.002% | Maps to existing slot 'net_tx'; semantic=network, unit=bytes, source=Container |
| 57 | `Container-DOCKER_CONTAINER_1bc4fc80d241--bcou-trace-st-uat-s` | `mem` | container | ratio | percent | identity | 204 | 0.002% | Maps to existing slot 'mem'; semantic=memory, unit=percent, source=Container |
| 58 | `Container-DOCKER_CONTAINER_1bc4fc80d241--bcou-trace-st-uat-s` | `cpu` | container | ratio | percent | identity | 204 | 0.002% | Maps to existing slot 'cpu'; semantic=cpu, unit=percent, source=Container |
| 59 | `Container-DOCKER_CONTAINER_464dc801314b--bcou-trace-st-uat-s` | `net_rx` | container | unknown | bytes | log1p | 181 | 0.001% | Maps to existing slot 'net_rx'; semantic=network, unit=bytes, source=Container |
| 60 | `Container-DOCKER_CONTAINER_b6760337dc49--bcou-role-st-uat-st` | `net_tx` | container | unknown | bytes | log1p | 176 | 0.001% | Maps to existing slot 'net_tx'; semantic=network, unit=bytes, source=Container |
| 61 | `Container-DOCKER_CONTAINER_94eca4f96efe--bcou-role-st-uat-st` | `net_rx` | container | unknown | bytes | log1p | 171 | 0.001% | Maps to existing slot 'net_rx'; semantic=network, unit=bytes, source=Container |
| 62 | `Container-DOCKER_CONTAINER_9e8c309d0aab--bcou-trace-st-uat-s` | `net_rx` | container | unknown | bytes | log1p | 171 | 0.001% | Maps to existing slot 'net_rx'; semantic=network, unit=bytes, source=Container |
| 63 | `Container-DOCKER_CONTAINER_b6760337dc49--bcou-role-st-uat-st` | `mem` | container | ratio | percent | identity | 165 | 0.001% | Maps to existing slot 'mem'; semantic=memory, unit=percent, source=Container |
| 64 | `Container-DOCKER_CONTAINER_d27123361435--bcou-trace-st-uat-s` | `cpu` | container | ratio | percent | identity | 156 | 0.001% | Maps to existing slot 'cpu'; semantic=cpu, unit=percent, source=Container |
| 65 | `Container-DOCKER_CONTAINER_d27123361435--bcou-trace-st-uat-s` | `mem` | container | ratio | percent | identity | 156 | 0.001% | Maps to existing slot 'mem'; semantic=memory, unit=percent, source=Container |
| 66 | `Container-DOCKER_CONTAINER_d27123361435--bcou-trace-st-uat-s` | `net_tx` | container | unknown | bytes | log1p | 156 | 0.001% | Maps to existing slot 'net_tx'; semantic=network, unit=bytes, source=Container |
| 67 | `Container-DOCKER_CONTAINER_3bf443a64876--bcou-role-st-uat-st` | `net_rx` | container | unknown | bytes | log1p | 154 | 0.001% | Maps to existing slot 'net_rx'; semantic=network, unit=bytes, source=Container |
| 68 | `Container-DOCKER_CONTAINER_7b4b80f345e0--bcou-role-st-uat-st` | `net_rx` | container | unknown | bytes | log1p | 150 | 0.001% | Maps to existing slot 'net_rx'; semantic=network, unit=bytes, source=Container |
| 69 | `Container-DOCKER_CONTAINER_2d16f5b2e830--bcou-trace-st-uat-s` | `net_rx` | container | unknown | bytes | log1p | 145 | 0.001% | Maps to existing slot 'net_rx'; semantic=network, unit=bytes, source=Container |
| 70 | `Container-DOCKER_CONTAINER_cb2bbb5e3f90--bcou-role-st-uat-st` | `net_rx` | container | unknown | bytes | log1p | 142 | 0.001% | Maps to existing slot 'net_rx'; semantic=network, unit=bytes, source=Container |
| 71 | `Container-DOCKER_CONTAINER_350771a68ac2--bcou-trace-st-uat-s` | `mem` | container | ratio | percent | identity | 140 | 0.001% | Maps to existing slot 'mem'; semantic=memory, unit=percent, source=Container |
| 72 | `Container-DOCKER_CONTAINER_350771a68ac2--bcou-trace-st-uat-s` | `cpu` | container | ratio | percent | identity | 140 | 0.001% | Maps to existing slot 'cpu'; semantic=cpu, unit=percent, source=Container |
| 73 | `Container-DOCKER_CONTAINER_350771a68ac2--bcou-trace-st-uat-s` | `net_tx` | container | unknown | bytes | log1p | 139 | 0.001% | Maps to existing slot 'net_tx'; semantic=network, unit=bytes, source=Container |
| 74 | `Container-DOCKER_CONTAINER_c67422614c81--bcou-role-st-uat-st` | `mem` | container | ratio | percent | identity | 138 | 0.001% | Maps to existing slot 'mem'; semantic=memory, unit=percent, source=Container |
| 75 | `Container-DOCKER_CONTAINER_c67422614c81--bcou-role-st-uat-st` | `cpu` | container | ratio | percent | identity | 138 | 0.001% | Maps to existing slot 'cpu'; semantic=cpu, unit=percent, source=Container |
| 76 | `Container-DOCKER_CONTAINER_c67422614c81--bcou-role-st-uat-st` | `net_tx` | container | unknown | bytes | log1p | 137 | 0.001% | Maps to existing slot 'net_tx'; semantic=network, unit=bytes, source=Container |
| 77 | `Container-DOCKER_CONTAINER_2fa7eaebac26--bcou-role-st-uat-st` | `mem` | container | ratio | percent | identity | 120 | 0.001% | Maps to existing slot 'mem'; semantic=memory, unit=percent, source=Container |
| 78 | `Container-DOCKER_CONTAINER_2fa7eaebac26--bcou-role-st-uat-st` | `net_tx` | container | unknown | bytes | log1p | 120 | 0.001% | Maps to existing slot 'net_tx'; semantic=network, unit=bytes, source=Container |
| 79 | `Container-DOCKER_CONTAINER_2fa7eaebac26--bcou-role-st-uat-st` | `cpu` | container | ratio | percent | identity | 120 | 0.001% | Maps to existing slot 'cpu'; semantic=cpu, unit=percent, source=Container |
| 80 | `Container-DOCKER_CONTAINER_23bdcf67c7e3--bcou-role-st-uat-st` | `net_tx` | container | unknown | bytes | log1p | 113 | 0.001% | Maps to existing slot 'net_tx'; semantic=network, unit=bytes, source=Container |
| 81 | `Container-DOCKER_CONTAINER_b6760337dc49--bcou-role-st-uat-st` | `net_rx` | container | unknown | bytes | log1p | 103 | 0.001% | Maps to existing slot 'net_rx'; semantic=network, unit=bytes, source=Container |
| 82 | `Container-DOCKER_CONTAINER_89f0c1e5346c--bcou-trace-st-uat-s` | `net_rx` | container | unknown | bytes | log1p | 94 | 0.001% | Maps to existing slot 'net_rx'; semantic=network, unit=bytes, source=Container |
| 83 | `Container-DOCKER_CONTAINER_d27123361435--bcou-trace-st-uat-s` | `net_rx` | container | unknown | bytes | log1p | 90 | 0.001% | Maps to existing slot 'net_rx'; semantic=network, unit=bytes, source=Container |
| 84 | `Container-DOCKER_CONTAINER_bbc780df1fce--bcou-role-st-uat-st` | `net_rx` | container | unknown | bytes | log1p | 89 | 0.001% | Maps to existing slot 'net_rx'; semantic=network, unit=bytes, source=Container |
| 85 | `Container-DOCKER_CONTAINER_1bc4fc80d241--bcou-trace-st-uat-s` | `net_rx` | container | unknown | bytes | log1p | 87 | 0.001% | Maps to existing slot 'net_rx'; semantic=network, unit=bytes, source=Container |
| 86 | `Container-DOCKER_CONTAINER_23bdcf67c7e3--bcou-role-st-uat-st` | `mem` | container | ratio | percent | identity | 75 | 0.001% | Maps to existing slot 'mem'; semantic=memory, unit=percent, source=Container |
| 87 | `Container-DOCKER_CONTAINER_ef6cb138bb10--bcou-role-st-uat-st` | `mem` | container | ratio | percent | identity | 74 | 0.001% | Maps to existing slot 'mem'; semantic=memory, unit=percent, source=Container |
| 88 | `Container-DOCKER_CONTAINER_23bdcf67c7e3--bcou-role-st-uat-st` | `cpu` | container | ratio | percent | identity | 74 | 0.001% | Maps to existing slot 'cpu'; semantic=cpu, unit=percent, source=Container |
| 89 | `Container-DOCKER_CONTAINER_ef6cb138bb10--bcou-role-st-uat-st` | `cpu` | container | ratio | percent | identity | 73 | 0.001% | Maps to existing slot 'cpu'; semantic=cpu, unit=percent, source=Container |
| 90 | `Container-DOCKER_CONTAINER_ef6cb138bb10--bcou-role-st-uat-st` | `net_tx` | container | unknown | bytes | log1p | 73 | 0.001% | Maps to existing slot 'net_tx'; semantic=network, unit=bytes, source=Container |
| 91 | `Container-DOCKER_CONTAINER_23bdcf67c7e3--bcou-role-st-uat-st` | `net_rx` | container | unknown | bytes | log1p | 69 | 0.001% | Maps to existing slot 'net_rx'; semantic=network, unit=bytes, source=Container |
| 92 | `Container-DOCKER_CONTAINER_0f6f3aa7920c--bcou-role-st-uat-st` | `cpu` | container | ratio | percent | identity | 60 | 0.000% | Maps to existing slot 'cpu'; semantic=cpu, unit=percent, source=Container |
| 93 | `Container-DOCKER_CONTAINER_0f6f3aa7920c--bcou-role-st-uat-st` | `mem` | container | ratio | percent | identity | 60 | 0.000% | Maps to existing slot 'mem'; semantic=memory, unit=percent, source=Container |
| 94 | `Container-DOCKER_CONTAINER_0f6f3aa7920c--bcou-role-st-uat-st` | `net_tx` | container | unknown | bytes | log1p | 60 | 0.000% | Maps to existing slot 'net_tx'; semantic=network, unit=bytes, source=Container |
| 95 | `Container-DOCKER_CONTAINER_6d83a96887c3--bcou-role-st-uat-st` | `cpu` | container | ratio | percent | identity | 58 | 0.000% | Maps to existing slot 'cpu'; semantic=cpu, unit=percent, source=Container |
| 96 | `Container-DOCKER_CONTAINER_6d83a96887c3--bcou-role-st-uat-st` | `mem` | container | ratio | percent | identity | 58 | 0.000% | Maps to existing slot 'mem'; semantic=memory, unit=percent, source=Container |
| 97 | `Container-DOCKER_CONTAINER_6d83a96887c3--bcou-role-st-uat-st` | `net_tx` | container | unknown | bytes | log1p | 58 | 0.000% | Maps to existing slot 'net_tx'; semantic=network, unit=bytes, source=Container |
| 98 | `Container-DOCKER_CONTAINER_b2fc383d2438--bcou-trace-st-uat-s` | `mem` | container | ratio | percent | identity | 56 | 0.000% | Maps to existing slot 'mem'; semantic=memory, unit=percent, source=Container |
| 99 | `Container-DOCKER_CONTAINER_b2fc383d2438--bcou-trace-st-uat-s` | `net_tx` | container | unknown | bytes | log1p | 55 | 0.000% | Maps to existing slot 'net_tx'; semantic=network, unit=bytes, source=Container |
| 100 | `Container-DOCKER_CONTAINER_b2fc383d2438--bcou-trace-st-uat-s` | `cpu` | container | ratio | percent | identity | 55 | 0.000% | Maps to existing slot 'cpu'; semantic=cpu, unit=percent, source=Container |
| 101 | `Container-DOCKER_CONTAINER_01eeda2c9f0b--bcou-trace-st-uat-s` | `cpu` | container | ratio | percent | identity | 50 | 0.000% | Maps to existing slot 'cpu'; semantic=cpu, unit=percent, source=Container |
| 102 | `Container-DOCKER_CONTAINER_01eeda2c9f0b--bcou-trace-st-uat-s` | `mem` | container | ratio | percent | identity | 50 | 0.000% | Maps to existing slot 'mem'; semantic=memory, unit=percent, source=Container |
| 103 | `Container-DOCKER_CONTAINER_01eeda2c9f0b--bcou-trace-st-uat-s` | `net_tx` | container | unknown | bytes | log1p | 50 | 0.000% | Maps to existing slot 'net_tx'; semantic=network, unit=bytes, source=Container |
| 104 | `Container-DOCKER_CONTAINER_350771a68ac2--bcou-trace-st-uat-s` | `net_rx` | container | unknown | bytes | log1p | 29 | 0.000% | Maps to existing slot 'net_rx'; semantic=network, unit=bytes, source=Container |
| 105 | `Container-DOCKER_CONTAINER_6d83a96887c3--bcou-role-st-uat-st` | `net_rx` | container | unknown | bytes | log1p | 26 | 0.000% | Maps to existing slot 'net_rx'; semantic=network, unit=bytes, source=Container |
| 106 | `Container-DOCKER_CONTAINER_ef6cb138bb10--bcou-role-st-uat-st` | `net_rx` | container | unknown | bytes | log1p | 12 | 0.000% | Maps to existing slot 'net_rx'; semantic=network, unit=bytes, source=Container |
| 107 | `Container-DOCKER_CONTAINER_c67422614c81--bcou-role-st-uat-st` | `net_rx` | container | unknown | bytes | log1p | 10 | 0.000% | Maps to existing slot 'net_rx'; semantic=network, unit=bytes, source=Container |
| 108 | `Container-DOCKER_CONTAINER_cd2b4a29291e--bcou-role-st-uat-st` | `net_rx` | container | unknown | bytes | log1p | 10 | 0.000% | Maps to existing slot 'net_rx'; semantic=network, unit=bytes, source=Container |
| 109 | `Container-DOCKER_CONTAINER_0f6f3aa7920c--bcou-role-st-uat-st` | `net_rx` | container | unknown | bytes | log1p | 5 | 0.000% | Maps to existing slot 'net_rx'; semantic=network, unit=bytes, source=Container |
| 110 | `Container-DOCKER_CONTAINER_ac3ba8476104--bcou-trace-st-uat-s` | `net_tx` | container | unknown | bytes | log1p | 4 | 0.000% | Maps to existing slot 'net_tx'; semantic=network, unit=bytes, source=Container |
| 111 | `Container-DOCKER_CONTAINER_ac3ba8476104--bcou-trace-st-uat-s` | `mem` | container | ratio | percent | identity | 4 | 0.000% | Maps to existing slot 'mem'; semantic=memory, unit=percent, source=Container |
| 112 | `Container-DOCKER_CONTAINER_ac3ba8476104--bcou-trace-st-uat-s` | `cpu` | container | ratio | percent | identity | 4 | 0.000% | Maps to existing slot 'cpu'; semantic=cpu, unit=percent, source=Container |
| 113 | `Container-DOCKER_CONTAINER_2fa7eaebac26--bcou-role-st-uat-st` | `net_rx` | container | unknown | bytes | log1p | 1 | 0.000% | Maps to existing slot 'net_rx'; semantic=network, unit=bytes, source=Container |
| 114 | `Container-DOCKER_CONTAINER_f861998c398e--bcou-role-st-uat-st` | `net_tx` | container | unknown | bytes | log1p | 1 | 0.000% | Maps to existing slot 'net_tx'; semantic=network, unit=bytes, source=Container |
| 115 | `Container-DOCKER_CONTAINER_f861998c398e--bcou-role-st-uat-st` | `net_rx` | container | unknown | bytes | log1p | 1 | 0.000% | Maps to existing slot 'net_rx'; semantic=network, unit=bytes, source=Container |
| 116 | `Container-DOCKER_CONTAINER_f861998c398e--bcou-role-st-uat-st` | `mem` | container | ratio | percent | identity | 1 | 0.000% | Maps to existing slot 'mem'; semantic=memory, unit=percent, source=Container |
| 117 | `Container-DOCKER_CONTAINER_f861998c398e--bcou-role-st-uat-st` | `cpu` | container | ratio | percent | identity | 1 | 0.000% | Maps to existing slot 'cpu'; semantic=cpu, unit=percent, source=Container |
| 118 | `Container-DOCKER_CONTAINER_76d31070b844--bcou-role-st-uat-st` | `net_rx` | container | unknown | bytes | log1p | 1 | 0.000% | Maps to existing slot 'net_rx'; semantic=network, unit=bytes, source=Container |

### Coverage Impact

- **Current accepted coverage**: 1.34% (163,849 rows)
- **After high-confidence v2 candidates**: 9.28% (1,134,233 rows)
- **Coverage gain**: +7.94% (+970,384 rows)

## 4. New-Slot Candidates

### By proposed new slot

**`cache_metrics`** (20 KPIs, 449,396 rows, 3.68% coverage)
  - `redis-Redis_6379_Redis  (redis_git_dirty)` (23,658 rows, Redis, cache, unknown, unknown)
  - `redis-Redis_6379_Redis  (instantaneous_ops_per_sec)` (23,657 rows, Redis, cache, rate, seconds)
  - `redis-Redis_6379_Redis  (keyspace_hits)` (23,657 rows, Redis, cache, counter, count)
  - `redis-Redis_6379_Redis  (latest_fork_usec)` (23,657 rows, Redis, cache, unknown, seconds)
  - `redis-Redis_6379_Redis  (aof_enabled)` (23,657 rows, Redis, cache, unknown, unknown)
  - ... and 15 more KPIs

**`db_latency`** (4 KPIs, 86,877 rows, 0.71% coverage)
  - `Mysql-MySQL_3306_GetResponseTimeOfMysqld` (22,110 rows, MySQL, latency, duration, milliseconds)
  - `Mysql-MySQL_3306_Innodb Row Lock Time` (21,590 rows, MySQL, latency, duration, milliseconds)
  - `Mysql-MySQL_3306_Innodb row lock time avg` (21,589 rows, MySQL, latency, duration, milliseconds)
  - `Mysql-MySQL_3306_Innodb row lock time max` (21,588 rows, MySQL, latency, duration, milliseconds)

**`disk_io`** (30 KPIs, 1,434,201 rows, 11.74% coverage)
  - `OSLinux-OSLinux_LOCALDISK_LOCALDISK-sdb_DSKWTps` (78,689 rows, OSLinux, disk, unknown, rate)
  - `OSLinux-OSLinux_LOCALDISK_LOCALDISK-sda_DSKWrite` (78,679 rows, OSLinux, disk, unknown, unknown)
  - `OSLinux-OSLinux_LOCALDISK_LOCALDISK-sda_DSKWTps` (78,679 rows, OSLinux, disk, unknown, rate)
  - `OSLinux-OSLinux_LOCALDISK_LOCALDISK-sdb_DSKRTps` (78,661 rows, OSLinux, disk, unknown, rate)
  - `OSLinux-OSLinux_LOCALDISK_LOCALDISK-sda_DSKRead` (78,661 rows, OSLinux, disk, unknown, unknown)
  - ... and 25 more KPIs

**`filesystem_usage`** (60 KPIs, 1,843,131 rows, 15.09% coverage)
  - `OSLinux-OSLinux_FILESYSTEM_-tmp_FSCapacity` (72,439 rows, OSLinux, filesystem, gauge, unknown)
  - `OSLinux-OSLinux_FILESYSTEM_-tmp_FSInodeUsedPercent` (72,375 rows, OSLinux, filesystem, ratio, percent)
  - `OSLinux-OSLinux_FILESYSTEM_-_FSCapacity` (72,360 rows, OSLinux, filesystem, gauge, unknown)
  - `OSLinux-OSLinux_FILESYSTEM_-home_FSCapacity` (72,306 rows, OSLinux, filesystem, gauge, unknown)
  - `OSLinux-OSLinux_FILESYSTEM_-boot_FSCapacity` (72,302 rows, OSLinux, filesystem, gauge, unknown)
  - ... and 55 more KPIs

**`jvm_cpu`** (2 KPIs, 36,468 rows, 0.30% coverage)
  - `JVM-Operating System_7778_JVM_JVM_CPULoad` (18,237 rows, JVM, cpu, gauge, unknown)
  - `JVM-Operating System_7779_JVM_JVM_CPULoad` (18,231 rows, JVM, cpu, gauge, unknown)

**`jvm_mem`** (12 KPIs, 251,537 rows, 2.06% coverage)
  - `JVM-Memory_7779_JVM_Memory_NoHeapMemoryUsed` (28,755 rows, JVM, memory, gauge, unknown)
  - `JVM-Memory_7779_JVM_Memory_HeapMemoryUsed` (28,754 rows, JVM, memory, gauge, unknown)
  - `JVM-Memory_7779_JVM_Memory_HeapMemoryMax` (28,754 rows, JVM, memory, gauge, unknown)
  - `JVM-Memory_7778_JVM_Memory_NoHeapMemoryUsed` (28,708 rows, JVM, memory, gauge, unknown)
  - `JVM-Memory_7778_JVM_Memory_HeapMemoryMax` (28,708 rows, JVM, memory, gauge, unknown)
  - ... and 7 more KPIs

**`mysql_io`** (99 KPIs, 2,130,178 rows, 17.43% coverage)
  - `Mysql-MySQL_3306_Com Update` (21,598 rows, MySQL, database_io, counter, count)
  - `Mysql-MySQL_3306_Com Replace Select` (21,598 rows, MySQL, database_io, unknown, unknown)
  - `Mysql-MySQL_3306_Bytes Sent` (21,598 rows, MySQL, database_io, rate, bytes)
  - `Mysql-MySQL_3306_Handler Delete` (21,597 rows, MySQL, database_io, counter, count)
  - `Mysql-MySQL_3306_Com Update Multi` (21,597 rows, MySQL, database_io, counter, count)
  - ... and 94 more KPIs

**`os_network`** (11 KPIs, 780,383 rows, 6.39% coverage)
  - `OSLinux-OSLinux_NETWORK_NETWORK_TCP-CLOSE-WAIT` (160,049 rows, OSLinux, network, unknown, unknown)
  - `OSLinux-OSLinux_NETWORK_NETWORK_TCP-FIN-WAIT` (159,678 rows, OSLinux, network, unknown, unknown)
  - `OSLinux-OSLinux_NETWORK_NETWORK_TotalTcpConnNum` (107,316 rows, OSLinux, network, gauge, count)
  - `OSLinux-OSLinux_NETWORK_ens160_NETPacketsIn` (49,355 rows, OSLinux, network, unknown, unknown)
  - `OSLinux-OSLinux_NETWORK_ens160_NETPacketsOut` (49,354 rows, OSLinux, network, unknown, unknown)
  - ... and 6 more KPIs

**`process_health`** (4 KPIs, 380,590 rows, 3.11% coverage)
  - `OSLinux-OSLinux_PROCESS_zabbix_PROCPPCount` (160,489 rows, OSLinux, process, counter, count)
  - `OSLinux-OSLinux_PROCESS_PROCESS_PROCPPMem` (119,082 rows, OSLinux, process, unknown, unknown)
  - `OSLinux-OSLinux_PROCESS_PROCESS_PROCPPMemPerc` (79,241 rows, OSLinux, process, ratio, percent)
  - `OSLinux-OSLinux_PROCESS_apache_10001_PROCPPCount` (21,778 rows, OSLinux, process, counter, count)

**`sessions`** (12 KPIs, 639,242 rows, 5.23% coverage)
  - `Tomcat-Sessions_7441--_SESSIONRejectedSessions` (56,867 rows, Tomcat, session, counter, count)
  - `Tomcat-Sessions_7441--UOCP_SESSIONRejectedSessions` (56,866 rows, Tomcat, session, counter, count)
  - `Tomcat-Sessions_7441--logHome_IS_UNDEFINED_SESSIONKeepaliveCounter` (56,866 rows, Tomcat, session, counter, count)
  - `Tomcat-Sessions_7441--logHome_IS_UNDEFINED_SESSIONActiveCounter` (56,865 rows, Tomcat, session, counter, count)
  - `Tomcat-Sessions_7441--UOCP_SESSIONKeepaliveCounter` (56,865 rows, Tomcat, session, counter, count)
  - ... and 7 more KPIs

**`swap_io`** (2 KPIs, 81,241 rows, 0.66% coverage)
  - `OSLinux-OSLinux_SWAP_SWAP_SWPTotSwapUsedPercent` (78,563 rows, OSLinux, swap, ratio, percent)
  - `OSLinux-OSLinux_SWAP_SWAP_SWPTotSwapSize` (2,678 rows, OSLinux, swap, gauge, unknown)

**`threads`** (7 KPIs, 165,430 rows, 1.35% coverage)
  - `JVM-Threads_7779_JVM_ThreadCount_Threads` (28,752 rows, JVM, thread, counter, count)
  - `JVM-Threads_7778_JVM_ThreadCount_Threads` (28,707 rows, JVM, thread, counter, count)
  - `Mysql-MySQL_3306_Threads Cached` (21,596 rows, MySQL, thread, counter, count)
  - `Mysql-MySQL_3306_ThreadsRunning` (21,595 rows, MySQL, thread, counter, count)
  - `Mysql-MySQL_3306_ThreadsConnected` (21,594 rows, MySQL, thread, counter, count)
  - ... and 2 more KPIs

**`tomcat_requests`** (7 KPIs, 398,337 rows, 3.26% coverage)
  - `Tomcat-Threads_7441-"http-nio-8003"_CurrentThreadCountThreadInfo` (56,907 rows, Tomcat, request, counter, count)
  - `Tomcat-Requests_7441-"http-nio-8003"_MaxTimeRequestInfo` (56,906 rows, Tomcat, request, duration, milliseconds)
  - `Tomcat-Requests_7441-"http-nio-8003"_RequestCountRequestInfo` (56,906 rows, Tomcat, request, counter, count)
  - `Tomcat-Requests_7441-"http-nio-8003"_ErrorCountRequestInfo` (56,905 rows, Tomcat, request, counter, count)
  - `Tomcat-Threads_7441-"http-nio-8003"_MaxThreadsThreadInfo` (56,905 rows, Tomcat, request, counter, count)
  - ... and 2 more KPIs

## 5. Manual-Review List

109 KPIs require manual review:

| # | Raw KPI Name | Rows | Source | Semantic | Kind | Unit |
|---|---|---|---|---|---|---|
| 1 | `OSLinux-OSLinux_PROCESS_PROCESS_PROCNoZombies` | 160,491 | OSLinux | unknown | unknown | unknown |
| 2 | `OSLinux-OSLinux_ZABBIX_Host_Uptime` | 160,460 | OSLinux | uptime | duration | milliseconds |
| 3 | `OSLinux-CPU_CPU_CPULoad` | 158,529 | OSLinux | cpu | gauge | unknown |
| 4 | `OSLinux-CPU_CPU_CPUWio` | 158,523 | OSLinux | cpu | unknown | unknown |
| 5 | `OSLinux-CPU_CPU_CPUSysTime` | 158,515 | OSLinux | cpu | duration | milliseconds |
| 6 | `OSLinux-CPU_CPU_CPUUserTime` | 158,512 | OSLinux | cpu | duration | milliseconds |
| 7 | `OSLinux-OSLinux_MEMORY_MEMORY_UserMem` | 121,488 | OSLinux | memory | gauge | unknown |
| 8 | `OSLinux-OSLinux_MEMORY_MEMORY_CacheMem` | 118,963 | OSLinux | memory | gauge | unknown |
| 9 | `OSLinux-OSLinux_MEMORY_MEMORY_MEMFreeMem` | 118,949 | OSLinux | memory | gauge | unknown |
| 10 | `OSLinux-OSLinux_FILE_-tmp-zabbix_agentd.log_FileSizeMB` | 114,436 | OSLinux | monitoring | gauge | unknown |
| 11 | `OSLinux-OSLinux_SWAP_SWAP_So` | 79,669 | OSLinux | unknown | unknown | unknown |
| 12 | `OSLinux-OSLinux_SWAP_SWAP_Si` | 79,621 | OSLinux | unknown | unknown | unknown |
| 13 | `OSLinux-CPU_CPU-0_SingleCpuidle` | 78,710 | OSLinux | cpu | unknown | unknown |
| 14 | `OSLinux-CPU_CPU-2_SingleCpuidle` | 78,689 | OSLinux | cpu | unknown | unknown |
| 15 | `OSLinux-CPU_CPU-1_SingleCpuidle` | 54,324 | OSLinux | cpu | unknown | unknown |
| 16 | `OSLinux-CPU_CPU-3_SingleCpuidle` | 54,303 | OSLinux | cpu | unknown | unknown |
| 17 | `redis-Redis_6379_Redis  (used_memory_rss)` | 23,657 | Redis | memory | gauge | unknown |
| 18 | `redis-Redis_6379_Redis  (total_connections_received)` | 23,656 | Redis | connection | counter | count |
| 19 | `redis-Redis_6379_Redis  (blocked_clients)` | 23,656 | Redis | database_io | unknown | count |
| 20 | `redis-Redis_6379_Redis  (connected_clients)` | 23,655 | Redis | connection | counter | count |
| 21 | `redis-Redis_6379_Redis  (connected_slaves)` | 23,655 | Redis | connection | counter | count |
| 22 | `redis-Redis_6379_Redis  (lru_clock)` | 23,655 | Redis | database_io | unknown | count |
| 23 | `redis-Redis_6379_Redis  (rejected_connections)` | 23,655 | Redis | connection | counter | count |
| 24 | `redis-Redis_6379_Redis  (used_memory)` | 23,655 | Redis | memory | gauge | unknown |
| 25 | `redis-Redis_6379_Redis  (used_memory_peak)` | 23,655 | Redis | memory | gauge | unknown |
| 26 | `redis-Redis_6379_Redis  (used_cpu_sys)` | 23,456 | Redis | cpu | unknown | unknown |
| 27 | `redis-Redis_6379_Redis  (used_cpu_user)` | 23,456 | Redis | cpu | unknown | unknown |
| 28 | `redis-Redis_6379_Redis  (used_cpu_user_children)` | 23,456 | Redis | cpu | unknown | unknown |
| 29 | `redis-Redis_6379_Redis  (mem_fragmentation_ratio)` | 23,455 | Redis | memory | ratio | ratio |
| 30 | `redis-Redis_6379_Redis  (used_cpu_sys_children)` | 23,454 | Redis | cpu | unknown | unknown |
| 31 | `Mysql-MySQL_3306_GetConnectedStateOfMysqld` | 22,103 | MySQL | connection | counter | count |
| 32 | `Mysql-MySQL_3306_Aborted Clients` | 21,598 | MySQL | connection | counter | count |
| 33 | `Mysql-MySQL_3306_Connections` | 21,596 | MySQL | connection | counter | count |
| 34 | `Mysql-MySQL_3306_Aborted Connects` | 21,596 | MySQL | connection | counter | count |
| 35 | `Mysql-MySQL_3306_max trx lock memory bytes` | 21,596 | MySQL | memory | gauge | bytes |
| 36 | `Mysql-MySQL_3306_Innodb buffer pool write requests` | 21,594 | MySQL | request | gauge | count |
| 37 | `Mysql-MySQL_3306_Innodb buffer pool read requests` | 21,591 | MySQL | request | gauge | count |
| 38 | `Mysql-MySQL_3306_Max Used Connections` | 21,590 | MySQL | connection | counter | count |
| 39 | `Mysql-MySQL_3306_Key Read Requests` | 21,590 | MySQL | request | unknown | count |
| 40 | `Mysql-MySQL_3306_Qcache Free Memory` | 21,590 | MySQL | memory | gauge | count |
| 41 | `Mysql-MySQL_3306_Innodb log write requests` | 21,589 | MySQL | request | unknown | count |
| 42 | `Mysql-MySQL_3306_MaxConnections` | 21,583 | MySQL | connection | counter | count |
| 43 | `JVM-Runtime_7778_JVM_JVM_Uptime` | 19,053 | JVM | uptime | duration | milliseconds |
| 44 | `JVM-Runtime_7779_JVM_JVM_Uptime` | 19,051 | JVM | uptime | duration | milliseconds |
| 45 | `Mysql-MySQL_3306_Key Write Requests` | 17,871 | MySQL | request | unknown | count |
| 46 | `OSLinux-OSLinux_SYSTEM_SYSTEM_Check-Hostname` | 16,213 | OSLinux | system | unknown | unknown |
| 47 | `OSLinux-OSLinux_SYSTEM_SYSTEM_Check-DefaultRoute` | 16,067 | OSLinux | system | unknown | unknown |
| 48 | `OSLinux-OSLinux_FILE_-home-zabbix_DirSizeMB` | 12,003 | OSLinux | monitoring | gauge | unknown |
| 49 | `OSLinux-NTP_197.30.1.68_NtpServerTimeOffset` | 6,507 | OSLinux | time_sync | duration | milliseconds |
| 50 | `OSLinux-NTP_197.30.1.67_NtpServerTimeOffset` | 6,499 | OSLinux | time_sync | duration | milliseconds |
*(... and 59 more KPIs)*

## 6. Full Action Summary

| Recommended Action | KPI Count | Total Rows | Row % |
|---|---|---|---|
| map_safe_v2 | 118 | 970,384 | 7.94% |
| requires_new_slot | 270 | 8,677,011 | 71.02% |
| manual_review | 109 | 2,570,699 | 21.04% |

## 7. Recommendations for Phase 0.9.2.3d

### Minimal `bank_safe_v2` implementation

The following high-confidence KPIs should be added to `bank_safe_v2`:

- `OSLinux-CPU_CPU_CPUidleutil` → slot `cpu` (unknown, high confidence)
- `OSLinux-OSLinux_PROCESS_zabbix-zabbix_agentd_PROCPPCPUPerc` → slot `cpu` (identity, high confidence)
- `OSLinux-OSLinux_MEMORY_MEMORY_MEMUsedMemPerc` → slot `mem` (identity, high confidence)
- `OSLinux-CPU_CPU_CPUCpuUtil` → slot `cpu` (unknown, high confidence)
- `OSLinux-OSLinux_MEMORY_MEMORY_NoCacheMemPerc` → slot `mem` (identity, high confidence)
- `OSLinux-CPU_CPU-2_SingleCpuUtil` → slot `cpu` (unknown, high confidence)
- `OSLinux-CPU_CPU-0_SingleCpuUtil` → slot `cpu` (unknown, high confidence)
- `OSLinux-CPU_CPU-3_SingleCpuUtil` → slot `cpu` (unknown, high confidence)
- `OSLinux-CPU_CPU-1_SingleCpuUtil` → slot `cpu` (unknown, high confidence)
- `Tomcat-MEMORY_7441-MEMORY_JVMMemoryUsedPercent` → slot `mem` (identity, high confidence)
- `Container-DOCKER_CONTAINER_94eca4f96efe--bcou-role-st-uat-statefulset-1--bcou--UATWKR04_CpuPercent` → slot `cpu` (identity, high confidence)
- `Container-DOCKER_CONTAINER_94eca4f96efe--bcou-role-st-uat-statefulset-1--bcou--UATWKR04_MemPercent` → slot `mem` (identity, high confidence)
- `Container-DOCKER_CONTAINER_3bf443a64876--bcou-role-st-uat-statefulset-1--bcou--UATWKR03_CpuPercent` → slot `cpu` (identity, high confidence)
- `Container-DOCKER_CONTAINER_94eca4f96efe--bcou-role-st-uat-statefulset-1--bcou--UATWKR04_NetworkTxBytes` → slot `net_tx` (log1p, high confidence)
- `Container-DOCKER_CONTAINER_cd2b4a29291e--bcou-role-st-uat-statefulset-1--bcou--UATWKR04_NetworkTxBytes` → slot `net_tx` (log1p, high confidence)
- `Container-DOCKER_CONTAINER_cd2b4a29291e--bcou-role-st-uat-statefulset-1--bcou--UATWKR04_MemPercent` → slot `mem` (identity, high confidence)
- `Container-DOCKER_CONTAINER_cd2b4a29291e--bcou-role-st-uat-statefulset-1--bcou--UATWKR04_CpuPercent` → slot `cpu` (identity, high confidence)
- `Container-DOCKER_CONTAINER_3bf443a64876--bcou-role-st-uat-statefulset-1--bcou--UATWKR03_NetworkTxBytes` → slot `net_tx` (log1p, high confidence)
- `Container-DOCKER_CONTAINER_b30097144a13--bcou-trace-st-uat-statefulset-0--bcou--UATWKR04_NetworkTxBytes` → slot `net_tx` (log1p, high confidence)
- `Container-DOCKER_CONTAINER_9e8c309d0aab--bcou-trace-st-uat-statefulset-0--bcou--UATWKR04_NetworkTxBytes` → slot `net_tx` (log1p, high confidence)
- `Container-DOCKER_CONTAINER_b30097144a13--bcou-trace-st-uat-statefulset-0--bcou--UATWKR04_MemPercent` → slot `mem` (identity, high confidence)
- `Container-DOCKER_CONTAINER_b30097144a13--bcou-trace-st-uat-statefulset-0--bcou--UATWKR04_CpuPercent` → slot `cpu` (identity, high confidence)
- `Container-DOCKER_CONTAINER_3bf443a64876--bcou-role-st-uat-statefulset-1--bcou--UATWKR03_MemPercent` → slot `mem` (identity, high confidence)
- `Container-DOCKER_CONTAINER_2c2336e2994f--bcou-role-st-uat-statefulset-1--bcou--UATWKR06_NetworkTxBytes` → slot `net_tx` (log1p, high confidence)
- `Container-DOCKER_CONTAINER_2c2336e2994f--bcou-role-st-uat-statefulset-1--bcou--UATWKR06_NetworkRxBytes` → slot `net_rx` (log1p, high confidence)
- `Container-DOCKER_CONTAINER_2c2336e2994f--bcou-role-st-uat-statefulset-1--bcou--UATWKR06_MemPercent` → slot `mem` (identity, high confidence)
- `Container-DOCKER_CONTAINER_2c2336e2994f--bcou-role-st-uat-statefulset-1--bcou--UATWKR06_CpuPercent` → slot `cpu` (identity, high confidence)
- `Container-DOCKER_CONTAINER_7b4b80f345e0--bcou-role-st-uat-statefulset-0--bcou--UATWKR04_NetworkTxBytes` → slot `net_tx` (log1p, high confidence)
- `Container-DOCKER_CONTAINER_9e8c309d0aab--bcou-trace-st-uat-statefulset-0--bcou--UATWKR04_MemPercent` → slot `mem` (identity, high confidence)
- `Container-DOCKER_CONTAINER_7b4b80f345e0--bcou-role-st-uat-statefulset-0--bcou--UATWKR04_MemPercent` → slot `mem` (identity, high confidence)
- `Container-DOCKER_CONTAINER_7b4b80f345e0--bcou-role-st-uat-statefulset-0--bcou--UATWKR04_CpuPercent` → slot `cpu` (identity, high confidence)
- `Container-DOCKER_CONTAINER_9e8c309d0aab--bcou-trace-st-uat-statefulset-0--bcou--UATWKR04_CpuPercent` → slot `cpu` (identity, high confidence)
- `Container-DOCKER_CONTAINER_69b53a78b2eb--bcou-role-st-uat-statefulset-1--bcou--UATWKR06_NetworkTxBytes` → slot `net_tx` (log1p, high confidence)
- `Container-DOCKER_CONTAINER_89f0c1e5346c--bcou-trace-st-uat-statefulset-0--bcou--UATWKR04_CpuPercent` → slot `cpu` (identity, high confidence)
- `Container-DOCKER_CONTAINER_cb2bbb5e3f90--bcou-role-st-uat-statefulset-1--bcou--UATWKR02_NetworkTxBytes` → slot `net_tx` (log1p, high confidence)
- `Container-DOCKER_CONTAINER_cb2bbb5e3f90--bcou-role-st-uat-statefulset-1--bcou--UATWKR02_MemPercent` → slot `mem` (identity, high confidence)
- `Container-DOCKER_CONTAINER_cb2bbb5e3f90--bcou-role-st-uat-statefulset-1--bcou--UATWKR02_CpuPercent` → slot `cpu` (identity, high confidence)
- `Container-DOCKER_CONTAINER_76d31070b844--bcou-role-st-uat-statefulset-1--bcou--UATWKR06_NetworkTxBytes` → slot `net_tx` (log1p, high confidence)
- `Container-DOCKER_CONTAINER_b30097144a13--bcou-trace-st-uat-statefulset-0--bcou--UATWKR04_NetworkRxBytes` → slot `net_rx` (log1p, high confidence)
- `Container-DOCKER_CONTAINER_89f0c1e5346c--bcou-trace-st-uat-statefulset-0--bcou--UATWKR04_MemPercent` → slot `mem` (identity, high confidence)
- `Container-DOCKER_CONTAINER_89f0c1e5346c--bcou-trace-st-uat-statefulset-0--bcou--UATWKR04_NetworkTxBytes` → slot `net_tx` (log1p, high confidence)
- `Container-DOCKER_CONTAINER_69b53a78b2eb--bcou-role-st-uat-statefulset-1--bcou--UATWKR06_MemPercent` → slot `mem` (identity, high confidence)
- `Container-DOCKER_CONTAINER_69b53a78b2eb--bcou-role-st-uat-statefulset-1--bcou--UATWKR06_CpuPercent` → slot `cpu` (identity, high confidence)
- `Container-DOCKER_CONTAINER_464dc801314b--bcou-trace-st-uat-statefulset-0--bcou--UATWKR04_NetworkTxBytes` → slot `net_tx` (log1p, high confidence)
- `Container-DOCKER_CONTAINER_76d31070b844--bcou-role-st-uat-statefulset-1--bcou--UATWKR06_MemPercent` → slot `mem` (identity, high confidence)
- `Container-DOCKER_CONTAINER_464dc801314b--bcou-trace-st-uat-statefulset-0--bcou--UATWKR04_MemPercent` → slot `mem` (identity, high confidence)
- `Container-DOCKER_CONTAINER_464dc801314b--bcou-trace-st-uat-statefulset-0--bcou--UATWKR04_CpuPercent` → slot `cpu` (identity, high confidence)
- `Container-DOCKER_CONTAINER_76d31070b844--bcou-role-st-uat-statefulset-1--bcou--UATWKR06_CpuPercent` → slot `cpu` (identity, high confidence)
- `Container-DOCKER_CONTAINER_2d16f5b2e830--bcou-trace-st-uat-statefulset-1--bcou--UATWKR18_NetworkTxBytes` → slot `net_tx` (log1p, high confidence)
- `Container-DOCKER_CONTAINER_2d16f5b2e830--bcou-trace-st-uat-statefulset-1--bcou--UATWKR18_CpuPercent` → slot `cpu` (identity, high confidence)
- `Container-DOCKER_CONTAINER_2d16f5b2e830--bcou-trace-st-uat-statefulset-1--bcou--UATWKR18_MemPercent` → slot `mem` (identity, high confidence)
- `Container-DOCKER_CONTAINER_b6760337dc49--bcou-role-st-uat-statefulset-0--bcou--UATWKR18_CpuPercent` → slot `cpu` (identity, high confidence)
- `Container-DOCKER_CONTAINER_bbc780df1fce--bcou-role-st-uat-statefulset-0--bcou--UATWKR18_MemPercent` → slot `mem` (identity, high confidence)
- `Container-DOCKER_CONTAINER_bbc780df1fce--bcou-role-st-uat-statefulset-0--bcou--UATWKR18_CpuPercent` → slot `cpu` (identity, high confidence)
- `Container-DOCKER_CONTAINER_bbc780df1fce--bcou-role-st-uat-statefulset-0--bcou--UATWKR18_NetworkTxBytes` → slot `net_tx` (log1p, high confidence)
- `Container-DOCKER_CONTAINER_1bc4fc80d241--bcou-trace-st-uat-statefulset-1--bcou--UATWKR18_NetworkTxBytes` → slot `net_tx` (log1p, high confidence)
- `Container-DOCKER_CONTAINER_1bc4fc80d241--bcou-trace-st-uat-statefulset-1--bcou--UATWKR18_MemPercent` → slot `mem` (identity, high confidence)
- `Container-DOCKER_CONTAINER_1bc4fc80d241--bcou-trace-st-uat-statefulset-1--bcou--UATWKR18_CpuPercent` → slot `cpu` (identity, high confidence)
- `Container-DOCKER_CONTAINER_464dc801314b--bcou-trace-st-uat-statefulset-0--bcou--UATWKR04_NetworkRxBytes` → slot `net_rx` (log1p, high confidence)
- `Container-DOCKER_CONTAINER_b6760337dc49--bcou-role-st-uat-statefulset-0--bcou--UATWKR18_NetworkTxBytes` → slot `net_tx` (log1p, high confidence)
- `Container-DOCKER_CONTAINER_94eca4f96efe--bcou-role-st-uat-statefulset-1--bcou--UATWKR04_NetworkRxBytes` → slot `net_rx` (log1p, high confidence)
- `Container-DOCKER_CONTAINER_9e8c309d0aab--bcou-trace-st-uat-statefulset-0--bcou--UATWKR04_NetworkRxBytes` → slot `net_rx` (log1p, high confidence)
- `Container-DOCKER_CONTAINER_b6760337dc49--bcou-role-st-uat-statefulset-0--bcou--UATWKR18_MemPercent` → slot `mem` (identity, high confidence)
- `Container-DOCKER_CONTAINER_d27123361435--bcou-trace-st-uat-statefulset-1--bcou--UATWKR18_CpuPercent` → slot `cpu` (identity, high confidence)
- `Container-DOCKER_CONTAINER_d27123361435--bcou-trace-st-uat-statefulset-1--bcou--UATWKR18_MemPercent` → slot `mem` (identity, high confidence)
- `Container-DOCKER_CONTAINER_d27123361435--bcou-trace-st-uat-statefulset-1--bcou--UATWKR18_NetworkTxBytes` → slot `net_tx` (log1p, high confidence)
- `Container-DOCKER_CONTAINER_3bf443a64876--bcou-role-st-uat-statefulset-1--bcou--UATWKR03_NetworkRxBytes` → slot `net_rx` (log1p, high confidence)
- `Container-DOCKER_CONTAINER_7b4b80f345e0--bcou-role-st-uat-statefulset-0--bcou--UATWKR04_NetworkRxBytes` → slot `net_rx` (log1p, high confidence)
- `Container-DOCKER_CONTAINER_2d16f5b2e830--bcou-trace-st-uat-statefulset-1--bcou--UATWKR18_NetworkRxBytes` → slot `net_rx` (log1p, high confidence)
- `Container-DOCKER_CONTAINER_cb2bbb5e3f90--bcou-role-st-uat-statefulset-1--bcou--UATWKR02_NetworkRxBytes` → slot `net_rx` (log1p, high confidence)
- `Container-DOCKER_CONTAINER_350771a68ac2--bcou-trace-st-uat-statefulset-1--bcou--UATWKR18_MemPercent` → slot `mem` (identity, high confidence)
- `Container-DOCKER_CONTAINER_350771a68ac2--bcou-trace-st-uat-statefulset-1--bcou--UATWKR18_CpuPercent` → slot `cpu` (identity, high confidence)
- `Container-DOCKER_CONTAINER_350771a68ac2--bcou-trace-st-uat-statefulset-1--bcou--UATWKR18_NetworkTxBytes` → slot `net_tx` (log1p, high confidence)
- `Container-DOCKER_CONTAINER_c67422614c81--bcou-role-st-uat-statefulset-1--bcou--UATWKR02_MemPercent` → slot `mem` (identity, high confidence)
- `Container-DOCKER_CONTAINER_c67422614c81--bcou-role-st-uat-statefulset-1--bcou--UATWKR02_CpuPercent` → slot `cpu` (identity, high confidence)
- `Container-DOCKER_CONTAINER_c67422614c81--bcou-role-st-uat-statefulset-1--bcou--UATWKR02_NetworkTxBytes` → slot `net_tx` (log1p, high confidence)
- `Container-DOCKER_CONTAINER_2fa7eaebac26--bcou-role-st-uat-statefulset-1--bcou--UATWKR02_MemPercent` → slot `mem` (identity, high confidence)
- `Container-DOCKER_CONTAINER_2fa7eaebac26--bcou-role-st-uat-statefulset-1--bcou--UATWKR02_NetworkTxBytes` → slot `net_tx` (log1p, high confidence)
- `Container-DOCKER_CONTAINER_2fa7eaebac26--bcou-role-st-uat-statefulset-1--bcou--UATWKR02_CpuPercent` → slot `cpu` (identity, high confidence)
- `Container-DOCKER_CONTAINER_23bdcf67c7e3--bcou-role-st-uat-statefulset-0--bcou--UATWKR18_NetworkTxBytes` → slot `net_tx` (log1p, high confidence)
- `Container-DOCKER_CONTAINER_b6760337dc49--bcou-role-st-uat-statefulset-0--bcou--UATWKR18_NetworkRxBytes` → slot `net_rx` (log1p, high confidence)
- `Container-DOCKER_CONTAINER_89f0c1e5346c--bcou-trace-st-uat-statefulset-0--bcou--UATWKR04_NetworkRxBytes` → slot `net_rx` (log1p, high confidence)
- `Container-DOCKER_CONTAINER_d27123361435--bcou-trace-st-uat-statefulset-1--bcou--UATWKR18_NetworkRxBytes` → slot `net_rx` (log1p, high confidence)
- `Container-DOCKER_CONTAINER_bbc780df1fce--bcou-role-st-uat-statefulset-0--bcou--UATWKR18_NetworkRxBytes` → slot `net_rx` (log1p, high confidence)
- `Container-DOCKER_CONTAINER_1bc4fc80d241--bcou-trace-st-uat-statefulset-1--bcou--UATWKR18_NetworkRxBytes` → slot `net_rx` (log1p, high confidence)
- `Container-DOCKER_CONTAINER_23bdcf67c7e3--bcou-role-st-uat-statefulset-0--bcou--UATWKR18_MemPercent` → slot `mem` (identity, high confidence)
- `Container-DOCKER_CONTAINER_ef6cb138bb10--bcou-role-st-uat-statefulset-0--bcou--UATWKR18_MemPercent` → slot `mem` (identity, high confidence)
- `Container-DOCKER_CONTAINER_23bdcf67c7e3--bcou-role-st-uat-statefulset-0--bcou--UATWKR18_CpuPercent` → slot `cpu` (identity, high confidence)
- `Container-DOCKER_CONTAINER_ef6cb138bb10--bcou-role-st-uat-statefulset-0--bcou--UATWKR18_CpuPercent` → slot `cpu` (identity, high confidence)
- `Container-DOCKER_CONTAINER_ef6cb138bb10--bcou-role-st-uat-statefulset-0--bcou--UATWKR18_NetworkTxBytes` → slot `net_tx` (log1p, high confidence)
- `Container-DOCKER_CONTAINER_23bdcf67c7e3--bcou-role-st-uat-statefulset-0--bcou--UATWKR18_NetworkRxBytes` → slot `net_rx` (log1p, high confidence)
- `Container-DOCKER_CONTAINER_0f6f3aa7920c--bcou-role-st-uat-statefulset-0--bcou--UATWKR18_CpuPercent` → slot `cpu` (identity, high confidence)
- `Container-DOCKER_CONTAINER_0f6f3aa7920c--bcou-role-st-uat-statefulset-0--bcou--UATWKR18_MemPercent` → slot `mem` (identity, high confidence)
- `Container-DOCKER_CONTAINER_0f6f3aa7920c--bcou-role-st-uat-statefulset-0--bcou--UATWKR18_NetworkTxBytes` → slot `net_tx` (log1p, high confidence)
- `Container-DOCKER_CONTAINER_6d83a96887c3--bcou-role-st-uat-statefulset-0--bcou--UATWKR18_CpuPercent` → slot `cpu` (identity, high confidence)
- `Container-DOCKER_CONTAINER_6d83a96887c3--bcou-role-st-uat-statefulset-0--bcou--UATWKR18_MemPercent` → slot `mem` (identity, high confidence)
- `Container-DOCKER_CONTAINER_6d83a96887c3--bcou-role-st-uat-statefulset-0--bcou--UATWKR18_NetworkTxBytes` → slot `net_tx` (log1p, high confidence)
- `Container-DOCKER_CONTAINER_b2fc383d2438--bcou-trace-st-uat-statefulset-1--bcou--UATWKR18_MemPercent` → slot `mem` (identity, high confidence)
- `Container-DOCKER_CONTAINER_b2fc383d2438--bcou-trace-st-uat-statefulset-1--bcou--UATWKR18_NetworkTxBytes` → slot `net_tx` (log1p, high confidence)
- `Container-DOCKER_CONTAINER_b2fc383d2438--bcou-trace-st-uat-statefulset-1--bcou--UATWKR18_CpuPercent` → slot `cpu` (identity, high confidence)
- `Container-DOCKER_CONTAINER_01eeda2c9f0b--bcou-trace-st-uat-statefulset-1--bcou--UATWKR18_CpuPercent` → slot `cpu` (identity, high confidence)
- `Container-DOCKER_CONTAINER_01eeda2c9f0b--bcou-trace-st-uat-statefulset-1--bcou--UATWKR18_MemPercent` → slot `mem` (identity, high confidence)
- `Container-DOCKER_CONTAINER_01eeda2c9f0b--bcou-trace-st-uat-statefulset-1--bcou--UATWKR18_NetworkTxBytes` → slot `net_tx` (log1p, high confidence)
- `Container-DOCKER_CONTAINER_350771a68ac2--bcou-trace-st-uat-statefulset-1--bcou--UATWKR18_NetworkRxBytes` → slot `net_rx` (log1p, high confidence)
- `Container-DOCKER_CONTAINER_6d83a96887c3--bcou-role-st-uat-statefulset-0--bcou--UATWKR18_NetworkRxBytes` → slot `net_rx` (log1p, high confidence)
- `Container-DOCKER_CONTAINER_ef6cb138bb10--bcou-role-st-uat-statefulset-0--bcou--UATWKR18_NetworkRxBytes` → slot `net_rx` (log1p, high confidence)
- `Container-DOCKER_CONTAINER_c67422614c81--bcou-role-st-uat-statefulset-1--bcou--UATWKR02_NetworkRxBytes` → slot `net_rx` (log1p, high confidence)
- `Container-DOCKER_CONTAINER_cd2b4a29291e--bcou-role-st-uat-statefulset-1--bcou--UATWKR04_NetworkRxBytes` → slot `net_rx` (log1p, high confidence)
- `Container-DOCKER_CONTAINER_0f6f3aa7920c--bcou-role-st-uat-statefulset-0--bcou--UATWKR18_NetworkRxBytes` → slot `net_rx` (log1p, high confidence)
- `Container-DOCKER_CONTAINER_ac3ba8476104--bcou-trace-st-uat-statefulset-1--bcou--UATWKR18_NetworkTxBytes` → slot `net_tx` (log1p, high confidence)
- `Container-DOCKER_CONTAINER_ac3ba8476104--bcou-trace-st-uat-statefulset-1--bcou--UATWKR18_MemPercent` → slot `mem` (identity, high confidence)
- `Container-DOCKER_CONTAINER_ac3ba8476104--bcou-trace-st-uat-statefulset-1--bcou--UATWKR18_CpuPercent` → slot `cpu` (identity, high confidence)
- `Container-DOCKER_CONTAINER_2fa7eaebac26--bcou-role-st-uat-statefulset-1--bcou--UATWKR02_NetworkRxBytes` → slot `net_rx` (log1p, high confidence)
- `Container-DOCKER_CONTAINER_f861998c398e--bcou-role-st-uat-statefulset-0--bcou--UATWKR18_NetworkTxBytes` → slot `net_tx` (log1p, high confidence)
- `Container-DOCKER_CONTAINER_f861998c398e--bcou-role-st-uat-statefulset-0--bcou--UATWKR18_NetworkRxBytes` → slot `net_rx` (log1p, high confidence)
- `Container-DOCKER_CONTAINER_f861998c398e--bcou-role-st-uat-statefulset-0--bcou--UATWKR18_MemPercent` → slot `mem` (identity, high confidence)
- `Container-DOCKER_CONTAINER_f861998c398e--bcou-role-st-uat-statefulset-0--bcou--UATWKR18_CpuPercent` → slot `cpu` (identity, high confidence)
- `Container-DOCKER_CONTAINER_76d31070b844--bcou-role-st-uat-statefulset-1--bcou--UATWKR06_NetworkRxBytes` → slot `net_rx` (log1p, high confidence)

### Implementation steps for Phase 0.9.2.3d

1. Add the above KPIs (or their regex families) to `bank_safe_v2` in the adapter
2. Add corresponding transform rules (identity/log1p/delta_then_log1p)
3. Do NOT add new slots — keep to existing 4-slot structure
4. Re-run Phase 0.9.2.3b scale audit on the expanded `bank_safe_v2` output
5. Defer unit normalization, new slots, and per-container sub-slotting

### Deferred to later phases

- New-slot candidates (Phase 0.9.2.4): mysql_io, disk_io, tomcat_requests, sessions, threads, cache_metrics, filesystem_usage, process_health, swap_io
- Unit normalization for mixed-scale slots (Phase 0.9.2.5)
- Per-container sparse-signal treatment (Phase 0.9.2.6)
- 77.4% unmapped OSLinux metrics: comprehensive regex rewrite (Phase 0.9.3)
