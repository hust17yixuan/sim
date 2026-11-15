# mindie-sim 0.9.12

[English](sim-0.9.12.en.md)

本版本面向昇腾 A5（`Ascend950DT_9582`）多机专家并行（如 4 机 × 8 卡 `EP32`）与新一代混合架构大模型推理，系统落地多级互联与 MoE 偏斜/负载均衡求解、部署命令解析与硬件物理极值（SOL）对照、四路守恒残差归因（`diff-trace`）、XPyD 多池分离容量与显存六分量账本、单算子可解释理论公式与通用算子流（`OpFlow`）评估，并合入可选 C11 原生事件引擎加速核：

- **A5（`Ascend950DT_9582`）`EP32` 多级通信、HCCL 展开模式与 MoE 偏斜/EPLB 建模（`mindie_sim/fabric_topology.py`、`mindie_sim/fabric_calibrated.py`、`mindie_sim/observations/a5_comm_calibration.py`、`mindie_sim/prediction/hccl_*.py`、`mindie_sim/moe_skew_eplb.py`、`mindie_sim/parallel_ep_knee.py`）**：
  - 支持双模通信织网定价：Mode A（`HierarchicalFabricTree`）按显式 `die -> package -> node -> superpod -> pod -> spine_roce` 物理拓扑树定价跨机架 RoCE 与非均匀通信域；Mode B（`CalibratedFabricProfile` / `comm-w32`）消费 4 机 × 8 卡 A5 UB-mesh/URMA 实测分段曲线，对未标定范围或非均匀 `AllToAllV` 接收端拥塞执行显式边界拒止而非静默外推。
  - 精细区分 `HCCL_OP_EXPANSION_MODE` 下 `AIV`、`AICPU` 与 `CCU_SCHED`（含双 Die 协同与 Z 轴绕行开销）的通信展开机制，支持双微批（dual-microbatch）计算/通信重叠与 MC2 通算融合链路评估。
  - 新增 `moe_skew_eplb` 与 `parallel_ep_knee`：基于有限多项分布与 Dirichlet 专家热度浓度计算最慢 EP Rank 离散词元负载（`max_rank_tokens`），求解整数最小最大热专家副本放置与周期重平衡净收益（`eplb_effect`），并在显式显存与路由资格约束下扫描离散 EP 拐点（`sweep_ep_knee` / `settle_optima`）。
- **启动命令解析、硬件物理极值（SOL）对照与部署寻优（`mindie_sim/launch_*.py`、`mindie_sim/feature_cli.py`、`mindie_sim/advise.py`、`mindie_sim/sweep.py`）**：
  - 新增 `mindie-sim inspect-launch`、`mindie-sim sol` 与 `mindie-sim feature-deltas` 子命令及 `parse_launch` / `compare_launch` API，直接解析 Shell、argv、JSON 或容器 YAML 启动配置，校验并行/特性组合合法性并输出相对硬件物理极值（Speed-of-Light）的差距分解与特性反事实增量。
  - 增强 `compose`、`advise`、`sweep` 与 `skew` 子命令，支持从实测 C7 剖析摘要合成仿真 Case、对候选干预方案按因果收益排序，以及跨拓扑网格与 PD 分离配比求解 Pareto 前沿。
- **实测残差四路守恒归因（`mindie-sim diff-trace`、`mindie_sim/diff_trace.py`）**：
  - 新增 `diff-trace` 子命令与 `attribute_diff` / `effective_operator_rows` 接口（`mindie-diff-trace/2`），通过累积反事实回放将预测与实测单步耗时残差严格守恒分解为单核机制、分块/Padding 膨胀、下发/图模式开销与多流重叠差异四项，自动校验固定算子 ABI 目录中的有效词元行数并生成可直接执行的 `mindie-profiler repro-op` 复现命令。
- **XPyD 分离池求解器、六分量显存账本与混合线性注意力 / DSpark 状态物理（`mindie_sim/xpyd_*.py`、`mindie_sim/memory_breakdown.py`、`mindie_sim/hybrid_kv_state_*.py`、`mindie_sim/dspark_*.py`、`mindie_sim/linear_state_*.py`）**：
  - 新增 `solve_xpyd_pools`：在总卡数预算与严格利用率上限下联合裁剪 P-HBM、P-NIC、RoCE、D-NIC、D-HBM、KV 传输线速与暂存池容量，最大化满足 TTFT/TPOT SLO 的有效完成请求吞吐（Goodput）。
  - 新增 `MemoryLedger` / `analyze_memory_breakdown`：按每 Rank 精确拆解分片权重、辅助权重（MTP/EPLB）、分页 KV 与稀疏/递推状态、通信域持久 Arena、ACLGraph 捕获池与运行时 Workspace 六大常驻 HBM 分量。
  - 完整建模混合线性注意力（GDN / KDA / Mamba 状态块与前缀重放）与 DSpark 稀疏 KV / 前缀缓存 / DCP 协同开销。
- **单算子理论公式引擎、通用算子流（`OpFlow`）、Day-0 图优化器与可选 C11 加速核（`mindie_sim/op_formulas/`、`mindie_sim/opflow/`、`mindie_sim/prediction/graph_optimizer.py`、`mindie_sim/prediction/weight_prefetch.py`、`mindie_sim/prediction/runtime/`）**：
  - 新增 `mindie_sim.op_formulas`（覆盖 18+ 算子族代数公式卡、硬物理下界 `t_floor` 与参考下界 `t_ref`）与 `mindie_sim.opflow`（UB/L1/L0 分块、Fixpipe 路径、Triton 降级门禁及 `ACLGraph` / `SuperKernel` 启动下界有序建模）。
  - 新增确定性 Day-0 图融合与多流调度优化器（`run_optimizer`）、权重预取掩盖建模（`weight_prefetch`）以及 `mindie-sim build-accel` 可选 C11 原生事件引擎加速核（Windows 编译子进程默认携带 `CREATE_NO_WINDOW`），并保留 `v0.9.11` 全部 7 款多机集群离线模型目录、`native-call-binding/1` 与 `evaluate_primitive_spec` 契约。

本版本的实际测试与基准结果以随包 `benchmark-results.json`、`benchmark-results.md` 及对应 CI 记录为准。
