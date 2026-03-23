# 创新点实验计划（可直接执行）

本文件给出三项创新的对照实验设计，确保可复现、可量化、可写论文。

## A. 创新点1：遥感知识增强工具描述

### A1. 对照组设置

- Baseline：仅使用通用工具描述（不含遥感先验）。
- Ours-V1：加入 `rs_knowledge`（场景先验、尺度先验、混淆对、GSD规则）。

### A2. 评估指标

- `ToolSelection@1`：首个关键工具是否正确。
- `ParamValidRate`：参数合法率（bbox、单位、GSD 相关）。
- `Pass@1`：第一轮 verifier 全通过比例。
- `AvgRetries`：平均重试次数。

### A3. 预期结论

- Ours-V1 在工具选择与参数合法性上显著优于 Baseline。

## B. 创新点2：视觉感知增强空间推理与验证

### B1. 消融设置

- B0：关闭 `spatial_verifier`。
- B1：仅 Pixel-level（边界/像素/GSD）。
- B2：Pixel + Region（重叠、尺度、密度、形状）。
- B3：Pixel + Region + Global（语义一致性 + 跨轮一致性）。

### B2. 指标

- `SpatialConsistencyScore`：空间规则通过率。
- `SemanticConsistencyScore`：目标-场景一致性通过率。
- `CrossRoundStability`：多轮证据稳定性（IoU/计数方差）。
- `FalsePositiveRecoveryRate`：误检恢复比例。

### B3. 预期结论

- B3 在稳定性和误检恢复上最好。

## C. 创新点3：自检多步规划机制

### C1. 对照设置

- C0：无反思回路（单次执行）。
- C1：有 verifier + reflector 但不重规划。
- C2：完整闭环（planner-reasoner-reflector 重规划）。

### C2. 指标

- `TaskSuccessRate`：任务最终正确率。
- `RecoverySuccessRate`：首次失败后恢复成功率。
- `ErrorTypeBreakdown`：感知/参数/逻辑/一致性错误占比变化。

### C3. 预期结论

- C2 在复杂任务上的成功率和鲁棒性显著提升。

## D. 数据与日志建议

- 使用你现有 `tasks/thinkgeo_*.json` 作为起点，补充“高混淆样本”。
- 每次运行保存：
  - step 级工具调用与输入输出；
  - verifier verdict；
  - reflector action；
  - 是否触发 replan；
  - 最终答案。

## E. 论文呈现建议

- 表1：三创新点分模块增益（逐步叠加）。
- 表2：错误类型迁移（由感知错误向可恢复参数错误转移）。
- 图1：闭环推理流程图（planner→reasoner→verifier→reflector）。
- 图2：典型案例可视化（误检前后、重规划前后）。

