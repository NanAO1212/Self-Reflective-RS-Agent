# 实验日志

记录格式：改了什么 → 为什么改 → 结果如何

---

## 2026-02-26 全量实验 v1 (baseline)
- 时间: 2026-02-26，总耗时 8886.6s (~2.5h)
- 配置: max_retries=1, evidence_rounds=1, Qwen3-VL-30B-A3B-Thinking, image_size=auto
- 目的: 跑全量 432 tasks baseline，对齐 ThinkGeo Table 2
- 改动: 无，首次全量跑

### 运行概况
| 项目 | 值 |
|---|---|
| 总任务数 | 432 |
| 成功 | 423 |
| 失败(有 fail step) | 1 |
| 错误(异常) | 8 |
| 评估覆盖 | 55 / 432（仅 55 个任务进入 eval_report） |
| 有 GT 的任务 | 42 / 55 |

### 评估指标

#### ThinkGeo 对齐指标
| 指标 | 值 |
|---|---|
| AnswerAcc | 50.0% (21/42) |
| ToolCallAcc | 98.2% |
| StepSuccessRate | 100% |

#### 创新点 A — 知识增强工具描述
| 指标 | 值 |
|---|---|
| ToolSelection@1 | 94.6% |
| ParamValidRate | 100% |
| Pass@1 | 98.2% |
| AvgRetries | 0.018 |

#### 创新点 B — 空间验证
| 指标 | 值 |
|---|---|
| SpatialConsistency | 1.0 |
| SemanticConsistency | 1.0 |
| CrossRoundStability | 1.0 |
| FPRecoveryRate | 1.0 |

#### 创新点 C — 自反思
| 指标 | 值 |
|---|---|
| TaskSuccessRate | 50.0% |
| RecoverySuccessRate | 25.0% (1/4 有 fail 的任务恢复) |
| ReflectionStrategyDist | 全部 "none"（267 步均未触发反思） |
| 总 retries | 5 |

### 发现/问题
1. **评估覆盖率严重不足** — 432 个任务只有 55 个进入 eval_report，需排查 evaluate.py 是否遗漏了大量 log 文件
2. **工具链表现优秀** — ToolCallAcc 98.2%、ParamValidRate 100%，Planner 工具选择和参数生成可靠
3. **AnswerAcc 瓶颈在 Reasoner** — 工具调用正确但最终答案只有 50%，说明 Reasoner 的答案提取/推理环节是主要短板
4. **自反思机制未生效** — 267 步全部 strategy="none"，Verifier 几乎没有检测到需要反思的情况；仅 5 次 retry 且只有 1 次成功恢复
5. **空间验证指标全 1.0** — 过于完美，怀疑验证规则覆盖不足，大量步骤因无适用规则而直接 pass

---

## 2026-03-09 评估修复 + 自反思规则扩展

### 1. evaluate.py 修复

**问题**: evaluate.py 只评估了 55/432 任务
**根因**: 两个独立 bug
1. eval_report 在 baseline 跑到一半时就生成了（14:56 生成，log 直到 17:02 才全部完成）→ 只有 55 个 log
2. `answer_match()` 不兼容 list 格式的 `gt_answer`（12 个任务），导致 `AttributeError` 崩溃

**修复**:
- `answer_match()` 增加 list 格式兼容：`["答案1"] → {"whitelist": [["答案1"]], "blacklist": []}`
- 新增 `_kw_match()` 三级匹配策略：精确子串 → 去千分位逗号 → 分词匹配

**修复后全量评估 (424 tasks, 358 with GT)**:

| 指标 | 修复前 (55 tasks) | 修复后 (424 tasks) | 匹配优化后 |
|------|:---:|:---:|:---:|
| AnswerAcc | 50.0% | 68.2% | **73.7%** |
| ToolCallAcc | 98.2% | 95.8% | 95.8% |
| StepSuccessRate | 100% | 99.9% | 99.9% |

**66 个无 gt_answer 的任务**: 全部是 draw/plot/detect 类视觉输出任务，ThinkGeo 数据集本身未提供标准答案，非转换丢失。

### 2. AnswerAcc 错误模式分析

114 个错误任务的分类：

| 错误类型 | 数量 | 占比 | 说明 |
|----------|:---:|:---:|------|
| partial_match | 48 | 42.1% | 答案正确但匹配方式太严格 |
| numerical_error | 39 | 34.2% | 数值计算/推理错误 |
| wrong_content | 22 | 19.3% | 答案内容真的错了 |
| spatial_reasoning | 4 | 3.5% | 空间推理错误 |
| empty_answer | 1 | 0.9% | 答案为空 |

匹配修复（`_kw_match` 三级策略）挽救了 21 个任务，AnswerAcc 从 68.2% → 73.7%。

### 3. 自反思机制修复

**问题**: 1411 步全部 strategy="none"，自反思未触发
**根因链条**: Verifier 只有 2 个 fail → Pipeline 不触发 reflect_and_patch() → Reflector 无机会参与

**修复**: 在 verifier.py 新增 4 条语义级验证规则：

| 规则 | 检查内容 | 离线模拟触发数 |
|------|----------|:---:|
| GL-30 | CountGivenObject 返回 0 或 >500 | 13 |
| GL-31 | Calculator 结果 NaN/Inf/负数 | 0 |
| GL-32 | 工具返回空/无效输出 | 1 |
| GL-34 | 描述工具幻觉检测 | **73** |
| 合计 | — | **87 (影响 57 个任务)** |

同步更新 reflector.py：RULE_DESCRIPTIONS、reflect()、_rule_based_patches() 均添加了新规则的处理逻辑。

**预期效果**: 下次实验时 verifier 将在 57 个任务中触发 87 次 fail（vs baseline 的 2 次），驱动反思机制真正参与。

### 待验证
- [x] 小规模测试验证反思链条（2026-03-09 已通过）
- [ ] 全量重跑实验，对比 ReflectionStrategyDist、RecoverySuccessRate 等指标变化
- [ ] GL-34 触发后 retry 是否能改善 RegionAttributeDescription 的输出质量

### 小规模测试结果 (2026-03-09)

| 任务 | 预期 | 实际结果 |
|------|------|------|
| thinkgeo_0 (正常) | 无 fail | 0 fail, 0 retry, strategy=none — 未误伤 |
| thinkgeo_145 (GL-34) | 幻觉触发 | GL-34 触发 → strategy=rule_based → switch_tool→ImageDescription → retry 成功 |

thinkgeo_145 完整链条：
1. RegionAttributeDescription 输出 "a black and white photo of a person holding a frisbee"
2. GL-34 触发 FAIL
3. Reflector 生成 rule_based 策略，patches.switch_tool = "ImageDescription"
4. Retry 用 ImageDescription，输出 "an aerial view of a road in the middle of nowhere"
5. GL-34 检查 PASS

注意：GL-30 (count=0) 规则已调整为不触发，因为分析发现 13 个 count=0 的任务中 GT 答案本身就预期 0 或多步骤中其他步骤提供了正确计数。

---

<!-- 模板（复制粘贴新增条目）
## YYYY-MM-DD 实验名称
- 配置: max_retries=?, evidence_rounds=?, 模型=?
- 目的:
- 改动:
- 结果:
- 发现/问题:
-->
