# 遥感知识增强工具描述模板（创新点1）

本模板用于把“通用工具描述”升级为“遥感任务可执行描述”，核心目标是降低工具误选、参数误配与单位错误。

## 1) 设计原则

- 面向遥感：显式编码 GSD、尺度先验、场景先验、常见混淆。
- 面向执行：每个工具都给出输入约束、输出解释、失败处理与替代策略。
- 面向验证：每个工具都给出可检查规则，便于接入 Verifier/Reflector。

## 2) 统一字段规范（建议）

每个工具建议包含以下一级字段：

- `tool`: 工具名。
- `purpose`: 工具在遥感场景中的用途。
- `input_contract`: 输入要求（字段、单位、必填、默认值）。
- `output_contract`: 输出结构与语义解释。
- `rs_knowledge`: 遥感知识增强字段（见下）。
- `failure_modes`: 常见失败模式与触发条件。
- `recovery_policy`: 失败后推荐恢复策略。
- `validation_hooks`: 可直接映射到验证规则的检查项。

### `rs_knowledge` 子字段（核心）

- `scene_priors`: 类别-场景先验映射（如 ship -> water/port）。
- `size_priors_m`: 类别真实尺度范围（米）。
- `density_priors`: 单位面积密度先验（可选）。
- `common_confusions`: 常见误判对（如 building vs ship）。
- `gsd_rules`: 像素与米制换算规则。
- `temporal_priors`: 时序任务变化先验（仅时序工具）。

## 3) 模板实例（TextToBbox）

```json
{
  "tool": "TextToBbox",
  "purpose": "在遥感影像中按文本目标进行定位，输出 bbox。",
  "input_contract": {
    "required": ["image", "text"],
    "optional": {"top1": true},
    "units": {"bbox": "pixel", "gsd_m_per_px": "meter/pixel"}
  },
  "output_contract": {
    "fields": ["bbox_px", "score"],
    "score_range": [0, 1]
  },
  "rs_knowledge": {
    "scene_priors": {
      "ship": ["water", "port", "harbor"],
      "airplane": ["airport", "runway"],
      "vehicle": ["urban", "road", "parking"]
    },
    "size_priors_m": {
      "vehicle": [2, 15],
      "ship": [30, 400],
      "airplane": [10, 80]
    },
    "common_confusions": [
      "building vs ship",
      "road vs river"
    ],
    "gsd_rules": [
      "distance_m = distance_px * gsd_m_per_px",
      "area_m2 = area_px * gsd_m_per_px^2"
    ]
  },
  "failure_modes": [
    "bbox out of image bounds",
    "multiple highly-overlapping boxes",
    "class-scene mismatch"
  ],
  "recovery_policy": [
    "top1=false rerun",
    "switch to ObjectDetection cross-check",
    "require ImageDescription scene verification"
  ],
  "validation_hooks": [
    "PX-01", "PX-02", "RG-22", "GL-20"
  ]
}
```

## 4) 在你当前代码中的落点

- Planner/Reasoner 读取增强描述：`planner.py`、`reasoner.py`。
- 规则验证映射：`spatial_verifier.py`。
- 反思恢复动作：`reflector.py`。

建议把 `failure_modes/recovery_policy/validation_hooks` 显式注入 planner/reasoner prompt，减少模型“自由发挥”。

## 5) 最小实施清单（1周可完成）

- 先覆盖 6 个核心工具：`TextToBbox/ObjectDetection/CountGivenObject/SegmentObjectPixels/ImageDescription/ChangeDetection`。
- 每个工具至少补齐：`scene_priors + common_confusions + gsd_rules + recovery_policy`。
- 建立 20~50 条任务集，记录“工具选择正确率”和“一次通过率”。

## 6) 产出指标（用于论文）

- Tool Selection Accuracy（工具选择正确率）。
- Parameter Validity Rate（参数合法率）。
- Verification Pass@1（首轮验证通过率）。
- Replan Rate（触发重规划比例，越低越好）。

