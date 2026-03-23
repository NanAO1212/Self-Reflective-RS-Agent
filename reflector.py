"""
反思器模块 — 上下文感知的 LLM-based 反思 + 规则兜底。

两层反思策略：
1. reflect()          — 轻量规则映射，零延迟，用于简单/明确错误的快速修复
2. reflect_with_llm() — 调用 LLM，综合失败步骤、验证结果、执行历史、
                         场景上下文生成针对性修复方案，返回结构化 patch
"""

import json
import os
from pathlib import Path

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# ---------------------------------------------------------------------------
# 环境 & API 工具函数（复用 planner/reasoner 的模式）
# ---------------------------------------------------------------------------

def _api_url():
    base = os.getenv(
        "REFLECTOR_BASE_URL",
        os.getenv("REASONER_BASE_URL",
                   os.getenv("LLM_BASE_URL", "http://127.0.0.1:8001/v1")),
    )
    return base.rstrip("/") + "/chat/completions"


def _retry_session():
    """创建带自动重试的 requests session"""
    session = requests.Session()
    retries = Retry(
        total=3,
        backoff_factor=10,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["POST"],
        raise_on_status=False,
    )
    adapter = HTTPAdapter(max_retries=retries)
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    return session


def _env_bool(name, default=None):
    val = os.getenv(name)
    if val is None:
        return default
    return val.strip().lower() in ("1", "true", "yes", "y", "on")


def _maybe_dump_payload(tag, payload):
    if not _env_bool("LLM_DEBUG_PAYLOAD"):
        return
    safe = json.loads(json.dumps(payload))
    for msg in safe.get("messages", []):
        content = msg.get("content")
        if isinstance(content, list):
            for item in content:
                if item.get("type") == "image_url":
                    url = item.get("image_url", {}).get("url", "")
                    item["image_url"]["url"] = f"<base64 len={len(url)}>"
    Path("logs").mkdir(parents=True, exist_ok=True)
    Path(f"logs/{tag}_payload.json").write_text(
        json.dumps(safe, ensure_ascii=False, indent=2), encoding="utf-8"
    )


# ---------------------------------------------------------------------------
# 规则 ID → 领域知识映射，给 LLM 提供验证规则的语义解释
# ---------------------------------------------------------------------------

RULE_DESCRIPTIONS = {
    # Pixel-level
    "PX-01": "检测框超出图像边界",
    "PX-02": "检测框尺寸为零或负值",
    "PX-03": "分割像素计数为负",
    "PX-04": "计算面积时缺少 GSD（地面采样距离）",
    # Region-level
    "RG-01": "多个检测框高度重叠（IoU>0.9），可能重复检测",
    "RG-10": "检测框覆盖几乎整张图像，目标尺度可能错误",
    "RG-11": "检测框面积极小（<4px），可能是噪声",
    "RG-12": "检测框宽高比极端（>20 或 <0.05）",
    "RG-20": "目标物理尺寸超出该类别的先验范围（结合 GSD）",
    "RG-21": "所有检测框总面积占图像 >90%，密度异常",
    "RG-22": "存在近乎重复的检测框（IoU>0.95）",
    # Perception-level (evidence)
    "PV-10": "GSD 值非正或无效",
    "PV-20": "多轮证据中目标定位不一致（IoU<0.4）",
    "PV-21": "多轮证据中计数/像素值波动过大（>25%）",
    "PV-30": "与上一次相同查询的定位结果不一致（IoU<0.3）",
    # Global-level
    "GL-10": "文本描述工具返回空结果",
    "GL-20": "目标类别与全局场景语义冲突（如水域中检测到汽车）",
    "GL-30": "目标计数为零或异常大，可能检测失败或类别不匹配",
    "GL-31": "计算结果为 NaN/Inf 或面积距离为负值",
    "GL-32": "工具返回空或无效输出",
    "GL-34": "描述工具输出疑似模型幻觉（与遥感场景无关的描述）",
}

# ---------------------------------------------------------------------------
# 错误类型 → 修复策略模板（给 LLM 的先验指导）
# ---------------------------------------------------------------------------

ERROR_TYPE_HINTS = {
    "parameter": (
        "参数类错误：通常是 bbox 越界、GSD 缺失或格式错误。"
        "优先尝试 clamp bbox 到图像范围、补充 GSD、修正输入格式。"
    ),
    "perception": (
        "感知类错误：检测/分割结果不稳定或为空。"
        "可尝试：设置 top1=false 获取更多候选、降低置信度阈值、"
        "换用替代工具（如 TextToBbox→ObjectDetection）、调整 prompt 描述。"
    ),
    "logic": (
        "逻辑类错误：单位换算、面积/距离计算有误。"
        "应插入 Calculator/Solver 步骤，确保 GSD 单位一致（m/px），"
        "检查是否混淆了像素坐标和物理坐标。"
    ),
    "consistency": (
        "一致性错误：跨步骤或跨轮次结果矛盾。"
        "需要回溯检查：场景描述是否与目标类别匹配、"
        "多次检测的 bbox 是否稳定、是否存在重复检测需要 NMS。"
    ),
}


# ---------------------------------------------------------------------------
# 第一层：轻量规则反思（原有逻辑，保留作为 fallback）
# ---------------------------------------------------------------------------

def reflect(verifier_results):
    """规则映射反思 — 零延迟，用于 LLM 不可用或简单错误场景。"""
    actions = []
    for r in verifier_results:
        if r.get("status") != "fail":
            continue
        error_type = r.get("error_type")
        rule_id = r.get("rule_id", "")
        details = r.get("details", "")
        suggested_fix = r.get("suggested_fix", "")

        # 比原版更具体：结合 rule_id 给出针对性建议
        if error_type == "parameter":
            if rule_id == "PX-01":
                actions.append("Clamp bbox coordinates to image bounds and re-call.")
            elif rule_id == "PX-04":
                actions.append("Obtain GSD (gsd_m_per_px) from task context before computing area.")
            elif rule_id == "PV-10":
                actions.append("Set a valid positive gsd_m_per_px value.")
            else:
                actions.append(f"Fix parameter issue ({details}): {suggested_fix}")
        elif error_type == "perception":
            if rule_id == "PV-20":
                actions.append("Localization unstable across rounds; re-run with top1=false or use ObjectDetection as anchor.")
            elif rule_id == "PV-21":
                actions.append("Count/pixel values unstable; re-run segmentation with stricter threshold.")
            elif rule_id == "PV-30":
                actions.append("Result inconsistent with previous call; re-run with fixed seed or adjust text prompt.")
            elif rule_id == "GL-10":
                actions.append("Description tool returned empty; re-run or switch to ImageDescription.")
            else:
                actions.append(f"Re-run perception tool ({details}): {suggested_fix}")
        elif error_type == "logic":
            if rule_id == "GL-30":
                actions.append("Count is zero or abnormally large; verify object category keyword or use TextToBbox to confirm.")
            elif rule_id == "GL-31":
                actions.append("Calculator result is NaN/Inf/negative; check expression for division by zero or coordinate order.")
            else:
                actions.append(f"Insert Calculator/Solver to fix: {details}")
        elif error_type == "consistency":
            if rule_id == "GL-20":
                actions.append("Object-scene semantic conflict; verify scene description first, then re-detect.")
            elif rule_id in ("RG-01", "RG-22"):
                actions.append("Duplicate detections; apply NMS or set top1=true.")
            elif rule_id == "RG-20":
                actions.append("Object scale out of prior range; verify GSD value or re-check object category.")
            else:
                actions.append(f"Consistency issue ({details}): {suggested_fix}")
        else:
            if rule_id == "GL-32":
                actions.append("Tool returned empty output; re-run with adjusted parameters or use alternative tool.")
            elif rule_id == "GL-34":
                actions.append("Description tool produced hallucination; re-run with tighter bbox crop or switch to ImageDescription.")
            else:
                actions.append(f"Re-run step with safer defaults ({details}).")
    return actions


# ---------------------------------------------------------------------------
# 第二层：LLM-based 上下文感知反思
# ---------------------------------------------------------------------------

REFLECTOR_SYSTEM_PROMPT = """\
You are a self-reflection module in a remote sensing agent pipeline.
Your job: given a failed tool execution step, its verification verdicts, and the
execution context, produce a structured corrective plan.

You receive:
- failed_step: the tool call that failed (tool_name, input, parsed output)
- verdicts: list of verification failures (rule_id, error_type, details)
- context: scene description, GSD, execution history of previous steps
- failure_history: past failures in this session (to avoid repeating the same fix)

You MUST return a JSON object with these fields:
{
  "diagnosis": "1-2 sentence root cause analysis in Chinese",
  "strategy": "corrective strategy in Chinese",
  "actions": ["human-readable action strings"],
  "patches": {
    "input_overrides": {"key": "new_value", ...},
    "add_steps_before": [{"tool_name": "...", "tool_type": "...", "input": {...}}],
    "add_steps_after": [{"tool_name": "...", "tool_type": "...", "input": {...}}],
    "switch_tool": "alternative_tool_name or null",
    "retry_params": {"top1": false, "flag": false, ...}
  },
  "confidence": 0.0-1.0,
  "should_replan": false
}

Rules:
- patches.input_overrides: concrete key-value pairs to override in the tool input for retry.
- patches.add_steps_before: if the fix requires a prerequisite step (e.g., get scene description first).
- patches.add_steps_after: if a follow-up verification step is needed.
- patches.switch_tool: if the current tool is fundamentally wrong, suggest an alternative.
- patches.retry_params: parameters to merge into tool input on retry.
- should_replan: true only if the error is systemic and local retry cannot fix it.
- confidence: your confidence that the proposed fix will resolve the issue.

Domain knowledge for remote sensing:
- GSD (Ground Sampling Distance) converts pixels to meters: length_m = pixels * gsd_m_per_px
- Common objects and expected scenes: ship→water, airplane→airport, car→urban
- Size priors (meters): car 2-6m, truck 5-15m, ship 30-400m, airplane 10-80m, building 5-200m
- If bbox covers >98% of image, the detection likely failed — try segmentation instead.
- If multi-round IoU < 0.4, the perception model is unstable — try alternate prompt or tool.
- Always prefer concrete parameter fixes over vague "re-run" suggestions.
"""


def _build_reflection_context(failed_step, verdicts, context, failure_history):
    """构建发送给 LLM 的反思上下文。"""
    # 提取失败的验证规则及其语义描述
    failure_details = []
    for v in verdicts:
        if v.get("status") != "fail":
            continue
        rule_id = v.get("rule_id", "unknown")
        failure_details.append({
            "rule_id": rule_id,
            "error_type": v.get("error_type"),
            "details": v.get("details"),
            "suggested_fix": v.get("suggested_fix"),
            "rule_description": RULE_DESCRIPTIONS.get(rule_id, ""),
        })

    # 提取错误类型的领域提示
    error_types = list({f["error_type"] for f in failure_details if f.get("error_type")})
    domain_hints = [ERROR_TYPE_HINTS[et] for et in error_types if et in ERROR_TYPE_HINTS]

    # 精简执行历史（只保留最近 5 步，避免 token 爆炸）
    history_summary = []
    for tool_name, records in (context.get("history") or {}).items():
        for rec in records[-2:]:  # 每个工具最多保留最近 2 次
            history_summary.append({
                "tool": tool_name,
                "input_keys": list((rec.get("input") or {}).keys()),
                "parsed_keys": list((rec.get("parsed") or {}).keys()),
                "had_bboxes": bool((rec.get("spatial") or {}).get("bboxes_px")),
            })

    return {
        "failed_step": {
            "tool_name": failed_step.get("tool_name") or failed_step.get("tool"),
            "tool_type": failed_step.get("tool_type"),
            "input": failed_step.get("tool_input") or failed_step.get("input", {}),
            "parsed_output": failed_step.get("parsed"),
            "raw_output_preview": str(failed_step.get("tool_output", ""))[:500],
        },
        "verdicts": failure_details,
        "domain_hints": domain_hints,
        "context": {
            "scene_text": context.get("scene_text"),
            "gsd_m_per_px": context.get("gsd_m_per_px"),
            "history_summary": history_summary[-5:],
        },
        "failure_history": [
            {"tool": f.get("tool"), "rule_id": f.get("rule_id"), "error_type": f.get("error_type")}
            for f in (failure_history or [])[-5:]
        ],
    }


def reflect_with_llm(failed_step, verdicts, context=None, failure_history=None):
    """
    LLM-based 上下文感知反思。

    Parameters
    ----------
    failed_step : dict
        失败的步骤记录（包含 tool_name, input, parsed, tool_output 等）
    verdicts : list[dict]
        该步骤的 verifier 结果列表
    context : dict | None
        流水线上下文（scene_text, gsd_m_per_px, history）
    failure_history : list[dict] | None
        本次 session 中之前的失败记录，用于避免重复修复

    Returns
    -------
    dict  包含 diagnosis, strategy, actions, patches, confidence, should_replan
    """
    context = context or {}
    failure_history = failure_history or []

    # 如果没有失败，直接返回空结果
    has_fail = any(v.get("status") == "fail" for v in verdicts)
    if not has_fail:
        return {
            "diagnosis": "所有验证规则通过，无需反思。",
            "strategy": "none",
            "actions": [],
            "patches": {},
            "confidence": 1.0,
            "should_replan": False,
        }

    reflection_context = _build_reflection_context(
        failed_step, verdicts, context, failure_history
    )

    user_message = json.dumps(reflection_context, ensure_ascii=False, indent=2)

    messages = [
        {"role": "system", "content": REFLECTOR_SYSTEM_PROMPT},
        {"role": "user", "content": user_message},
    ]

    api_key = os.getenv(
        "REFLECTOR_API_KEY",
        os.getenv("REASONER_API_KEY", os.getenv("LLM_API_KEY", "")),
    )
    model = os.getenv(
        "REFLECTOR_MODEL",
        os.getenv("LLM_MODEL", "Qwen3-8B"),
    )
    enable_thinking = _env_bool("REFLECTOR_ENABLE_THINKING")
    thinking_budget = os.getenv("REFLECTOR_THINKING_BUDGET")

    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    payload = {
        "model": model,
        "messages": messages,
        "temperature": 0.3,
    }
    # qwen3 系列非流式调用必须显式关闭 thinking
    if enable_thinking is not None:
        payload["enable_thinking"] = enable_thinking
    else:
        payload["enable_thinking"] = False
    if thinking_budget:
        try:
            payload["thinking_budget"] = int(thinking_budget)
        except Exception:
            pass

    _maybe_dump_payload("reflector", payload)

    try:
        session = _retry_session()
        resp = session.post(_api_url(), json=payload, headers=headers, timeout=60)
        if not resp.ok:
            raise RuntimeError(f"Reflector API error {resp.status_code}: {resp.text}")
        content = resp.json()["choices"][0]["message"]["content"]
        result = _parse_llm_response(content)
    except Exception as e:
        # LLM 调用失败时，降级到规则反思
        rule_actions = reflect(verdicts)
        result = {
            "diagnosis": f"LLM 反思调用失败 ({type(e).__name__})，降级为规则反思。",
            "strategy": "rule_fallback",
            "actions": rule_actions,
            "patches": _rule_based_patches(failed_step, verdicts),
            "confidence": 0.4,
            "should_replan": False,
            "_fallback": True,
            "_error": str(e),
        }

    return result


def _parse_llm_response(content):
    """从 LLM 输出中提取结构化 JSON，容错处理。"""
    try:
        result = json.loads(content)
    except Exception:
        start = content.find("{")
        end = content.rfind("}")
        if start >= 0 and end > start:
            try:
                result = json.loads(content[start:end + 1])
            except Exception:
                result = None
        else:
            result = None

    if not isinstance(result, dict):
        return {
            "diagnosis": "LLM 返回格式异常，降级为规则反思。",
            "strategy": "parse_fallback",
            "actions": [content[:300] if content else "No response"],
            "patches": {},
            "confidence": 0.3,
            "should_replan": False,
            "_raw": content[:500] if content else "",
        }

    # 确保所有必需字段存在
    result.setdefault("diagnosis", "")
    result.setdefault("strategy", "")
    result.setdefault("actions", [])
    result.setdefault("patches", {})
    result.setdefault("confidence", 0.5)
    result.setdefault("should_replan", False)

    # 规范化 patches 结构
    patches = result["patches"]
    patches.setdefault("input_overrides", {})
    patches.setdefault("add_steps_before", [])
    patches.setdefault("add_steps_after", [])
    patches.setdefault("switch_tool", None)
    patches.setdefault("retry_params", {})

    return result


def _rule_based_patches(failed_step, verdicts):
    """当 LLM 不可用时，基于规则生成结构化 patches。"""
    patches = {
        "input_overrides": {},
        "add_steps_before": [],
        "add_steps_after": [],
        "switch_tool": None,
        "retry_params": {},
    }
    tool_name = failed_step.get("tool_name") or failed_step.get("tool", "")

    for v in verdicts:
        if v.get("status") != "fail":
            continue
        rule_id = v.get("rule_id", "")
        error_type = v.get("error_type", "")

        if rule_id == "PX-01":
            # bbox 越界 — pipeline 中的 _clamp_bbox 会处理，这里标记
            patches["retry_params"]["_needs_clamp"] = True

        elif rule_id in ("RG-10", "RG-11", "RG-12"):
            # 检测框异常 — 尝试换用分割
            if tool_name in ("TextToBbox", "ObjectDetection"):
                patches["switch_tool"] = "SegmentObjectPixels"

        elif rule_id == "RG-20":
            # 尺度先验不匹配 — 可能 GSD 有误，插入场景描述步骤
            patches["add_steps_before"].append({
                "tool_name": "ImageDescription",
                "tool_type": "perception",
                "input": {"image": (failed_step.get("tool_input") or failed_step.get("input", {})).get("image", "")},
            })

        elif rule_id in ("PV-20", "PV-30"):
            # 定位不稳定 — 放宽 top1 限制
            if tool_name == "TextToBbox":
                patches["retry_params"]["top1"] = False

        elif rule_id == "PV-21":
            # 计数不稳定 — 降低 flag
            if tool_name == "SegmentObjectPixels":
                patches["retry_params"]["flag"] = False

        elif rule_id == "GL-10":
            # 空描述 — 换工具
            if tool_name == "RegionAttributeDescription":
                patches["switch_tool"] = "ImageDescription"

        elif rule_id == "GL-20":
            # 语义冲突 — 先获取场景描述再重新检测
            patches["add_steps_before"].append({
                "tool_name": "ImageDescription",
                "tool_type": "perception",
                "input": {"image": (failed_step.get("tool_input") or failed_step.get("input", {})).get("image", "")},
            })

        elif rule_id == "GL-30":
            # 计数异常 — 换用 TextToBbox 确认目标是否存在
            if tool_name == "CountGivenObject":
                patches["switch_tool"] = "TextToBbox"

        elif rule_id == "GL-32":
            # 空输出 — 换工具或调整参数
            if tool_name == "CountGivenObject":
                patches["switch_tool"] = "TextToBbox"
            elif tool_name in ("RegionAttributeDescription",):
                patches["switch_tool"] = "ImageDescription"

        elif rule_id == "GL-34":
            # 幻觉描述 — 换用 ImageDescription
            if tool_name == "RegionAttributeDescription":
                patches["switch_tool"] = "ImageDescription"

        elif error_type == "logic":
            patches["add_steps_after"].append({
                "tool_name": "Calculator",
                "tool_type": "logic",
                "input": {"expression": ""},
            })

    return patches


# ---------------------------------------------------------------------------
# 对外统一接口：自动选择反思策略
# ---------------------------------------------------------------------------

def reflect_and_patch(failed_step, verdicts, context=None, failure_history=None,
                      use_llm=None):
    """
    统一反思入口 — pipeline 调用此函数。

    Parameters
    ----------
    failed_step : dict
        失败步骤的完整记录
    verdicts : list[dict]
        verifier 结果
    context : dict | None
        流水线上下文
    failure_history : list[dict] | None
        历史失败记录
    use_llm : bool | None
        是否强制使用 LLM 反思。None = 自动判断（复杂错误用 LLM，简单错误用规则）

    Returns
    -------
    dict  {diagnosis, strategy, actions, patches, confidence, should_replan}
    """
    failures = [v for v in verdicts if v.get("status") == "fail"]
    if not failures:
        return {
            "diagnosis": "无失败，跳过反思。",
            "strategy": "none",
            "actions": [],
            "patches": {},
            "confidence": 1.0,
            "should_replan": False,
        }

    # 自动判断是否需要 LLM
    if use_llm is None:
        use_llm = _should_use_llm(failures, failure_history)

    if use_llm:
        return reflect_with_llm(failed_step, verdicts, context, failure_history)
    else:
        actions = reflect(verdicts)
        patches = _rule_based_patches(failed_step, verdicts)
        return {
            "diagnosis": "规则反思：" + "; ".join(v.get("details", "") for v in failures),
            "strategy": "rule_based",
            "actions": actions,
            "patches": patches,
            "confidence": 0.6,
            "should_replan": False,
        }


def _should_use_llm(failures, failure_history=None):
    """
    判断是否需要调用 LLM 进行深度反思。

    触发 LLM 的条件（满足任一）：
    1. 存在多种不同类型的错误（需要综合分析）
    2. 出现语义一致性错误（GL-20），需要理解场景
    3. 同一 rule_id 在历史中重复出现（说明规则修复无效，需要更深层分析）
    4. 存在 perception 类错误且历史中也有 perception 失败（持续不稳定）
    """
    # 环境变量强制控制
    force = _env_bool("REFLECTOR_USE_LLM")
    if force is not None:
        return force

    error_types = {f.get("error_type") for f in failures}
    rule_ids = {f.get("rule_id") for f in failures}

    # 条件 1：多种错误类型
    if len(error_types) >= 2:
        return True

    # 条件 2：语义一致性错误
    if "GL-20" in rule_ids:
        return True

    # 条件 3 & 4：历史重复
    if failure_history:
        hist_rules = {f.get("rule_id") for f in failure_history}
        if rule_ids & hist_rules:
            return True
        hist_types = {f.get("error_type") for f in failure_history}
        if "perception" in error_types and "perception" in hist_types:
            return True

    return False


# ---------------------------------------------------------------------------
# apply_patches: 将反思结果应用到工具输入
# ---------------------------------------------------------------------------

def apply_patches(tool_name, tool_input, patches):
    """
    将 reflector 返回的 patches 应用到工具输入，返回新的 (tool_name, tool_input)。
    """
    if not patches:
        return tool_name, dict(tool_input)

    new_input = dict(tool_input)

    # 合并 input_overrides
    for k, v in (patches.get("input_overrides") or {}).items():
        new_input[k] = v

    # 合并 retry_params
    for k, v in (patches.get("retry_params") or {}).items():
        if k.startswith("_"):
            continue  # 跳过内部标记
        new_input[k] = v

    # 工具切换
    new_tool = patches.get("switch_tool")
    if new_tool:
        tool_name = new_tool

    return tool_name, new_input


# ---------------------------------------------------------------------------
# CLI 入口
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Reflector — 规则反思 or LLM 反思")
    ap.add_argument("--results", required=True, help="verifier results JSON string")
    ap.add_argument("--step", default="{}", help="failed step JSON string")
    ap.add_argument("--context", default="{}", help="pipeline context JSON string")
    ap.add_argument("--use-llm", action="store_true", help="强制使用 LLM 反思")
    args = ap.parse_args()

    verdicts = json.loads(args.results)
    step = json.loads(args.step)
    ctx = json.loads(args.context)

    result = reflect_and_patch(step, verdicts, context=ctx, use_llm=args.use_llm or None)
    print(json.dumps(result, ensure_ascii=False, indent=2))






