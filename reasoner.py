import base64
import io
import json
import mimetypes
import os
import time
from pathlib import Path
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


SYSTEM_PROMPT = """You are a remote sensing analyst agent.
Given an image path (or two temporal image paths) and a natural language question,
produce a tool-executable plan in JSON.

Requirements:
- Use ONLY tools from the provided tool list.
- Use image paths directly as tool inputs.
- Return a JSON object with:
  - steps: list of tool calls (tool_name, tool_type, input)
  - final_answer: concise text answer
- outputs: list of visualization actions (DrawBox/DrawMask/AddText/Plot) to produce multi-modal outputs
  - optional gsd_m_per_px if you can infer or it is provided

Multi-modal outputs:
- If you localize objects, add DrawBox with bbox.
- If you compute numeric stats, add Plot with a simple chart.
- If you want labels, add AddText.

Temporal change:
- If question is about changes over time, MUST include ChangeDetection.
- For change areas/counts, include SegmentObjectPixels or CountGivenObject + Calculator.

Spatial reasoning:
- Use TextToBbox or ObjectDetection for localization.
- Use Calculator for GSD-based distance/area.
- Always be explicit with bbox and GSD usage if available in the question.

Self-check:
- If reflector feedback is provided, adjust the plan to fix the failed steps.
- If validation depends on global scene, include an ImageDescription step early.

Mandatory ops verification:
- Always include ObjectDetection and SegmentObjectPixels at least once to verify detection/segmentation pipelines.

Output format (MUST follow strictly):
- steps: each item MUST be {"tool_name": "...", "tool_type": "perception|logic|operation", "input": {...}}
- outputs: each item MUST be {"tool_name": "DrawBox|AddText|Plot", "tool_type": "operation", "input": {...}}
- Do NOT use fields like "action", "bbox" at the top level of outputs. Put everything inside "input".
- Always include required inputs:
  - DrawBox: input must include image, bbox, annotation(optional)
  - DrawMask: input must include image and one of {mask_path, polygon, bbox}
  - AddText: input must include image, text, position (e.g., "top-left")
  - Plot: input must include command
  - Solver/Plot: the command MUST define a Python function named solution()
    - Solver: solution() returns a string answer (no prints)
    - Plot: solution() returns a matplotlib figure object

Detection/segmentation guidance:
- ObjectDetection should be class-agnostic (no category filter) to detect all objects first.
- Only use class-specific segmentation when the question asks about a class (e.g., airplane), then SegmentObjectPixels text should be that class.
- If the target class is ambiguous, prefer a generic segmentation target (text="object") instead of forcing a wrong class.

Example outputs:
{
  "outputs": [
    {"tool_name": "DrawBox", "tool_type": "operation", "input": {"image": "...", "bbox": [x1,y1,x2,y2], "annotation": "airplane"}},
    {"tool_name": "DrawMask", "tool_type": "operation", "input": {"image": "...", "bbox": [x1,y1,x2,y2], "color": "red", "alpha": 0.4}},
    {"tool_name": "AddText", "tool_type": "operation", "input": {"image": "...", "text": "airplane", "position": "top-left"}}
  ]
}
"""


SEG_LABEL_KEYWORDS = {
    "airplane": ("airplane", "plane", "aircraft", "jet"),
    "building": ("building", "house", "terminal", "hangar", "warehouse"),
    "runway": ("runway", "taxiway"),
    "road": ("road", "highway", "street"),
    "ship": ("ship", "boat", "vessel", "dock", "harbor", "port"),
    "vehicle": ("vehicle", "car", "truck", "bus", "parking"),
}


def _infer_seg_hint(question: str | None):
    if not question:
        return {
            "primary": None,
            "candidates": [],
            "confidence": 0.0,
            "ambiguous": True,
            "reason": "missing_query",
        }

    query = question.lower()
    matches = []
    for label, keywords in SEG_LABEL_KEYWORDS.items():
        hit_keywords = [k for k in keywords if k in query]
        if hit_keywords:
            matches.append((label, hit_keywords))

    if not matches:
        return {
            "primary": None,
            "candidates": [],
            "confidence": 0.0,
            "ambiguous": True,
            "reason": "no_keyword_match",
        }

    matches.sort(key=lambda item: len(item[1]), reverse=True)
    candidates = [label for label, _ in matches]
    primary = candidates[0]

    if len(matches) == 1:
        confidence = 0.85 if len(matches[0][1]) >= 2 else 0.72
        ambiguous = False
    else:
        top_hits = len(matches[0][1])
        total_hits = sum(len(hit_keywords) for _, hit_keywords in matches)
        confidence = (top_hits / total_hits) if total_hits > 0 else 0.0
        ambiguous = True

    return {
        "primary": primary,
        "candidates": candidates[:3],
        "confidence": round(float(confidence), 3),
        "ambiguous": ambiguous,
        "reason": "keyword_match",
    }


def _default_seg_text(question: str | None):
    hint = _infer_seg_hint(question)
    if hint.get("primary") and not hint.get("ambiguous"):
        return hint.get("primary"), hint
    return "object", hint


def _api_url():
    base = os.getenv("REASONER_BASE_URL", os.getenv("LLM_BASE_URL", "http://127.0.0.1:8001/v1"))
    return base.rstrip("/") + "/chat/completions"


def _retry_session():
    """创建带自动重试的 requests session（应对网络抖动/超时）"""
    session = requests.Session()
    retries = Retry(
        total=3,
        backoff_factor=10,  # 10s, 20s, 40s
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


# DashScope data-uri 限制 10MB，留余量设 9.5MB
MAX_DATA_URI_BYTES = int(os.getenv("MAX_IMAGE_BYTES", 9_500_000))


def _compress_image(path, max_bytes=MAX_DATA_URI_BYTES):
    """如果图片 base64 编码后超过 max_bytes，逐步缩放+压缩直到满足限制"""
    from PIL import Image

    data = Path(path).read_bytes()
    b64_len = len(base64.b64encode(data))

    if b64_len <= max_bytes:
        return data, "image/jpeg"

    img = Image.open(path)
    if img.mode == "RGBA":
        img = img.convert("RGB")

    # 先尝试只压缩质量
    for quality in (90, 80, 70, 60):
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=quality)
        if len(base64.b64encode(buf.getvalue())) <= max_bytes:
            return buf.getvalue(), "image/jpeg"

    # 质量压缩不够，逐步缩放（每次缩到 80%）
    scale = 0.8
    for _ in range(10):
        w, h = int(img.width * scale), int(img.height * scale)
        if w < 100 or h < 100:
            break
        resized = img.resize((w, h), Image.LANCZOS)
        buf = io.BytesIO()
        resized.save(buf, format="JPEG", quality=70)
        if len(base64.b64encode(buf.getvalue())) <= max_bytes:
            return buf.getvalue(), "image/jpeg"
        scale *= 0.8

    # 兜底
    buf = io.BytesIO()
    resized.save(buf, format="JPEG", quality=60)
    return buf.getvalue(), "image/jpeg"


def _encode_image(path):
    mime, _ = mimetypes.guess_type(path)
    if not mime:
        mime = "image/jpeg"
    raw_data = Path(path).read_bytes()
    b64_raw = base64.b64encode(raw_data)

    if len(b64_raw) > MAX_DATA_URI_BYTES:
        data, mime = _compress_image(path)
        b64 = base64.b64encode(data).decode("utf-8")
        return f"data:{mime};base64,{b64}"

    b64 = b64_raw.decode("utf-8")
    return f"data:{mime};base64,{b64}"


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
        json.dumps(safe, ensure_ascii=False, indent=2),
        encoding="utf-8"
    )


def _build_messages(question, image, pre_image, post_image, user_payload):
    image_mode = os.getenv("REASONER_IMAGE_MODE", os.getenv("LLM_IMAGE_MODE", "path")).lower()
    if image_mode == "base64":
        detail = os.getenv("REASONER_IMAGE_DETAIL", os.getenv("LLM_IMAGE_DETAIL"))
        image_item = {"type": "image_url", "image_url": {"url": _encode_image(image)}}
        if detail:
            image_item["image_url"]["detail"] = detail
        content = [image_item]
        if pre_image and post_image:
            pre_item = {"type": "image_url", "image_url": {"url": _encode_image(pre_image)}}
            post_item = {"type": "image_url", "image_url": {"url": _encode_image(post_image)}}
            if detail:
                pre_item["image_url"]["detail"] = detail
                post_item["image_url"]["detail"] = detail
            content.append(pre_item)
            content.append(post_item)
        content.append({"type": "text", "text": question})
        return [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": content}
        ]
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": json.dumps(user_payload, ensure_ascii=False)}
    ]


def _load_tool_descs():
    merged = {}
    skip_knowledge = os.getenv("DISABLE_RS_KNOWLEDGE", "").strip().lower() in ("1", "true", "yes")
    for name in ("tool_descs_rs.json", "tool_descs_rs_knowledge.json"):
        if skip_knowledge and "knowledge" in name:
            continue
        path = Path(__file__).with_name(name)
        if path.exists():
            data = json.loads(path.read_text(encoding="utf-8-sig"))
            merged.update(data)
    return merged


def generate_plan(question, image, tools, pre_image=None, post_image=None, planner_plan=None, reflector_feedback=None):
    api_key = os.getenv("REASONER_API_KEY", os.getenv("LLM_API_KEY", ""))
    model = os.getenv("REASONER_MODEL", os.getenv("LLM_MODEL", "Qwen3-VL-30B-A3B-Thinking"))
    enable_thinking = _env_bool("REASONER_ENABLE_THINKING")
    thinking_budget = os.getenv("REASONER_THINKING_BUDGET")

    user_payload = {
        "question": question,
        "image": image,
        "pre_image": pre_image,
        "post_image": post_image,
        "tools": tools,
        "output_requirements": {
            "text": True,
            "annotated_image": True,
            "plot": True
        },
        "planner_plan": planner_plan,
        "reflector_feedback": reflector_feedback,
        "tool_descs": _load_tool_descs()
    }

    messages = _build_messages(question, image, pre_image, post_image, user_payload)

    if pre_image and post_image:
        prompt_path = os.path.join(os.path.dirname(__file__), "prompts", "change_prompt.md")
        if os.path.exists(prompt_path):
            change_prompt = Path(prompt_path).read_text(encoding="utf-8")
            messages.insert(1, {"role": "system", "content": change_prompt})

    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    payload = {
        "model": model,
        "messages": messages,
        "temperature": 0.2
    }
    if enable_thinking is not None:
        payload["enable_thinking"] = enable_thinking
    if thinking_budget:
        try:
            payload["thinking_budget"] = int(thinking_budget)
        except Exception:
            pass

    _maybe_dump_payload("reasoner", payload)
    session = _retry_session()
    resp = session.post(_api_url(), json=payload, headers=headers, timeout=300)
    if not resp.ok:
        raise RuntimeError(f"Reasoner API error {resp.status_code}: {resp.text}")
    content = resp.json()["choices"][0]["message"]["content"]

    # Extract JSON from model output
    try:
        plan = json.loads(content)
    except Exception:
        # best-effort: find first JSON block
        start = content.find("{")
        end = content.rfind("}")
        if start >= 0 and end > start:
            raw_json = content[start:end + 1]
            try:
                plan = json.loads(raw_json)
            except json.JSONDecodeError:
                # 尝试修复常见 JSON 问题：尾逗号、注释等
                import re
                cleaned = re.sub(r',\s*([}\]])', r'\1', raw_json)  # 去尾逗号
                cleaned = re.sub(r'//.*?\n', '\n', cleaned)  # 去行注释
                try:
                    plan = json.loads(cleaned)
                except json.JSONDecodeError:
                    # 最后兜底：返回空 plan 而不是崩溃
                    plan = {"steps": [], "final_answer": "", "outputs": [],
                            "_parse_error": f"Failed to parse JSON from model output (len={len(content)})"}
        else:
            plan = {"steps": [], "final_answer": "", "outputs": [],
                    "_parse_error": f"No JSON found in model output (len={len(content)})"}

    seg_text_default, seg_hint = _default_seg_text(question)

    def _normalize_step(step):
        if "tool_name" not in step:
            step["tool_name"] = step.get("tool") or step.get("action")
        if "tool_type" not in step:
            step["tool_type"] = step.get("tool_type") or step.get("type") or "perception"
        tool_input = step.get("input")
        if tool_input is None:
            tool_input = step.get("tool_input") or step.get("action_input") or {}
        if isinstance(tool_input, str):
            tool_input = {"text": tool_input}
        if not isinstance(tool_input, dict):
            tool_input = {}
        tool_name = step.get("tool_name")
        if tool_name in {
            "ImageDescription", "ObjectDetection", "SegmentObjectPixels", "TextToBbox",
            "ChangeDetection", "CountGivenObject", "SegmentGivenObject", "SegmentChange",
            "ClassifyScene"
        } and "image" not in tool_input:
            tool_input["image"] = image
        if tool_name == "SegmentObjectPixels" and "text" not in tool_input:
            tool_input["text"] = seg_text_default
            if seg_hint.get("candidates") and seg_hint.get("ambiguous"):
                tool_input["candidate_labels"] = seg_hint.get("candidates")
        step["input"] = tool_input
        return step

    def _normalize_output(out):
        if "tool_name" not in out:
            out["tool_name"] = out.get("action") or out.get("tool")
        if "tool_type" not in out:
            out["tool_type"] = "operation"
        tool_input = out.get("input")
        if tool_input is None:
            tool_input = {}
        if not isinstance(tool_input, dict):
            tool_input = {}
        for key in ("image", "bbox", "text", "mask_path", "polygon"):
            if key in out and key not in tool_input:
                tool_input[key] = out[key]
        if "image" not in tool_input:
            tool_input["image"] = image
        if out.get("tool_name") == "AddText" and "position" not in tool_input:
            tool_input["position"] = "lt"
        out["input"] = tool_input
        return out

    plan["steps"] = [_normalize_step(s) for s in plan.get("steps", []) if isinstance(s, dict)]
    plan["outputs"] = [_normalize_output(o) for o in plan.get("outputs", []) if isinstance(o, dict)]

    steps = plan.get("steps", [])
    has_det = any(s.get("tool_name") == "ObjectDetection" for s in steps)
    has_seg = any(s.get("tool_name") == "SegmentObjectPixels" for s in steps)
    prepend = []
    if not has_det:
        prepend.append({
            "tool_name": "ObjectDetection",
            "tool_type": "perception",
            "input": {"image": image}
        })
    if not has_seg:
        prepend.append({
            "tool_name": "SegmentObjectPixels",
            "tool_type": "perception",
            "input": {
                "image": image,
                "text": seg_text_default,
                "flag": False,
                **({"candidate_labels": seg_hint.get("candidates")} if seg_hint.get("candidates") and seg_hint.get("ambiguous") else {}),
            }
        })
    if prepend:
        plan["steps"] = prepend + steps
    return plan
