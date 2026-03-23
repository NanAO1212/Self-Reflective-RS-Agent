import base64
import io
import json
import mimetypes
import os
from pathlib import Path
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


SYSTEM_PROMPT = """You are a planning agent for remote sensing tasks.
Given image(s) and a natural language question, output a high-level plan with task classification.

Return JSON:
{
  "task_type": "counting|area|distance|change_detection|attribute|scene_description|multi_task|other",
  "subtasks": [
    {"goal": "...", "tool_category": "Perception|Logic|Operation"}
  ],
  "notes": "optional"
}

Classification hints:
- counting: asks "how many", "count", "number of"
- area: asks "area", "coverage", "size"
- distance: asks "distance", "how far", "spacing"
- change_detection: mentions change, before/after, temporal, time1/time2
- attribute: asks color/shape/type/material or region attribute
- scene_description: asks to describe the scene
- multi_task: combines two or more above
- other: unclear

Rules:
- Use tool_category only from: Perception, Logic, Operation.
- If change over time is asked, include a subtask for temporal change (Perception).
- If measurements are needed, include a Logic subtask.
- If visual output is needed, include an Operation subtask.
"""


def _api_url():
    base = os.getenv("PLANNER_BASE_URL", os.getenv("LLM_BASE_URL", "http://127.0.0.1:8001/v1"))
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


# DashScope data-uri 限制 10MB，留点余量设 9.5MB
MAX_DATA_URI_BYTES = int(os.getenv("MAX_IMAGE_BYTES", 9_500_000))


def _compress_image(path, max_bytes=MAX_DATA_URI_BYTES):
    """如果图片 base64 编码后超过 max_bytes，逐步缩放+压缩直到满足限制"""
    from PIL import Image

    data = Path(path).read_bytes()
    b64_len = len(base64.b64encode(data))  # base64 大约膨胀 4/3

    if b64_len <= max_bytes:
        return data, "image/jpeg"

    img = Image.open(path)
    if img.mode == "RGBA":
        img = img.convert("RGB")

    # 先尝试只压缩质量，不缩放
    for quality in (90, 80, 70, 60):
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=quality)
        if len(base64.b64encode(buf.getvalue())) <= max_bytes:
            return buf.getvalue(), "image/jpeg"

    # 质量压缩不够，逐步缩放（每次缩小到 80%）
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

    # 兜底：返回最后一次压缩结果
    buf = io.BytesIO()
    resized.save(buf, format="JPEG", quality=60)
    return buf.getvalue(), "image/jpeg"


def _encode_image(path):
    mime, _ = mimetypes.guess_type(path)
    if not mime:
        mime = "image/jpeg"
    raw_data = Path(path).read_bytes()
    b64_raw = base64.b64encode(raw_data)

    # 检查是否超限，超限则压缩
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


def _build_messages(question, image, pre_image, post_image):
    image_mode = os.getenv("PLANNER_IMAGE_MODE", "path").lower()
    if image_mode == "base64":
        detail = os.getenv("PLANNER_IMAGE_DETAIL", os.getenv("LLM_IMAGE_DETAIL"))
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

    user_payload = {
        "question": question,
        "image": image,
        "pre_image": pre_image,
        "post_image": post_image,
        "tool_descs": _load_tool_descs()
    }
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": json.dumps(user_payload, ensure_ascii=False)}
    ]


def plan(question, image, pre_image=None, post_image=None):
    api_key = os.getenv("PLANNER_API_KEY", os.getenv("LLM_API_KEY", ""))
    model = os.getenv("PLANNER_MODEL", "Qwen3-VL-30B-A3B-Thinking")
    enable_thinking = _env_bool("PLANNER_ENABLE_THINKING")
    thinking_budget = os.getenv("PLANNER_THINKING_BUDGET")

    messages = _build_messages(question, image, pre_image, post_image)

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

    _maybe_dump_payload("planner", payload)
    session = _retry_session()
    resp = session.post(_api_url(), json=payload, headers=headers, timeout=300)
    if not resp.ok:
        raise RuntimeError(f"Planner API error {resp.status_code}: {resp.text}")
    content = resp.json()["choices"][0]["message"]["content"]

    try:
        return json.loads(content)
    except Exception:
        start = content.find("{")
        end = content.rfind("}")
        if start >= 0 and end > start:
            return json.loads(content[start:end + 1])
        raise
