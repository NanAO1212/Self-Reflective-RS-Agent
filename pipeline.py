import json
import os
import time
from pathlib import Path

from parser import load_rules, wrap_tool_result
from verifier import verify
from reflector import reflect, reflect_and_patch, apply_patches
from tool_registry import ToolRegistry
from adapters import build_default_registry, call_tool
from reasoner import generate_plan
from planner import plan as plan_task


PERCEPTION_TOOLS = {
    "TextToBbox", "ObjectDetection", "CountGivenObject", "SegmentObjectPixels",
    "RegionAttributeDescription", "ImageDescription", "ChangeDetection", "OCR",
}


SEG_LABEL_KEYWORDS = {
    "airplane": ("airplane", "plane", "aircraft", "jet", "飞机", "航空器"),
    "building": ("building", "house", "terminal", "hangar", "warehouse", "建筑", "房屋", "航站楼", "机库"),
    "runway": ("runway", "taxiway", "跑道", "滑行道", "机场跑道"),
    "road": ("road", "highway", "street", "道路", "公路", "街道", "高速"),
    "ship": ("ship", "boat", "vessel", "dock", "harbor", "port", "船", "轮船", "港口"),
    "vehicle": ("vehicle", "car", "truck", "bus", "parking", "车辆", "汽车", "卡车", "巴士", "停车"),
}


def _get_task_query(task):
    if not isinstance(task, dict):
        return ""
    for key in ("question", "query"):
        value = task.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


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


def _infer_seg_text(question: str | None):
    return _infer_seg_hint(question).get("primary")


def _merge_candidate_labels(seg_hint, detections, max_labels=3):
    query_labels = []
    if seg_hint.get("primary"):
        query_labels.append(seg_hint["primary"])
    for label in seg_hint.get("candidates") or []:
        if label not in query_labels:
            query_labels.append(label)

    det_labels = _select_top_labels(detections, max_labels=max_labels, score_threshold=0.5)

    chosen = []
    seen = set()

    def _add(label):
        if not label:
            return
        key = str(label).strip().lower()
        if not key or key in seen:
            return
        seen.add(key)
        chosen.append(str(label).strip())

    det_map = {(label or "").strip().lower(): label for label in det_labels}
    for query_label in query_labels:
        key = query_label.lower()
        if key in det_map:
            _add(det_map[key])

    for query_label in query_labels:
        _add(query_label)
    for det_label in det_labels:
        _add(det_label)

    return chosen[:max_labels]


def _clamp_bbox(bbox, image_size):
    if not image_size:
        return bbox
    try:
        x1, y1, x2, y2 = [float(v) for v in bbox.strip("()").split(",")]
    except Exception:
        return bbox
    w, h = image_size
    x1 = max(0, min(x1, w))
    x2 = max(0, min(x2, w))
    y1 = max(0, min(y1, h))
    y2 = max(0, min(y2, h))
    if x2 <= x1:
        x2 = min(w, x1 + 1)
    if y2 <= y1:
        y2 = min(h, y1 + 1)
    return f"({x1:.0f}, {y1:.0f}, {x2:.0f}, {y2:.0f})"


def _next_input(tool_name, tool_input, error_types, image_size):
    new_input = dict(tool_input)
    if "parameter" in error_types:
        if "bbox" in new_input:
            new_input["bbox"] = _clamp_bbox(new_input["bbox"], image_size)
    if "perception" in error_types:
        if tool_name == "TextToBbox":
            new_input["top1"] = False
        if tool_name == "SegmentObjectPixels":
            new_input["flag"] = False
    return new_input


def _call_with_evidence(tool, tool_name, tool_type, tool_input, rules, evidence_rounds):
    raw_outputs = []
    records = []
    for _ in range(max(1, evidence_rounds)):
        raw_output = call_tool(tool, tool_name, tool_input)
        record = wrap_tool_result(tool_name, tool_type, tool_input, raw_output, rules)
        raw_outputs.append(raw_output)
        records.append(record)

    primary = records[0]
    if evidence_rounds > 1:
        primary["evidence"] = {
            "rounds": evidence_rounds,
            "records": [{"parsed": r.get("parsed"), "spatial": r.get("spatial")} for r in records],
            "raw_outputs": [str(o) for o in raw_outputs],
        }
    return primary, raw_outputs


def _select_top_labels(detections, max_labels=3, score_threshold=0.6):
    if not detections:
        return []
    best = {}
    for det in detections:
        label = (det.get("label") or "").strip()
        if not label:
            continue
        score = det.get("score", 0)
        if label not in best or score > best[label]:
            best[label] = score
    ranked = sorted(best.items(), key=lambda x: x[1], reverse=True)
    chosen = [label for label, score in ranked if score >= score_threshold]
    if not chosen and ranked:
        chosen = [ranked[0][0]]
    return chosen[:max_labels]


def _maybe_save_mask(raw_output, output_dir, tag):
    try:
        from PIL import Image
    except Exception:
        Image = None
    mask_path = None
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if hasattr(raw_output, "save"):
        mask_path = str(output_dir / f"{tag}.png")
        try:
            raw_output.save(mask_path)
            return mask_path
        except Exception:
            mask_path = None

    if isinstance(raw_output, str):
        try:
            p = Path(raw_output)
            if p.exists() and p.suffix.lower() in {".png", ".jpg", ".jpeg"}:
                return str(p)
        except Exception:
            pass

    if Image is not None:
        try:
            import numpy as np

            if isinstance(raw_output, np.ndarray):
                arr = raw_output
                if arr.dtype != np.uint8:
                    arr = (arr > 0).astype(np.uint8) * 255
                if arr.ndim == 2:
                    img = Image.fromarray(arr, mode="L")
                else:
                    img = Image.fromarray(arr)
                mask_path = str(output_dir / f"{tag}.png")
                img.save(mask_path)
                return mask_path
        except Exception:
            pass

    return None


def _try_segment_mask(tool, tool_input, output_dir, tag, bboxes=None):
    if tool is None:
        return None
    if not hasattr(tool, "grounding") or not hasattr(tool, "sam_predictor"):
        return None
    image_path = tool_input.get("image")
    if not image_path:
        return None
    try:
        from PIL import Image
        import numpy as np
        import torch
    except Exception:
        return None

    try:
        image = np.array(Image.open(image_path).convert("RGB"))
        if bboxes:
            boxes_filt = torch.tensor(bboxes, dtype=torch.float32)
        else:
            text = str(tool_input.get("text") or "").strip()
            if not text or text.lower() == "object":
                return None
            results = tool.grounding(
                inputs=image[:, :, ::-1],
                texts=text,
                no_save_vis=True,
                return_datasamples=True,
            )
            preds = results["predictions"][0].pred_instances
            boxes_filt = preds.bboxes
            scores = preds.scores
            boxes_filt = boxes_filt[scores > 0.4]
        if boxes_filt.shape[0] == 0:
            return None
        masks = tool.get_mask_with_boxes(image, boxes_filt)
        if masks == []:
            return None
        union = np.zeros(image.shape[:2], dtype=np.uint8)
        for mask in masks:
            mask_arr = mask[0].detach().cpu().numpy().astype(bool)
            union[mask_arr] = 255
        return _maybe_save_mask(union, output_dir, tag)
    except Exception:
        return None

def _execute_outputs(outputs, registry, output_dir, default_image=None, default_text=None):
    results = []
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for out in outputs or []:
        tool_name = out.get("tool_name")
        if not tool_name:
            results.append({
                "tool_name": None,
                "tool_type": out.get("tool_type", "operation"),
                "input": out.get("input", {}),
                "raw_output": "SKIPPED: missing tool_name",
                "saved_path": None
            })
            continue
        tool_input = out.get("input", {}) or {}
        if tool_name in ("DrawBox", "AddText", "Plot", "DrawMask"):
            if default_image and "image" not in tool_input:
                tool_input["image"] = default_image
            if default_image and isinstance(tool_input.get("image"), str):
                try:
                    if not Path(tool_input["image"]).exists():
                        tool_input["image"] = default_image
                except Exception:
                    tool_input["image"] = default_image
        if tool_name == "AddText" and "text" not in tool_input and default_text:
            tool_input["text"] = default_text
        if tool_name == "AddText" and "position" not in tool_input:
            tool_input["position"] = "lt"
        tool_type = out.get("tool_type", "operation")
        tool = registry.get(tool_name)
        try:
            raw_output = call_tool(tool, tool_name, tool_input)
        except BaseException as e:
            raw_output = f"ERROR:{tool_name}: {e}"
        saved_path = None
        try:
            if hasattr(raw_output, "save"):
                saved_path = str(output_dir / f"{tool_name}_{len(results)+1}.png")
                raw_output.save(saved_path)
            elif hasattr(raw_output, "to_path"):
                saved_path = raw_output.to_path()
            elif isinstance(raw_output, str) and Path(raw_output).exists() and Path(raw_output).suffix.lower() in (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"):
                # DrawBox/AddText 等工具返回的是生成图片的路径字符串，复制到 output_dir
                import shutil
                dest = str(output_dir / f"{tool_name}_{len(results)+1}{Path(raw_output).suffix}")
                shutil.copy2(raw_output, dest)
                saved_path = dest
        except Exception:
            saved_path = None
        results.append({
            "tool_name": tool_name,
            "tool_type": tool_type,
            "input": tool_input,
            "raw_output": str(raw_output),
            "saved_path": saved_path
        })
    return results


def _normalize_outputs(outputs):
    normalized = []
    for out in outputs or []:
        if not isinstance(out, dict):
            continue
        tool_name = out.get("tool_name") or out.get("action")
        if not tool_name:
            normalized.append(out)
            continue
        tool_input = dict(out.get("input", {}) or {})
        if "image" in out and "image" not in tool_input:
            tool_input["image"] = out.get("image")
        if "bbox" in out and "bbox" not in tool_input:
            tool_input["bbox"] = out.get("bbox")
        if "text" in out and "text" not in tool_input:
            tool_input["text"] = out.get("text")
        if tool_name == "AddText" and "position" not in tool_input:
            tool_input["position"] = "lt"
        normalized.append({
            "tool_name": tool_name,
            "tool_type": out.get("tool_type", "operation"),
            "input": tool_input,
        })
    return normalized


# AddText position 别名映射（Reasoner 常用自然语言描述，agentlego 只接受两字母缩写）
_POSITION_ALIASES = {
    "top-left": "lt", "top-center": "mt", "top-right": "rt",
    "center-left": "lm", "center": "mm", "center-right": "rm",
    "bottom-left": "lb", "bottom-center": "mb", "bottom-right": "rb",
    "top left": "lt", "top center": "mt", "top right": "rt",
    "bottom left": "lb", "bottom center": "mb", "bottom right": "rb",
    "left": "lm", "right": "rm", "top": "mt", "bottom": "mb",
    "topleft": "lt", "topcenter": "mt", "topright": "rt",
    "bottomleft": "lb", "bottomcenter": "mb", "bottomright": "rb",
}


def _patch_outputs_from_results(outputs, step_logs, default_image):
    """用步骤执行的真实结果替换 outputs 中的占位符，使 E2E 可视化能成功执行"""
    import re as _re

    # 收集所有真实结果
    all_bboxes = []       # 来自 TextToBbox
    all_detections = []   # 来自 ObjectDetection
    mask_paths = []       # 来自 SegmentObjectPixels
    calc_results = []     # 来自 Calculator/Solver

    for s in step_logs:
        parsed = s.get("parsed") or {}
        tool = s.get("tool", "")
        if tool == "TextToBbox":
            for b in parsed.get("bboxes", []):
                all_bboxes.append({
                    "bbox": b["bbox_px"],
                    "text": (s.get("tool_input") or {}).get("text", ""),
                })
        if tool == "ObjectDetection":
            for d in parsed.get("detections", []):
                all_detections.append(d)
        if tool == "SegmentObjectPixels":
            mp = parsed.get("mask_path")
            if mp:
                mask_paths.append(mp)
            mps = parsed.get("mask_paths") or []
            mask_paths.extend(mps)
        if tool in ("Calculator", "Solver"):
            val = parsed.get("value") or s.get("tool_output")
            if val and not str(val).startswith("UNAVAILABLE"):
                calc_results.append(str(val))

    bbox_idx = 0
    det_idx = 0
    mask_idx = 0

    for out in outputs or []:
        inp = out.get("input") or {}
        tn = out.get("tool_name", "")

        # DrawBox: 替换占位符 bbox
        if tn == "DrawBox":
            bbox = inp.get("bbox", "")
            is_placeholder = isinstance(bbox, str) and not any(c.isdigit() for c in str(bbox))
            if is_placeholder:
                if bbox_idx < len(all_bboxes):
                    inp["bbox"] = all_bboxes[bbox_idx]["bbox"]
                    bbox_idx += 1
                elif det_idx < len(all_detections):
                    inp["bbox"] = all_detections[det_idx]["bbox_px"]
                    inp.setdefault("annotation", all_detections[det_idx].get("label", ""))
                    det_idx += 1

        # DrawMask: 替换假 mask 路径
        if tn == "DrawMask":
            mp = inp.get("mask_path", "")
            if isinstance(mp, str) and mp and not Path(mp).exists():
                if mask_idx < len(mask_paths):
                    inp["mask_path"] = mask_paths[mask_idx]
                    mask_idx += 1

        # AddText: position 映射 + 文本占位符替换
        if tn == "AddText":
            pos = inp.get("position", "")
            if isinstance(pos, str):
                mapped = _POSITION_ALIASES.get(pos.lower().strip())
                if mapped:
                    inp["position"] = mapped
                elif pos and pos not in ("lt", "lm", "lb", "mt", "mm", "mb", "rt", "rm", "rb"):
                    inp["position"] = "lt"  # 兜底默认左上角
            text = inp.get("text", "")
            if calc_results and isinstance(text, str) and "X" in text:
                inp["text"] = text.replace("X", calc_results[0], 1)

        # Plot: 剥离 markdown 代码块 + 确保 def solution()
        if tn == "Plot":
            cmd = inp.get("command", "")
            if isinstance(cmd, str):
                cmd = _re.sub(r'^```\w*\n?', '', cmd)
                cmd = _re.sub(r'\n?```$', '', cmd)
                cmd = cmd.strip()
                if 'def solution()' not in cmd:
                    lines = cmd.split('\n')
                    indented = '\n'.join('    ' + l for l in lines)
                    cmd = f"def solution():\n{indented}\n    return fig"
                inp["command"] = cmd

        # 确保 image 字段存在且有效
        if tn in ("DrawBox", "AddText", "DrawMask") and default_image:
            if "image" not in inp:
                inp["image"] = default_image
            elif isinstance(inp.get("image"), str):
                try:
                    if not Path(inp["image"]).exists():
                        inp["image"] = default_image
                except Exception:
                    inp["image"] = default_image


def run_pipeline(task, rules_path, max_retries=1, image_size=None, output_dir="outputs", evidence_rounds=1, skip_seg_fallback=False):
    rules = load_rules(rules_path)
    registry = ToolRegistry()
    build_default_registry(registry)

    context = {
        "scene_text": None,
        "gsd_m_per_px": task.get("gsd_m_per_px"),
        "history": {},
    }

    logs = []
    step_logs = []
    last_detections = []
    failure_history = []
    outputs = list(task.get("outputs", []) or [])
    pipeline_warnings = []
    warning_set = set()
    task_query = _get_task_query(task)
    seg_hint = _infer_seg_hint(task_query)

    if not task_query:
        warning = "Missing question/query text; segmentation target inference may be ambiguous."
        pipeline_warnings.append(warning)
        warning_set.add(warning)

    for step_idx, step in enumerate(task["steps"]):
        tool_name = step["tool_name"]
        tool_type = step["tool_type"]
        tool_input = step["input"]
        step_warnings = []
        print(f"    [Pipeline] Executing step {step_idx}/{len(task['steps'])}: {tool_name} ({tool_type}) ...", end=" ", flush=True)
        if isinstance(tool_input, str):
            try:
                tool_input = json.loads(tool_input)
            except Exception:
                tool_input = {"text": tool_input}
        if isinstance(tool_input, dict):
            if ("image" not in tool_input or not isinstance(tool_input.get("image"), str)) and task.get("image"):
                tool_input["image"] = task["image"]
            if isinstance(tool_input.get("image"), str):
                image_path = tool_input["image"]
                if image_path and task.get("image"):
                    try:
                        if not Path(image_path).exists():
                            tool_input["image"] = task["image"]
                    except Exception:
                        tool_input["image"] = task["image"]
            if tool_name == "ChangeDetection":
                if "pre_image" not in tool_input and task.get("pre_image"):
                    tool_input["pre_image"] = task["pre_image"]
                if "post_image" not in tool_input and task.get("post_image"):
                    tool_input["post_image"] = task["post_image"]
            if tool_name == "ObjectDetection":
                tool_input.pop("text", None)
                tool_input.pop("classes", None)
                tool_input.pop("category", None)
            if tool_name == "SegmentObjectPixels" and "text" not in tool_input:
                inferred = seg_hint.get("primary")
                if inferred and not seg_hint.get("ambiguous"):
                    tool_input["text"] = inferred
                else:
                    tool_input["text"] = "object"
                    if seg_hint.get("candidates"):
                        step_warnings.append(
                            "Ambiguous segmentation target in query; using detection-assisted candidates: "
                            + ", ".join(seg_hint.get("candidates", []))
                        )
                    else:
                        step_warnings.append(
                            "No segmentation target keyword detected in query; using detection-assisted fallback."
                        )

        retries = 0
        retry_reflections = []  # 记录每轮 retry 的反思信息
        while True:
            tool = registry.get(tool_name)
            if tool_type == "perception" or tool_name in PERCEPTION_TOOLS:
                record, raw_outputs = _call_with_evidence(
                    tool, tool_name, tool_type, tool_input, rules, evidence_rounds
                )
                raw_output = raw_outputs[0]
            else:
                raw_output = call_tool(tool, tool_name, tool_input)
                record = wrap_tool_result(tool_name, tool_type, tool_input, raw_output, rules)
            print(f"done (output={str(raw_output)[:80]})", flush=True)

            record["context"] = {
                "scene_text": context.get("scene_text"),
                "gsd_m_per_px": context.get("gsd_m_per_px"),
                "history": context.get("history"),
            }
            if tool_name == "ObjectDetection":
                last_detections = record.get("parsed", {}).get("detections", []) or []
            if tool_name == "SegmentObjectPixels":
                if step_warnings:
                    record.setdefault("meta", {}).setdefault("warnings", []).extend(step_warnings)
                    for warning in step_warnings:
                        if warning not in warning_set:
                            pipeline_warnings.append(warning)
                            warning_set.add(warning)
                mask_path = _maybe_save_mask(raw_output, output_dir, f"seg_mask_{len(step_logs)+1}")
                if not mask_path and not skip_seg_fallback:
                    target = tool_input.get("text")
                    candidate_labels = _merge_candidate_labels(seg_hint, last_detections)
                    if (not target or target == "object") and candidate_labels:
                        colors = ["red", "green", "blue", "yellow", "cyan"]
                        mask_paths = []
                        for idx, label in enumerate(candidate_labels):
                            det_boxes = [
                                det.get("bbox_px")
                                for det in last_detections
                                if (det.get("label", "").lower() == label.lower())
                            ]
                            label_input = dict(tool_input)
                            label_input["text"] = label
                            label_mask_path = _try_segment_mask(
                                tool,
                                label_input,
                                output_dir,
                                f"seg_mask_{len(step_logs)+1}_{label}",
                                bboxes=det_boxes or None,
                            )
                            if label_mask_path:
                                mask_paths.append(label_mask_path)
                                outputs.append({
                                    "tool_name": "DrawMask",
                                    "tool_type": "operation",
                                    "input": {
                                        "image": task.get("image"),
                                        "mask_path": label_mask_path,
                                        "color": colors[idx % len(colors)],
                                        "alpha": 0.35,
                                    }
                                })
                        if mask_paths:
                            record.setdefault("parsed", {})["mask_paths"] = mask_paths
                            record.setdefault("spatial", {})["mask_paths"] = mask_paths
                        mask_path = None
                    else:
                        det_boxes = []
                        if target and last_detections:
                            for det in last_detections:
                                if det.get("label", "").lower() == target.lower():
                                    det_boxes.append(det.get("bbox_px"))
                        mask_path = _try_segment_mask(
                            tool,
                            tool_input,
                            output_dir,
                            f"seg_mask_{len(step_logs)+1}",
                            bboxes=det_boxes or None
                        )
                if mask_path:
                    record.setdefault("parsed", {})["mask_path"] = mask_path
                    record.setdefault("spatial", {})["mask_path"] = mask_path
                    outputs.append({
                        "tool_name": "DrawMask",
                        "tool_type": "operation",
                        "input": {
                            "image": task.get("image"),
                            "mask_path": mask_path,
                            "color": "red",
                            "alpha": 0.4,
                        }
                    })

            # 消融实验 B 开关：关闭验证
            _disable_verify = os.getenv("DISABLE_VERIFICATION", "").strip().lower() in ("1", "true", "yes")
            if _disable_verify:
                verdicts = []
            else:
                verdicts = verify(record, image_size=image_size)
            has_fail = any(v.get("status") == "fail" for v in verdicts)

            # 构建失败步骤记录供反思器使用
            failed_step_record = {
                "tool_name": tool_name,
                "tool_type": tool_type,
                "tool_input": tool_input,
                "tool_output": raw_output,
                "parsed": record.get("parsed"),
            }

            if has_fail and retries < max_retries:
                # 上下文感知反思：自动选择规则/LLM
                reflection = reflect_and_patch(
                    failed_step_record, verdicts,
                    context=context,
                    failure_history=failure_history,
                )
                # 确保 strategy 字段存在
                if "strategy" not in reflection:
                    reflection["strategy"] = "rule_based"
                actions = reflection.get("actions", [])
                patches = reflection.get("patches", {})
                should_replan = reflection.get("should_replan", False)

                # 记录本轮反思信息
                retry_reflections.append({
                    "retry_round": retries + 1,
                    "strategy": reflection.get("strategy", "rule_based"),
                    "diagnosis": reflection.get("diagnosis", ""),
                    "confidence": reflection.get("confidence", 0.0),
                    "actions": actions,
                    "failed_rules": [v.get("rule_id") for v in verdicts if v.get("status") == "fail"],
                    "patches_summary": {
                        "switch_tool": patches.get("switch_tool"),
                        "has_input_overrides": bool(patches.get("input_overrides")),
                        "has_retry_params": bool(patches.get("retry_params")),
                    },
                })

                # 记录本次失败到历史
                for v in verdicts:
                    if v.get("status") == "fail":
                        failure_history.append({
                            "tool": tool_name,
                            "rule_id": v.get("rule_id"),
                            "error_type": v.get("error_type"),
                            "details": v.get("details"),
                        })
            else:
                actions = reflect(verdicts)
                reflection = {
                    "actions": actions,
                    "patches": {},
                    "should_replan": False,
                    "strategy": "none",
                    "diagnosis": "",
                    "confidence": 1.0,
                }
                patches = {}
                should_replan = False

            log_entry = {
                "step": step,
                "record": record,
                "verdicts": verdicts,
                "actions": actions,
                "reflection": reflection,
                "retries": retries,
            }
            logs.append(log_entry)

            if not has_fail or retries >= max_retries:
                # 最终写入 step_log 时，合并反思历史
                final_reflection = dict(reflection)
                if retry_reflections:
                    # 用最后一次 retry 的 strategy 作为主 strategy
                    final_reflection["strategy"] = retry_reflections[-1].get("strategy", "rule_based")
                    final_reflection["retry_history"] = retry_reflections
                step_logs.append({
                    "step_id": len(step_logs) + 1,
                    "tool": tool_name,
                    "tool_type": tool_type,
                    "tool_input": tool_input,
                    "tool_output": raw_output,
                    "action": tool_name,
                    "action_input": tool_input,
                    "observation": raw_output,
                    "parsed": record.get("parsed"),
                    "verdicts": verdicts,
                    "actions": actions,
                    "reflection": final_reflection,
                    "warnings": (record.get("meta", {}) or {}).get("warnings", []),
                    "retries": retries,
                })
                break

            retries += 1
            # 应用反思 patches（结构化修复）
            tool_name, tool_input = apply_patches(tool_name, tool_input, patches)
            # 兜底：仍然执行原有的 _next_input 做 bbox clamp 等基础修复
            error_types = [v.get("error_type") for v in verdicts if v.get("status") == "fail"]
            tool_input = _next_input(tool_name, tool_input, error_types, image_size)
            time.sleep(0.2)

        # Update context after step
        if tool_name == "ImageDescription":
            text = record.get("parsed", {}).get("text")
            if isinstance(text, str) and text.strip():
                context["scene_text"] = text
        if tool_name == "RegionAttributeDescription":
            attr = (record.get("input", {}) or {}).get("attribute", "")
            text = record.get("parsed", {}).get("text")
            if isinstance(text, str) and text.strip() and attr in ("landcover", "scene", "context"):
                context["scene_text"] = text
        context["history"].setdefault(tool_name, []).append({
            "parsed": record.get("parsed"),
            "spatial": record.get("spatial"),
            "input": record.get("input"),
        })

    output = {
        "task_id": task.get("task_id"),
        "steps": step_logs,
        "raw_logs": logs,
        "final_answer": task.get("final_answer"),
        "outputs": outputs,
        "warnings": pipeline_warnings,
    }

    # Execute visualization outputs (DrawBox/AddText/Plot)
    # 先用真实步骤结果修补 outputs 中的占位符
    normalized_outputs = _normalize_outputs(outputs)
    _patch_outputs_from_results(normalized_outputs, step_logs, task.get("image"))

    output["output_artifacts"] = _execute_outputs(
        normalized_outputs,
        registry,
        output_dir,
        default_image=task.get("image"),
        default_text=output.get("final_answer"),
    )
    return output


def _collect_failures(step_logs):
    failures = []
    for step in step_logs:
        for v in step.get("verdicts", []):
            if v.get("status") == "fail":
                failures.append({
                    "tool": step.get("tool"),
                    "tool_type": step.get("tool_type"),
                    "rule_id": v.get("rule_id"),
                    "error_type": v.get("error_type"),
                    "details": v.get("details"),
                    "suggested_fix": v.get("suggested_fix"),
                })
    return failures


def run_from_query(question, image, rules_path, max_retries=1, image_size=None, pre_image=None, post_image=None, output_dir="outputs", max_replans=1, evidence_rounds=1):
    tools = [
        "TextToBbox", "ObjectDetection", "CountGivenObject", "SegmentObjectPixels",
        "RegionAttributeDescription", "ImageDescription", "ChangeDetection", "OCR",
        "Calculator", "Solver", "Plot", "DrawBox", "DrawMask", "AddText", "GoogleSearch"
    ]
    import sys
    print(f"  [E2E] Calling Planner ...", end=" ", flush=True)
    planner_plan = plan_task(question, image, pre_image=pre_image, post_image=post_image)
    print(f"OK (type={planner_plan.get('task_type', '?')})", flush=True)
    reflector_feedback = None

    for round_id in range(max_replans + 1):
        print(f"  [E2E] Round {round_id}: Calling Reasoner ...", end=" ", flush=True)
        plan = generate_plan(
            question,
            image,
            tools,
            pre_image=pre_image,
            post_image=post_image,
            planner_plan=planner_plan,
            reflector_feedback=reflector_feedback,
        )
        steps = plan.get("steps", [])
        print(f"OK ({len(steps)} steps, answer={str(plan.get('final_answer',''))[:60]})", flush=True)
        for si, s in enumerate(steps):
            print(f"    Step {si}: {s.get('tool_name','?')} ({s.get('tool_type','?')})", flush=True)
        task = {
            "task_id": plan.get("task_id", f"query_task_{round_id}"),
            "question": question,
            "query": question,
            "image": image,
            "pre_image": pre_image,
            "post_image": post_image,
            "steps": plan.get("steps", []),
            "final_answer": plan.get("final_answer", ""),
            "outputs": plan.get("outputs", []),
            "gsd_m_per_px": plan.get("gsd_m_per_px"),
        }
        output = run_pipeline(
            task,
            rules_path,
            max_retries=max_retries,
            image_size=image_size,
            output_dir=output_dir,
            evidence_rounds=evidence_rounds,
            skip_seg_fallback=True,  # E2E 模式跳过多标签分割 fallback
        )
        output["final_answer"] = plan.get("final_answer", output.get("final_answer"))
        output["outputs"] = plan.get("outputs", output.get("outputs", []))
        output["planner_plan"] = planner_plan
        output["round_id"] = round_id

        # 收集本轮所有失败，用 LLM 反思生成 replan 级反馈
        failures = _collect_failures(output.get("steps", []))
        if not failures or round_id >= max_replans:
            output["reflector_feedback"] = reflector_feedback
            return output

        # 用最后一个失败步骤的完整信息做 replan 级反思
        last_failed_step = {}
        for s in reversed(output.get("steps", [])):
            if any(v.get("status") == "fail" for v in s.get("verdicts", [])):
                last_failed_step = s
                break

        replan_reflection = reflect_and_patch(
            last_failed_step, failures,
            context={"scene_text": None, "gsd_m_per_px": None, "history": {}},
            failure_history=failures,
            use_llm=True,
        )

        reflector_feedback = {
            "failures": failures,
            "actions": replan_reflection.get("actions", []),
            "diagnosis": replan_reflection.get("diagnosis", ""),
            "patches": replan_reflection.get("patches", {}),
            "should_replan": replan_reflection.get("should_replan", True),
        }

    return output


def main():
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True, help="Task JSON file")
    ap.add_argument("--rules", default="tool_parsing_rules.json")
    ap.add_argument("--max_retries", type=int, default=1)
    ap.add_argument("--evidence_rounds", type=int, default=1)
    ap.add_argument("--image_size", default="")
    ap.add_argument("--out", default="pipeline_log.json")
    args = ap.parse_args()

    task = json.loads(Path(args.task).read_text(encoding="utf-8"))
    size = tuple(map(int, args.image_size.split(","))) if args.image_size else None
    output = run_pipeline(
        task,
        args.rules,
        max_retries=args.max_retries,
        image_size=size,
        evidence_rounds=args.evidence_rounds,
    )
    Path(args.out).write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
