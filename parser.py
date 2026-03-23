import json
import re
from datetime import datetime, timezone


def _utc_now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def load_rules(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)["rules"]


def parse_tool_output(tool_name, raw_output, rules):
    rule = rules.get(tool_name)
    if rule is None:
        return {"text": str(raw_output)}

    if rule.get("identity"):
        return {rule["output"]: str(raw_output)}

    if tool_name in ("TextToBbox", "ObjectDetection", "OCR"):
        line_pattern = re.compile(rule["line_pattern"])
        items = []
        for line in str(raw_output).splitlines():
            m = line_pattern.search(line.strip())
            if not m:
                continue
            gd = m.groupdict()
            bbox = [
                float(gd["x1"]),
                float(gd["y1"]),
                float(gd["x2"]),
                float(gd["y2"]),
            ]
            if tool_name == "TextToBbox":
                score = float(gd["score"]) * rule.get("score_scale", 1.0)
                items.append({"bbox_px": bbox, "score": score})
            elif tool_name == "ObjectDetection":
                score = float(gd["score"]) * rule.get("score_scale", 1.0)
                items.append({"label": gd["label"].strip(), "bbox_px": bbox, "score": score})
            else:
                items.append({"bbox_px": bbox, "text": gd["text"].strip()})

        if tool_name == "TextToBbox":
            return {"bboxes": items}
        if tool_name == "ObjectDetection":
            return {"detections": items}
        return {"items": items}

    if tool_name == "SegmentObjectPixels":
        split_pattern = re.compile(rule["split_pattern"])
        parts = [p for p in split_pattern.split(str(raw_output).strip()) if p]
        counts = []
        for p in parts:
            try:
                counts.append(int(float(p)))
            except Exception:
                continue
        if not counts:
            return {"text": str(raw_output)}
        parsed = {"pixel_counts": counts}
        if rule.get("sum"):
            parsed["sum"] = sum(counts)
        return parsed

    if tool_name == "CountGivenObject":
        m = re.search(rule["pattern"], str(raw_output))
        return {"count": int(m.group("count"))} if m else {"count": 0}

    return {"text": str(raw_output)}


def wrap_tool_result(tool_name, tool_type, tool_input, raw_output, rules, image_id=None):
    parsed = parse_tool_output(tool_name, raw_output, rules)
    spatial = {}
    if tool_name in ("TextToBbox", "ObjectDetection"):
        bboxes = [i["bbox_px"] for i in parsed.get("bboxes", [])] or [i["bbox_px"] for i in parsed.get("detections", [])]
        if bboxes:
            spatial["bboxes_px"] = bboxes
    if tool_name == "OCR":
        bboxes = [i["bbox_px"] for i in parsed.get("items", [])]
        if bboxes:
            spatial["bboxes_px"] = bboxes
    if tool_name == "SegmentObjectPixels":
        if "sum" in parsed:
            spatial["mask_area_px"] = parsed["sum"]

    return {
        "tool_name": tool_name,
        "tool_type": tool_type,
        "input": tool_input,
        "raw_output": raw_output,
        "parsed": parsed,
        "confidence": None,
        "spatial": spatial,
        "meta": {
            "image_id": image_id or tool_input.get("image", "n/a"),
            "timestamp": _utc_now(),
            "warnings": []
        }
    }


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--rules", required=True)
    ap.add_argument("--tool", required=True)
    ap.add_argument("--type", required=True)
    ap.add_argument("--input", required=True, help="JSON string")
    ap.add_argument("--raw", required=True)
    args = ap.parse_args()

    rules = load_rules(args.rules)
    tool_input = json.loads(args.input)
    wrapped = wrap_tool_result(args.tool, args.type, tool_input, args.raw, rules)
    print(json.dumps(wrapped, ensure_ascii=False, indent=2))
