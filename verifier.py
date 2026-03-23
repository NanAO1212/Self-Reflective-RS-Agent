def _fail(rule_id, error_type, details, fix):
    return {
        "status": "fail",
        "error_type": error_type,
        "rule_id": rule_id,
        "details": details,
        "suggested_fix": fix
    }


def _pass(rule_id):
    return {"status": "pass", "rule_id": rule_id}


def verify(record, image_size=None):
    from spatial_verifier import verify_spatial

    parsed = record.get("parsed", {})
    spatial = record.get("spatial", {})
    tool = record.get("tool_name")
    results = []

    # PX-01 bbox bounds
    bboxes = spatial.get("bboxes_px", [])
    if image_size and bboxes:
        w, h = image_size
        for b in bboxes:
            x1, y1, x2, y2 = b
            if x1 < 0 or y1 < 0 or x2 > w or y2 > h:
                results.append(_fail(
                    "PX-01", "parameter",
                    "bbox out of image bounds",
                    "clip bbox or re-run detection"
                ))
                break
        else:
            results.append(_pass("PX-01"))

    # PX-02 bbox size sanity
    if bboxes:
        for b in bboxes:
            x1, y1, x2, y2 = b
            if (x2 - x1) <= 0 or (y2 - y1) <= 0:
                results.append(_fail(
                    "PX-02", "parameter",
                    "bbox has non-positive size",
                    "re-run detection"
                ))
                break
        else:
            results.append(_pass("PX-02"))

    # PX-03 pixel count non-negative
    if tool == "SegmentObjectPixels":
        counts = parsed.get("pixel_counts", [])
        if any(c < 0 for c in counts):
            results.append(_fail(
                "PX-03", "parameter",
                "negative pixel count detected",
                "re-run segmentation"
            ))
        else:
            results.append(_pass("PX-03"))

    # PX-04 GSD required for area
    if spatial.get("area_m2") is not None and spatial.get("gsd_m_per_px") is None:
        results.append(_fail(
            "PX-04", "logic",
            "area requested without GSD",
            "obtain gsd_m_per_px and re-calc area"
        ))

    # RG-01 overlap sanity (simple IoU check if multiple bboxes)
    if len(bboxes) >= 2:
        def iou(a, b):
            x1 = max(a[0], b[0])
            y1 = max(a[1], b[1])
            x2 = min(a[2], b[2])
            y2 = min(a[3], b[3])
            inter = max(0, x2 - x1) * max(0, y2 - y1)
            area_a = max(0, a[2] - a[0]) * max(0, a[3] - a[1])
            area_b = max(0, b[2] - b[0]) * max(0, b[3] - b[1])
            union = area_a + area_b - inter
            return inter / union if union > 0 else 0

        high_iou = any(iou(bboxes[i], bboxes[j]) > 0.9 for i in range(len(bboxes)) for j in range(i + 1, len(bboxes)))
        if high_iou:
            results.append(_fail(
                "RG-01", "consistency",
                "high overlap between different boxes",
                "verify labels or re-run detection"
            ))
        else:
            results.append(_pass("RG-01"))

    # GL-03 unit consistency placeholder
    if tool == "Calculator" and "value" in parsed and "m" in record.get("raw_output", ""):
        results.append(_pass("GL-03"))

    # GL-30 count sanity (仅检查异常大的值，count=0 可能是合理结果)
    if tool == "CountGivenObject":
        count = parsed.get("count")
        if count is not None:
            try:
                c = int(count)
                if c > 500:
                    results.append(_fail(
                        "GL-30", "logic",
                        f"count={c} seems unreasonably large",
                        "verify detection threshold or object category"
                    ))
                else:
                    results.append(_pass("GL-30"))
            except (ValueError, TypeError):
                results.append(_pass("GL-30"))

    # GL-31 calculator result sanity
    if tool == "Calculator":
        val = parsed.get("value")
        if val is not None:
            try:
                import math
                v = float(val)
                if not math.isfinite(v):
                    results.append(_fail(
                        "GL-31", "logic",
                        "calculator returned NaN or Inf",
                        "check expression for division by zero or invalid input"
                    ))
                elif v < 0 and any(k in record.get("raw_output", "").lower()
                                  for k in ("area", "distance", "length", "width", "height")):
                    results.append(_fail(
                        "GL-31", "logic",
                        "negative value for area/distance",
                        "check coordinate order or absolute value"
                    ))
                else:
                    results.append(_pass("GL-31"))
            except (ValueError, TypeError):
                results.append(_pass("GL-31"))

    # GL-32 empty/invalid tool output
    raw = record.get("raw_output", "")
    raw_str = str(raw).strip().lower() if raw is not None else ""
    if raw_str in ("", "none", "null", "error", "failed"):
        results.append(_fail(
            "GL-32", "perception",
            "tool returned empty or invalid output",
            "re-run tool or check input parameters"
        ))

    # GL-34 hallucination detection for description tools
    if tool in ("RegionAttributeDescription", "ImageDescription"):
        desc = parsed.get("text", "") if isinstance(parsed, dict) else ""
        desc_lower = str(desc).lower()
        hallucination_patterns = [
            "blurry photo of", "a photo of a", "stock photo",
            "close up of", "picture of a", "image of a cat",
            "image of a dog", "a black and white photo",
        ]
        if any(p in desc_lower for p in hallucination_patterns):
            results.append(_fail(
                "GL-34", "perception",
                f"description looks like model hallucination: '{desc[:80]}'",
                "re-run with clearer bbox or switch to ImageDescription"
            ))

    # Spatial reasoning & validation
    results.extend(verify_spatial(record, image_size=image_size))

    return results


if __name__ == "__main__":
    import json
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--record", required=True, help="JSON string")
    ap.add_argument("--image_size", default="")
    args = ap.parse_args()

    record = json.loads(args.record)
    size = tuple(map(int, args.image_size.split(","))) if args.image_size else None
    res = verify(record, size)
    print(json.dumps(res, ensure_ascii=False, indent=2))
