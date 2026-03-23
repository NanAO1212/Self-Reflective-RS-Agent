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


def _iou(a, b):
    x1 = max(a[0], b[0])
    y1 = max(a[1], b[1])
    x2 = min(a[2], b[2])
    y2 = min(a[3], b[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area_a = max(0, a[2] - a[0]) * max(0, a[3] - a[1])
    area_b = max(0, b[2] - b[0]) * max(0, b[3] - b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def _scene_categories(text):
    text = (text or "").lower()
    categories = set()
    if any(k in text for k in ("water", "sea", "ocean", "river", "lake", "coast", "harbor", "port", "dock", "shore")):
        categories.add("water")
    if any(k in text for k in ("airport", "runway", "airfield", "tarmac", "hangar")):
        categories.add("airport")
    if any(k in text for k in ("urban", "city", "residential", "industrial", "building", "road", "street", "parking")):
        categories.add("urban")
    if any(k in text for k in ("farmland", "agriculture", "field", "crop", "irrigation", "plantation")):
        categories.add("agriculture")
    if any(k in text for k in ("forest", "woodland", "trees", "vegetation")):
        categories.add("forest")
    if any(k in text for k in ("desert", "sand", "dune")):
        categories.add("desert")
    if any(k in text for k in ("snow", "ice", "glacier")):
        categories.add("snow")
    return categories


def _object_expected_scene(label):
    label = (label or "").lower()
    mapping = {
        "ship": {"water"},
        "boat": {"water"},
        "vessel": {"water"},
        "dock": {"water"},
        "harbor": {"water"},
        "port": {"water"},
        "airplane": {"airport"},
        "aircraft": {"airport"},
        "runway": {"airport"},
        "hangar": {"airport"},
        "car": {"urban"},
        "truck": {"urban"},
        "bus": {"urban"},
        "building": {"urban"},
        "road": {"urban"},
        "bridge": {"urban", "water"},
        "farm": {"agriculture"},
        "field": {"agriculture"},
    }
    return mapping.get(label, set())


def _size_priors_m(label):
    label = (label or "").lower()
    priors = {
        "car": (2.0, 6.0),
        "truck": (5.0, 15.0),
        "bus": (8.0, 20.0),
        "ship": (30.0, 400.0),
        "boat": (5.0, 50.0),
        "building": (5.0, 200.0),
        "airplane": (10.0, 80.0),
        "runway": (500.0, 5000.0),
        "bridge": (20.0, 2000.0),
    }
    return priors.get(label)


def verify_spatial(record, image_size=None):
    parsed = record.get("parsed", {})
    spatial = record.get("spatial", {})
    tool = record.get("tool_name", "")
    context = record.get("context", {}) or {}
    results = []

    bboxes = spatial.get("bboxes_px", [])

    # Pixel-level: GSD sanity if provided
    gsd = spatial.get("gsd_m_per_px")
    if gsd is None:
        gsd = record.get("input", {}).get("gsd_m_per_px")
    if gsd is None:
        gsd = context.get("gsd_m_per_px")
    if gsd is not None:
        try:
            if float(gsd) <= 0:
                results.append(_fail(
                    "PV-10", "parameter",
                    "non-positive GSD value",
                    "set a positive gsd_m_per_px"
                ))
            else:
                results.append(_pass("PV-10"))
        except Exception:
            results.append(_fail(
                "PV-10", "parameter",
                "invalid GSD value",
                "set numeric gsd_m_per_px"
            ))

    # Region-level: size/aspect sanity
    # 仅对非 ObjectDetection 工具检查（ObjectDetection 返回大量 bbox 是正常行为）
    if image_size and bboxes and tool not in ("ObjectDetection",):
        w, h = image_size
        img_area = max(w, 1) * max(h, 1)
        for b in bboxes:
            x1, y1, x2, y2 = b
            bw = max(0, x2 - x1)
            bh = max(0, y2 - y1)
            area = bw * bh
            if area / img_area > 0.995:
                results.append(_fail(
                    "RG-10", "consistency",
                    "bbox covers almost entire image",
                    "verify object scale or use segmentation"
                ))
                break
            if area < 4:
                results.append(_fail(
                    "RG-11", "consistency",
                    "bbox area extremely small",
                    "re-check detection threshold"
                ))
                break
            ar = bw / bh if bh > 0 else 0
            if ar > 20 or ar < 0.05:
                results.append(_fail(
                    "RG-12", "consistency",
                    "extreme bbox aspect ratio",
                    "verify detection or use segmentation"
                ))
                break
        else:
            results.append(_pass("RG-10"))
            results.append(_pass("RG-11"))
            results.append(_pass("RG-12"))

    # Region-level: size priors with GSD
    if gsd and bboxes:
        label = None
        if tool in ("TextToBbox", "CountGivenObject", "SegmentObjectPixels"):
            label = record.get("input", {}).get("text")
        elif tool == "ObjectDetection":
            detections = parsed.get("detections") or []
            if detections:
                label = detections[0].get("label")
        if label:
            priors = _size_priors_m(label)
            if priors:
                min_m, max_m = priors
                x1, y1, x2, y2 = bboxes[0]
                length_m = max(x2 - x1, y2 - y1) * float(gsd)
                if length_m < min_m or length_m > max_m:
                    results.append(_fail(
                        "RG-20", "consistency",
                        f"object scale out of expected range ({min_m}-{max_m} m)",
                        "verify GSD or re-check object category"
                    ))
                else:
                    results.append(_pass("RG-20"))

    # Region-level: density sanity
    # 仅对非 ObjectDetection 工具检查；ObjectDetection 返回大量 bbox 是正常行为
    if image_size and bboxes and tool not in ("ObjectDetection",):
        w, h = image_size
        img_area = max(w, 1) * max(h, 1)
        total_area = 0.0
        for b in bboxes:
            total_area += max(0, b[2] - b[0]) * max(0, b[3] - b[1])
        if total_area / img_area > 0.95:
            results.append(_fail(
                "RG-21", "consistency",
                "total bbox area too large for image",
                "verify detection threshold or use segmentation"
            ))
        else:
            results.append(_pass("RG-21"))

    # Region-level: duplicate boxes
    if len(bboxes) >= 2:
        dup = any(_iou(bboxes[i], bboxes[j]) > 0.95 for i in range(len(bboxes)) for j in range(i + 1, len(bboxes)))
        if dup:
            results.append(_fail(
                "RG-22", "consistency",
                "near-duplicate bboxes detected",
                "deduplicate boxes or re-run detection"
            ))
        else:
            results.append(_pass("RG-22"))

    # Evidence-level: multi-round consistency
    evidence = record.get("evidence", {})
    evidence_records = evidence.get("records") or []
    if len(evidence_records) >= 2:
        if bboxes:
            anchors = []
            for r in evidence_records:
                eb = (r.get("spatial") or {}).get("bboxes_px") or []
                if eb:
                    anchors.append(eb[0])
            if len(anchors) >= 2:
                ious = []
                for i in range(len(anchors)):
                    for j in range(i + 1, len(anchors)):
                        ious.append(_iou(anchors[i], anchors[j]))
                mean_iou = sum(ious) / len(ious) if ious else 1.0
                if mean_iou < 0.4:
                    results.append(_fail(
                        "PV-20", "perception",
                        "inconsistent localization across evidence rounds",
                        "re-run perception or adjust prompt"
                    ))
                else:
                    results.append(_pass("PV-20"))
        elif tool in ("CountGivenObject", "SegmentObjectPixels"):
            vals = []
            for r in evidence_records:
                p = r.get("parsed", {})
                if tool == "CountGivenObject" and "count" in p:
                    vals.append(float(p["count"]))
                if tool == "SegmentObjectPixels":
                    if "sum" in p:
                        vals.append(float(p["sum"]))
                    elif "pixel_counts" in p and p["pixel_counts"]:
                        vals.append(float(sum(p["pixel_counts"])))
            if len(vals) >= 2:
                vmin, vmax = min(vals), max(vals)
                if vmax > 0 and (vmax - vmin) / vmax > 0.25:
                    results.append(_fail(
                        "PV-21", "perception",
                        "unstable counts across evidence rounds",
                        "re-run with adjusted thresholds or segmentation"
                    ))
                else:
                    results.append(_pass("PV-21"))

    # Cross-step consistency (history)
    history = (context.get("history") or {}).get(tool, [])
    if history and bboxes:
        last = history[-1]
        last_bboxes = (last.get("spatial") or {}).get("bboxes_px") or []
        same_query = True
        if tool in ("TextToBbox", "CountGivenObject", "SegmentObjectPixels"):
            last_text = (last.get("input") or {}).get("text")
            curr_text = (record.get("input") or {}).get("text")
            same_query = (last_text or "").lower() == (curr_text or "").lower()
        if last_bboxes and same_query:
            if _iou(bboxes[0], last_bboxes[0]) < 0.3:
                results.append(_fail(
                    "PV-30", "perception",
                    "inconsistent localization vs previous call",
                    "re-run with fixed seed or adjust prompt"
                ))
            else:
                results.append(_pass("PV-30"))

    # Global-level: empty text description
    if tool in ("ImageDescription", "RegionAttributeDescription", "ChangeDetection"):
        text = parsed.get("text", "") if isinstance(parsed, dict) else ""
        if isinstance(text, str) and not text.strip():
            results.append(_fail(
                "GL-10", "perception",
                "empty text description",
                "re-run tool or adjust prompt"
            ))
        else:
            results.append(_pass("GL-10"))

    # Global-level: semantic consistency
    scene_text = context.get("scene_text")
    if scene_text:
        scene_cats = _scene_categories(scene_text)
        label = None
        if tool in ("TextToBbox", "CountGivenObject", "SegmentObjectPixels"):
            label = record.get("input", {}).get("text")
        elif tool == "ObjectDetection":
            detections = parsed.get("detections") or []
            if detections:
                label = detections[0].get("label")
        if label:
            expected = _object_expected_scene(label)
            if expected and scene_cats and not (expected & scene_cats):
                results.append(_fail(
                    "GL-20", "consistency",
                    "object label conflicts with global scene description",
                    "verify scene description or re-run detection"
                ))
            else:
                results.append(_pass("GL-20"))

    return results
