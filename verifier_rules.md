# Verifier Rules Template (ThinkGeo)

This file defines validation rules for the **Reasoner → Verifier → Reflector** loop.
Rules are grouped into **Pixel-level / Region-level / Global-level**.

---

## Output Format (Verifier)

```json
{
  "status": "pass|fail",
  "error_type": "perception|parameter|logic|consistency",
  "rule_id": "PX-01",
  "details": "bbox out of image bounds",
  "suggested_fix": "clip bbox or re-run TextToBbox with higher threshold"
}
```

---

## Pixel-level Rules (PX)

**PX-01 BBox bounds**
- Condition: x1 < 0 or y1 < 0 or x2 > image_width or y2 > image_height
- Fail type: parameter
- Fix: clamp bbox or re-run detection

**PX-02 BBox size sanity**
- Condition: (x2-x1) <= 0 or (y2-y1) <= 0
- Fail type: parameter
- Fix: re-run tool

**PX-03 Pixel count non-negative**
- Condition: pixel_counts has negative values or sum < 0
- Fail type: parameter
- Fix: re-run segmentation

**PX-04 GSD required for area**
- Condition: requesting area_m2 but gsd_m_per_px missing
- Fail type: logic
- Fix: ask for GSD, insert Calculator step

---

## Region-level Rules (RG)

**RG-01 Overlap sanity**
- Condition: two boxes overlap > 0.9 IoU but represent different classes
- Fail type: consistency
- Fix: verify class labels or re-detect

**RG-02 Density sanity**
- Condition: count / area_m2 exceeds expected range for class
- Fail type: perception
- Fix: re-run detection or expand bbox

**RG-03 Area ratio**
- Condition: segmented area exceeds bbox area by > 10%
- Fail type: consistency
- Fix: re-run SegmentObjectPixels

---

## Global-level Rules (GL)

**GL-01 Scene consistency**
- Condition: ImageDescription says “waterbody” but target is “airplane”
- Fail type: consistency
- Fix: re-run ImageDescription or check wrong object

**GL-02 Temporal consistency**
- Condition: ChangeDetection says “no change” but pixel diff significant
- Fail type: perception
- Fix: re-run ChangeDetection or verify inputs

**GL-03 Multi-step unit consistency**
- Condition: mixed units (m, km, m2) without conversion
- Fail type: logic
- Fix: add Calculator conversion step

---

## Reflector Routing (Example)

| error_type | recommended action |
|-----------|--------------------|
| parameter | re-format input, re-call same tool |
| perception | call alternate perception tool or rerun with top1=false |
| logic | insert Calculator / Solver step |
| consistency | re-check scene/global context |

