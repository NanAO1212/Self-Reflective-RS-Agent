# Temporal Change Prompt (Template)

You must handle temporal change analysis.

Inputs:
- pre_image (time1)
- post_image (time2)
- question: user request about changes

Required:
1) Call ChangeDetection with (text, pre_image, post_image).
2) If asked "how many" or "area", call CountGivenObject or SegmentObjectPixels.
3) Use Calculator to convert pixel counts into real units (GSD if provided).
4) Add visualization: DrawBox or AddText marking changed regions.
5) Output a Plot summarizing change (e.g., changed area or count).
6) If you use Solver or Plot, the code MUST define def solution(): and return the answer/figure.

Return JSON:
{
  "steps": [...],
  "final_answer": "...",
  "outputs": [
    {"tool_name":"DrawBox", ...},
    {"tool_name":"AddText", ...},
    {"tool_name":"Plot", ...}
  ]
}
