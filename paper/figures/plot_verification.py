"""
绘制 SR²A 三层验证框架图 (Fig.2)
输出: verification_levels.pdf
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

# ── 颜色方案 ──
C_BORDER  = "#37474F"
C_ARROW   = "#455A64"
C_GLOBAL  = "#FFCC80"   # 暖橙 - Global
C_REGION  = "#FFE0B2"   # 浅橙 - Region
C_PIXEL   = "#FFF3E0"   # 最浅橙 - Pixel
C_TITLE   = "#E65100"   # 深橙标题
C_RULE    = "#FFF8E1"   # 规则条目背景
C_INPUT   = "#E3F2FD"   # 输入
C_OUTPUT  = "#C8E6C9"   # 输出

fig, ax = plt.subplots(1, 1, figsize=(8.5, 9))
ax.set_xlim(0, 10)
ax.set_ylim(0, 10.5)
ax.set_aspect("equal")
ax.axis("off")

cx = 5.0

def draw_box(x, y, w, h, color, label=None, fontsize=11, bold=True,
             label_y_offset=0):
    box = FancyBboxPatch((x, y), w, h,
                         boxstyle="round,pad=0.12",
                         facecolor=color, edgecolor=C_BORDER,
                         linewidth=1.5, zorder=2)
    ax.add_patch(box)
    if label:
        weight = "bold" if bold else "normal"
        ax.text(x + w/2, y + h/2 + label_y_offset,
                label, ha="center", va="center",
                fontsize=fontsize, fontweight=weight, zorder=3)

def draw_rule_item(x, y, rule_id, desc, fontsize=8.5):
    """绘制单条验证规则"""
    ax.text(x, y, rule_id, ha="left", va="center",
            fontsize=fontsize, fontweight="bold", color=C_TITLE, zorder=3)
    ax.text(x + 0.65, y, desc, ha="left", va="center",
            fontsize=fontsize, color="#333333", zorder=3)

def draw_arrow_down(x, y1, y2, color=C_ARROW, lw=2.0):
    arrow = FancyArrowPatch((x, y1), (x, y2),
                            arrowstyle="-|>", color=color,
                            linewidth=lw, mutation_scale=15, zorder=1)
    ax.add_patch(arrow)

# ── 布局参数 ──
level_w = 8.0
level_h = 1.8
level_x = cx - level_w / 2
gap = 0.5

# Y 坐标（从上到下：Global → Region → Pixel）
y_input  = 9.5
y_global = 7.2
y_region = 4.9
y_pixel  = 2.6
y_output = 0.8

# ── 输入 ──
draw_box(cx - 2.0, y_input, 4.0, 0.7, C_INPUT,
         label="Tool Output (Parsed)", fontsize=11)

# ── Global Level ──
draw_box(level_x, y_global, level_w, level_h, C_GLOBAL,
         label="Global-Level Verification (GL)", fontsize=12,
         label_y_offset=0.55)

rules_global = [
    ("GL-10", "Empty Description Detection"),
    ("GL-20", "Scene–Object Semantic Consistency"),
    ("GL-30", "Count Sanity (>500 threshold)"),
    ("GL-34", "VLM Hallucination Pattern Detection"),
]
for i, (rid, desc) in enumerate(rules_global):
    draw_rule_item(level_x + 0.3, y_global + level_h - 0.55 - i * 0.35, rid, desc)

# ── Region Level ──
draw_box(level_x, y_region, level_w, level_h, C_REGION,
         label="Region-Level Verification (RG)", fontsize=12,
         label_y_offset=0.55)

rules_region = [
    ("RG-10", "Coverage Check (>98% image area)"),
    ("RG-11", "Minimum Size (<4 pixels)"),
    ("RG-12", "Aspect Ratio (>20 or <0.05)"),
    ("RG-20", "GSD-Based Physical Size Priors"),
]
for i, (rid, desc) in enumerate(rules_region):
    draw_rule_item(level_x + 0.3, y_region + level_h - 0.55 - i * 0.35, rid, desc)

# ── Pixel Level ──
draw_box(level_x, y_pixel, level_w, level_h, C_PIXEL,
         label="Pixel-Level Verification (PX)", fontsize=12,
         label_y_offset=0.55)

rules_pixel = [
    ("PX-01", "Bounding Box Bounds Check"),
    ("PX-02", "Box Size Sanity (positive dims)"),
    ("PX-03", "Pixel Count Sign (non-negative)"),
    ("PX-04", "GSD Requirement for Area"),
]
for i, (rid, desc) in enumerate(rules_pixel):
    draw_rule_item(level_x + 0.3, y_pixel + level_h - 0.55 - i * 0.35, rid, desc)

# ── 输出 ──
draw_box(cx - 2.5, y_output, 5.0, 0.7, C_OUTPUT,
         label="Verdicts: {(rule_id, pass/fail, error_type, fix)}", fontsize=9.5)

# ── 箭头 ──
draw_arrow_down(cx, y_input, y_global + level_h)
draw_arrow_down(cx, y_global, y_region + level_h)
draw_arrow_down(cx, y_region, y_pixel + level_h)
draw_arrow_down(cx, y_pixel, y_output + 0.7)

# ── 左侧层级标注 ──
for y_pos, label in [(y_global, "Semantic\nCoherence"),
                      (y_region, "Spatial\nReasonableness"),
                      (y_pixel, "Geometric\nIntegrity")]:
    ax.text(level_x - 0.15, y_pos + level_h / 2, label,
            ha="right", va="center", fontsize=9,
            fontstyle="italic", color="#555555",
            rotation=0, zorder=3)

# ── 保存 ──
plt.tight_layout()
fig.savefig("figures/verification_levels.pdf", dpi=300, bbox_inches="tight",
            pad_inches=0.1, facecolor="white")
fig.savefig("figures/verification_levels.png", dpi=200, bbox_inches="tight",
            pad_inches=0.1, facecolor="white")
print("Saved: figures/verification_levels.pdf")
print("Saved: figures/verification_levels.png")
