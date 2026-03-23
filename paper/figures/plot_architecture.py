"""
绘制 SR²A 系统架构图 (Fig.1)
输出: architecture.pdf
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np

# ── 颜色方案 ──
C_INPUT   = "#E8EAF6"  # 淡紫灰 - 输入/输出
C_CORE    = "#E3F2FD"  # 淡蓝 - 核心模块 (Planner, Reasoner)
C_TOOL    = "#F3E5F5"  # 淡紫 - Tool Execution
C_KNOW    = "#BBDEFB"  # 蓝色 - Innovation A (RS Knowledge)
C_VERIFY  = "#FFE0B2"  # 橙色 - Innovation B (Verifier)
C_REFLECT = "#C8E6C9"  # 绿色 - Innovation C (Reflector)
C_PASS    = "#4CAF50"  # 绿色箭头
C_FAIL    = "#F44336"  # 红色箭头
C_BORDER  = "#37474F"  # 深灰边框
C_ARROW   = "#455A64"  # 箭头颜色
C_REPLAN  = "#FF9800"  # 橙色 replan 箭头

fig, ax = plt.subplots(1, 1, figsize=(8.5, 11))
ax.set_xlim(0, 10)
ax.set_ylim(0, 13)
ax.set_aspect("equal")
ax.axis("off")

def draw_box(x, y, w, h, label, color, sublabel=None, fontsize=11, bold=True,
             label_y_offset=0, sublabel_y_offset=0):
    """绘制圆角矩形模块"""
    box = FancyBboxPatch((x, y), w, h,
                         boxstyle="round,pad=0.15",
                         facecolor=color, edgecolor=C_BORDER,
                         linewidth=1.5, zorder=2)
    ax.add_patch(box)
    weight = "bold" if bold else "normal"
    ax.text(x + w/2, y + h/2 + (0.12 if sublabel else 0) + label_y_offset,
            label, ha="center", va="center",
            fontsize=fontsize, fontweight=weight, zorder=3)
    if sublabel:
        ax.text(x + w/2, y + h/2 - 0.22 + sublabel_y_offset,
                sublabel, ha="center", va="center",
                fontsize=8, fontstyle="italic", color="#555555", zorder=3)

def draw_arrow(x1, y1, x2, y2, color=C_ARROW, style="-|>", lw=1.8, ls="-", connectionstyle="arc3,rad=0"):
    """绘制箭头"""
    arrow = FancyArrowPatch((x1, y1), (x2, y2),
                            arrowstyle=style,
                            color=color, linewidth=lw,
                            linestyle=ls,
                            connectionstyle=connectionstyle,
                            mutation_scale=15, zorder=4)
    ax.add_patch(arrow)

def draw_label_on_arrow(x, y, text, fontsize=8, color="#333333"):
    ax.text(x, y, text, ha="center", va="center",
            fontsize=fontsize, color=color,
            bbox=dict(boxstyle="round,pad=0.15", facecolor="white",
                      edgecolor="none", alpha=0.9), zorder=5)

# ── 坐标布局（从上到下） ──
# 中心 x = 5.0
cx = 5.0
box_w = 3.0
box_h = 0.7
gap = 0.5

# Y 坐标（从上到下）
y_input    = 11.8
y_planner  = 10.4
y_reasoner = 9.0
y_tool     = 7.4
y_verifier = 5.8
y_reflect  = 4.0
y_output   = 2.2

# ── 绘制模块 ──

# Input
draw_box(cx - 1.5, y_input, 3.0, 0.65, "Input", C_INPUT,
         sublabel="Query q + Image(s)", fontsize=10)

# Planner
draw_box(cx - box_w/2, y_planner, box_w, box_h, "Planner", C_CORE,
         sublabel="Task classification + Plan decomposition")

# Reasoner
draw_box(cx - box_w/2, y_reasoner, box_w, box_h, "Reasoner", C_CORE,
         sublabel="Tool sequence generation")

# RS Knowledge (Innovation A) — 右侧
know_x = 7.5
know_w = 2.0
know_h = 1.8
draw_box(know_x, y_reasoner - 0.1, know_w, know_h + 0.7,
         "RS Domain\nKnowledge", C_KNOW,
         sublabel="Innovation A", fontsize=10)

# 子标签
ax.text(know_x + know_w/2, y_reasoner + 0.55,
        "Scene-Object Priors\nSize Constraints\nGSD Rules\nConfusion Pairs",
        ha="center", va="center", fontsize=7.5, color="#1565C0", zorder=3)

# Tool Execution — 更宽更高
tool_w = 4.5
tool_h = 1.2
draw_box(cx - tool_w/2, y_tool, tool_w, tool_h, "Tool Execution", C_TOOL, fontsize=11,
         label_y_offset=0.25)

# Tool 子模块
sub_w = 1.2
sub_h = 0.35
sub_y = y_tool + 0.08
for i, (label, clr) in enumerate([("Perception", "#CE93D8"),
                                    ("Logic", "#B39DDB"),
                                    ("Operation", "#9FA8DA")]):
    sx = cx - 1.7 + i * 1.35
    box = FancyBboxPatch((sx, sub_y), sub_w, sub_h,
                         boxstyle="round,pad=0.08",
                         facecolor=clr, edgecolor=C_BORDER,
                         linewidth=0.8, alpha=0.7, zorder=2)
    ax.add_patch(box)
    ax.text(sx + sub_w/2, sub_y + sub_h/2, label,
            ha="center", va="center", fontsize=8, zorder=3)

# Verifier (Innovation B) — 更宽显示三层
verify_w = 4.0
verify_h = 1.2
draw_box(cx - verify_w/2, y_verifier, verify_w, verify_h,
         "Hierarchical Verifier", C_VERIFY,
         sublabel="Innovation B", fontsize=11,
         label_y_offset=0.25, sublabel_y_offset=0.25)

# 三层标签
for i, (label, rid) in enumerate([("Pixel-Level", "PX"),
                                   ("Region-Level", "RG"),
                                   ("Global-Level", "GL")]):
    vx = cx - 1.5 + i * 1.2
    vy = y_verifier + 0.08
    box = FancyBboxPatch((vx, vy), 1.1, 0.32,
                         boxstyle="round,pad=0.06",
                         facecolor="#FFCC80", edgecolor=C_BORDER,
                         linewidth=0.6, alpha=0.8, zorder=2)
    ax.add_patch(box)
    ax.text(vx + 0.55, vy + 0.16, f"{label}\n({rid})",
            ha="center", va="center", fontsize=7, zorder=3)

# Reflector (Innovation C)
reflect_w = 3.5
reflect_h = 1.2
draw_box(cx - reflect_w/2, y_reflect, reflect_w, reflect_h,
         "Self-Reflective Module", C_REFLECT,
         sublabel="Innovation C", fontsize=10,
         label_y_offset=0.25, sublabel_y_offset=0.25)

# 双层标签
for i, label in enumerate(["Tier 1: Rule-Based", "Tier 2: LLM-Based"]):
    rx = cx - 1.4 + i * 1.5
    ry = y_reflect + 0.08
    box = FancyBboxPatch((rx, ry), 1.4, 0.3,
                         boxstyle="round,pad=0.06",
                         facecolor="#A5D6A7", edgecolor=C_BORDER,
                         linewidth=0.6, alpha=0.8, zorder=2)
    ax.add_patch(box)
    ax.text(rx + 0.7, ry + 0.15, label,
            ha="center", va="center", fontsize=7.5, zorder=3)

# Output
draw_box(cx - 1.5, y_output, 3.0, 0.6, "Final Answer", C_INPUT, fontsize=10)

# ── 绘制箭头 ──

# Input → Planner
draw_arrow(cx, y_input, cx, y_planner + box_h)

# Planner → Reasoner
draw_arrow(cx, y_planner, cx, y_reasoner + box_h)

# Reasoner → Tool Execution
draw_arrow(cx, y_reasoner, cx, y_tool + tool_h)

# Tool Execution → Verifier
draw_arrow(cx, y_tool, cx, y_verifier + verify_h)

# RS Knowledge → Planner (虚线)
draw_arrow(know_x, y_planner + box_h/2 + 0.35, cx + box_w/2, y_planner + box_h/2,
           color="#1565C0", ls="--", lw=1.5,
           connectionstyle="arc3,rad=0.2")

# RS Knowledge → Reasoner (虚线)
draw_arrow(know_x, y_reasoner + box_h/2, cx + box_w/2, y_reasoner + box_h/2,
           color="#1565C0", ls="--", lw=1.5)

# Verifier → Reflector (always)
draw_arrow(cx, y_verifier, cx, y_reflect + reflect_h,
           color=C_ARROW, lw=2.0)
draw_label_on_arrow(cx + 0.8, (y_verifier + y_reflect + reflect_h) / 2,
                    "Verdicts", fontsize=9, color=C_ARROW)

# Reflector → All Pass → Output
draw_arrow(cx - 0.5, y_reflect, cx - 0.5, y_output + 0.6,
           color=C_PASS, lw=2.0)
draw_label_on_arrow(cx - 1.5, (y_reflect + y_output + 0.6) / 2,
                    "All Pass\n(Next Step\n/ Output)", fontsize=9, color=C_PASS)

# Reflector → Fail → Retry (left loop)
# 注释说明: Fail 时 Reflector 生成 patches，回到 Tool Execution

# Reflector → Retry → Tool Execution (左侧回环，折线路径)
# 从 Reflector 左侧 → 向左拐 → 向上 → 到 Tool Execution 左侧
retry_x_offset = 1.8  # 折线向左偏移量
retry_from_x = cx - reflect_w/2
retry_from_y = y_reflect + 0.6
retry_to_x = cx - tool_w/2
retry_to_y = y_tool + 0.6
retry_mid_x = min(retry_from_x, retry_to_x) - retry_x_offset

# 用 3 段直线 + 箭头
ax.annotate("",
            xy=(retry_to_x, retry_to_y),
            xytext=(retry_mid_x, retry_to_y),
            arrowprops=dict(arrowstyle="-|>", color=C_REPLAN,
                           linewidth=2.5),
            zorder=6)
# 垂直线段
ax.plot([retry_mid_x, retry_mid_x], [retry_from_y, retry_to_y],
        color=C_REPLAN, linewidth=2.5, zorder=6)
# 水平线段（从 Reflector 到折点）
ax.plot([retry_from_x, retry_mid_x], [retry_from_y, retry_from_y],
        color=C_REPLAN, linewidth=2.5, zorder=6)
draw_label_on_arrow(retry_mid_x + 0.75, (retry_from_y + retry_to_y) / 2,
                    "Retry\n(Corrective\n Patches)", fontsize=9, color=C_REPLAN)

# Reflector → Replan → Reasoner (更外侧折线路径)
replan_x_offset = 3.2  # 更大偏移，明确区分
replan_from_x = cx - reflect_w/2
replan_from_y = y_reflect + 0.3
replan_to_x = cx - box_w/2
replan_to_y = y_reasoner + 0.35
replan_mid_x = min(replan_from_x, replan_to_x) - replan_x_offset

ax.annotate("",
            xy=(replan_to_x, replan_to_y),
            xytext=(replan_mid_x, replan_to_y),
            arrowprops=dict(arrowstyle="-|>", color="#D32F2F",
                           linewidth=2.0, linestyle="--"),
            zorder=6)
ax.plot([replan_mid_x, replan_mid_x], [replan_from_y, replan_to_y],
        color="#D32F2F", linewidth=2.0, linestyle="--", zorder=6)
ax.plot([replan_from_x, replan_mid_x], [replan_from_y, replan_from_y],
        color="#D32F2F", linewidth=2.0, linestyle="--", zorder=6)
draw_label_on_arrow(replan_mid_x - 0.55, (replan_from_y + replan_to_y) / 2,
                    "Replan", fontsize=9, color="#D32F2F")

# ── 图例（竖直排列） ──
legend_x = 7.2
legend_y_start = 2.0
legend_gap = 0.45
legend_items = [
    (C_KNOW,    "Innovation A: RS Domain Knowledge"),
    (C_VERIFY,  "Innovation B: Hierarchical Verification"),
    (C_REFLECT, "Innovation C: Self-Reflective Module"),
]
for i, (color, label) in enumerate(legend_items):
    ly = legend_y_start - i * legend_gap
    box = FancyBboxPatch((legend_x, ly), 0.35, 0.25,
                         boxstyle="round,pad=0.05",
                         facecolor=color, edgecolor=C_BORDER,
                         linewidth=0.8, zorder=2)
    ax.add_patch(box)
    ax.text(legend_x + 0.5, ly + 0.12, label,
            ha="left", va="center", fontsize=7.5, zorder=3)

# ── 保存 ──
plt.tight_layout()
out_path = "figures/architecture_v2.pdf"
fig.savefig(out_path, dpi=300, bbox_inches="tight",
            pad_inches=0.1, facecolor="white")
print(f"Saved: {out_path}")

# 同时保存 PNG 预览
fig.savefig("figures/architecture_v2.png", dpi=200, bbox_inches="tight",
            pad_inches=0.1, facecolor="white")
print("Saved: figures/architecture.png")
