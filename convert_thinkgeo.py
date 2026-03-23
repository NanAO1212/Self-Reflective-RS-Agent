"""
convert_thinkgeo.py — 将 ThinkGeoBench.json 转换为 self_reflective_agent 的 task JSON 格式。

用法:
    python convert_thinkgeo.py --input ThinkGeoBench.json --output_dir ../tasks/thinkgeo_full
"""

import json
import re
import os
import argparse
from pathlib import Path


# 工具名 → tool_type 映射
TOOL_TYPE_MAP = {
    "TextToBbox": "perception",
    "ObjectDetection": "perception",
    "CountGivenObject": "perception",
    "SegmentObjectPixels": "perception",
    "RegionAttributeDescription": "perception",
    "ImageDescription": "perception",
    "ChangeDetection": "perception",
    "OCR": "perception",
    "Calculator": "logic",
    "Solver": "logic",
    "Plot": "operation",
    "DrawBox": "operation",
    "DrawMask": "operation",
    "AddText": "operation",
    "GoogleSearch": "logic",
}

GSD_PATTERN = re.compile(r"GSD\s*[=:]\s*([\d.]+)\s*m/px", re.IGNORECASE)


def extract_gsd(query):
    """从 query 文本中提取 GSD 值。"""
    m = GSD_PATTERN.search(query)
    if m:
        return float(m.group(1))
    return None


def convert_task(task_id, item, image_base_dir):
    """将单个 ThinkGeoBench 条目转换为 pipeline task 格式。"""
    # 提取 user query
    query = ""
    for d in item["dialogs"]:
        if d["role"] == "user":
            query = d["content"]
            break

    # 提取图片路径（绝对路径）
    files = item.get("files", [])
    image = None
    pre_image = None
    post_image = None

    if len(files) == 1:
        image = str(Path(image_base_dir) / files[0]["path"])
    elif len(files) == 2:
        # change detection: pre + post
        f0 = files[0]["path"]
        f1 = files[1]["path"]
        if "pre" in f0.lower():
            pre_image = str(Path(image_base_dir) / f0)
            post_image = str(Path(image_base_dir) / f1)
        elif "post" in f0.lower():
            pre_image = str(Path(image_base_dir) / f1)
            post_image = str(Path(image_base_dir) / f0)
        else:
            pre_image = str(Path(image_base_dir) / f0)
            post_image = str(Path(image_base_dir) / f1)
        image = pre_image  # 默认主图用 pre

    # 提取 ground truth steps（从 assistant tool_calls 中）
    steps = []
    for d in item["dialogs"]:
        if d["role"] != "assistant" or not d.get("tool_calls"):
            continue
        for tc in d["tool_calls"]:
            fn = tc["function"]
            tool_name = fn["name"]
            tool_input = dict(fn.get("arguments", {}))

            # 将相对图片路径转为绝对路径
            for img_key in ("image", "pre_image", "post_image"):
                if img_key in tool_input:
                    rel = tool_input[img_key]
                    abs_path = str(Path(image_base_dir) / rel)
                    tool_input[img_key] = abs_path

            steps.append({
                "tool_name": tool_name,
                "tool_type": TOOL_TYPE_MAP.get(tool_name, "perception"),
                "input": tool_input,
            })

    # 提取 final answer
    final_answer = ""
    for d in reversed(item["dialogs"]):
        if d["role"] == "assistant" and d.get("content"):
            final_answer = d["content"]
            break

    # 提取 gt_answer 和 evaluation
    gt_answer = item.get("gt_answer")
    evaluation = item.get("evaluation")

    # 提取 GSD
    gsd = extract_gsd(query)

    # 分离 steps 和 outputs（operation 类型的放到 outputs）
    exec_steps = []
    outputs = []
    for s in steps:
        if s["tool_type"] == "operation":
            outputs.append(s)
        else:
            exec_steps.append(s)

    task = {
        "task_id": f"thinkgeo_{task_id}",
        "query": query,
        "image": image,
        "steps": exec_steps,
        "outputs": outputs,
        "final_answer": final_answer,
    }

    if pre_image:
        task["pre_image"] = pre_image
    if post_image:
        task["post_image"] = post_image
    if gsd is not None:
        task["gsd_m_per_px"] = gsd
    if gt_answer:
        task["gt_answer"] = gt_answer
    if evaluation:
        task["evaluation"] = evaluation

    return task


def main():
    ap = argparse.ArgumentParser(description="转换 ThinkGeoBench.json → pipeline task JSONs")
    ap.add_argument("--input", default=None,
                    help="ThinkGeoBench.json 路径（默认自动查找）")
    ap.add_argument("--output_dir", default="tasks/thinkgeo_full",
                    help="输出目录")
    ap.add_argument("--single_file", action="store_true",
                    help="输出为单个 JSON 文件（all_tasks.json）而非逐个文件")
    ap.add_argument("--skip_missing", action="store_true",
                    help="跳过图片缺失的任务")
    args = ap.parse_args()

    # 自动查找 ThinkGeoBench.json
    if args.input:
        bench_path = Path(args.input)
    else:
        bench_path = (
            Path(__file__).parent.parent
            / "thinkgeo" / "ThinkGeo" / "opencompass" / "data"
            / "ThinkGeo_dataset" / "ThinkGeoBench.json"
        )

    image_base_dir = str(bench_path.parent)

    if not bench_path.exists():
        print(f"Error: {bench_path} not found")
        return

    data = json.loads(bench_path.read_text(encoding="utf-8"))
    print(f"Loaded {len(data)} tasks from {bench_path}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    tasks = []
    skipped = 0
    converted = 0

    for task_id, item in data.items():
        task = convert_task(task_id, item, image_base_dir)

        # 检查图片是否存在
        if args.skip_missing:
            missing = False
            for key in ("image", "pre_image", "post_image"):
                p = task.get(key)
                if p and not Path(p).exists():
                    missing = True
                    break
            if missing:
                skipped += 1
                continue

        tasks.append(task)
        converted += 1

        if not args.single_file:
            out_path = output_dir / f"thinkgeo_{task_id}.json"
            out_path.write_text(
                json.dumps(task, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )

    if args.single_file:
        out_path = output_dir / "all_tasks.json"
        out_path.write_text(
            json.dumps(tasks, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(f"Wrote {converted} tasks to {out_path}")
    else:
        print(f"Wrote {converted} task files to {output_dir}/")

    if skipped:
        print(f"Skipped {skipped} tasks (missing images)")


if __name__ == "__main__":
    main()


