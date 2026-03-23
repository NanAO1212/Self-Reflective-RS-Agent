"""
run_all.py — 批量执行 ThinkGeo 任务并收集结果。

用法:
    # Step-by-Step 模式（使用 GT steps）
    python run_all.py --tasks_dir tasks/thinkgeo_full --out_dir logs/full_run
    # End-to-End 模式（Planner+Reasoner 自主规划）
    python run_all.py --mode e2e --tasks_dir tasks/thinkgeo_full --out_dir logs/e2e_run
"""

import json
import os
import time
import traceback
from pathlib import Path

from pipeline import run_pipeline, run_from_query


def _json_default(obj):
    try:
        import numpy as np
        if isinstance(obj, np.ndarray):
            return obj.tolist()
    except Exception:
        pass
    return str(obj)


def main():
    os.environ.setdefault("MPLBACKEND", "Agg")
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks_dir", default="tasks/thinkgeo_full")
    ap.add_argument("--mode", choices=["sbs", "e2e"], default="sbs",
                    help="sbs=Step-by-Step (GT steps), e2e=End-to-End (自主规划)")
    ap.add_argument("--rules", default="tool_parsing_rules.json")
    ap.add_argument("--max_retries", type=int, default=1)
    ap.add_argument("--max_replans", type=int, default=1, help="E2E 模式最大 replan 次数")
    ap.add_argument("--evidence_rounds", type=int, default=1)
    ap.add_argument("--image_size", default="")
    ap.add_argument("--out_dir", default="logs/full_run")
    ap.add_argument("--start", type=int, default=0, help="起始任务索引")
    ap.add_argument("--end", type=int, default=-1, help="结束任务索引（-1=全部）")
    ap.add_argument("--skip_existing", action="store_true",
                    help="跳过已有 _log.json 的任务（断点续跑）")
    args = ap.parse_args()

    tasks_dir = Path(args.tasks_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    size = tuple(map(int, args.image_size.split(","))) if args.image_size else None

    task_files = sorted(tasks_dir.glob("thinkgeo_*.json"))
    end = args.end if args.end >= 0 else len(task_files)
    task_files = task_files[args.start:end]
    print(f"Running {len(task_files)} tasks ({args.start}..{end}) from {tasks_dir}  [mode={args.mode}]")

    summary = []
    success = 0
    failed = 0
    errors = 0
    t0 = time.time()

    for i, task_file in enumerate(task_files):
        task = json.loads(task_file.read_text(encoding="utf-8"))
        task_id = task.get("task_id", task_file.stem)
        task_out_dir = str(out_dir / task_id)

        # 断点续跑：跳过已有结果的任务
        log_path = out_dir / f"{task_id}_log.json"
        if args.skip_existing and log_path.exists():
            print(f"[{i+1}/{len(task_files)}] {task_id} ... SKIP (already exists)")
            # 把已有结果加入 summary
            try:
                existing = json.loads(log_path.read_text(encoding="utf-8"))
                failed_steps = [
                    s for s in existing.get("steps", [])
                    if any(v.get("status") == "fail" for v in s.get("verdicts", []))
                ]
                has_fail = len(failed_steps) > 0
                if has_fail:
                    failed += 1
                else:
                    success += 1
                summary.append({
                    "task_id": task_id,
                    "status": "fail" if has_fail else "ok",
                    "log": str(log_path),
                    "elapsed_s": existing.get("elapsed_s", 0),
                    "total_steps": len(existing.get("steps", [])),
                    "failed_steps": len(failed_steps),
                    "skipped": True,
                    "final_answer": existing.get("final_answer", ""),
                    "gt_answer": task.get("gt_answer"),
                })
            except Exception:
                summary.append({"task_id": task_id, "status": "skipped", "skipped": True})
            continue

        # 自动获取图片尺寸（verifier 多项规则依赖此参数）
        img_size = size
        if img_size is None:
            img_path = task.get("image")
            if img_path and Path(img_path).exists():
                try:
                    from PIL import Image
                    with Image.open(img_path) as im:
                        img_size = im.size  # (width, height)
                except Exception:
                    pass

        print(f"[{i+1}/{len(task_files)}] {task_id} ...", end=" ", flush=True)
        ts = time.time()

        try:
            if args.mode == "e2e":
                # E2E: Planner + Reasoner 自主规划执行
                question = task.get("query") or task.get("question", "")
                image = task.get("image", "")
                if not question:
                    raise ValueError("Task missing query/question field")
                output = run_from_query(
                    question=question,
                    image=image,
                    rules_path=args.rules,
                    max_retries=args.max_retries,
                    image_size=img_size,
                    pre_image=task.get("pre_image"),
                    post_image=task.get("post_image"),
                    output_dir=task_out_dir,
                    max_replans=args.max_replans,
                    evidence_rounds=args.evidence_rounds,
                )
                # E2E 输出中保留 task_id
                output["task_id"] = task_id
            else:
                # SbS: 使用 GT steps 执行
                output = run_pipeline(
                    task,
                    args.rules,
                    max_retries=args.max_retries,
                    image_size=img_size,
                    output_dir=task_out_dir,
                    evidence_rounds=args.evidence_rounds,
                )

            # 保留 gt_answer 和 evaluation 到输出中
            output["gt_answer"] = task.get("gt_answer")
            output["evaluation"] = task.get("evaluation")

            out_path = out_dir / f"{task_id}_log.json"
            out_path.write_text(
                json.dumps(output, ensure_ascii=False, indent=2, default=_json_default),
                encoding="utf-8",
            )

            failed_steps = [
                s for s in output.get("steps", [])
                if any(v.get("status") == "fail" for v in s.get("verdicts", []))
            ]
            # 收集反思策略统计
            reflections = [
                s.get("reflection", {}).get("strategy", "none")
                for s in output.get("steps", [])
                if s.get("reflection")
            ]

            elapsed = time.time() - ts
            has_fail = len(failed_steps) > 0
            if has_fail:
                failed += 1
            else:
                success += 1

            print(f"{'FAIL' if has_fail else 'OK'} ({elapsed:.1f}s, {len(failed_steps)} failed steps)")

            summary.append({
                "task_id": task_id,
                "status": "fail" if has_fail else "ok",
                "log": str(out_path),
                "elapsed_s": round(elapsed, 2),
                "total_steps": len(output.get("steps", [])),
                "failed_steps": len(failed_steps),
                "failed_rules": [
                    v.get("rule_id")
                    for s in failed_steps
                    for v in s.get("verdicts", [])
                    if v.get("status") == "fail"
                ],
                "reflection_strategies": reflections,
                "retries_total": sum(s.get("retries", 0) for s in output.get("steps", [])),
                "final_answer": output.get("final_answer", ""),
                "gt_answer": task.get("gt_answer"),
            })

        except Exception as e:
            elapsed = time.time() - ts
            errors += 1
            print(f"ERROR ({elapsed:.1f}s): {e}")
            summary.append({
                "task_id": task_id,
                "status": "error",
                "error": str(e),
                "traceback": traceback.format_exc(),
                "elapsed_s": round(elapsed, 2),
            })

    # 写入汇总
    total_time = time.time() - t0
    run_summary = {
        "total": len(task_files),
        "success": success,
        "failed": failed,
        "errors": errors,
        "total_time_s": round(total_time, 1),
        "config": {
            "tasks_dir": str(tasks_dir),
            "mode": args.mode,
            "rules": args.rules,
            "max_retries": args.max_retries,
            "max_replans": args.max_replans if args.mode == "e2e" else 0,
            "evidence_rounds": args.evidence_rounds,
            "image_size": args.image_size or "auto",
            "task_range": f"{args.start}..{end}",
            "planner_url": os.environ.get("PLANNER_BASE_URL", ""),
            "reasoner_url": os.environ.get("REASONER_BASE_URL", ""),
            "reflector_url": os.environ.get("REFLECTOR_BASE_URL", ""),
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        },
        "tasks": summary,
    }
    summary_path = out_dir / "run_summary.json"
    summary_path.write_text(
        json.dumps(run_summary, ensure_ascii=False, indent=2, default=_json_default),
        encoding="utf-8",
    )
    print(f"\nDone: {success} ok, {failed} fail, {errors} error in {total_time:.1f}s")
    print(f"Summary: {summary_path}")


if __name__ == "__main__":
    main()

