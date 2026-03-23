"""
evaluate.py — 全量评估脚本，覆盖 ThinkGeo 对齐指标 + 三个创新点消融指标。

用法:
    python evaluate.py --log_dir logs/full_run
    python evaluate.py --log_dir logs/full_run --tasks_dir tasks/thinkgeo_full
"""

import argparse
import json
import re
import glob
from pathlib import Path
from collections import Counter, defaultdict


# ─────────────────────────── answer matching ───────────────────────────

def _kw_match_word_boundary(kw: str, text: str) -> bool:
    """使用词边界的关键词匹配（对齐 ThinkGeo 官方 iscorrect）。"""
    return bool(re.search(r'\b' + re.escape(kw) + r'\b', text, re.IGNORECASE))


def answer_match(final_answer: str, gt_answer: dict | list | None) -> bool:
    """对齐 ThinkGeo 官方 iscorrect 逻辑：
    - whitelist 中每个 group 内的关键词是 OR 关系（任一命中即可）
    - 所有 group 都必须通过（AND 关系）
    兼容两种 gt_answer 格式：
      - dict: {"whitelist": [[alias, ...], ...], "blacklist": [kw, ...]}
      - list: ["候选答案1", "候选答案2", ...]  → 视为每个元素单独成组的 whitelist
    """
    if not gt_answer or not final_answer:
        return False
    text = str(final_answer).lower().strip()

    # 统一为 dict 格式（list 格式视为单个 OR 组）
    if isinstance(gt_answer, list):
        gt_answer = {"whitelist": [[str(a) for a in gt_answer]], "blacklist": []}

    # blacklist 检查（对齐官方：用词边界 OR 匹配）
    blacklist = gt_answer.get("blacklist") or []
    if blacklist:
        flat = [alias for group in blacklist for alias in
                (group if isinstance(group, list) else [group])]
        pattern_bk = r'\b(?:' + '|'.join(re.escape(str(a).lower()) for a in flat) + r')\b'
        if re.search(pattern_bk, text, re.IGNORECASE):
            return False

    # whitelist 检查：每个 group 内 OR，所有 group AND
    whitelist = gt_answer.get("whitelist") or []
    if not whitelist:
        return False
    count = 0
    for aliases in whitelist:
        pattern = r'\b(?:' + '|'.join(re.escape(str(a).lower()) for a in aliases) + r')\b'
        if re.search(pattern, text, re.IGNORECASE):
            count += 1
    return count == len(whitelist)


# ─────────────────────────── per-task metrics ───────────────────────────

def _group_verdicts(verdicts: list) -> dict:
    """按 rule_id 前缀分组统计 pass/fail。"""
    groups = defaultdict(lambda: {"pass": 0, "fail": 0, "total": 0})
    for v in verdicts:
        rid = v.get("rule_id", "")
        prefix = rid.split("-")[0] if "-" in rid else rid
        status = v.get("status", "pass")
        groups[prefix][status] += 1
        groups[prefix]["total"] += 1
        groups["ALL"][status] += 1
        groups["ALL"]["total"] += 1
    return dict(groups)

def compute_task_metrics(log: dict, gt_task: dict | None = None) -> dict:
    """从单个任务 log 计算所有指标，返回 metrics dict。"""
    steps = log.get("steps", [])
    gt_answer = log.get("gt_answer") or (gt_task or {}).get("gt_answer")
    final_answer = log.get("final_answer", "")
    gt_steps = (gt_task or {}).get("steps", [])

    # ── ThinkGeo 对齐指标 ──
    ans_acc = answer_match(final_answer, gt_answer)

    # ToolCallAcc: 逐步比较工具名
    tool_matches = 0
    tool_total = max(len(steps), len(gt_steps))
    for i in range(min(len(steps), len(gt_steps))):
        pred_tool = steps[i].get("tool") or steps[i].get("action", "")
        gt_tool = gt_steps[i].get("tool_name", "")
        if pred_tool == gt_tool:
            tool_matches += 1
    tool_call_acc = tool_matches / tool_total if tool_total > 0 else 0.0

    # StepSuccessRate: 每步 verdicts 全 pass 的比例（无 verdict 视为通过）
    step_all_pass = 0
    for s in steps:
        verdicts = s.get("verdicts", [])
        if not verdicts or all(v.get("status") == "pass" for v in verdicts):
            step_all_pass += 1
    step_success_rate = step_all_pass / len(steps) if steps else 0.0

    # ── 创新点 A: 知识增强工具描述 ──
    # ToolSelection@1: 首个关键工具是否正确
    tool_sel_1 = False
    if steps and gt_steps:
        tool_sel_1 = (steps[0].get("tool") == gt_steps[0].get("tool_name"))

    # ParamValidRate: PX/PV 规则通过比例
    px_pv_pass = 0
    px_pv_total = 0
    for s in steps:
        for v in s.get("verdicts", []):
            rid = v.get("rule_id", "")
            if rid.startswith("PX") or rid.startswith("PV"):
                px_pv_total += 1
                if v.get("status") == "pass":
                    px_pv_pass += 1
    param_valid_rate = px_pv_pass / px_pv_total if px_pv_total > 0 else 1.0

    # Pass@1: 第一轮 verifier 全通过比例（每步首次执行即通过，无 verdict 视为通过）
    pass_at_1_steps = sum(1 for s in steps if s.get("retries", 0) == 0
                         and (not s.get("verdicts") or all(v.get("status") == "pass" for v in s.get("verdicts", []))))
    pass_at_1 = pass_at_1_steps / len(steps) if steps else 0.0

    # AvgRetries
    total_retries = sum(s.get("retries", 0) for s in steps)
    avg_retries = total_retries / len(steps) if steps else 0.0

    # ── 创新点 B: 空间验证 ──
    rg_pass = rg_total = 0
    gl_pass = gl_total = 0
    pv20_21_pass = pv20_21_total = 0
    for s in steps:
        for v in s.get("verdicts", []):
            rid = v.get("rule_id", "")
            is_pass = v.get("status") == "pass"
            if rid.startswith("RG"):
                rg_total += 1
                rg_pass += int(is_pass)
            if rid.startswith("GL"):
                gl_total += 1
                gl_pass += int(is_pass)
            if rid in ("PV-20", "PV-21"):
                pv20_21_total += 1
                pv20_21_pass += int(is_pass)

    spatial_consistency = rg_pass / rg_total if rg_total > 0 else 1.0
    semantic_consistency = gl_pass / gl_total if gl_total > 0 else 1.0
    cross_round_stability = pv20_21_pass / pv20_21_total if pv20_21_total > 0 else 1.0

    # FalsePositiveRecoveryRate: 首次 fail 后 retry 成功的比例
    first_fail_count = 0
    recovered_count = 0
    for s in steps:
        has_fail = any(v.get("status") == "fail" for v in s.get("verdicts", []))
        retries = s.get("retries", 0)
        if retries > 0:
            # 有 retry 说明首次有 fail
            first_fail_count += 1
            # 最终 verdicts 全 pass 说明恢复成功
            if all(v.get("status") == "pass" for v in s.get("verdicts", [])):
                recovered_count += 1
        elif has_fail:
            # 有 fail 但没 retry（达到上限或无需重试）
            first_fail_count += 1
    fp_recovery = recovered_count / first_fail_count if first_fail_count > 0 else None

    # ── 创新点 C: 自反思机制 ──
    # ErrorTypeBreakdown
    error_types = Counter()
    for s in steps:
        for v in s.get("verdicts", []):
            if v.get("status") == "fail":
                et = v.get("error_type", "unknown")
                error_types[et] += 1

    # ReflectionStrategyDist
    strategy_dist = Counter()
    for s in steps:
        ref = s.get("reflection", {})
        strategy = ref.get("strategy")
        # 兼容旧格式：如果没有 strategy 字段，根据 retries 推断
        if not strategy:
            if s.get("retries", 0) > 0:
                strategy = "rule_based"
            else:
                strategy = "none"
        strategy_dist[strategy] += 1

    # RecoverySuccessRate: 任务级——首次有 fail 步骤后最终 answer 正确
    has_any_fail = any(
        any(v.get("status") == "fail" for v in s.get("verdicts", []))
        or s.get("retries", 0) > 0
        for s in steps
    )

    return {
        # ThinkGeo 对齐
        "answer_acc": ans_acc,
        "tool_call_acc": tool_call_acc,
        "step_success_rate": step_success_rate,
        # 创新点 A
        "tool_selection_at_1": tool_sel_1,
        "param_valid_rate": param_valid_rate,
        "pass_at_1": pass_at_1,
        "avg_retries": avg_retries,
        # 创新点 B
        "spatial_consistency": spatial_consistency,
        "semantic_consistency": semantic_consistency,
        "cross_round_stability": cross_round_stability,
        "fp_recovery_rate": fp_recovery,
        # 创新点 C
        "task_success": ans_acc,  # 同 answer_acc
        "has_any_fail": has_any_fail,
        "recovery_success": ans_acc and has_any_fail,
        "error_types": dict(error_types),
        "strategy_dist": dict(strategy_dist),
        "total_retries": total_retries,
        "num_steps": len(steps),
        # 元信息
        "task_id": log.get("task_id"),
        "has_gt": gt_answer is not None,
    }


# ─────────────────────────── aggregation ───────────────────────────

def _safe_mean(values):
    vals = [v for v in values if v is not None]
    return sum(vals) / len(vals) if vals else 0.0


def _safe_ratio(count, total):
    return count / total if total > 0 else 0.0


def aggregate(task_metrics_list: list) -> dict:
    """汇总所有任务指标，生成最终 report。"""
    n = len(task_metrics_list)
    if n == 0:
        return {"error": "no tasks to evaluate"}

    # 只统计有 gt_answer 的任务做 accuracy 指标
    with_gt = [m for m in task_metrics_list if m["has_gt"]]
    n_gt = len(with_gt)

    # ── ThinkGeo 对齐 ──
    answer_acc = _safe_mean([m["answer_acc"] for m in with_gt])
    tool_call_acc = _safe_mean([m["tool_call_acc"] for m in task_metrics_list])
    step_success_rate = _safe_mean([m["step_success_rate"] for m in task_metrics_list])

    # ── 创新点 A ──
    tool_sel_1 = _safe_mean([m["tool_selection_at_1"] for m in task_metrics_list])
    param_valid_rate = _safe_mean([m["param_valid_rate"] for m in task_metrics_list])
    pass_at_1 = _safe_mean([m["pass_at_1"] for m in task_metrics_list])
    avg_retries = _safe_mean([m["avg_retries"] for m in task_metrics_list])

    # ── 创新点 B ──
    spatial_consistency = _safe_mean([m["spatial_consistency"] for m in task_metrics_list])
    semantic_consistency = _safe_mean([m["semantic_consistency"] for m in task_metrics_list])
    cross_round_stability = _safe_mean([m["cross_round_stability"] for m in task_metrics_list])
    fp_recoveries = [m["fp_recovery_rate"] for m in task_metrics_list if m["fp_recovery_rate"] is not None]
    fp_recovery_rate = _safe_mean(fp_recoveries) if fp_recoveries else None

    # ── 创新点 C ──
    task_success_rate = _safe_mean([m["task_success"] for m in with_gt])
    # RecoverySuccessRate: 在有 fail 的任务中，最终正确的比例
    tasks_with_fail = [m for m in with_gt if m["has_any_fail"]]
    recovery_success_rate = _safe_ratio(
        sum(1 for m in tasks_with_fail if m["recovery_success"]),
        len(tasks_with_fail),
    )

    # ErrorTypeBreakdown: 全局汇总
    error_type_total = Counter()
    for m in task_metrics_list:
        error_type_total.update(m["error_types"])

    # ReflectionStrategyDist: 全局汇总
    strategy_total = Counter()
    for m in task_metrics_list:
        strategy_total.update(m["strategy_dist"])

    total_retries_all = sum(m["total_retries"] for m in task_metrics_list)
    total_steps_all = sum(m["num_steps"] for m in task_metrics_list)

    return {
        "num_tasks": n,
        "num_tasks_with_gt": n_gt,
        "total_steps": total_steps_all,
        "total_retries": total_retries_all,
        "thinkgeo_aligned": {
            "AnswerAcc": round(answer_acc, 4),
            "ToolCallAcc": round(tool_call_acc, 4),
            "StepSuccessRate": round(step_success_rate, 4),
        },
        "innovation_A_tool_description": {
            "ToolSelection@1": round(tool_sel_1, 4),
            "ParamValidRate": round(param_valid_rate, 4),
            "Pass@1": round(pass_at_1, 4),
            "AvgRetries": round(avg_retries, 4),
        },
        "innovation_B_spatial_verification": {
            "SpatialConsistencyScore": round(spatial_consistency, 4),
            "SemanticConsistencyScore": round(semantic_consistency, 4),
            "CrossRoundStability": round(cross_round_stability, 4),
            "FalsePositiveRecoveryRate": round(fp_recovery_rate, 4) if fp_recovery_rate is not None else "N/A",
        },
        "innovation_C_self_reflection": {
            "TaskSuccessRate": round(task_success_rate, 4),
            "RecoverySuccessRate": round(recovery_success_rate, 4),
            "ErrorTypeBreakdown": dict(error_type_total),
            "ReflectionStrategyDist": dict(strategy_total),
        },
    }


# ─────────────────────────── pretty print ───────────────────────────

def print_report(report: dict):
    """终端打印汇总表格。"""
    print("=" * 64)
    print(f"  评估报告  |  {report['num_tasks']} tasks  "
          f"({report['num_tasks_with_gt']} with GT)  "
          f"|  {report['total_steps']} steps  "
          f"|  {report['total_retries']} retries")
    print("=" * 64)

    sections = [
        ("ThinkGeo 对齐指标", "thinkgeo_aligned"),
        ("创新点 A — 知识增强工具描述", "innovation_A_tool_description"),
        ("创新点 B — 空间验证", "innovation_B_spatial_verification"),
        ("创新点 C — 自反思机制", "innovation_C_self_reflection"),
    ]
    for title, key in sections:
        data = report[key]
        print(f"\n  [{title}]")
        for k, v in data.items():
            if isinstance(v, dict):
                print(f"    {k}:")
                for kk, vv in v.items():
                    print(f"      {kk:30s} {vv}")
            elif isinstance(v, float):
                print(f"    {k:30s} {v:.4f}")
            else:
                print(f"    {k:30s} {v}")
    print("\n" + "=" * 64)


# ─────────────────────────── main ───────────────────────────

def load_logs(log_dir: str) -> list[dict]:
    """加载所有 thinkgeo_*_log.json。"""
    pattern = str(Path(log_dir) / "thinkgeo_*_log.json")
    files = sorted(glob.glob(pattern))
    logs = []
    for f in files:
        try:
            logs.append(json.loads(Path(f).read_text(encoding="utf-8")))
        except Exception as e:
            print(f"  WARN: 跳过 {f}: {e}")
    return logs


def load_gt_tasks(tasks_dir: str) -> dict:
    """加载 GT 任务 JSON，返回 {task_id: task_dict}。"""
    if not tasks_dir:
        return {}
    pattern = str(Path(tasks_dir) / "thinkgeo_*.json")
    files = glob.glob(pattern)
    tasks = {}
    for f in files:
        try:
            t = json.loads(Path(f).read_text(encoding="utf-8"))
            tasks[t.get("task_id", Path(f).stem)] = t
        except Exception:
            pass
    return tasks


def _has_unavailable_tool(log: dict) -> bool:
    """检查任务日志中是否有工具返回 UNAVAILABLE。"""
    for s in log.get("steps", []):
        if "UNAVAILABLE" in str(s.get("tool_output", "")):
            return True
    return False


def main():
    ap = argparse.ArgumentParser(description="评估全量实验结果")
    ap.add_argument("--log_dir", default="logs/full_run", help="日志目录")
    ap.add_argument("--tasks_dir", default="tasks/thinkgeo_full", help="GT 任务目录")
    ap.add_argument("--out", default=None, help="输出 JSON 路径（默认 log_dir/eval_report.json）")
    ap.add_argument("--exclude_unavailable", action="store_true",
                    help="排除含 UNAVAILABLE 工具输出的任务")
    args = ap.parse_args()

    log_dir = Path(args.log_dir)
    out_path = Path(args.out) if args.out else log_dir / "eval_report.json"

    print(f"加载日志: {log_dir}")
    logs = load_logs(str(log_dir))
    print(f"  共 {len(logs)} 个任务日志")

    if args.exclude_unavailable:
        before = len(logs)
        logs = [l for l in logs if not _has_unavailable_tool(l)]
        excluded = before - len(logs)
        print(f"  排除 {excluded} 个含 UNAVAILABLE 工具的任务，剩余 {len(logs)} 个")

    print(f"加载 GT 任务: {args.tasks_dir}")
    gt_tasks = load_gt_tasks(args.tasks_dir)
    print(f"  共 {len(gt_tasks)} 个 GT 任务")

    # 逐任务计算指标
    all_metrics = []
    for log in logs:
        tid = log.get("task_id", "")
        gt_task = gt_tasks.get(tid)
        m = compute_task_metrics(log, gt_task)
        all_metrics.append(m)

    # 汇总
    report = aggregate(all_metrics)
    report["per_task"] = all_metrics

    # 打印
    print_report(report)

    # 保存
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"评估报告已保存: {out_path}")


if __name__ == "__main__":
    main()
