# SR²A: Self-Reflective Remote Sensing Agent

A multi-level self-reflective agent framework for remote sensing image analysis, built on the [ThinkGeo](https://github.com/Geo-BERT/ThinkGeo) benchmark.

## Architecture

```
Input (Query + Image)
        │
        ▼
   ┌─────────┐     ┌─────────────────────┐
   │ Planner  │◄····│  RS Domain Knowledge │  (Innovation A)
   └────┬─────┘     │  · Scene-Object Priors│
        │           │  · Size Constraints   │
        ▼           │  · GSD Rules          │
   ┌─────────┐     │  · Confusion Pairs    │
   │ Reasoner │◄····└─────────────────────┘
   └────┬─────┘
        │
        ▼
  ┌───────────────┐
  │ Tool Execution │  (Perception / Logic / Operation)
  └───────┬───────┘
          │
          ▼
  ┌──────────────────────┐
  │ Hierarchical Verifier │  (Innovation B)
  │  Pixel → Region →    │
  │  Global level checks  │
  └───────┬──────────────┘
          │ Verdicts
          ▼
  ┌──────────────────────┐
  │ Self-Reflective Module│  (Innovation C)
  │  Tier 1: Rule-Based   │
  │  Tier 2: LLM-Based    │
  └──┬─────────┬─────────┘
     │         │
   All Pass  Fail
     │         ├── Retry (corrective patches → Tool Execution)
     ▼         └── Replan (feedback → Reasoner)
 Final Answer
```

## Three Innovations

| | Innovation | Description |
|---|---|---|
| **A** | RS Domain Knowledge | Remote sensing priors (object sizes, scene-object mappings, GSD rules, confusion pairs) injected into Planner & Reasoner prompts |
| **B** | Hierarchical Verification | Three-level rule-based verifier: Pixel-level (bbox bounds, size), Region-level (overlap, aspect ratio, physical size priors), Global-level (semantic conflicts, hallucination detection) |
| **C** | Self-Reflective Module | Two-tier reflection: fast rule-based patches for common errors, LLM-based deep analysis for complex failures. Generates machine-readable corrective patches for automated retry |

## Two Evaluation Modes

- **SbS (Step-by-Step)**: Uses ground-truth tool call sequences from ThinkGeo. Evaluates verification & reflection quality independently of planning.
- **E2E (End-to-End)**: Planner classifies the task, Reasoner generates tool call plan autonomously, pipeline executes with full verify-reflect-retry loop.

## Project Structure

```
self_reflective_agent/
├── pipeline.py            # Main pipeline: SbS (run_pipeline) & E2E (run_from_query)
├── planner.py             # Task classification + plan decomposition via VLM
├── reasoner.py            # Tool call sequence generation via VLM
├── verifier.py            # Pixel-level & global-level rule verification
├── spatial_verifier.py    # Region-level spatial & semantic verification
├── reflector.py           # Two-tier reflection: rule-based + LLM-based
├── parser.py              # Tool output parsing (regex-based)
├── adapters.py            # Tool registry adapters (agentlego integration)
├── tool_registry.py       # Name → function registry
├── operations.py          # Mask overlay visualization utilities
├── evaluate.py            # Evaluation metrics (AnswerAcc, ToolCallAcc, etc.)
├── run_all.py             # Batch runner with checkpoint resume
├── convert_thinkgeo.py    # ThinkGeoBench.json → per-task JSON converter
│
├── tool_descs_rs.json             # Tool descriptions with RS-specific rules
├── tool_descs_rs_knowledge.json   # RS domain knowledge priors (Innovation A)
├── tool_parsing_rules.json        # Regex parsing rules per tool type
├── verifier_rules.md              # Verification rule documentation
│
├── prompts/               # Prompt templates (e.g., change detection)
├── tasks/                 # ThinkGeo benchmark task JSONs (432 tasks)
│   └── thinkgeo_full/     # Full dataset
│
├── paper/                 # Paper source
│   ├── main.tex           # LaTeX source
│   ├── main.pdf           # Compiled PDF
│   └── figures/           # Architecture & verification diagrams
│
└── docs/                  # Experiment plans & logs
```

## Core Modules

### Pipeline (`pipeline.py`)

Entry points:
- `run_pipeline(task, ...)` — SbS mode. Iterates GT steps, executes tools, verifies, reflects on failures, retries with patches.
- `run_from_query(question, image, ...)` — E2E mode. Calls Planner → Reasoner → `run_pipeline`. Supports replan cycles when failures persist.

**Retry logic** (per-step): execute → verify → if fail, `reflect_and_patch()` generates corrective patches → `apply_patches()` → retry (up to `max_retries`).

**Replan logic** (E2E only): after all steps, collect remaining failures → LLM reflector provides plan-level feedback → Reasoner generates new plan (up to `max_replans`).

### Planner (`planner.py`)

Classifies task type (counting / area / distance / change_detection / attribute / scene_description / multi_task) and generates subtask decomposition. Uses Qwen3-VL model with RS domain knowledge prompts.

### Reasoner (`reasoner.py`)

Generates executable tool call sequences as structured JSON. Ensures mandatory perception tools (ObjectDetection, SegmentObjectPixels) are included. Supports reflector feedback for replan cycles.

### Verifier (`verifier.py` + `spatial_verifier.py`)

22 verification rules across 4 levels:

| Level | Rules | Examples |
|-------|-------|---------|
| **Pixel (PX)** | PX-01~04 | Bbox out of bounds, negative pixel count, missing GSD |
| **Region (RG)** | RG-01, RG-10~22 | High IoU overlap, full-image bbox, extreme aspect ratio, physical size prior mismatch |
| **Evidence (PV)** | PV-10~30 | Cross-round localization inconsistency, unstable counts, cross-step drift |
| **Global (GL)** | GL-10~34 | Empty description, object-scene semantic conflict, abnormal count, hallucination |

### Reflector (`reflector.py`)

Two-tier reflection with automatic tier selection:
- **Tier 1 (Rule-based)**: Zero-latency. Maps rule IDs to specific fix patches (bbox clamping, tool switching, parameter overrides).
- **Tier 2 (LLM-based)**: Activated for complex cases (multiple error types, semantic conflicts, repeated failures). Returns structured diagnosis + corrective patches.

Patch format: `{input_overrides, switch_tool, retry_params, add_steps_before, add_steps_after}`

### Evaluation (`evaluate.py`)

Computes metrics aligned with ThinkGeo benchmark:
- **Core**: AnswerAcc, ToolCallAcc, StepSuccessRate
- **Innovation A**: ToolSelection@1, ParamValidRate
- **Innovation B**: SpatialConsistencyScore, SemanticConsistencyScore, FalsePositiveRecoveryRate
- **Innovation C**: RecoverySuccessRate, ErrorTypeBreakdown, ReflectionStrategyDist

## Running Experiments

### Prerequisites

- Python 3.10+, [agentlego](https://github.com/InternLM/agentlego) toolkit
- VLM API endpoint (Qwen3-VL-30B-A3B-Thinking or compatible)
- ThinkGeo benchmark images (not included in repo)

### Environment Variables

```bash
# Model endpoints
export PLANNER_BASE_URL="http://your-vlm-api/v1"
export REASONER_BASE_URL="http://your-vlm-api/v1"
export REFLECTOR_BASE_URL="http://your-vlm-api/v1"
export PLANNER_API_KEY="your-key"
export REASONER_API_KEY="your-key"
export REFLECTOR_API_KEY="your-key"

# Ablation switches
export DISABLE_RS_KNOWLEDGE=1      # Turn off Innovation A
export DISABLE_VERIFICATION=1      # Turn off Innovation B
export REFLECTOR_USE_LLM=0         # Force rule-based only (partial Innovation C ablation)
```

### Batch Execution

```bash
# Step-by-Step mode (uses ground-truth tool sequences)
python run_all.py --tasks_dir tasks/thinkgeo_full --out_dir logs/full_run --mode sbs

# End-to-End mode (autonomous planning)
python run_all.py --tasks_dir tasks/thinkgeo_full --out_dir logs/e2e_run --mode e2e

# Resume from checkpoint
python run_all.py --tasks_dir tasks/thinkgeo_full --out_dir logs/full_run --skip_existing

# Run subset of tasks
python run_all.py --tasks_dir tasks/thinkgeo_full --out_dir logs/test --start 0 --end 10
```

### Evaluation

```bash
python evaluate.py --log_dir logs/full_run --task_dir tasks/thinkgeo_full
```

## Tools (14 total)

| Category | Tools |
|----------|-------|
| **Perception** (8) | TextToBbox, ObjectDetection, CountGivenObject, SegmentObjectPixels, RegionAttributeDescription, ImageDescription, ChangeDetection, OCR |
| **Logic** (3) | Calculator, Solver, Plot |
| **Operation** (3) | DrawBox, DrawMask, AddText |

## Ablation Configurations

| Config | Innovation A | Innovation B | Innovation C |
|--------|:-----------:|:-----------:|:-----------:|
| **Full (SR²A)** | ON | ON | ON |
| w/o RS Knowledge | OFF | ON | ON |
| w/o Verification | ON | OFF | ON |
| w/o Reflection | ON | ON | OFF |
| Baseline | OFF | OFF | OFF |

## License

This project is for academic research purposes.
