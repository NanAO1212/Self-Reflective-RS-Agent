from __future__ import annotations

import argparse
import json
from pathlib import Path


def _load_tool_descs(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _infer_input_type(name: str) -> str:
    if name in {"image", "pre_image", "post_image"}:
        return "image"
    if name in {"top1", "flag"}:
        return "bool"
    if name in {"k"}:
        return "integer"
    return "text"


def _infer_output_type(name: str) -> str:
    # ThinkGeo dataset uses text outputs in tool schema.
    return "text"


def _build_io_entries(io_dict: dict, is_input: bool) -> list[dict]:
    entries = []
    for name, desc in io_dict.items():
        entry = {
            "type": _infer_input_type(name) if is_input else _infer_output_type(name),
            "name": name if is_input else None,
            "description": desc if desc else None,
            "optional": False,
            "default": None,
            "filetype": None,
        }
        # Heuristic: optional boolean flags
        if is_input and name in {"top1", "flag"}:
            entry["optional"] = True
            entry["default"] = True
        if is_input and name == "k":
            entry["optional"] = True
            entry["default"] = 10
        entries.append(entry)
    return entries


def map_to_thinkgeo(tool_descs: dict) -> dict:
    tools = []
    for tool_name, meta in tool_descs.items():
        tool = {
            "name": tool_name,
            "description": meta.get("purpose"),
            "inputs": _build_io_entries(meta.get("inputs", {}), is_input=True),
            "outputs": _build_io_entries(meta.get("outputs", {}), is_input=False),
        }
        tools.append(tool)
    return {"tools": tools}


def main():
    parser = argparse.ArgumentParser(description="Map tool_descs_rs.json to ThinkGeo-style tool schema list")
    parser.add_argument("--src", default="tool_descs_rs.json", help="Source tool descriptions JSON")
    parser.add_argument("--out", default="thinkgeo_tools.json", help="Output ThinkGeo tool list JSON")
    args = parser.parse_args()

    src = Path(args.src)
    out = Path(args.out)
    tool_descs = _load_tool_descs(src)
    mapped = map_to_thinkgeo(tool_descs)
    out.write_text(json.dumps(mapped, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
