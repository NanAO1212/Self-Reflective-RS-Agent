def _unavailable_tool(name, error):
    def _tool(*args, **kwargs):
        return f"UNAVAILABLE:{name}: {error}"
    return _tool


def _lazy_import_tool(name):
    if name == "ChangeDetection":
        try:
            from agentlego.benchmark import ChangeDetection
            return ChangeDetection(device="cpu")
        except Exception:
            try:
                from benchmark import ChangeDetection  # type: ignore
                return ChangeDetection(device="cpu")
            except Exception:
                # fall back to loading from a local benchmark.py if provided
                try:
                    import importlib.util
                    import os
                    from pathlib import Path

                    hint = os.getenv("AGENTLEGO_BENCHMARK_PATH", "").strip()
                    candidates = []
                    if hint:
                        candidates.append(Path(hint))
                    # Common local layout under ThinkGeo repo
                    candidates.append(Path("E:/Agentic remote sensing reasoning/thinkgeo/ThinkGeo/agentlego/benchmark.py"))
                    for path in candidates:
                        if path.exists():
                            spec = importlib.util.spec_from_file_location("agentlego_benchmark", path)
                            if spec and spec.loader:
                                module = importlib.util.module_from_spec(spec)
                                spec.loader.exec_module(module)
                                ChangeDetection = getattr(module, "ChangeDetection", None)
                                if ChangeDetection is not None:
                                    return ChangeDetection(device="cpu")
                except Exception:
                    pass
            # fall back to standard load_tool path
            pass
    try:
        from agentlego.apis import load_tool
    except Exception as exc:
        raise RuntimeError(
            "agentlego is not available. Install dependencies or "
            "wire your own tool functions."
        ) from exc
    tool = load_tool(name, device="cpu")
    return tool


def _safe_register(registry, name):
    try:
        registry.register(name, _lazy_import_tool(name))
    except Exception as exc:
        registry.register(name, _unavailable_tool(name, exc))


def build_default_registry(registry):
    from operations import draw_mask
    from serpapi_search import serpapi_search
    import os
    # Perception tools
    _safe_register(registry, "TextToBbox")
    _safe_register(registry, "ObjectDetection")
    _safe_register(registry, "CountGivenObject")
    _safe_register(registry, "SegmentObjectPixels")
    _safe_register(registry, "RegionAttributeDescription")
    # Backward-compat: use ImageRegionDescription when RegionAttributeDescription is missing.
    _safe_register(registry, "ImageRegionDescription")
    _safe_register(registry, "ImageDescription")
    _safe_register(registry, "ChangeDetection")
    _safe_register(registry, "OCR")
    # Logic tools
    _safe_register(registry, "Calculator")
    _safe_register(registry, "Solver")
    _safe_register(registry, "Plot")
    # Operation tools
    _safe_register(registry, "DrawBox")
    registry.register("DrawMask", draw_mask)
    _safe_register(registry, "AddText")
    if os.getenv("GOOGLE_SEARCH_PROVIDER", "").lower() == "serpapi":
        registry.register("GoogleSearch", serpapi_search)
    else:
        _safe_register(registry, "GoogleSearch")

    # Backward-compat: if ImageRegionDescription is available, alias it.
    try:
        registry.register("RegionAttributeDescription", registry.get("ImageRegionDescription"))
    except Exception:
        pass


def call_tool(tool, tool_name, tool_input):
    # Tool signatures are heterogeneous; map by name.
    try:
        if tool_name in ("TextToBbox", "ObjectDetection", "OCR", "ImageDescription"):
            return tool(tool_input["image"], *(tool_input.get("text"),) if "text" in tool_input else ())

        if tool_name == "CountGivenObject":
            text = tool_input.get("text") or tool_input.get("object", "object")
            return tool(tool_input["image"], text, tool_input.get("bbox"))

        if tool_name == "SegmentObjectPixels":
            return tool(tool_input["image"], tool_input.get("text", "object"), tool_input.get("flag", True))

        if tool_name == "RegionAttributeDescription" or tool_name == "ImageRegionDescription":
            return tool(tool_input["image"], tool_input["bbox"], tool_input["attribute"])

        if tool_name == "ChangeDetection":
            return tool(tool_input["text"], tool_input["pre_image"], tool_input["post_image"])

        if tool_name == "Calculator":
            # Reasoner 可能用 "expression" 或 "formula" 字段
            expr = tool_input.get("expression") or tool_input.get("formula") or tool_input.get("command", "")
            # 如果提供了 variables 字典，替换变量后再求值
            variables = tool_input.get("variables")
            if variables and isinstance(variables, dict):
                for var_name, var_val in variables.items():
                    expr = expr.replace(str(var_name), str(var_val))
            # 如果没有可用的表达式，尝试从结构化参数构建
            if not expr or expr.isspace():
                op = tool_input.get("operation", "")
                if op == "distance":
                    # 从 bbox 中心点计算距离
                    import math
                    try:
                        def _bbox_center(bbox_val):
                            if isinstance(bbox_val, str):
                                nums = [float(x) for x in bbox_val.strip("[]() ").split(",") if x.strip().replace('.','').replace('-','').isdigit()]
                            elif isinstance(bbox_val, (list, tuple)):
                                nums = [float(x) for x in bbox_val]
                            else:
                                return None
                            if len(nums) == 4:
                                return ((nums[0]+nums[2])/2, (nums[1]+nums[3])/2)
                            return None
                        c1 = _bbox_center(tool_input.get("bbox1"))
                        c2 = _bbox_center(tool_input.get("bbox2"))
                        gsd = float(tool_input.get("gsd_m_per_px", tool_input.get("gsd", 1)))
                        if c1 and c2:
                            dist = math.sqrt((c2[0]-c1[0])**2 + (c2[1]-c1[1])**2) * gsd
                            return f"{dist:.2f}"
                    except Exception:
                        pass
                elif op == "area":
                    try:
                        pixels = float(tool_input.get("pixels", tool_input.get("pixel_count", 0)))
                        gsd = float(tool_input.get("gsd_m_per_px", tool_input.get("gsd", 1)))
                        area = pixels * gsd * gsd
                        return f"{area:.2f}"
                    except Exception:
                        pass
                return f"UNAVAILABLE:Calculator: no valid expression, got {list(tool_input.keys())}"
            return tool(expr)

        if tool_name == "Solver":
            return tool(tool_input["command"])

        if tool_name == "Plot":
            return tool(tool_input["command"])

        if tool_name == "DrawBox":
            bbox = tool_input.get("bbox", "")
            # 跳过占位符 bbox（如 "calculated_flooded_house_bboxes"）
            if isinstance(bbox, str) and not any(c.isdigit() for c in bbox):
                return f"SKIPPED:DrawBox: bbox is a placeholder '{bbox}', not real coordinates."
            return tool(tool_input["image"], bbox, tool_input.get("annotation"))

        if tool_name == "DrawMask":
            return tool(
                tool_input["image"],
                tool_input.get("mask_path"),
                tool_input.get("polygon"),
                tool_input.get("bbox"),
                tool_input.get("color", "red"),
                tool_input.get("alpha", 0.4),
            )

        if tool_name == "AddText":
            return tool(tool_input["image"], tool_input["text"], tool_input["position"], tool_input.get("color", "red"))

        if tool_name == "GoogleSearch":
            return tool(tool_input["query"], tool_input.get("k", 10))

        raise ValueError(f"Unsupported tool: {tool_name}")
    except Exception as exc:
        return f"UNAVAILABLE:{tool_name}: {exc}"
