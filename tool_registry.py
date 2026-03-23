class ToolRegistry:
    def __init__(self):
        self._tools = {}

    def register(self, name, func):
        self._tools[name] = func

    def get(self, name):
        if name not in self._tools:
            # 未注册的工具返回一个兜底函数，不崩溃
            def _unregistered(*args, **kwargs):
                return f"UNAVAILABLE:{name}: Tool '{name}' is not registered."
            return _unregistered
        return self._tools[name]

    def list(self):
        return sorted(self._tools.keys())
