class Registry:
    def __init__(self):
        self._registry = {}

    def register(self, name, obj):
        if name in self._registry:
            raise KeyError(f"Object with name '{name}' is already registered.")
        self._registry[name] = obj

    def get(self, name):
        if name not in self._registry:
            raise KeyError(f"Object with name '{name}' is not registered.")
        return self._registry[name]

    def list(self):
        return list(self._registry.keys())