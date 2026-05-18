from typing import Dict, Any, Optional
from dataclasses import fields
import importlib
import inspect


def load_model_class(identifier: str, prefix: str = "models."):
    module_path, class_name = identifier.split('@')

    # Import the module
    module = importlib.import_module(prefix + module_path)
    cls = getattr(module, class_name)
    
    return cls


def get_model_source_path(identifier: str, prefix: str = "models."):
    module_path, class_name = identifier.split('@')

    module = importlib.import_module(prefix + module_path)
    return inspect.getsourcefile(module)


def dict_view(obj: Any) -> Dict[str, Any]:
    """Returns a dictionary view of a dataclass, without copy. dataclasses.asdict(...) will make a deepcopy of every field."""
    return {k.name: getattr(obj, k.name) for k in fields(obj)}


def last_boxed_only_string(string: str) -> Optional[str]:
    idx = string.rfind("\\boxed")
    if idx < 0:
        idx = string.rfind("\\fbox")
        if idx < 0:
            return None

    i = idx
    left_brace_idx = None
    right_brace_idx = None
    num_left_braces_open = 0
    while i < len(string):
        if string[i] == "{":
            num_left_braces_open += 1
            if left_brace_idx is None:
                left_brace_idx = i
        elif string[i] == "}":
            num_left_braces_open -= 1
            if num_left_braces_open == 0:
                right_brace_idx = i
                break

        i += 1
    
    if left_brace_idx is None or right_brace_idx is None:
        return None

    return string[left_brace_idx + 1: right_brace_idx].strip()


def compute_benchmark_micro_macro_avg(stats: dict[str, dict[str, int | float]]):
    results = {}
    total_acc = total_invalid = 0.0
    for k, v in stats.items():
        results[f"n_{k}"] = v["n"]
        results[f"acc_{k}"] = acc = v["correct"] / max(1, v["n"])
        results[f"invalid_{k}"] = invalid = v["invalid"] / max(1, v["n"])

        total_acc += acc
        total_invalid += invalid
    
    results["n"] = len(stats)
    results["acc"] = total_acc / max(1, len(stats))
    results["invalid"] = total_invalid / max(1, len(stats))
    return results
