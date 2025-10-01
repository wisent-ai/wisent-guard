from __future__ import annotations
import re
from typing import Dict, List, Optional, Tuple

__all__ = [
    "to_bool", "parse_layers", "parse_kv", "parse_natural_tokens",
    "BOOL_TRUE", "BOOL_FALSE", "DTYPE_MAP", "TOP_KEYS"
]

BOOL_TRUE = {"1", "true", "yes", "y", "on"}
BOOL_FALSE = {"0", "false", "no", "n", "off"}

DTYPE_MAP = {None: None, "float32": "float32", "float16": "float16", "bfloat16": "bfloat16"}

TOP_KEYS = {
    "model", "loader", "loaders_location", "methods_location", "method", "layers",
    "aggregation", "device", "store_device", "dtype", "save_dir",
    "return_full_sequence", "normalize_layers", "interactive", "plan-only", "plan_only", "confirm"
}

def to_bool(s: str) -> bool:
    ls = s.lower()
    if ls in BOOL_TRUE: return True
    if ls in BOOL_FALSE: return False
    raise ValueError(f"Expected a boolean (true/false), got {s!r}")

def parse_layers(spec: Optional[str]) -> Optional[str]:
    if not spec: return None
    s = spec.strip().replace(" ", "")
    if re.match(r"^\d+(,\d+)*$", s):  # 5,7,9
        return s
    s = s.replace("-", "..").replace(":", "..")
    if re.match(r"^\d+\.\.(\d+)$", s): return s
    if re.match(r"^\d+$", s): return s
    raise ValueError("Invalid layers spec. Use '12' or '5,10,20' or '10..20' (also '10-20'/'10:20').")

def parse_kv(items: List[str]) -> Dict[str, object]:
    out: Dict[str, object] = {}
    for item in items:
        if "=" not in item:
            raise ValueError(f"Expected key=value, got: {item!r}")
        k, v = item.split("=", 1)
        k, v = k.strip(), v.strip()
        lv = v.lower()
        if lv in {"true", "false"}:
            out[k] = (lv == "true")
        else:
            try:
                out[k] = float(v) if "." in v else int(v)
            except ValueError:
                out[k] = v
    return out

def parse_natural_tokens(tokens: List[str]) -> Tuple[Dict[str, str], Dict[str, str], Dict[str, str]]:
    """
    Return (top_level, loader_kwargs, method_kwargs).
    Understands:
      - key value
      - key=value
      - sections: loader ..., method ...
      - dotted keys: loader.path=..., method.alpha=...
    """
    top: Dict[str, str] = {}
    lkw: Dict[str, str] = {}
    mkw: Dict[str, str] = {}
    ctx: Optional[str] = None

    i = 0
    while i < len(tokens):
        tok = tokens[i]

        if "=" in tok:
            k, v = tok.split("=", 1)
            k, v = k.strip(), v.strip()
            if k.startswith("loader."):
                lkw[k.split(".", 1)[1]] = v
            elif k.startswith("method."):
                mkw[k.split(".", 1)[1]] = v
            elif k in TOP_KEYS:
                top[k] = v
            elif ctx == "loader":
                lkw[k] = v
            elif ctx == "method":
                mkw[k] = v
            else:
                top[k] = v
            i += 1
            continue

        lt = tok.lower()
        if lt in {"loader", "method"}:
            ctx = lt
            if i + 1 < len(tokens) and "=" not in tokens[i + 1] and tokens[i + 1].lower() not in TOP_KEYS:
                if lt == "loader":
                    top["loader"] = tokens[i + 1]
                else:
                    top["method"] = tokens[i + 1]
                i += 2
            else:
                i += 1
            continue

        if lt in TOP_KEYS:
            if i + 1 >= len(tokens) or tokens[i + 1].lower() in TOP_KEYS or "=" in tokens[i + 1]:
                top[lt] = "true"
                i += 1
            else:
                top[lt] = tokens[i + 1]
                i += 2
            ctx = None
            continue

        if ctx in {"loader", "method"}:
            if i + 1 < len(tokens) and "=" not in tokens[i + 1] and tokens[i + 1].lower() not in TOP_KEYS:
                key = tok
                val = tokens[i + 1]
                (lkw if ctx == "loader" else mkw)[key] = val
                i += 2
            else:
                (lkw if ctx == "loader" else mkw)[tok] = "true"
                i += 1
            continue

        i += 1

    return top, lkw, mkw
