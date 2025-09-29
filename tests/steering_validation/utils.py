import gc

import torch

from wisent_guard.core.utils.device import empty_device_cache, resolve_default_device


def aggressive_memory_cleanup():
    """Aggressively clean GPU memory."""
    gc.collect()
    device_kind = resolve_default_device()
    empty_device_cache(device_kind)
    if device_kind == "cuda" and torch.cuda.is_available():
        torch.cuda.synchronize()
        # Force garbage collection multiple times
        for _ in range(3):
            gc.collect()
            empty_device_cache("cuda")
