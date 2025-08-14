import gc

import torch


def aggressive_memory_cleanup():
    """Aggressively clean GPU memory."""
    gc.collect()
    torch.cuda.empty_cache()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        # Force garbage collection multiple times
        for _ in range(3):
            gc.collect()
            torch.cuda.empty_cache()
