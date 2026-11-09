"""Explicit allocator budget plus free-memory margin, not GPU reservation."""
import math


def budget_ready(readings,*,allocator_gib=42.,free_margin_gib=2.5):
    """Rows are (used MiB, free MiB, utilization percent).

    Margin covers memory outside the allocator and some neighbor growth. It
    cannot guarantee future external allocations or eliminate interference.
    This is opt-in; existing supervisors retain their stricter default guard.
    """
    if (not math.isfinite(allocator_gib) or not math.isfinite(free_margin_gib) or
            allocator_gib<=0 or free_margin_gib<2.):
        raise ValueError('Positive allocator budget and at least 2 GiB free margin required')
    required=math.ceil((allocator_gib+free_margin_gib)*1024)
    return len(readings)>=4 and all(free>=required and 0<=util<10
        for used,free,util in readings[:4])
