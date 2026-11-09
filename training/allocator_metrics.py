"""Untimed per-step native allocator event counters, not estimated time savings."""
COUNTERS = ('num_alloc_retries','num_ooms','num_sync_all_streams',
            'num_device_alloc','num_device_free')


def counter_deltas(before, after):
    result={}
    for name in COUNTERS:
        if name not in before or name not in after:
            result[name]=None;continue
        a,b=before[name],after[name]
        if type(a) is not int or type(b) is not int or a<0 or b<a:
            raise ValueError('Allocator counters must be nonnegative monotonic integers')
        result[name]=b-a
    return result
