import time
import functools

def timeit(key: str = "time/total_time", multi_out_idx=0):
    """
    Decorator with parameter `key` indicating where to store runtime.
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            t0 = time.perf_counter()
            out = func(*args, **kwargs)
            t1 = time.perf_counter()

            if out is None:
                return {key: t1 - t0}
            
            if isinstance(out, (tuple, list)):
                out[multi_out_idx][key] = t1 - t0
            else:
                out[key] = t1 - t0
            return out
        return wrapper
    return decorator