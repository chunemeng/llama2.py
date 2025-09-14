from ops.time_util import time_in_ms, store_time


def cast(value, to_type):
    """
    Casts the given value to the specified type.

    Parameters:
    value: The value to be cast.
    to_type: The type to which the value should be cast. This can be a type object (e.g., int, float, str).

    Returns:
    The value cast to the specified type.
    """
    t0 = time_in_ms()
    try:
        v = value.to(to_type)
        t1 = time_in_ms()
        store_time('cast', t1 - t0)
        return v
    except (ValueError, TypeError) as e:
        raise ValueError(f"Cannot cast {value} to {to_type}: {e}")
