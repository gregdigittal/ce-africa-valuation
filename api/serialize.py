from __future__ import annotations

import math
from datetime import date, datetime
from typing import Any, Dict, List, Union


JSONScalar = Union[str, int, float, bool, None]
JSONType = Union[JSONScalar, List["JSONType"], Dict[str, "JSONType"]]


def to_jsonable(obj: Any) -> JSONType:
    """
    Convert arbitrary python objects into JSON-serializable structures.
    - Converts NaN/inf -> None
    - Converts datetime/date -> isoformat
    - Converts dict/list/tuple recursively
    - Best-effort fallback: str(obj)
    """
    if obj is None:
        return None

    if isinstance(obj, (str, bool, int)):
        return obj

    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None

    if isinstance(obj, (datetime, date)):
        return obj.isoformat()

    if isinstance(obj, dict):
        out: Dict[str, JSONType] = {}
        for k, v in obj.items():
            out[str(k)] = to_jsonable(v)
        return out

    if isinstance(obj, (list, tuple)):
        return [to_jsonable(v) for v in obj]

    # Numpy/pandas scalars often behave like numbers but aren't instances of float/int.
    try:
        if hasattr(obj, "item") and callable(obj.item):
            return to_jsonable(obj.item())
    except Exception:
        pass

    return str(obj)

