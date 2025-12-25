"""
Supabase pagination helpers.

Supabase (PostgREST) commonly caps response payloads to 1000 rows per request.
If you `.select(...).execute()` without paging, you may silently receive only the
first ~1000 rows. This module provides a tiny helper to fetch all rows via
`.range()` pagination.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional


def fetch_all_rows(
    query: Any,
    *,
    page_size: int = 1000,
    max_rows: Optional[int] = None,
    order_by: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """
    Fetch all rows for a supabase-py query using `.range()` pagination.

    Args:
        query: A supabase-py request builder (e.g., client.table(...).select(...))
        page_size: Page size (Supabase max rows per request is typically 1000)
        max_rows: Optional safety cap to prevent runaway fetches
        order_by: Optional column name for deterministic pagination (e.g. "id")
    """
    if page_size is None or int(page_size) <= 0:
        page_size = 1000
    page_size = int(page_size)

    # Best-effort deterministic ordering (avoids surprises when paging)
    if order_by:
        try:
            query = query.order(str(order_by))
        except Exception:
            pass

    # If the builder doesn't support `.range()`, fall back to single execute
    if not hasattr(query, "range"):
        try:
            resp = query.execute()
            return resp.data or []
        except Exception:
            return []

    out: List[Dict[str, Any]] = []
    start = 0
    while True:
        end = start + page_size - 1
        try:
            resp = query.range(start, end).execute()
        except Exception:
            # Fail gracefully (callers treat empty list as "no data")
            break
        data = resp.data or []
        out.extend(data)

        if len(data) < page_size:
            break
        start += page_size

        if max_rows is not None and len(out) >= int(max_rows):
            out = out[: int(max_rows)]
            break

    return out


