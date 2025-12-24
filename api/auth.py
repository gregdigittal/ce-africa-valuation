from __future__ import annotations

import json
import os
import time
import urllib.request
from dataclasses import dataclass
from typing import Any, Dict, Optional

from jose import jwt
from jose.exceptions import JWTError


@dataclass
class AuthContext:
    user_id: str
    claims: Dict[str, Any]


_JWKS_CACHE: Dict[str, Any] = {"fetched_at": 0.0, "jwks": None}


def _get_supabase_url() -> str:
    url = os.getenv("SUPABASE_URL", "").strip().rstrip("/")
    if not url:
        raise RuntimeError("SUPABASE_URL is not set")
    return url


def _fetch_jwks() -> Dict[str, Any]:
    # Supabase JWKS endpoint
    url = _get_supabase_url() + "/auth/v1/keys"
    req = urllib.request.Request(url, headers={"Content-Type": "application/json"}, method="GET")
    with urllib.request.urlopen(req, timeout=10) as resp:
        raw = resp.read().decode("utf-8")
    return json.loads(raw)


def get_jwks(ttl_seconds: int = 3600) -> Dict[str, Any]:
    now = time.time()
    cached = _JWKS_CACHE.get("jwks")
    fetched_at = float(_JWKS_CACHE.get("fetched_at") or 0.0)
    if cached and (now - fetched_at) < ttl_seconds:
        return cached
    jwks = _fetch_jwks()
    _JWKS_CACHE["jwks"] = jwks
    _JWKS_CACHE["fetched_at"] = now
    return jwks


def parse_bearer_token(authorization_header: Optional[str]) -> Optional[str]:
    if not authorization_header:
        return None
    parts = authorization_header.strip().split()
    if len(parts) != 2:
        return None
    scheme, token = parts
    if scheme.lower() != "bearer":
        return None
    return token.strip() or None


def verify_supabase_jwt(token: str) -> AuthContext:
    """
    Verify Supabase JWT signature using JWKS and return user id (sub).
    """
    jwks = get_jwks()
    # Audience varies by project; in many Supabase projects, aud is "authenticated".
    # We avoid hard-failing on missing aud by not enforcing audience here.
    try:
        claims = jwt.decode(
            token,
            jwks,
            algorithms=["RS256"],
            options={"verify_aud": False},
        )
    except JWTError as e:
        raise RuntimeError(f"Invalid JWT: {e}") from e

    sub = claims.get("sub")
    if not sub:
        raise RuntimeError("JWT missing 'sub'")
    return AuthContext(user_id=str(sub), claims=dict(claims))

