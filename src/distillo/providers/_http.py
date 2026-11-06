"""HTTP helper for the providers that speak plain JSON over HTTPS."""

from __future__ import annotations

import time
from typing import Any

import httpx

from .base import LLMError, QuotaError

RETRYABLE = {429, 500, 502, 503, 504}


def post_json(
    url: str,
    *,
    provider: str,
    headers: dict[str, str] | None = None,
    payload: dict[str, Any] | None = None,
    files: Any = None,
    data: dict[str, Any] | None = None,
    timeout: float = 300.0,
    retries: int = 3,
    max_wait: float = 30.0,
) -> Any:
    """POST and return the decoded JSON body.

    Rate limits and server errors are retried with backoff (honouring ``Retry-After``);
    if they persist a :class:`QuotaError` is raised so a fallback provider can take over.
    """
    for attempt in range(retries + 1):
        try:
            response = httpx.post(
                url, headers=headers, json=payload, files=files, data=data, timeout=timeout
            )
        except httpx.TransportError as exc:
            if attempt == retries:
                raise QuotaError(f"cannot reach {provider}: {exc}") from exc
            time.sleep(min(2 ** (attempt + 1), max_wait))
            continue
        if response.status_code in RETRYABLE:
            if attempt == retries:
                raise QuotaError(
                    f"{provider} error {response.status_code} (rate limit, quota or outage): "
                    f"{response.text[:200]}"
                )
            time.sleep(_wait(response, attempt, max_wait))
            continue
        if response.status_code in (401, 403):
            raise LLMError(f"{provider} rejected the API key ({response.status_code})")
        if response.status_code >= 400:
            raise LLMError(f"{provider} error {response.status_code}: {response.text[:300]}")
        try:
            return response.json()
        except ValueError as exc:
            raise LLMError(f"{provider} returned a non-JSON response") from exc
    raise AssertionError("unreachable")  # pragma: no cover


def _wait(response: httpx.Response, attempt: int, max_wait: float) -> float:
    header = response.headers.get("retry-after")
    try:
        if header is not None:
            return min(float(header), max_wait)
    except ValueError:
        pass
    return float(min(2 ** (attempt + 1), max_wait))
