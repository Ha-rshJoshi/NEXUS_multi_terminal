"""Shared Finnhub error classification (retryable rate-limit/5xx errors vs.
permanent 401/403 permission errors), used by data_engine.py and
news_scraper.py's retry wrappers."""


def get_status_code(exc: Exception):
    return getattr(exc, "status_code", None)


def is_rate_limit_error(exc: Exception) -> bool:
    status = get_status_code(exc)
    message = str(exc)
    return status == 429 or "429" in message or "Too Many Requests" in message


def is_permission_error(exc: Exception) -> bool:
    status = get_status_code(exc)
    message = str(exc).lower()
    return status in (401, 403) or "don't have access" in message or "access to this resource" in message


def is_retryable(exc: Exception) -> bool:
    """True only for errors worth another attempt (429 / 5xx / no status
    code at all, e.g. a transient network error). False for 4xx permission
    errors, which will never succeed on retry."""
    if is_permission_error(exc):
        return False
    status = get_status_code(exc)
    if status is not None and 400 <= status < 500:
        return False
    return True
