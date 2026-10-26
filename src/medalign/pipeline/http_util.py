"""极简 HTTP 客户端（仅标准库），带重试与审计日志。

刻意不引入 requests：减少环境依赖，方便在本机受限环境下安装。
"""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from typing import Any

DEFAULT_UA = "MedAlignFlywheel/0.1 (research data ingestion; contact: local user)"


def _get(url: str, timeout: float, retries: int, headers: dict | None) -> bytes:
    last_err: Exception | None = None
    for attempt in range(retries + 1):
        try:
            req = urllib.request.Request(
                url, headers={"User-Agent": DEFAULT_UA, **(headers or {})}
            )
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return resp.read()
        except urllib.error.HTTPError as exc:
            last_err = exc
            if exc.code in (429, 503, 502, 500) and attempt < retries:
                # 限流/服务端抖动：退避要够久，HF datasets-server 的窗口通常按秒级恢复
                wait = min(60.0, 8.0 * (2**attempt))
                time.sleep(wait)
                continue
            raise
        except Exception as exc:  # noqa: BLE001 - 网络错误统一重试
            last_err = exc
            if attempt < retries:
                time.sleep(min(8.0, 1.5 * (2**attempt)))
    raise RuntimeError(f"GET failed after {retries + 1} attempts: {url} :: {last_err}")


def fetch_bytes(url: str, timeout: float = 60.0, retries: int = 3, headers: dict | None = None) -> bytes:
    return _get(url, timeout=timeout, retries=retries, headers=headers)


def fetch_json(url: str, timeout: float = 90.0, retries: int = 3, headers: dict | None = None) -> Any:
    raw = _get(url, timeout=timeout, retries=retries, headers=headers)
    return json.loads(raw.decode("utf-8"))
