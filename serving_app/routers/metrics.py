"""대시보드 운영 지표 조회 API."""

from __future__ import annotations

import json
import os
import time
from datetime import datetime, timezone
from typing import Literal

from fastapi import APIRouter, Query

from serving_app.monitoring.drift_detector import (
    RMSE_THRESHOLD,
    WINDOW_SIZE,
    compute_rmse,
)
from serving_app.routers.predict import recent_predictions

router = APIRouter(prefix="/metrics", tags=["dashboard"])

REQUEST_LOG_PATH = "logs/requests.log"
Window = Literal["5m", "1h", "6h", "24h"]
WINDOW_SECONDS: dict[Window, int] = {
    "5m": 5 * 60,
    "1h": 60 * 60,
    "6h": 6 * 60 * 60,
    "24h": 24 * 60 * 60,
}


def should_record_request(method: str, path: str) -> bool:
    """사용자 기능 API만 집계하고 대시보드 자체 조회는 제외한다."""

    return (
        path == "/health"
        or path.startswith("/predict")
        or (method.upper() == "POST" and path == "/data/upload")
    )


def append_request_metric(*, method: str, path: str, status_code: int, latency_ms: float) -> None:
    os.makedirs(os.path.dirname(REQUEST_LOG_PATH), exist_ok=True)
    now = time.time()
    record = {
        "timestamp": now,
        "datetime": datetime.fromtimestamp(now, tz=timezone.utc).isoformat(),
        "method": method,
        "path": path,
        "status_code": status_code,
        "latency_ms": round(latency_ms, 3),
    }
    with open(REQUEST_LOG_PATH, "a", encoding="utf-8") as file:
        file.write(json.dumps(record, ensure_ascii=False) + "\n")


def _read_records(since: float) -> list[dict]:
    if not os.path.isfile(REQUEST_LOG_PATH):
        return []

    records: list[dict] = []
    with open(REQUEST_LOG_PATH, encoding="utf-8") as file:
        for line in file:
            try:
                record = json.loads(line)
                if float(record.get("timestamp", 0)) >= since:
                    records.append(record)
            except (json.JSONDecodeError, TypeError, ValueError):
                # 한 줄이 손상되어도 나머지 운영 지표는 계속 집계한다.
                continue
    return records


@router.get("/summary")
def get_summary(window: Window = Query("5m")):
    now = time.time()
    records = _read_records(now - WINDOW_SECONDS[window])

    request_count = len(records)
    avg_latency_ms = None
    success_rate = None
    if records:
        avg_latency_ms = round(
            sum(float(record["latency_ms"]) for record in records) / request_count,
            2,
        )
        successes = sum(1 for record in records if int(record["status_code"]) < 400)
        success_rate = round(successes / request_count * 100, 1)

    prediction_count = len(recent_predictions)
    recent_rmse = compute_rmse(recent_predictions) if recent_predictions else None
    drift_ready = prediction_count >= WINDOW_SIZE
    drift_detected = (
        recent_rmse > RMSE_THRESHOLD if drift_ready and recent_rmse is not None else None
    )

    return {
        "window": window,
        "has_request_data": bool(records),
        "request_count": request_count if records else None,
        "avg_latency_ms": avg_latency_ms,
        "success_rate": success_rate,
        "latency_trend": [float(record["latency_ms"]) for record in records[-12:]],
        "recent_rmse": recent_rmse,
        "prediction_count": prediction_count,
        "drift_window_size": WINDOW_SIZE,
        "drift_threshold": RMSE_THRESHOLD,
        "drift_detected": drift_detected,
    }
