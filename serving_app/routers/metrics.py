"""
대시보드의 "운영 지표 요약"(KPI 5개)용 - 요청 로그 기록 + 집계.

/predict 와 /predict/batch-test 요청만 logs/requests.log 에 한 줄씩(JSON) 쌓고(main.py의 미들웨어가 호출),
GET /metrics/summary 가 그 파일을 읽어 시간 구간별로 집계한다. 대시보드 자체의 폴링 요청
(/health, /models, /metrics ...)은 기록하지 않아 숫자가 부풀지 않는다.

모델 성능(RMSE)은 MLflow 레지스트리, 드리프트 점수는 서버 메모리의 recent_predictions에서 가져온다.
"""
import json
import os
import threading
import time

from fastapi import APIRouter, HTTPException

from serving_app.monitoring.drift_detector import RMSE_THRESHOLD, WINDOW_SIZE, compute_rmse
from serving_app.routers import models as models_router
from serving_app.routers import predict as predict_router

router = APIRouter(prefix="/metrics")

REQUEST_LOG = os.path.join("logs", "requests.log")
LOGGED_PREFIX = "/predict"

WINDOWS = {"5m": 300, "1h": 3600, "6h": 21600, "24h": 86400}  # 대시보드 버튼 -> 초
N_BUCKETS = 12  # 스파크라인 점 개수

_lock = threading.Lock()


def log_request(path: str, status: int, latency_ms: float) -> None:
    """미들웨어가 호출한다. 한 요청 = 한 줄."""
    line = json.dumps({"ts": time.time(), "path": path, "status": status, "ms": round(latency_ms, 1)})
    with _lock:
        os.makedirs(os.path.dirname(REQUEST_LOG), exist_ok=True)
        with open(REQUEST_LOG, "a", encoding="utf-8") as f:
            f.write(line + "\n")


def _read_requests(since: float) -> list[dict]:
    if not os.path.isfile(REQUEST_LOG):
        return []
    rows = []
    with open(REQUEST_LOG, encoding="utf-8") as f:
        for raw in f:
            try:
                r = json.loads(raw)
            except ValueError:
                continue  # 깨진 줄은 건너뛴다
            if r.get("ts", 0) >= since:
                rows.append(r)
    return rows


def _request_stats(window: str) -> dict:
    seconds = WINDOWS[window]
    since = time.time() - seconds
    rows = _read_requests(since)

    count = len(rows)
    ok = sum(1 for r in rows if r["status"] < 400)

    # 스파크라인: 구간을 N_BUCKETS 개로 나눠 버킷별 값을 만든다 (요청이 없는 버킷은 None -> 선을 이어 그림)
    size = seconds / N_BUCKETS
    cnt = [0] * N_BUCKETS
    lat = [0.0] * N_BUCKETS
    good = [0] * N_BUCKETS
    for r in rows:
        i = min(N_BUCKETS - 1, max(0, int((r["ts"] - since) / size)))
        cnt[i] += 1
        lat[i] += r["ms"]
        good[i] += 1 if r["status"] < 400 else 0

    return {
        "requests": {"value": count, "series": cnt},
        "latency": {
            "value": (sum(r["ms"] for r in rows) / count) if count else 0,
            "series": [lat[i] / cnt[i] if cnt[i] else None for i in range(N_BUCKETS)],
        },
        "success": {
            "value": (ok / count * 100) if count else 0,
            "series": [good[i] / cnt[i] * 100 if cnt[i] else None for i in range(N_BUCKETS)],
        },
    }


def _model_stats() -> dict:
    try:
        history = models_router.list_versions()
    except Exception:
        return {"value": None, "series": [], "gate": RMSE_THRESHOLD}
    current = next((r for r in history if r["stage"] == "Production"), None)
    series = [r["rmse"] for r in sorted(history, key=lambda r: r["version"]) if r["rmse"] is not None]
    return {
        "value": current["rmse"] if current else None,
        "series": series[-12:],
        "gate": RMSE_THRESHOLD,
    }


def _drift_stats() -> dict:
    recent = list(predict_router.recent_predictions[-WINDOW_SIZE:])
    if not recent:
        return {"score": None, "threshold": RMSE_THRESHOLD, "n": 0, "exceeded": False, "series": []}
    score = compute_rmse(recent)
    return {
        "score": score,
        "threshold": RMSE_THRESHOLD,
        "n": len(recent),
        "exceeded": len(recent) >= WINDOW_SIZE and score > RMSE_THRESHOLD,
        "series": [abs(p["actual"] - p["predicted"]) for p in recent],
    }


@router.get("/summary")
def summary(window: str = "5m"):
    if window not in WINDOWS:
        raise HTTPException(400, f"window는 {sorted(WINDOWS)} 중 하나여야 합니다.")
    return {"window": window, **_request_stats(window), "model": _model_stats(), "drift": _drift_stats()}
