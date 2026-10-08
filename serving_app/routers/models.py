"""MLflow Registry 기반 현재 모델 및 재학습 이력 API."""

from __future__ import annotations

import os
from datetime import datetime, timezone

from fastapi import APIRouter

from serving_app.mlflow_config import MODEL_NAME, configure_mlflow

router = APIRouter(prefix="/models", tags=["dashboard"])

def _format_timestamp(timestamp_ms: int | None) -> str | None:
    if not timestamp_ms:
        return None
    return datetime.fromtimestamp(timestamp_ms / 1000, tz=timezone.utc).isoformat()


def _version_number(value: object) -> int:
    try:
        return int(str(value))
    except ValueError:
        return 0


@router.get("/status")
def model_status():
    """Registry를 읽기만 하며 모델 파일 자체는 로드하지 않는다."""

    try:
        from mlflow.tracking import MlflowClient

        uri = configure_mlflow()
        client = MlflowClient(tracking_uri=uri)
        versions = list(client.search_model_versions(f"name='{MODEL_NAME}'"))
    except Exception:
        return {
            "available": False,
            "model_name": MODEL_NAME,
            "model_source": os.getenv("MODEL_SOURCE", "local"),
            "current": None,
            "history": [],
        }

    history = []
    for version in sorted(versions, key=lambda item: _version_number(item.version), reverse=True):
        metrics: dict[str, float] = {}
        params: dict[str, str] = {}
        if version.run_id:
            try:
                run = client.get_run(version.run_id)
                metrics = dict(run.data.metrics)
                params = dict(run.data.params)
            except Exception:
                pass

        history.append(
            {
                "version": str(version.version),
                "created_at": _format_timestamp(version.creation_timestamp),
                "stage": version.current_stage or None,
                "mode": params.get("mode"),
                "rmse": metrics.get("rmse"),
                "run_id": version.run_id,
            }
        )

    current = next(
        (item for item in history if (item["stage"] or "").lower() == "production"),
        None,
    )
    return {
        "available": True,
        "model_name": MODEL_NAME,
        "model_source": os.getenv("MODEL_SOURCE", "local"),
        "current": current,
        "history": history,
    }
