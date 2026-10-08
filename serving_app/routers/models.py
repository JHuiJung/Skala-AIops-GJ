"""
대시보드의 "현재 운영 모델" 카드와 "재학습 이력" 표용 - MLflow Model Registry를 읽기 전용으로 조회한다.

train_and_register.py가 "HAIC_Predictor" 이름으로 등록한 버전들을 그대로 보여 줄 뿐,
학습·승격 로직은 없다. (RMSE와 학습 방식은 각 버전을 만든 MLflow run의 metric/param에서 읽는다.)
"""
import os

from fastapi import APIRouter

from serving_app.monitoring.drift_detector import RMSE_THRESHOLD

router = APIRouter(prefix="/models")

MODEL_NAME = "HAIC_Predictor"


def list_versions() -> list[dict]:
    """등록된 모든 버전을 최신 버전이 먼저 오도록 반환한다. 레지스트리를 못 읽으면 예외를 그대로 던진다."""
    from mlflow.tracking import MlflowClient  # 서버 시작 속도를 위해 호출 시점에 import

    client = MlflowClient()
    versions = client.search_model_versions(f"name='{MODEL_NAME}'")

    result = []
    for v in versions:
        rmse, mode = None, None
        try:
            run = client.get_run(v.run_id)
            rmse = run.data.metrics.get("rmse")
            mode = run.data.params.get("mode")
        except Exception:
            pass  # run이 지워진 버전도 목록에는 남긴다
        result.append(
            {
                "version": int(v.version),
                "created_at": v.creation_timestamp / 1000,  # epoch 초
                "mode": mode,
                "rmse": rmse,
                "stage": v.current_stage,
                "run_id": v.run_id,
            }
        )
    result.sort(key=lambda r: r["version"], reverse=True)
    return result


@router.get("")
def models():
    base = {
        "name": MODEL_NAME,
        "source": os.getenv("MODEL_SOURCE", "local"),
        "gate": RMSE_THRESHOLD,
    }
    try:
        history = list_versions()
    except Exception as e:
        return {**base, "available": False, "detail": str(e), "current": None, "history": []}

    # 서버는 "Production" 단계 중 가장 최근 버전을 불러온다
    current = next((r for r in history if r["stage"] == "Production"), None)
    return {**base, "available": True, "current": current, "history": history}
