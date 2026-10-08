"""프로젝트 로컬 MLflow 저장소 설정."""

from __future__ import annotations

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RUNTIME_DIR = PROJECT_ROOT / "runtime"
MLFLOW_DB_PATH = RUNTIME_DIR / "mlflow.db"
MLFLOW_ARTIFACT_DIR = RUNTIME_DIR / "mlartifacts"
DEFAULT_TRACKING_URI = f"sqlite:///{MLFLOW_DB_PATH}"

EXPERIMENT_NAME = "SKHY_Voltage"
MODEL_NAME = "SKHY_Voltage_Predictor"


def tracking_uri() -> str:
    """환경변수가 없으면 Git에서 제외된 프로젝트 로컬 DB를 사용한다."""

    return os.getenv("MLFLOW_TRACKING_URI", DEFAULT_TRACKING_URI)


def configure_mlflow(*, select_experiment: bool = False) -> str:
    """MLflow 클라이언트가 모든 실행 경로에서 같은 저장소를 보도록 설정한다."""

    import mlflow

    RUNTIME_DIR.mkdir(parents=True, exist_ok=True)
    uri = tracking_uri()
    mlflow.set_tracking_uri(uri)

    if select_experiment:
        from mlflow.tracking import MlflowClient

        client = MlflowClient(tracking_uri=uri)
        experiment = client.get_experiment_by_name(EXPERIMENT_NAME)
        if experiment is None:
            MLFLOW_ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
            client.create_experiment(
                EXPERIMENT_NAME,
                artifact_location=MLFLOW_ARTIFACT_DIR.resolve().as_uri(),
            )
        mlflow.set_experiment(EXPERIMENT_NAME)

    return uri
