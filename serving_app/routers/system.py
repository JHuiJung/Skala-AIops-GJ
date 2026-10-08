"""대시보드에 노출할 읽기 전용 시스템 설정 API."""

import os

from fastapi import APIRouter

from data.voltage_preprocessing import INTERVAL, SEQUENCE_LENGTH, STRIDE, TRAIN_RATIO
from serving_app.monitoring.drift_detector import RMSE_THRESHOLD, WINDOW_SIZE
from serving_app.mlflow_config import MODEL_NAME, tracking_uri

router = APIRouter(prefix="/system", tags=["dashboard"])


@router.get("/info")
def system_info():
    # 학습 모듈을 import하면 TensorFlow까지 로드되므로 현재 학습 설정값만 읽기 전용으로 노출한다.
    return {
        "sequence_length": SEQUENCE_LENGTH,
        "interval": INTERVAL,
        "stride": STRIDE,
        "train_ratio": TRAIN_RATIO,
        "drift_window": WINDOW_SIZE,
        "rmse_threshold": RMSE_THRESHOLD,
        "base_epochs": 30,
        "base_batch_size": 512,
        "fine_tune_epochs": 10,
        "fine_tune_learning_rate": 0.0001,
        "fine_tune_batch_size": 8,
        "model_source": os.getenv("MODEL_SOURCE", "local"),
        "loading_mode": os.getenv("LOADING_MODE", "lazy"),
        "mlflow_model_name": MODEL_NAME,
        "mlflow_tracking_uri": tracking_uri(),
    }
