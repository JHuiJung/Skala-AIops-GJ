"""
Day2: MLflow로 SKHY 전압 LSTM(Input_V -> E)을 학습 -> 기록(Tracking) -> 게이트 검증 -> 등록(Registry) -> Production 승격.
Day3: 드리프트 감지 후 Production 가중치에서 이어서 학습하는 fine-tuning 재학습.

전처리(로드, 3행 간격 150개 입력, 449행 뒤 E 타깃)는 data/voltage_preprocessing.py가 전담하고,
스케일러는 쓰지 않는다(입력·타깃 모두 원본 값 그대로).

실습 시나리오:
    1) SKHY_train.csv로 base 모델 학습(early stopping) -> SKHY_test_answer.csv로 RMSE 확인
    2) 게이트(RMSE_GATE) 통과 시 Production으로 승격
    3) (Day3) 드리프트 감지 시 Production 가중치에서 warm-start -> 최근 구간(약 470행 = 샘플 21건)으로
       10 epoch만 fine-tuning (처음부터 다시 학습하지 않음 - 샘플 21건으로는 스크래치 학습이 불안정)

실행:
    python serving_app/train_and_register.py     # 프로젝트 루트에서
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import mlflow
import mlflow.tensorflow
import numpy as np
from mlflow.tracking import MlflowClient
from tensorflow import keras

from data.voltage_preprocessing import (
    VoltageData,
    build_sequences,
    load_voltage_data,
    split_train_validation,
)
from serving_app.lstm_model import build_model

# 시드 고정: LSTM 가중치 초기화가 랜덤이라 시드 없이는 실행마다 RMSE가 흔들려
# 게이트 통과 여부가 운에 좌우됩니다. numpy/tensorflow/python random을 한 번에 고정합니다.
SEED = 42
keras.utils.set_random_seed(SEED)

TRAIN_CSV = "data/SKHY_train.csv"
TEST_CSV = "data/SKHY_test_answer.csv"  # 게이트 평가용 (학습에 쓰지 않은 구간)

RMSE_GATE = 0.012  # drift_detector.RMSE_THRESHOLD와 같은 기준 (E 전압 단위)
MODEL_NAME = "SKHY_Predictor"

BASE_EPOCHS = 30  # early stopping이 있으므로 "최대" 에폭
BASE_BATCH_SIZE = 512
EARLY_STOP_PATIENCE = 5

FINE_TUNE_EPOCHS = 10
FINE_TUNE_LR = 1e-4  # base 학습(1e-3)보다 낮은 학습률로 살짝만 갱신
FINE_TUNE_BATCH_SIZE = 8  # fine-tuning 샘플은 20건 안팎이라 배치를 작게 해야 갱신이 일어난다
FINE_TUNE_TEST_RATIO = 0.2


def rmse(y_true, y_pred) -> float:
    diff = np.asarray(y_true, dtype="float64").reshape(-1) - np.asarray(y_pred, dtype="float64").reshape(-1)
    return float(np.sqrt(np.mean(diff**2)))


def _register_if_gate_passed(model, run_id: str, score: float) -> dict:
    result = {"run_id": run_id, "rmse": score, "promoted": False}
    if score <= RMSE_GATE:
        v = mlflow.register_model(f"runs:/{run_id}/model", MODEL_NAME)
        MlflowClient().transition_model_version_stage(name=MODEL_NAME, version=v.version, stage="Production")
        result["promoted"] = True
        result["version"] = v.version
        print(f"[GATE PASSED] rmse={score:.6f} -> {MODEL_NAME} v{v.version} promoted to Production")
    else:
        print(f"[GATE FAILED] rmse={score:.6f} > {RMSE_GATE} -> 배포 차단, 기존 Production 유지")
    return result


def train_and_register(train_data: VoltageData | None = None) -> dict:
    """Day2: 처음부터(scratch) 학습. 데이터가 충분한 base 학습에서만 사용합니다.

    train_data를 지정하지 않으면 SKHY_train.csv를 읽습니다. 앞 90%로 학습, 뒤 10%로 early stopping용
    검증을 하고, 게이트 점수는 SKHY_test_answer.csv(한 번도 쓰지 않은 구간)로 계산합니다.
    """
    if train_data is None:
        train_data = load_voltage_data(TRAIN_CSV, require_target=True)
    train, valid = split_train_validation(train_data)
    test = load_voltage_data(TEST_CSV, require_target=True)

    X_train, y_train = build_sequences(train.input_v, train.target_e)
    X_valid, y_valid = build_sequences(valid.input_v, valid.target_e)
    X_test, y_test = build_sequences(test.input_v, test.target_e)
    X_train, X_valid, X_test = (a.astype("float32") for a in (X_train, X_valid, X_test))
    y_train, y_valid = y_train.astype("float32"), y_valid.astype("float32")

    with mlflow.start_run(run_name="base-train"):
        model = build_model()
        early_stop = keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=EARLY_STOP_PATIENCE, min_delta=1e-6, restore_best_weights=True
        )
        history = model.fit(
            X_train,
            y_train,
            validation_data=(X_valid, y_valid),
            epochs=BASE_EPOCHS,
            batch_size=BASE_BATCH_SIZE,
            callbacks=[early_stop],
            verbose=2,
        )

        valid_score = rmse(y_valid, model.predict(X_valid, verbose=0))
        score = rmse(y_test, model.predict(X_test, verbose=0))

        mlflow.log_param("mode", "scratch")
        mlflow.log_param("epochs_max", BASE_EPOCHS)
        mlflow.log_param("epochs_run", len(history.history["loss"]))
        mlflow.log_metric("valid_rmse", valid_score)
        mlflow.log_metric("rmse", score)
        mlflow.tensorflow.log_model(model, name="model", input_example=X_train[:1])

        return _register_if_gate_passed(model, mlflow.active_run().info.run_id, score)


def fine_tune(data: VoltageData) -> dict:
    """
    Day3: 현재 Production 모델 가중치에서 이어서(warm start), 넘겨받은 data(최근 구간)로
    짧게 fine-tuning합니다. 샘플이 적을 때(예: 최근 21건)도 스크래치 학습보다 훨씬 안정적입니다.

    data는 연속된 파형 구간이어야 하며, stride=1로 잘라 가능한 모든 샘플을 만듭니다.
    앞 80%로 학습하고 뒤 20%로 게이트 점수를 계산합니다 (시간 순서 유지).
    """
    X, y = build_sequences(data.input_v, data.target_e, stride=1)
    X, y = X.astype("float32"), y.astype("float32")
    if len(X) < 2:
        raise ValueError(f"fine-tuning에는 샘플이 최소 2건 필요합니다. 현재 {len(X)}건입니다.")

    split_idx = min(max(1, int(len(X) * (1 - FINE_TUNE_TEST_RATIO))), len(X) - 1)
    X_train, y_train = X[:split_idx], y[:split_idx]
    X_test, y_test = X[split_idx:], y[split_idx:]

    model = mlflow.tensorflow.load_model(f"models:/{MODEL_NAME}/Production")
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=FINE_TUNE_LR), loss="mse")

    with mlflow.start_run(run_name="fine-tune"):
        model.fit(X_train, y_train, epochs=FINE_TUNE_EPOCHS, batch_size=FINE_TUNE_BATCH_SIZE, verbose=0)

        score = rmse(y_test, model.predict(X_test, verbose=0))

        mlflow.log_param("mode", "fine-tune")
        mlflow.log_param("epochs", FINE_TUNE_EPOCHS)
        mlflow.log_param("n_samples", len(X))
        mlflow.log_metric("rmse", score)
        mlflow.tensorflow.log_model(model, name="model", input_example=X_train[:1])

        return _register_if_gate_passed(model, mlflow.active_run().info.run_id, score)


if __name__ == "__main__":
    train_and_register()
