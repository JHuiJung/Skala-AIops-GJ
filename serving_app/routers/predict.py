"""
[Day1 → Day3] 예측 API  —  serving_app/routers/predict.py
【실습용】 ___ (밑줄 3개)만 채우세요. 채울 곳은 [빈칸 N] 으로 표시되어 있습니다.
   ___ 가 남은 채 실행하면 "name '___' is not defined" 에러가 나며, 그 줄이 채울 곳입니다.

■ 이 파일이 하는 일 (한 줄 요약)
   외부 요청을 받아 모델에게 전달하고, 결과를 돌려주는 "창구"입니다.
   계산은 직접 하지 않고, 모델(model_loader)과 감시 도구(retrain_trigger)에게 맡깁니다.

■ 엔드포인트
   [Day1] POST /predict             : 20일치 데이터 → 다음날 종가 1개   (완성 — 읽고 흐름만 이해하세요)
   [Day3] POST /predict/batch-test  : 긴 가격 목록 → 여러 번 예측 → 드리프트 검사

■ 이 파일의 빈칸 : [빈칸 6]  (batch_test 의 슬라이딩 윈도우)
"""
from fastapi import APIRouter
import numpy as np
from data.features import SEQ_LEN  # = 20
from data.voltage_preprocessing import (
    SEQUENCE_LENGTH,
    INTERVAL,
    build_sequences,
    load_voltage_data,
)
from serving_app import model_loader
from serving_app.schemas import PredictRequest, PredictResponse, BatchTestRequest, BatchTestResponse
from serving_app.monitoring.drift_detector import WINDOW_SIZE, compute_rmse
from serving_app.monitoring.retrain_trigger import check_and_trigger

router = APIRouter()

# (Day3) 최근 예측 기록을 모아 두는 목록.  예: [{"predicted": 161.2, "actual": 163.0}, ...]
#        드리프트 판단은 "최근 21건"(drift_detector.py 의 WINDOW_SIZE)만 보므로 21개까지만 유지합니다.
recent_predictions: list[dict] = []

# (Day3) 시뮬레이션은 종가만 보내므로, 거래량은 이 값으로 고정해서 채웁니다.
SIMULATED_VOLUME = 1_200_000

# 목업 데이터 소스: 정답 E가 들어 있는 테스트 파일. 요청마다 읽지 않도록 한 번만 읽어 둡니다.
MOCK_CSV = "data/SKHY_test_answer.csv"
DRIFT_MULTIPLIER = 3
_mock_source = None


def _get_mock_source():
    global _mock_source
    if _mock_source is None:
        _mock_source = load_voltage_data(MOCK_CSV, require_target=True)
    return _mock_source


def make_mock_batch(kind: str):
    """
    테스트 파형에서 랜덤 구간을 잘라 (X, y)를 만든다.
    반환: X (21, 150, 1), y (21, 1)  - 드리프트 판정 윈도우(21건)가 정확히 채워진다.
    drift면 입력과 정답을 모두 DRIFT_MULTIPLIER배 한다.
    """
    data = _get_mock_source()
    # 샘플 21개를 만들려면 (150 * 3 - 1) + 21 = 470행 구간이 필요하다
    span = SEQUENCE_LENGTH * INTERVAL - 1 + WINDOW_SIZE
    start = np.random.randint(0, len(data) - span + 1)
    segment = data.slice(start, start + span)

    # stride=1 고정: 학습용 STRIDE와 무관하게 21건이 연속으로 채워져야 한다
    X, y = build_sequences(segment.input_v, segment.target_e, stride=1)

    if kind == "drift":
        X, y = X * DRIFT_MULTIPLIER, y * DRIFT_MULTIPLIER
    return X, y

@router.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    """
    [Day1] 다음날 종가 예측  (완성)
    받는 것  : {"sequence": [{"close": 160.0, "volume": 1200000}, ... 20개]}
               20개가 아니면 schemas.py 가 알아서 422 에러를 돌려줍니다.
    돌려줄 것: {"predicted_close": 161.37, "model_version": "v1-local"}

    흐름: 모델 가져오기(get_model) → dict 목록으로 변환 → predict_one → 응답 포장
    핵심 계산은 모두 model_loader.predict_one() 안에 있습니다. ([빈칸 2], [빈칸 3])
    """
    model = model_loader.get_model()
    sequence = req.sequence[::INTERVAL]  # 연속 450행 -> 학습 때와 같은 3행 간격 150개
    predicted_e = model.predict_one(sequence)
    return PredictResponse(predicted_e=predicted_e, model_version=model.version)


@router.post("/predict/batch-test", response_model=BatchTestResponse)
def batch_test(req: BatchTestRequest):
    """
    [Day3] 드리프트 시뮬레이션
    받는 것  : {"kind": "normal"} 또는 {"kind": "drift"}   (웹 UI, scripts/simulate_drift.py 가 보냄)
    돌려줄 것: {"predictions": [예측값 21개], "drift_check": {"status": "ok"} 또는 재학습 결과}

    서버가 kind에 맞는 변동폭(sigma)으로 가격 41개(SEQ_LEN + WINDOW_SIZE)를 랜덤워크로 만든 뒤
    슬라이딩 윈도우로 예측합니다.

    ■ 핵심 아이디어: 슬라이딩 윈도우 (20칸짜리 창문을 한 칸씩 밀기)
      가격 41개를 20개씩 잘라 "그다음 날"을 예측하고 실제 값과 비교합니다.

        i=0 : [p0  ~ p19] → 예측   vs  실제 p20
        i=1 : [p1  ~ p20] → 예측   vs  실제 p21
        ...
        i=20: [p20 ~ p39] → 예측   vs  실제 p40
        → 총 41 - 20 = 21번 예측 = 드리프트 판단에 필요한 21건이 딱 채워집니다.

    확인 방법
      python scripts/simulate_drift.py
        [normal] drift_check = {'status': 'ok'}
        [drift]  drift_check = {'status': 'retrain_triggered', 'promoted': True, ...}
      /docs 에서 {"kind": "normal"}로 호출하면 predictions가 21개인지 확인하세요.
    """
    print(req.kind);
    X, y = make_mock_batch(req.kind)

    model = model_loader.get_model()
    predictions: list[float] = []
    actuals: list[float] = []

    for i in range(len(X)):
        pred = model.predict_one(X[i, :, 0])  # Input_V 150개 (슬라이딩은 build_sequences가 이미 처리)
        actual = float(y[i, 0])
        predictions.append(pred)
        actuals.append(actual)
        recent_predictions.append({"predicted": pred, "actual": actual})

    # 최근 21건만 남기기 — 오래된 기록까지 섞이면 "지금" 상태를 판단할 수 없습니다.
    # (recent_predictions = ... 로 쓰면 함수 안의 새 변수가 되므로, [:] 로 목록 내용을 바꿉니다)
    recent_predictions[:] = recent_predictions[-WINDOW_SIZE:]

    # 대시보드 표시용: 판정에 쓰인 최근 21건의 RMSE (21건 미만이면 판정 대기라 None)
    window_rmse = compute_rmse(recent_predictions) if len(recent_predictions) >= WINDOW_SIZE else None

    # 드리프트 판단·재학습은 retrain_trigger.py 가 합니다. 여기서는 넘겨주기만!
    drift_check = check_and_trigger(recent_predictions)
    return BatchTestResponse(
        predictions=predictions, actuals=actuals, window_rmse=window_rmse, drift_check=drift_check
    )
