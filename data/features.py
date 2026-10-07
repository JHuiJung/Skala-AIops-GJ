"""SKHY 전압 전처리의 호환성 인터페이스.

실제 데이터 로드와 시퀀스 생성 규칙은
``data.voltage_preprocessing``에서만 관리한다. 이 모듈은 기존 코드가
``data.features``를 import하는 동안 함수 이름을 유지하기 위한 임시
호환 계층이다.

전처리 기준:
- 입력: 3행 간격의 ``Input_V`` 150개
- 타깃: 시작 위치에서 449행 뒤의 ``E``
- 샘플 시작 간격: 10행
- 스케일링: 없음
"""

from __future__ import annotations

from os import PathLike

import numpy as np

from data.voltage_preprocessing import (
    INTERVAL,
    SEQUENCE_LENGTH,
    STRIDE,
    TRAIN_RATIO,
    PreprocessedVoltageData,
    VoltageData,
    build_sequences as _build_voltage_sequences,
    load_voltage_data,
    prepare_voltage_datasets,
    split_train_validation,
)


# 기존 호출부가 import하는 이름을 유지한다.
SEQ_LEN = SEQUENCE_LENGTH


def load_rows(
    csv_path: str | PathLike[str] = "data/SKHY_train.csv",
) -> list[dict[str, float]]:
    """SKHY CSV를 기존 ``list[dict]`` 형식으로 반환한다."""

    data = load_voltage_data(csv_path, require_target=True)
    assert data.target_e is not None
    return [
        {"TIME": float(time), "Input_V": float(input_v), "E": float(target_e)}
        for time, input_v, target_e in zip(
            data.time, data.input_v, data.target_e, strict=True
        )
    ]


def build_sequences(
    rows: list[dict[str, float]],
    scaler: object | None = None,
    seq_len: int = SEQ_LEN,
) -> tuple[np.ndarray, np.ndarray]:
    """기존 함수명을 유지하며 SKHY 전압 시퀀스를 생성한다.

    ``scaler``는 기존 호출부와의 시그니처 호환을 위해 받지만 사용하지
    않는다. 입력과 타깃은 원본 전압값을 그대로 사용한다.
    """

    del scaler
    input_values = np.asarray([row["Input_V"] for row in rows], dtype=np.float64)
    target_values = np.asarray([row["E"] for row in rows], dtype=np.float64)
    X, y = _build_voltage_sequences(
        input_values,
        target_values,
        sequence_length=seq_len,
        interval=INTERVAL,
        stride=STRIDE,
    )
    assert y is not None
    return X, y


def train_test_split(
    X: list | np.ndarray,
    y: list | np.ndarray,
    test_ratio: float = 1 - TRAIN_RATIO,
) -> tuple:
    """시간 순서를 유지하며 앞 90%와 뒤 10%를 나눈다."""

    if not 0 < test_ratio < 1:
        raise ValueError("test_ratio는 0과 1 사이여야 합니다.")
    if len(X) != len(y):
        raise ValueError("X와 y의 길이가 같아야 합니다.")
    split_index = int(len(X) * (1 - test_ratio))
    return X[:split_index], y[:split_index], X[split_index:], y[split_index:]


__all__ = [
    "INTERVAL",
    "PreprocessedVoltageData",
    "SEQ_LEN",
    "SEQUENCE_LENGTH",
    "STRIDE",
    "TRAIN_RATIO",
    "VoltageData",
    "build_sequences",
    "load_rows",
    "load_voltage_data",
    "prepare_voltage_datasets",
    "split_train_validation",
    "train_test_split",
]
