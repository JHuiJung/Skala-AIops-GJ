"""SKHY 전압 파형 데이터 전처리.

입력은 ``Input_V`` 파형 150개이고, 타깃은 노트북을 따라 마지막
입력보다 두 행 뒤의 ``E`` 전압이다. 예를 들어 첫 샘플은
``Input_V[0, 3, ..., 447]``로 ``E[449]``를 예측한다.

입력과 타깃은 스케일링하지 않고 원본 전압값을 그대로 사용한다.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from os import PathLike

import numpy as np


SEQUENCE_LENGTH = 150
INTERVAL = 3
STRIDE = 1
TRAIN_RATIO = 0.9


@dataclass(frozen=True)
class VoltageData:
    """시간, 입력 전압, 선택적 E 타깃을 담는 원본 데이터."""

    time: np.ndarray
    input_v: np.ndarray
    target_e: np.ndarray | None

    def __len__(self) -> int:
        return len(self.input_v)

    def slice(self, start: int | None, stop: int | None) -> "VoltageData":
        section = slice(start, stop)
        target_e = None if self.target_e is None else self.target_e[section]
        return VoltageData(
            time=self.time[section],
            input_v=self.input_v[section],
            target_e=target_e,
        )


@dataclass(frozen=True)
class PreprocessedVoltageData:
    """원본 전압값으로 구성한 학습·검증·테스트 시퀀스."""

    X_train: np.ndarray
    y_train: np.ndarray
    X_valid: np.ndarray
    y_valid: np.ndarray
    X_test: np.ndarray | None
    y_test: np.ndarray | None
    scaler: "VoltageScaler | None"


def _parse_float(value: str | None, column: str, row_number: int) -> float:
    if value is None or not value.strip():
        raise ValueError(f"{row_number}행의 {column} 값이 비어 있습니다.")
    try:
        return float(value)
    except ValueError as exc:
        raise ValueError(
            f"{row_number}행의 {column} 값을 실수로 변환할 수 없습니다: {value!r}"
        ) from exc


def load_voltage_data(
    path: str | PathLike[str], *, require_target: bool = True
) -> VoltageData:
    """SKHY CSV/TXT를 로드한다.

    ``require_target=True``면 ``E``가 빈 행을 오류로 처리한다.
    ``False``면 E가 모두 빈 테스트 파일도 로드할 수 있다. E가
    일부 행에만 있는 파일은 데이터 오류로 간주한다.
    """

    with open(path, encoding="utf-8", newline="") as file:
        reader = csv.DictReader(file)
        fieldnames = set(reader.fieldnames or [])
        required_columns = {"TIME", "Input_V"}
        missing = required_columns - fieldnames
        if missing:
            raise ValueError(f"필수 컬럼이 없습니다: {sorted(missing)}")
        if require_target and "E" not in fieldnames:
            raise ValueError("타깃 컬럼 E가 없습니다.")

        times: list[float] = []
        inputs: list[float] = []
        target_values: list[float | None] = []

        for row_number, row in enumerate(reader, start=2):
            times.append(_parse_float(row.get("TIME"), "TIME", row_number))
            inputs.append(_parse_float(row.get("Input_V"), "Input_V", row_number))

            raw_target = row.get("E")
            if raw_target is None or not raw_target.strip():
                if require_target:
                    raise ValueError(f"{row_number}행의 E 값이 비어 있습니다.")
                target_values.append(None)
            else:
                target_values.append(_parse_float(raw_target, "E", row_number))

    if not inputs:
        raise ValueError("데이터 행이 없습니다.")

    has_targets = [value is not None for value in target_values]
    if any(has_targets) and not all(has_targets):
        raise ValueError("E 컬럼에 값이 있는 행과 빈 행이 섞여 있습니다.")

    target_e = None
    if target_values and all(has_targets):
        target_e = np.asarray(target_values, dtype=np.float64)

    return VoltageData(
        time=np.asarray(times, dtype=np.float64),
        input_v=np.asarray(inputs, dtype=np.float64),
        target_e=target_e,
    )


def split_train_validation(
    data: VoltageData, train_ratio: float = TRAIN_RATIO
) -> tuple[VoltageData, VoltageData]:
    """시간 순서를 유지하며 앞 90%를 학습, 뒤 10%를 검증으로 나눈다."""

    if not 0 < train_ratio < 1:
        raise ValueError("train_ratio는 0과 1 사이여야 합니다.")
    split_index = int(len(data) * train_ratio)
    return data.slice(None, split_index), data.slice(split_index, None)


class VoltageScaler:
    """학습 구간의 평균·표준편차로 Input_V와 E를 각각 표준화한다."""

    def __init__(self) -> None:
        self.input_mean: float | None = None
        self.input_std: float | None = None
        self.target_mean: float | None = None
        self.target_std: float | None = None

    def fit(self, train_data: VoltageData) -> "VoltageScaler":
        if train_data.target_e is None:
            raise ValueError("스케일러 fit에는 E 타깃이 필요합니다.")

        self.input_mean = float(train_data.input_v.mean())
        self.input_std = float(train_data.input_v.std())
        self.target_mean = float(train_data.target_e.mean())
        self.target_std = float(train_data.target_e.std())

        if self.input_std == 0:
            raise ValueError("학습 데이터의 Input_V 표준편차가 0입니다.")
        if self.target_std == 0:
            raise ValueError("학습 데이터의 E 표준편차가 0입니다.")
        return self

    def _check_fitted(self) -> None:
        if any(
            value is None
            for value in (
                self.input_mean,
                self.input_std,
                self.target_mean,
                self.target_std,
            )
        ):
            raise RuntimeError("VoltageScaler.fit()을 먼저 호출해야 합니다.")

    def transform_input(self, values: np.ndarray) -> np.ndarray:
        self._check_fitted()
        return (np.asarray(values, dtype=np.float64) - self.input_mean) / self.input_std

    def transform_target(self, values: np.ndarray) -> np.ndarray:
        self._check_fitted()
        return (np.asarray(values, dtype=np.float64) - self.target_mean) / self.target_std

    def inverse_target(self, values: np.ndarray) -> np.ndarray:
        self._check_fitted()
        return np.asarray(values, dtype=np.float64) * self.target_std + self.target_mean


def build_sequences(
    input_values: np.ndarray,
    target_values: np.ndarray | None = None,
    *,
    sequence_length: int = SEQUENCE_LENGTH,
    interval: int = INTERVAL,
    stride: int = STRIDE,
) -> tuple[np.ndarray, np.ndarray | None]:
    """1차원 파형을 RNN 입력 ``(N, sequence_length, 1)``로 변환한다.

    타깃 오프셋은 노트북과 같이 ``sequence_length * interval - 1``이다.
    기본값에서 입력 오프셋은 0, 3, ..., 447이고 타깃은 449이다.
    """

    if sequence_length <= 0 or interval <= 0 or stride <= 0:
        raise ValueError("sequence_length, interval, stride는 모두 양수여야 합니다.")

    inputs = np.asarray(input_values, dtype=np.float64).reshape(-1)
    targets = None
    if target_values is not None:
        targets = np.asarray(target_values, dtype=np.float64).reshape(-1)
        if len(targets) != len(inputs):
            raise ValueError("input_values와 target_values의 길이가 같아야 합니다.")

    target_offset = sequence_length * interval - 1
    sample_count = len(inputs) - target_offset
    if sample_count <= 0:
        raise ValueError(
            f"시퀀스 생성에 최소 {target_offset + 1}행이 필요합니다. "
            f"현재 {len(inputs)}행입니다."
        )

    starts = np.arange(0, sample_count, stride)
    input_offsets = np.arange(0, sequence_length * interval, interval)
    X = inputs[starts[:, None] + input_offsets].reshape(-1, sequence_length, 1)
    y = None if targets is None else targets[starts + target_offset].reshape(-1, 1)
    return X, y


def prepare_voltage_datasets(
    train_path: str | PathLike[str],
    test_path: str | PathLike[str] | None = None,
    *,
    test_has_targets: bool = True,
    train_ratio: float = TRAIN_RATIO,
    sequence_length: int = SEQUENCE_LENGTH,
    interval: int = INTERVAL,
    stride: int = STRIDE,
) -> PreprocessedVoltageData:
    """파일을 로드하고 원본 전압값으로 시퀀스를 생성한다."""

    raw_train = load_voltage_data(train_path, require_target=True)
    train_data, valid_data = split_train_validation(raw_train, train_ratio)

    X_train, y_train = build_sequences(
        train_data.input_v,
        train_data.target_e,
        sequence_length=sequence_length,
        interval=interval,
        stride=stride,
    )
    X_valid, y_valid = build_sequences(
        valid_data.input_v,
        valid_data.target_e,
        sequence_length=sequence_length,
        interval=interval,
        stride=stride,
    )

    X_test = None
    y_test = None
    if test_path is not None:
        test_data = load_voltage_data(test_path, require_target=test_has_targets)
        X_test, y_test = build_sequences(
            test_data.input_v,
            test_data.target_e,
            sequence_length=sequence_length,
            interval=interval,
            stride=stride,
        )

    # require_target=True인 학습/검증 경로에서 y는 항상 ndarray이다.
    assert y_train is not None and y_valid is not None
    return PreprocessedVoltageData(
        X_train=X_train,
        y_train=y_train,
        X_valid=X_valid,
        y_valid=y_valid,
        X_test=X_test,
        y_test=y_test,
        scaler=None,
    )
