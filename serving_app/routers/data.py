"""SKHY 전압 데이터 CSV 업로드.

/data 폴더는 이 라우터로 업로드된 CSV만 쌓이는 곳입니다(data/uploads/). 여러 번
업로드하면 계속 쌓이고, 학습(train_and_register.py, fine_tune 등)은 항상 가장
최근 파일 하나를 사용합니다(data/storage.py의 latest_upload()).

대시보드(static/index.html)에서 파일을 올리면 이 엔드포인트가 호출됩니다.
"""
import csv
import io
import math
import os
import time

from fastapi import APIRouter, File, HTTPException, UploadFile

from data.storage import UPLOAD_DIR, latest_upload
from data.voltage_preprocessing import (
    INTERVAL,
    SEQUENCE_LENGTH,
    STRIDE,
    load_voltage_data,
)
from serving_app.monitoring.drift_detector import WINDOW_SIZE

router = APIRouter(prefix="/data")

REQUIRED_COLUMNS = ("TIME", "Input_V", "E")
TARGET_OFFSET = SEQUENCE_LENGTH * INTERVAL - 1
MIN_ROWS = TARGET_OFFSET + 1 + (WINDOW_SIZE - 1) * STRIDE  # 최소행


def _validate_rows(reader: csv.DictReader) -> list[dict[str, str]]:
    """업로드 CSV의 컬럼·숫자값·최소 행 수를 검증한다."""

    fieldnames = set(reader.fieldnames or [])
    missing_columns = set(REQUIRED_COLUMNS) - fieldnames
    if missing_columns:
        raise HTTPException(
            400,
            f"CSV에 필수 컬럼이 없습니다: {sorted(missing_columns)}",
        )

    rows = list(reader)
    if len(rows) < MIN_ROWS:
        raise HTTPException(400, f"최소 {MIN_ROWS}행 이상의 데이터가 필요합니다.")

    for row_number, row in enumerate(rows, start=2):
        if None in row:
            raise HTTPException(400, f"{row_number}행에 컬럼 수보다 많은 값이 있습니다.")
        for column in REQUIRED_COLUMNS:
            value = row.get(column)
            if value is None or not value.strip():
                raise HTTPException(400, f"{row_number}행의 {column} 값이 비어 있습니다.")
            try:
                number = float(value)
            except ValueError as exc:
                raise HTTPException(
                    400, f"{row_number}행의 {column} 값은 숫자여야 합니다."
                ) from exc
            if not math.isfinite(number):
                raise HTTPException(
                    400, f"{row_number}행의 {column} 값은 유한한 숫자여야 합니다."
                )
    return rows


@router.post("/upload")
async def upload(file: UploadFile = File(...)):
    if not file.filename or not file.filename.lower().endswith(".csv"):
        raise HTTPException(400, "CSV 파일만 업로드할 수 있습니다.")

    raw = await file.read()
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError:
        raise HTTPException(400, "UTF-8로 인코딩된 CSV 파일만 업로드할 수 있습니다.")

    reader = csv.DictReader(io.StringIO(text))
    rows = _validate_rows(reader)

    os.makedirs(UPLOAD_DIR, exist_ok=True)
    dest = os.path.join(UPLOAD_DIR, f"skhy_{time.time_ns()}.csv")
    with open(dest, "w", encoding="utf-8", newline="") as f:
        f.write(text)

    return {"filename": os.path.basename(dest), "rows": len(rows)}


@router.get("/status")
def status():
    try:
        path = latest_upload()
    except FileNotFoundError:
        return {"exists": False}

    data = load_voltage_data(path, require_target=True)
    assert data.target_e is not None
    return {
        "exists": True,
        "filename": os.path.basename(path),
        "rows": len(data),
        "start_time": float(data.time[0]),
        "end_time": float(data.time[-1]),
        "min_input_v": float(data.input_v.min()),
        "max_input_v": float(data.input_v.max()),
        "min_e": float(data.target_e.min()),
        "max_e": float(data.target_e.max()),
    }
