"""업로드된 SKHY 전압 데이터 파일 관리.

대시보드에서 ``TIME,Input_V,E`` CSV를 직접 업로드합니다
(serving_app/routers/data.py 참고).
업로드된 파일은 이 디렉터리(data/uploads/)에 타임스탬프가 붙은 이름으로 계속
쌓이고(과거 파일을 덮어쓰지 않습니다), 학습(train_and_register.py 등)은 항상
가장 최근에 올라온 파일 하나를 사용합니다.
"""
import glob
import os

UPLOAD_DIR = "data/uploads"


def latest_upload(upload_dir: str = UPLOAD_DIR) -> str:
    """data/uploads/ 에 쌓인 CSV 중 가장 최근에 업로드된 파일의 경로를 반환한다."""
    files = sorted(glob.glob(os.path.join(upload_dir, "*.csv")), key=os.path.getmtime)
    if not files:
        raise FileNotFoundError(
            "업로드된 SKHY 전압 데이터가 없습니다. "
            f"대시보드에서 TIME, Input_V, E CSV를 먼저 업로드하세요 -> {upload_dir}/"
        )
    return files[-1]
