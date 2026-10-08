"""로컬 MLflow 저장소를 백업·초기화하고 SKHY base 모델을 등록한다."""

from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from serving_app.mlflow_config import MLFLOW_ARTIFACT_DIR, MLFLOW_DB_PATH, RUNTIME_DIR


def backup_existing_store() -> list[Path]:
    """기존 로컬 저장소를 삭제하지 않고 runtime/backups로 이동한다."""

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_dir = RUNTIME_DIR / "backups"
    moved: list[Path] = []

    if MLFLOW_DB_PATH.exists() or MLFLOW_ARTIFACT_DIR.exists():
        backup_dir.mkdir(parents=True, exist_ok=True)
    if MLFLOW_DB_PATH.exists():
        destination = backup_dir / f"mlflow_{timestamp}.db"
        MLFLOW_DB_PATH.replace(destination)
        moved.append(destination)
    if MLFLOW_ARTIFACT_DIR.exists():
        destination = backup_dir / f"mlartifacts_{timestamp}"
        MLFLOW_ARTIFACT_DIR.replace(destination)
        moved.append(destination)

    return moved


def main() -> None:
    backups = backup_existing_store()
    if backups:
        print("[BACKUP] " + ", ".join(str(path) for path in backups))
    else:
        print("[BACKUP] 기존 runtime MLflow 저장소 없음")

    from serving_app.train_and_register import train_and_register

    result = train_and_register()
    print(f"[RESULT] {result}")
    if not result.get("promoted"):
        raise SystemExit("RMSE 게이트를 통과하지 못해 Production 모델이 등록되지 않았습니다.")


if __name__ == "__main__":
    main()
