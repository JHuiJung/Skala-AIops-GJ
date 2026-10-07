"""./data 안의 SKHY CSV 3개가 제대로 로딩되는지 확인하는 스크립트."""
from __future__ import annotations

import csv
import sys
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent / "data"
FILES = [
    "SKHY_train.csv",
    "SKHY_test_answer.csv",
    "SKHY_test_noanswer.csv",
]


def _to_float(value: str | None) -> float | None:
    """빈 칸이거나 컬럼이 없으면 None, 아니면 float."""
    if value is None or value.strip() == "":
        return None
    return float(value)


def load_rows(csv_path: str | Path) -> list[dict]:
    # utf-8-sig: 엑셀에서 저장한 CSV의 BOM 때문에 첫 컬럼명이 깨지는 것 방지
    with open(csv_path, encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError(f"{csv_path}: 빈 파일입니다")
        # 헤더 앞뒤 공백 제거
        reader.fieldnames = [name.strip() for name in reader.fieldnames]

        missing = [c for c in ("TIME", "Input_V") if c not in reader.fieldnames]
        if missing:
            raise KeyError(
                f"{csv_path}: 컬럼 {missing} 없음 (실제 헤더: {reader.fieldnames})"
            )

        rows = []
        for r in reader:
            try:
                rows.append(
                    {
                        "TIME": (r["TIME"] or "").strip(),
                        "Input_V": _to_float(r["Input_V"]),
                        "E": _to_float(r.get("E")),  # noanswer 파일은 E가 없거나 비어 있을 수 있음
                    }
                )
            except ValueError as e:
                raise ValueError(
                    f"{csv_path} {reader.line_num}번째 줄 숫자 변환 실패: {r}"
                ) from e
    return rows


def main() -> int:
    loaded: dict[str, list[dict]] = {}
    ok = True

    for name in FILES:
        path = DATA_DIR / name
        print(f"\n=== {name} ===")

        if not path.exists():
            print(f"  [FAIL] 파일 없음: {path}")
            ok = False
            continue

        try:
            rows = load_rows(path)
        except Exception as e:
            print(f"  [FAIL] {e}")
            ok = False
            continue

        loaded[name] = rows
        n = len(rows)
        v_missing = sum(r["Input_V"] is None for r in rows)
        e_missing = sum(r["E"] is None for r in rows)

        print(f"  행 수: {n}")
        print(f"  Input_V 결측: {v_missing}  |  E 결측: {e_missing}")
        for r in rows[:3]:
            print(f"  {r}")
        if n > 3:
            print("  ...")
            print(f"  {rows[-1]}")

        if n == 0:
            print("  [WARN] 데이터 행이 없습니다")
        print("  [OK]")

    # test_answer 와 test_noanswer 가 같은 시점들을 가지고 있는지 확인
    a = loaded.get("SKHY_test_answer.csv")
    b = loaded.get("SKHY_test_noanswer.csv")
    if a is not None and b is not None:
        print("\n=== test_answer vs test_noanswer ===")
        if len(a) != len(b):
            print(f"  [WARN] 행 수 다름: {len(a)} vs {len(b)}")
        else:
            diff = sum(x["TIME"] != y["TIME"] for x, y in zip(a, b))
            print(f"  행 수 같음 ({len(a)}), TIME 불일치 {diff}건")

    print("\n모두 로딩 성공" if ok else "\n로딩 실패한 파일이 있습니다")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())