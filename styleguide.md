# AIOps 대시보드 스타일 가이드

> 이 문서는 AIOps(HAIC ML Pipeline for Reliable Service) 대시보드의 시각 스타일 규칙이다.
> UI를 새로 만들거나 수정할 때 아래 토큰과 컴포넌트 규칙을 그대로 따른다.
> 여기에 없는 색·크기·간격을 임의로 만들지 말고, 가장 가까운 토큰을 사용한다.

---

## 1. 디자인 원칙

- **밝은 회색 배경 위 흰색 카드.** 정보 밀도는 높게, 구분은 여백과 얇은 테두리로 한다.
- **색은 의미가 있을 때만 쓴다.**
  - 인디고 = 주요 동작 · 선택됨 · 정보
  - 초록 = 정상 · 완료 · Production
  - 주황/앰버 = 경고 · 감지됨
  - 빨강 = 임계치 초과 · 실패
  - 그 외 모든 것은 회색 계열
- **그림자보다 테두리.** 그림자는 거의 보이지 않을 정도로만 쓴다.
- **숫자가 주인공.** KPI 값은 크고 굵게, 라벨·캡션은 작고 흐리게.
- **장식 금지.** 배경 그라데이션, 일러스트, 배경 이미지 사용하지 않는다. (예외: 로고 마크)
- **라이트 테마 전용.**

---

## 2. 디자인 토큰

```css
:root {
  /* Surface */
  --bg-page: #F4F5F7;
  --bg-surface: #FFFFFF;
  --bg-subtle: #F3F4F6;        /* 중립 칩, 보조 버튼 */
  --bg-hover: #FAFAFB;         /* 표 행 hover */
  --border: #E5E7EB;
  --divider: #EEF0F2;

  /* Text */
  --text-primary: #111827;
  --text-secondary: #374151;
  --text-muted: #6B7280;
  --text-faint: #9CA3AF;

  /* Primary (Indigo) — 유일한 브랜드 색 */
  --primary: #4F46E5;
  --primary-hover: #4338CA;
  --primary-soft: #EEF2FF;
  --primary-soft-border: #C7D2FE;

  /* Status */
  --success: #16A34A;
  --success-strong: #15803D;   /* 배지 글자 */
  --success-ring: #22C55E;     /* 스테퍼 원 테두리 */
  --success-soft: #DCFCE7;

  --warning: #EA580C;          /* 파이프라인 '감지됨' 단계 */
  --warning-soft: #FFEDD5;

  --caution: #D97706;          /* [WARN] 알람 아이콘 */
  --caution-soft: #FEF3C7;

  --danger: #DC2626;           /* 임계치 초과 스파크라인, 실패 */
  --danger-soft: #FEE2E2;

  /* Radius */
  --radius-card: 16px;         /* 섹션 카드 */
  --radius-tile: 12px;         /* KPI 타일 */
  --radius-control: 8px;       /* 버튼, 세그먼트, 로고 */
  --radius-pill: 999px;        /* 배지, 칩 */

  /* Shadow */
  --shadow-card: 0 1px 2px rgba(16, 24, 40, 0.04);

  /* Spacing (4px 기준) */
  --space-1: 4px;
  --space-2: 8px;
  --space-3: 12px;
  --space-4: 16px;
  --space-5: 20px;
  --space-6: 24px;
  --space-8: 32px;

  /* Font */
  --font-sans: "Pretendard", "Inter", -apple-system, BlinkMacSystemFont,
               "Apple SD Gothic Neo", "Noto Sans KR", sans-serif;
  --font-mono: "JetBrains Mono", "SF Mono", Menlo, Consolas, monospace;
}

body {
  background: var(--bg-page);
  color: var(--text-primary);
  font-family: var(--font-sans);
  font-size: 14px;
  line-height: 1.5;
  -webkit-font-smoothing: antialiased;
}
```

---

## 3. 타이포그래피

| 용도 | 크기 | 굵기 | 색 | 예시 |
|---|---|---|---|---|
| 앱 이름 | 16px | 800 | text-primary | AIOps |
| 앱 부제 | 11px | 500 | text-muted | HAIC ML Pipeline for Reliable Service |
| 내비 탭 | 14px | 600 | text-secondary (활성: primary) | Dashboard |
| 섹션 제목 | 18px | 700 | text-primary | 운영 지표 요약 |
| 섹션 설명 | 13px | 400 | text-muted | requests.log를 실제로 읽어… |
| KPI 라벨 | 13px | 600 | text-primary | 요청 수 (Requests) |
| KPI 값 | 26px | 800 | text-primary | 946 |
| 모델명 | 18px | 800 | text-primary | HAIC_Predictor |
| 본문 / 표 셀 | 14px | 400–500 | text-primary | — |
| 키-값 라벨 | 13px | 400 | text-secondary | 스테이지 |
| 표 헤더 | 12px | 500 | text-muted | 버전, 등록 시각 |
| 칩 / 배지 | 11–12px | 600 | 상태색 | Production |
| 캡션 | 11px | 400 | text-faint | 최근 5분 |

- line-height: 본문 1.5, 제목·KPI 값 1.2.
- 숫자가 들어가는 곳(KPI 값, 표 숫자, 시각)은 `font-variant-numeric: tabular-nums;`
- 영문을 병기하는 라벨은 `한글 (English)` 형식: `요청 수 (Requests)`, `모델 성능 (RMSE)`.
- 기계적인 값은 mono 폰트: `MODEL_SOURCE=mlflow`, `fine-tune`, `scratch`, `[WARN]`.
- 대문자 변환(uppercase), 기울임체는 쓰지 않는다.

---

## 4. 레이아웃

### 4.1 페이지 구조
- **상단 헤더**: 높이 64px, 흰 배경, 하단 1px `--border`. 상단 고정(sticky).
  - 좌측: 로고 마크 + 앱명/부제 → 내비 탭
  - 우측: 현재 시각 → 모델 소스 배지
- **본문**: 최대 폭 1280px 중앙 정렬, 좌우 패딩 24px (모바일 16px), 상단 패딩 24px.
- **섹션 간격**: 16px.

### 4.2 그리드
- 1행: `운영 지표 요약` 카드 — 전체 폭.
- 2행 이후: 2열, 좌 `2fr` / 우 `1fr`.
  - 좌측 열: 드리프트 감지 기반 재학습 파이프라인 → 재학습 이력
  - 우측 열: 현재 운영 모델 → 최근 알람
- 1024px 이하: 1열로 쌓는다 (순서: 지표 → 파이프라인 → 현재 운영 모델 → 재학습 이력 → 최근 알람).

### 4.3 카드 내부
- 패딩 24px (모바일 16px).
- 제목 ↔ 설명 4px, 설명 ↔ 본문 16–20px.

---

## 5. 컴포넌트

### 5.1 헤더 / 내비게이션
- **로고 마크**: 32×32, radius 8px, 인디고 계열 배경(미세한 세로 그라데이션 허용) + 흰색 펄스(심전도) 라인 아이콘.
- **앱명/부제**: 두 줄 스택, 간격 0–2px.
- **탭**: 텍스트만, 좌우 패딩 16px, 헤더 전체 높이를 차지.
  - 활성: 글자 `--primary` + 하단 2px `--primary` 언더라인 (헤더 하단 경계선에 겹치게).
  - 비활성: `--text-secondary`, hover 시 `--text-primary`.
- **시계**: 13px, `--text-secondary`, tabular-nums, 형식 `2026. 10. 8. 9시 44분 46초`.
- **소스 배지**: pill, 배경 `--primary-soft`, 글자 `--primary`, mono 11px 700, 좌우 패딩 10px. 예: `MODEL_SOURCE=mlflow`.

### 5.2 섹션 카드
- 배경 `--bg-surface`, 1px `--border`, radius 16px, `--shadow-card`.
- 헤더 행: 좌측 제목+설명, 우측에 컨트롤(세그먼트 버튼) 또는 메타 배지 / 상태 배지.
- **메타 배지**: 중립 pill — 배경 `--bg-subtle`, 글자 `--text-secondary`, 11px 600. 예: `마지막 실행 0초 전`.

### 5.3 세그먼트 버튼 (기간 선택)
- 버튼 간 간격 6px (붙여 그룹화하지 않음).
- 각 버튼: 높이 28px, 좌우 패딩 12px, radius 8px, 12px 600.
- 비활성: 흰 배경, 1px `--border`, 글자 `--text-secondary`. hover 시 배경 `--bg-subtle`.
- 활성: 배경 `--primary`, 글자 흰색, 테두리 없음.
- 레이블: `5분` `1시간` `6시간` `24시간`.

### 5.4 KPI 타일
- 배경 흰색, 1px `--border`, radius 12px, 패딩 16px.
- 5개를 한 줄에 균등 배치: `grid-template-columns: repeat(5, 1fr); gap: 12px;`
  - 768px 이하 2열, 480px 이하 1열.
- 세로 순서: **라벨 → 값 → 스파크라인 → 캡션** (간격 4–8px).
- **스파크라인**
  - 높이 36–40px, 선 두께 2px, `stroke-linecap: round; stroke-linejoin: round;`
  - 색 `--primary`. 채우기·축·눈금·점·툴팁 없음.
  - 데이터가 없으면 평탄한 기준선.
  - **임계치를 넘은 지표는 선 색만 `--danger`로 바꾼다.** 값 텍스트 색은 그대로 둔다.
- **캡션**: 집계 구간 또는 기준값. 예: `최근 5분`, `MLflow 등록 버전별 추이`, `임계치 $4.00`.

### 5.5 파이프라인 스테퍼
- 7단계 가로 배치, 균등 간격. 단계 사이는 2px `--border` 선으로 원의 세로 중앙 높이에서 연결.
- **단계 원**: 지름 40px, 2px 테두리 + 연한 채움, 중앙에 18px 이모지.
- 원 아래 텍스트 (가운데 정렬):
  1. 단계명 — 13px 700 `--text-primary`
  2. 설명 — 11px `--text-muted`
  3. 상태 — 11px 700 상태색

| 상태 | 원 테두리 | 원 채움 | 상태 텍스트 |
|---|---|---|---|
| 완료 | `--success-ring` | `--success-soft` | `완료` · `--success` |
| 감지됨 / 주의 | `--warning` | `--warning-soft` | `감지됨` · `--warning` |
| 진행 중 | `--primary` | `--primary-soft` | `진행 중` · `--primary` (원 테두리 pulse 허용) |
| 대기 | `--border` | `--bg-subtle` | `대기` · `--text-faint` |
| 실패 | `--danger` | `--danger-soft` | `실패` · `--danger` |

- 단계·아이콘·설명:

| 단계 | 아이콘 | 설명 |
|---|---|---|
| 데이터 수집 | 📥 | 요청 수신 |
| 데이터 모니터링 | 📊 | 예측 기록 누적 |
| 드리프트 감지 | 🔍 | RMSE vs 임계치 |
| 재학습 트리거 | ⚙️ | 조건 충족 시 시작 |
| 모델 학습 | 📦 | fine-tuning / scratch |
| 모델 등록 | 🗃️ | MLflow Registry |
| 배포 | ☁️ | Production 승격 |

- 1024px 이하: 가로 스크롤(스크롤바 숨김) 또는 세로 스테퍼로 전환.

### 5.6 배지 · 칩
공통: pill, 높이 20–22px, 좌우 패딩 8–10px, 11–12px 600.

**상태 배지** (앞에 6px 원형 점 + 텍스트, 점 색 = 글자 색)

| 상태 | 배경 | 글자 | 예시 |
|---|---|---|---|
| Healthy / Production | `--success-soft` | `--success-strong` | `● Healthy`, `● Production` |
| Staging | `--primary-soft` | `--primary` | `● Staging` |
| Archived / None | `--bg-subtle` | `--text-muted` | `● Archived` |
| Degraded / Warning | `--caution-soft` | `--caution` | `● Degraded` |
| Down / Failed | `--danger-soft` | `--danger` | `● Failed` |

**중립 칩** (점 없음): 배경 `--bg-subtle`, 글자 `--text-secondary`, mono 11px 600.
예: `fine-tune`, `scratch`, `v99`.

- 버전 칩은 모델명 바로 오른쪽에 8px 간격으로 붙인다: **HAIC_Predictor** `v99`.

### 5.7 키-값 리스트 (현재 운영 모델)
- 각 행: 좌측 라벨(13px `--text-secondary`) / 우측 값(14px 700 `--text-primary`, 우측 정렬).
- 행 높이 약 32px, 행 사이 구분선 없음.
- 카드 헤더(제목 + 상태 배지) → 모델명+버전 칩 → 키-값 리스트 → 버튼 순서.
- 복합 값은 슬래시로 연결: `$1.47 / 게이트 $4.00`.

### 5.8 버튼
- **보조(기본)**: 배경 `--bg-subtle` (hover `--border`), 글자 `--text-primary` 13px 700, 높이 36px, 좌우 패딩 14px, radius 8px, 테두리 없음. 예: `새로고침`.
- **주요**: 배경 `--primary` (hover `--primary-hover`), 글자 흰색.
- **위험**: 배경 `--danger`, 글자 흰색.
- 포커스: `outline: 2px solid var(--primary-soft-border); outline-offset: 2px;`

### 5.9 표 (재학습 이력)
- 카드 안에 외곽 테두리 없이 배치.
- 헤더 행: 12px 500 `--text-muted`, 좌측 정렬, 하단 1px `--border`.
- 본문 행: 높이 약 44px, 셀 좌우 패딩 12px, 하단 1px `--divider`, hover 배경 `--bg-hover`.
- 열 규칙:
  - 버전: 14px 700
  - 등록 시각 / RMSE: tabular-nums
  - 모드: 중립 칩
  - 스테이지: 상태 배지
- 정렬: 최신 버전이 맨 위.

### 5.10 알람 리스트 (최근 알람)
- 각 항목: 좌측 24px 원형 아이콘 + 우측(메시지, 시간). 아이콘↔텍스트 12px.
- 항목 세로 패딩 12px, 항목 사이 1px `--divider`.
- 메시지: 13px `--text-primary`, 로그 원문 그대로, 줄바꿈 허용.
  예: `[OK] new_rmse=1.47 - production promoted: HAIC_Predictor v99`
- 시간: 11px `--text-faint`, 상대 시각 (`1분 전`).
- 설명 문구에서 로거 이름·태그는 mono: `"aiops"`, `[WARN]/[INFO]/[OK]`.

| 로그 레벨 | 아이콘 글자 | 배경 | 글자 |
|---|---|---|---|
| `[OK]`, `[INFO]` | i | `--primary-soft` | `--primary` |
| `[WARN]` | ! | `--caution-soft` | `--caution` |
| `[ERROR]` | × | `--danger-soft` | `--danger` |

---

## 6. 데이터 표기 규칙

| 항목 | 형식 | 예시 |
|---|---|---|
| 표 날짜·시각 | `MM. DD. 오전/오후 hh:mm` (0 채움) | `10. 08. 오전 09:43` |
| 헤더 시계 | `YYYY. M. D. H시 m분 s초` | `2026. 10. 8. 9시 44분 46초` |
| 상대 시각 | `N초 전`, `N분 전`, `N시간 전` | `1분 전` |
| 정수 | 천 단위 콤마 | `1,946` |
| 지연시간 | 값 + 공백 + 단위 | `37 ms` |
| 비율 | 소수 1자리, % 붙임 | `100.0%` |
| RMSE · 드리프트 점수 | `$` + 소수 2자리 | `$1.47` |
| 데이터 없음 | `0`으로 표시 (빈칸·하이픈 금지), 섹션 설명에 이 규칙을 명시 | `0` |

---

## 7. 하지 말 것

- 카드에 진한 그림자, 두꺼운 테두리, 컬러 배경을 쓰지 않는다.
- KPI 값 텍스트에 색을 입히지 않는다. 상태는 스파크라인 색과 배지로만 표현한다.
- 상태색을 장식으로 쓰지 않는다 (정상 수치를 초록 글씨로 칠하는 것 등).
- 인디고 외의 파랑·보라 계열을 새로 추가하지 않는다.
- 아이콘을 섞지 않는다. 파이프라인 단계는 이모지, 그 외 UI 아이콘은 단색 라인 아이콘 한 세트로 통일한다.
- uppercase 라벨, 기울임체, 12px 미만 본문 텍스트(캡션·칩 제외)를 쓰지 않는다.