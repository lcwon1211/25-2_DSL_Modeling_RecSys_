# Running Route Generator

## 0. 개요

**Running Route Generator**는 사용자 선호도를 직접적으로 반영할 수 있는 **맞춤형 러닝 코스 생성 알고리즘**이다.

**Dijkstra 알고리즘**에 러닝 특화 비용 함수(customized cost function)를 적용하여 경사도, 야간 안전성, 자연경관, 편의시설을 종합적으로 고려한 루프형 경로를 생성한다.

📍 **주의**: 현재 버전은 서울특별시 서대문구에 한정한다.

---

## 1. 주요 기능

- **맞춤형 경로**: 출발지, 목표 거리, 4단계 난이도 설정, 루프형 코스 자동 생성
- **안전성**: 야간 모드(CCTV/가로등 밀도 고려), 횡단보도 정책 설정
- **환경 고려**: 자연경관 비율 조절(0-100%), 필수 경유지 지정(편의점/화장실/지하철/공원)
- **러닝 코스 시각화**: 인터랙티브 HTML 지도 및 상세 경로 통계 제공

---

## 2. 데이터 및 구조 (Data & Architecture)

### 1) 지역 제한

현재 Mega DB는 **서울특별시 서대문구(Seodaemun-gu)** 전용으로 구성되어 있다.

다른 지역에서 실행하려면 해당 지역의 Mega DB를 직접 구축해야 한다.

### 2) 프로그램 구조

```
Dijkstra/
├── main.py                      # 경로 생성 알고리즘 메인 실행 파일
├── dataset/
│   └── Mega_DB.csv              # 서대문구 도로망 + 외부 환경 데이터 통합본
├── running_loop_route.html      # 기본 옵션 반영된 러닝 코스 생성 예시
└── requirements.txt             # 필요 라이브러리 목록
```

---

## 3. 실행 방법 (Usage)

### 1) 환경 설정

```bash
pip install -r requirements.txt
```

### 2) Mega DB 준비

`dataset/` 폴더에 `Mega_DB.csv` 파일을 업로드한다.

### 3) 기본 실행

```bash
python main.py --lat 37.5603066 --lon 126.9368383
```

연세대학교 정문에서 출발하는 5km, 난이도 "최하" 러닝코스 생성

### 4) 주요 옵션

#### 필수 옵션

- `--lat LATITUDE` : 시작 지점 위도 (필수)
- `--lon LONGITUDE` : 시작 지점 경도 (필수)
- `--distance METERS` : 목표 거리 (단위: m, 기본값 5000)
- `--difficulty {최하,하,중,상}` : 난이도 레벨 (기본값: 최하)

#### 안전 관련

- `--night-mode` : 야간 모드 활성화 (가로등·CCTV 고려하여 안전 경로 보장)

#### 횡단보도 정책

- `--crossing-policy {forbid,penalize,allow}`
  - `forbid` : 횡단보도 있는 구간 제외
  - `penalize` : 횡단보도 구간에 페널티 부여 (기본값)
  - `allow` : 횡단보도 허용

#### 자연경관 옵션

- `--scenic-ratio RATIO` : 목표 자연경관 비율 (0–1, 기본값 0.3)
- `--require-scenic` : 자연경관(공원·수변) 필수 경유

#### 편의시설 옵션

- `--require-convenience` : 편의점 필수 경유
- `--require-toilets` : 화장실 필수 경유
- `--require-subway` : 지하철역 필수 경유

#### 파일 경로

- `--csv PATH` : Mega DB 파일 경로 (기본값: `dataset/Mega_DB.csv`)
- `--output FILE` : 출력 HTML 파일명 (기본값: `running_loop_route.html`)

---

## 4. 실행 결과 (Output)

프로그램 실행 후 다음과 같은 결과를 얻을 수 있다:

1. **콘솔 출력**: 경로 생성 과정 및 최종 결과 통계
2. **HTML 파일**: 인터랙티브 지도(`running_loop_route.html`)
   - 시작/종료 지점: 녹색 마커
   - 앵커 포인트: 파란색 마커
   - 러닝코스 경로: 파란색 라인
   - 마커 클릭 시 상세 정보 팝업
