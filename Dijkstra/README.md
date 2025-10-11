# Running Route Generator with Dijkstra Algorithm

사용자의 필요를 만족하는 **최적의 러닝 경로를 생성하는 Python 프로그램**입니다. 경사, 야간 안전성, 자연경관, 편의시설 등 다양한 요소를 고려하여 사용자 맞춤형 순환 러닝 코스를 만듭니다.

## 0. 중요 사항: 지역 제한

**현재 Mega DB는 서울특별시 서대문구(Seodaemun-gu) 전용입니다.**

- 기본 지역: `Seodaemun-gu, Seoul, South Korea`
- 다른 지역 사용 시: 해당 지역의 **Mega DB를 직접 구축**해야 합니다
- `--place` 옵션으로 다른 지역을 지정할 수 있지만, MegaDB가 해당 지역의 데이터를 포함하지 않으면 경로 생성이 실패하거나 부정확한 결과가 나올 수 있습니다

### 다른 지역을 위한 MegaDB 구축 방법

다른 지역에서 사용하려면:
1. 해당 지역의 OSM 데이터 다운로드 및 외부 데이터 결합
2. 같은 로직으로 edge 속성 계산 (경사도, cctv 및 가로등 개수 등)
3. `dataset/Mega DB.csv` 교체

## 1. 실행 방법

### 1. 필요한 패키지 설치

```bash
pip install -r requirements.txt
```

### 2. Mega DB.csv 파일 준비

`dataset/` 폴더에 `Mega DB.csv` 파일을 넣어주세요.

### 3. 기본 사용법

```bash
python main.py --lat 37.5603066 --lon 126.9368383
```

## 2. 주요 옵션

#### 필수 옵션
- `--lat LATITUDE`: 시작 지점 위도 (필수)
- `--lon LONGITUDE`: 시작 지점 경도 (필수)
- `--distance METERS`: 목표 거리(미터) (기본값: 5000)
- `--difficulty {최하,하,중,상}`: 난이도 레벨 (기본값: 최하)

#### 안전 설정
- `--night-mode`: 야간 모드 활성화 (가로등/CCTV 고려하여 안전 경로 보장)

#### 횡단보도 설정
- `--crossing-policy {forbid,penalize,allow}`: 횡단보도 정책
  - `forbid`: 횡단보도 있는 구간 제외
  - `penalize`: 횡단보도에 페널티 부여 (기본값)
  - `allow`: 횡단보도 허용

#### 자연경관 설정
- `--scenic-ratio RATIO`: 목표 자연경관 비율 0-1 (기본값: 0.3 = 30%)

#### 필수 경유지 설정
- `--require-convenience`: 편의점 필수 경유
- `--require-toilets`: 화장실 필수 경유
- `--require-scenic`: 자연경관(공원/수변) 필수 경유
- `--require-subway`: 지하철역 필수 경유

#### 파일 경로
- `--csv PATH`: Mega DB CSV 경로 (기본값: dataset/Mega DB.csv)
- `--output FILE`: 출력 HTML 파일명 (기본값: running_loop_route.html)

## 3. 사용 예시

### 1. 기본 5km 코스 (연세대학교 정문 출발, 쉬운 난이도)
```bash
python main.py --lat 37.5603066 --lon 126.9368383
```

### 2. 8km 중급 난이도 코스
```bash
python main.py --lat 37.5603066 --lon 126.9368383 \
  --distance 8000 \
  --difficulty 중
```

### 3. 야간 러닝 코스 (횡단보도 금지)
```bash
python main.py --lat 37.5603066 --lon 126.9368383 \
  --night-mode \
  --crossing-policy forbid \
  --night-weight 15.0
```

### 4. 자연경관 중심 코스 (40% 자연경관)
```bash
python main.py --lat 37.5603066 --lon 126.9368383 \
  --scenic-ratio 0.4 \
  --require-scenic
```

## 4. 실행 결과

프로그램 실행 후 다음과 같은 결과를 얻을 수 있습니다:

1. **콘솔 출력**: 경로 생성 과정 및 최종 결과 통계
2. **HTML 파일**: 인터랙티브 지도 (브라우저에서 열기)
   - 시작/종료 지점 (녹색 마커)
   - 앵커 포인트 (파란색 마커)
   - 경로 라인 (파란색)
   - 상세 정보 팝업