## **0. 개요**

**DCN + DeepFM** 모델은 **Deep & Cross Network (DCN)** 과 **Deep Factorization Machine (DeepFM)** 을 결합하여 피처 간의 저차(명시적) 및 고차(비선형) 상호작용을 동시에 학습하는 하이브리드 추천 구조이다.

본 프로젝트에서는 사용자·러닝 코스의 속성(feature) 간 복합적 관계를 포착하기 위해 두 네트워크의 장점을 결합하였으며, **BPR Loss** 기반 순위 학습과 **Hard Negative Sampling**을 적용하여 러닝 코스 추천의 정밀도를 높였다.

------------------------------------------------------------------------

## **1. 주요 기능**

-   **DCN (Deep & Cross Network)**\
    명시적인 교차항 계산을 통해 **고차 피처 상호작용**을 효율적으로 학습

-   **DeepFM (Deep Factorization Machine)**\
    임베딩 기반 **비선형 피처 조합**을 학습하여 일반화 성능 향상

-   **Soft Ensemble (0.6 : 0.4)**\
    DCN과 DeepFM의 출력을 가중 평균(0.6 : 0.4)으로 결합하여 서로 보완적인 특성을 반영

-   **BPR Loss 기반 순위 학습 + Hard Negative Sampling**\
    순위 최적화 목적의 학습으로 추천 정확도 개선

-   **Warm-Start / Cold-Start 시나리오 모두 지원**

------------------------------------------------------------------------

## **2. 모델 구조 및 데이터 구성 (Architecture & Data)**

### **1) 모델 구조**

<img width="1421" height="560" alt="스크린샷 2025-10-11 182715" src="https://github.com/user-attachments/assets/f9780dbf-4e1b-4c55-b3ff-20799d1a6fff" />

------------------------------------------------------------------------

### **2) 데이터 구성**

    data/
    ├── output4(1).csv                  # 사용자 피처: 러닝 빈도, 시간대, 거리·경사 선호도 등
    ├── df_route_capped_normalized.csv  # 코스(아이템) 피처: 
    ├── user_preferred_route.csv        # 사용자–코스 상호작용 데이터 (선호도 매칭)
    ├── warm_train_data.csv             # Warm-Start 훈련 데이터
    ├── warm_test_data.csv              # Warm-Start 테스트 데이터

------------------------------------------------------------------------

## **3. 실행 방법 (Usage)**

### **1) 환경 설정**

``` bash
pip install -r requirements.txt
```

### **2) Warm-Start 학습**

``` bash
python DCNDeepFM_Warmstart_train.py   --user_feats output4(1).csv   --item_feats df_route_capped_normalized.csv   --train_pairs warm_train_data.csv   --eval_pairs warm_test_data.csv
```

### **3) Cold-Start 학습**

``` bash
python DCNDeepFM_Coldstart_train.py   --user_feats output4(1).csv   --item_feats df_route_capped_normalized.csv   --interactions user_preferred_route.csv
```

------------------------------------------------------------------------

## **4. 실행 결과**

- **평가 지표 정의**: HR@K는 “사용자가 선호한 아이템이 상위 K 안에 포함될 확률”로 계산했습니다.
- **Warm-Start**: HR@1 = 0.79, HR@10 = 1.00
- **Cold-Start**: HR@1 = 0.92, HR@10 = 1.00

HR@10 값이 1.00으로 측정되어 **데이터 샘플링 혹은 손실 함수 설정에 따른 과적합 가능성**이 확인되었다. 따라서 본 모델은 **성능 참고용**으로 사용되었으며, 실제 서비스 모델에는 반영되지 않았다.

