## **0. 개요**

**KGAT (Knowledge Graph Attention Network)** 는 사용자, 러닝 코스,
그리고 환경 속성 간의 복잡한 관계를 **지식그래프(Knowledge Graph)** 로
구축하고, **어텐션 메커니즘**을 통해 중요한 관계에 높은 가중치를
부여하여 사용자 맞춤형 러닝 코스를 추천하는 모델이다.

본 프로젝트는 KGAT 모델을 활용하여 **러닝 코스 추천 시스템을
개발·학습·성능 검증까지 완성**하는 것을 목표로 했으며, 사용자 선호도와
실제 도로망 데이터를 결합한 데이터셋을 기반으로 **Recall@K**,
**NDCG@K**, **HitRate@K** 등의 지표로 모델의 실효성을 평가하였다.

------------------------------------------------------------------------

## **1. 주요 기능**

-   **지식그래프 기반 추천**\
    사용자, 코스, 속성(feature) 간의 관계를 **(head, relation, tail)**
    형태의 트리플로 구성하여\
    아이템의 의미적 표현을 학습함으로써 추천 정확도를 높임.

-   **어텐션 메커니즘 적용**\
    모델이 사용자별로 더 중요한 관계(예: 경사, 조명, 안전도 등)에 **동적
    가중치(attention weight)** 를 부여하여\
    개인화된 추천을 수행함.

-   **Hard Negative Sampling**\
    유사한 코스 간의 차이를 학습하도록 구성해 Cold-Start 환경에서도
    일반화 성능을 강화함.

-   **Warm-Start / Cold-Start 시나리오 지원**\
    기존 사용자(Warm)와 신규 사용자(Cold)를 각각 독립적으로
    학습·평가하여,\
    실제 서비스 환경에서의 적용 가능성을 검증함.

------------------------------------------------------------------------

## **2. 모델 구조 및 데이터 구성 (Architecture & Data)**

### **1) 모델 구조**

<img width="1156" height="364" alt="25-2_DSL_RecSys_report" src="https://github.com/user-attachments/assets/c0c7006f-0231-42b8-8d6c-c375ec075a25" />

### **2) 데이터 구성**

    data/
    ├── kg_triples_final.csv         # (head, relation, tail) 형태의 지식그래프 트리플
    ├── user_preferred_route.csv     # 사용자별 경로 선호도 전체 순위
    ├── train_interactions_warm.csv  # Warm-Start 훈련 데이터
    ├── test_interactions_warm.csv   # Warm-Start 테스트 데이터
    ├── train_interactions_cold.csv  # Cold-Start 훈련 데이터
    └── test_interactions_cold.csv   # Cold-Start 테스트 데이터

------------------------------------------------------------------------

## **3. 실행 방법 (Usage)**

### **1) 환경 설정**

``` bash
pip install -r requirements.txt
```

### **2) 데이터 생성 및 분할 (최초 1회 실행)**

``` bash
# 지식그래프 생성 및 학습용 데이터 분할
notebooks/triple.ipynb
notebooks/traintest.ipynb
```

### **3) 모델 학습**

-   **Warm-Start 모델**

    ``` bash
    python src/warm.py
    ```

    → 완료 시 **`models/best_model_warm_start.pth`** 생성

-   **Cold-Start 모델**

    ``` bash
    python src/cold.py
    ```

    → 완료 시 **`models/best_model_cold_start.pth`** 생성

------------------------------------------------------------------------

## **4. 실행 결과 **

-   **평가 지표 정의:** HR@K는 "사용자가 선호한 아이템이 상위 K 안에
    포함될 확률"로 계산했습니다.\
-   **Warm-Start:** HR@1 = 0.32, HR@10 = 0.85\
-   **Cold-Start:** HR@1 = 0.56, HR@10 = 0.84
