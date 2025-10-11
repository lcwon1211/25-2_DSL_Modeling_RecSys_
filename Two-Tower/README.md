## **0. 개요**

**Two-Tower (Dual Encoder)** 는 사용자와 아이템 임베딩을 각각 독립적으로
학습하여 효율적인 **유사도 기반 검색 및 추천 (Embedding-based
Retrieval)** 을 수행하는 모델이다.

본 프로젝트에서는 사용자와 아이템을 분리 인코딩한 뒤, 두 임베딩 벡터의
**Dot Product** 또는 **Cosine Similarity**를 계산하여 가장 유사한
후보군을 신속하게 검색하고 추천하도록 설계하였다.

------------------------------------------------------------------------

## **1. 주요 기능**

-   사용자 타워 (User Tower) / 아이템 타워 (Item Tower) 이중 구조
-   Negative Sampling 기반 대조 학습 (Contrastive Learning)
-   Dot Product / Cosine Similarity 기반 유사도 계산
-   TensorFlow 및 PyTorch 환경 호환
-   Recall@K, NDCG@K, HitRate@K 등 주요 추천 평가 지표 지원
-   FAISS / ANN을 통한 대규모 임베딩 검색 확장 가능

------------------------------------------------------------------------

## **2. 모델 구조 및 데이터 구성 (Architecture & Data)**

### **1) 모델 구조 (Text Description)**

Two-Tower 모델은 사용자(User)와 아이템(Item)을 각각 별도의
인코더(Encoder Tower)를 통해 임베딩 공간으로 투영한 뒤, 두 벡터의
유사도를 계산하여 추천을 수행한다.

-   **User Encoder (사용자 인코더)**\
    사용자 ID, 러닝 빈도, 시간대, 거리·경사 선호도 등의 feature를
    입력으로 받아 MLP를 통해 저차원 임베딩으로 변환한다.

-   **Item Encoder (아이템 인코더)**\
    러닝 코스의 거리, 경사도, 안전 등급, 루프형 여부 등 아이템 속성을
    입력으로 받아 동일한 차원의 임베딩을 생성한다.

-   **Similarity Calculation (유사도 계산)**\
    사용자 임베딩과 아이템 임베딩 간 **Dot Product** 혹은 **Cosine
    Similarity**를 계산하여 유사도를 측정하고, 가장 높은 유사도를 가진
    코스를 추천한다.

-   **Negative Sampling (음성 샘플링)**\
    사용자가 선택하지 않은 코스를 부정 샘플로 설정하고, Contrastive
    Learning 방식을 통해 "정답 쌍의 유사도 \> 부정 쌍의 유사도"가 되도록
    학습한다.

-   **Loss Function (손실 함수)**\
    **BPR Loss** 또는 **Contrastive Loss**를 사용하며, 다음과 같이
    정의된다:\
    \[ L = -`\log `{=tex}`\sigma`{=tex}(s\_{u,i\^+} - s\_{u,i\^-}) \]

------------------------------------------------------------------------

### **2) 데이터 구조**

    datasets/
    ├── users.csv           # user_id, feature1, feature2, ...
    ├── items.csv           # item_id, feature1, feature2, ...
    └── interactions.csv    # user_id, item_id, label

------------------------------------------------------------------------

## **3. 실행 방법 (Usage)**

### **1) 환경 설정**

``` bash
pip install -r requirements.txt
```

### **2) Warm-Start 학습**

``` bash
python warmTT.py train
python warmTT.py train --warm_start
python warmTT.py evaluate
```

### **3) Cold-Start 학습**

``` bash
python twotower.py
```

------------------------------------------------------------------------

## **4. 실행 결과**

-   **평가 지표 정의:** HR@K는 "사용자가 선호한 아이템이 상위 K 안에
    포함될 확률"로 계산했습니다.\
-   **Warm-Start:** HR@1 = 0.52, HR@10 = 0.68\
-   **Cold-Start:** HR@1 = 0.32, HR@10 = 0.60
