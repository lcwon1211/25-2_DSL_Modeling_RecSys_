# 25-2_DSL_Modeling_RecSys_

## **Running Course Recommendation System**

**Personalized Running Route Recommender for Seodaemun-gu, Seoul**

## **0. Overview**
This project develops a personalized running route recommendation system based on the real road network of Seodaemun-gu, Seoul.
It aims to generate and recommend optimized loop routes by considering various environmental and personal factors, such as slope, safety, scenery, and user preferences.
The full pipeline consists of four major stages:

<img width="940" height="457" alt="image" src="https://github.com/user-attachments/assets/b694df1e-fe0c-48f3-8bf2-a37dc519707a" />

1. Road Network Data Integration and Preprocessing
2. Running Route Generation via Customized Dijkstra Algorithm
3. User-Item Interaction Matrix Construction
4. Deep Learning-based Recommendation System Development

---

## **1. Repository Structure**

```
RunningCourse_Recommendation/
│
├── report/
│   ├── 25-2_DSL_RecSys_report.pdf             # Final project report
│   └── 25-2_RecSys_runningcourse_recsystem.pdf # Presentation slides
│
├── Dijkstra/                                  # Route generation module
│   ├── main.py                                # Running route generation script
│   ├── requirements.txt                       # Required packages
│   └── README.md                              # Usage guide and explanation
│
├── TwoTower/                                  # Two-Tower recommendation model
│   ├── TwoTower_Warmstart/
│   ├── TwoTower_Coldstart/
│   └── README.md
│
├── DCNDeepFM/                                 # Deep & Cross Network + DeepFM model
│   ├── DCNDeepFM_Warmstart_train.py
│   ├── DCNDeepFM_Coldstart_train.py
│   └── requirements.txt
│
├── KGAT/                                      # Knowledge Graph Attention Network
│   ├── cold.py
│   ├── warm.py
│   ├── triple.py
│   ├── traintest.py
│   └── requirements.txt
│
└── Dataset/                                   # Common datasets
    ├── df_route_capped_normalized.csv         # Normalized route data
    ├── kg_triples_final.csv                   # Knowledge graph triples
    ├── Mega_DB_processed.csv                  # Road network + external data
    ├── nodes_db.csv                           # Node-level road network info
    ├── output4 (1).csv                        # Intermediate output
    ├── pf_route.csv                           # Candidate route dataset
    └── user_preferred_route.csv               # User–route interaction data


```

---

## **2. Data Pipeline**

### **(1) Data Preparation**

- **MegaDB.csv / nodes_db.csv** — Raw road and node-level data from Seodaemun-gu.
- **MegaDB_processed.csv** — Standardized attributes per road segment, including slope, CCTV density, and lighting level.

### **(2) Route Generation**

- **generated_routes.csv** — 38,000 loop-shaped running routes
    
    Each route includes total distance, slope summary, and edge sequence information.
    

### **(3) Feature Engineering & Knowledge Graph**

- **routes_with_features.csv** — Normalized route-level features such as difficulty, safety, scenery, and convenience score.
- **kg_triples_v3.csv** — Structured knowledge graph representing route–feature–node relationships in (subject, predicate, object) triples.

### **(4) User Preference Modeling**

- **user_output.csv** — Running preference survey from 100 participants.
- **user_preferred_route.csv** — Interaction data aligning user preferences with route features, used as labeled training input for AI models.

---

## **3. Model Summary**

- **Dijkstra** is a weighted shortest-path model that generates feasible running routes from the actual road network.
- **Two-Tower** adopts a dual-encoder architecture for both Warm and Cold scenarios, independently encoding user and route embeddings for similarity-based retrieval.
- **KGAT** is a graph-based attention network applied to Warm and Cold settings, learning relational signals between users, routes, and features via a knowledge graph.
- **DCN + DeepFM** is a hybrid feature interaction model, also supporting Warm and Cold scenarios, combining explicit cross features and deep factorization to capture non-linear interactions.


---

## **5. Evaluation**

Model performance was evaluated using **HitRate@K.**
Each model was trained and validated under both **Warm-start** and **Cold-start** scenarios.

<img width="810" height="351" alt="image" src="https://github.com/user-attachments/assets/4743b52a-50e6-41db-b4f5-c4121b76ef06" />

- **Two-Tower** — Efficient embedding-based retrieval model with low latency
- **DCN + DeepFM** — Captures complex feature interactions; high  accuracy but risk of overfitting
- **KGAT** — Strong performance in Cold-start environments due to relational reasoning

---

## **6. References**

- OpenStreetMap (OSMnx)
- Seoul Public Data Portal
- He et al., *Neural Collaborative Filtering*, WWW 2017
- Wang et al., *KGAT: Knowledge Graph Attention Network*, KDD 2019
