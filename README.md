
## 유전체 정보 품종 분류 AI 경진대회

### (Topic) - 개체와 SNP 정보를 이용하여 품종 분류 AI 모델 개발

### (MODEL) - RandomForest + Catboost+ PCA
#### 데이터가 대부분 범주형으로 이루어져 있어, 범주형 변수일 때 효과적인 Catboost와 기본적으로 성능이 좋은 RandomForest 앙상블하여 진행
#### 데이터가 16개 이상 칼럼으로 구성되어 있어, PCA를 통하여 차원축소를 진행하여 모델링


### (RESULT) - 100 / 1095 (Macro F1 Score : 0.9686)
