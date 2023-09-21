import pandas as pd
from sklearn import preprocessing
import warnings
warnings.filterwarnings('ignore')
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from catboost import CatBoostClassifier

train = pd.read_csv('/train.csv')
test = pd.read_csv('/test.csv')
sub = pd.read_csv('/sample_submission.csv')
train_x = train.drop(columns=['father','mother','gender'])
test_x= test.drop(columns=['father','mother','gender'])

class_le = preprocessing.LabelEncoder()
snp_le = preprocessing.LabelEncoder()
snp_col = [f'SNP_{str(x).zfill(2)}' for x in range(1,16)]

train_x = train.drop(['id','class','father','mother','gender'], axis=1)
test_x = test.drop(['id','father','mother','gender'] , axis = 1)
train_y = train['class']
snp_data = []
for col in snp_col:
    snp_data += list(train_x[col].values)


train_y = class_le.fit_transform(train_y)
snp_le.fit(snp_data)


for col in train_x.columns:
    if col in snp_col:
        train_x[col] = snp_le.transform(train_x[col])
        test_x[col] = snp_le.transform(test_x[col])

train_x = train_x.drop(['SNP_06','SNP_09','SNP_10'], axis=1)
test_x = test_x.drop(['SNP_06','SNP_09','SNP_10'], axis=1)

x_train, x_test, y_train , y_test = train_test_split(train_x , train_y , random_state = 42)


# CatBoost 모델 초기화
catboost_model = CatBoostClassifier(iterations=100, depth=6, learning_rate=0.1, random_state=42)

# RandomForest 모델 초기화
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)

# 앙상블 모델 생성 (다수결 투표 방식)
ensemble_model = VotingClassifier(estimators=[('catboost', catboost_model), ('random_forest', rf_model)], voting='soft')

# 앙상블 모델 학습
ensemble_model.fit(x_train, y_train)


y_pred = ensemble_model.predict(x_test)

y_pred= class_le.inverse_transform(y_pred)
df = pd.DataFrame({'id':test['id'] , 'class':y_pred})

df.to_csv('/submission.csv', index = False)