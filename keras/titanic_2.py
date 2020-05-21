# @.@ coding:utf-8 ^-^
# @Author   : Leon Rein
# @Time     : 2020-05-18  ~  18:44 
# @File     : titanic_2.py
# @Software : PyCharm
# @Notice   : It's a WINDOWS version!


import pandas as pd
from tensorflow.keras import models
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = '2'  # Warnings or Errors ONLY  

dftest_raw = pd.read_csv(r'datasets/titanic\test.csv')
test_answer_raw = pd.read_csv(r'datasets/titanic\gender_submission.csv')


# A Function to Preprocess datasets sets
def preprocessing(dfdata):
    dfresult = pd.DataFrame()

    # Pclass
    dfPclass = pd.get_dummies(dfdata['Pclass'], prefix='Pclass_')  # One-hot
    dfresult = pd.concat([dfresult, dfPclass], axis=1)  # Merge arrays

    # Sex
    dfSex = pd.get_dummies(dfdata['Sex'])
    dfresult = pd.concat([dfresult, dfSex], axis=1)

    # Age
    dfresult['Age'] = dfdata['Age'].fillna(0)  # NaN considered to be 0
    dfresult['Age_null'] = pd.isna(dfdata['Age']).astype('int32')  # 1 for NaN

    # SibSp,Parch,Fare
    dfresult['SibSp'] = dfdata['SibSp']
    dfresult['Parch'] = dfdata['Parch']
    dfresult['Fare'] = dfdata['Fare']

    # Carbin
    dfresult['Cabin_null'] = pd.isna(dfdata['Cabin']).astype('int32')

    # Embarked
    dfEmbarked = pd.get_dummies(dfdata['Embarked'], dummy_na=True, prefix='Embarked_')
    dfresult = pd.concat([dfresult, dfEmbarked], axis=1)

    return dfresult


to_test = preprocessing(dftest_raw)  # Pandas DataFrame

model_loaded = models.load_model(r'datasets/titanic\tf_model_savedmodel')
answer = model_loaded.predict_classes(to_test)

test_answer_raw['Survived'] = answer

test_answer_raw.to_csv(r".\ans.csv", index=False)
