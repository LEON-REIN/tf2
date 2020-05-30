# @.@ coding:utf-8 ^-^
# @Author   : Leon Rein
# @Time     : 2020-05-16  ~  11:39 
# @File     : titanic.py
# @Software : PyCharm
# @Notice   : It's a WINDOWS version!


import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import models, layers
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = '2'  # Warnings or Errors ONLY  

'''
1. Load datasets-sets
'''

dftrain_raw = pd.read_csv(r'datasets/titanic\train.csv')
dftest_raw = pd.read_csv(r'datasets/titanic\test.csv')
test_answer_raw = pd.read_csv(r'datasets/titanic\gender_submission-0.76555.csv')
# test_answer_raw = pd.read_csv(r'titanic\answer-0.77033.csv')

# Data Visualization
#     -- Relationship between survival probability and age
# print(dftrain_raw.head(10))
# ax = dftrain_raw.query('Survived == 0')['Age'].plot(kind='density', figsize=(12, 8), fontsize=15)
# dftrain_raw.query('Survived == 1')['Age'].plot(kind='density', figsize=(12, 8), fontsize=15)
# ax.legend(['Survived==0', 'Survived==1'], fontsize=12)
# ax.set_ylabel('Density', fontsize=15)
# ax.set_xlabel('Age', fontsize=15)
# plt.show()

'''
2. Data Processing(The Most Important Part!)
* Survived: 0 代表死亡，1 代表存活【y标签】
* Pclass: 乘客所持票类，有三种值 (1,2,3) 【转换成 onehot 编码】
* Name: 乘客姓名 【舍去】
* Sex: 乘客性别 【转换成 onehot 编码】
* Age: 乘客年龄(有缺失) 【数值特征，添加“年龄是否缺失”作为辅助特征】
* SibSp: 乘客兄弟姐妹/配偶的个数(整数值) 【数值特征】
* Parch: 乘客父母/孩子的个数(整数值)【数值特征】
* Ticket: 票号(字符串)【舍去】
* Fare: 乘客所持票的价格(浮点数，0-500不等) 【数值特征】
* Cabin: 乘客所在船舱(有缺失) 【添加“所在船舱是否缺失”作为辅助特征】
* Embarked: 乘客登船港口: S、C、Q(有缺失)【转换成onehot编码，四维度 S,C,Q,nan】
'''


# A Function to Preprocess datasets sets
def preprocessing(dfdata):
    dfresult = pd.DataFrame()

    # # Pclass
    # dfPclass = pd.get_dummies(dfdata['Pclass'], prefix='Pclass_')  # One-hot
    # dfresult = pd.concat([dfresult, dfPclass], axis=1)  # Merge arrays

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


to_train = preprocessing(dftrain_raw)  # Pandas DataFrame
ans_train = dftrain_raw['Survived'].values  # Array

to_test = preprocessing(dftest_raw)  # Pandas DataFrame
ans_test = test_answer_raw['Survived'].values  # Array

'''
3. Keras Modeling (Using Sequential)
'''

# Destroys the current TF graph and session, and creates a new one
tf.keras.backend.clear_session()

model = models.Sequential()
model.add(layers.Dense(20, activation='relu', input_shape=(to_train.shape[1],)))
model.add(layers.Dense(10, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))
# model.summary()

model.compile(optimizer='adam',
              loss='binary_crossentropy',  # It has its advantages for Bi-classification problems
              metrics=['AUC'])  # One kind of evaluation function -- AUC

'''
4. Train the Model
'''

history_fit = model.fit(to_train, ans_train,
                        batch_size=60,  # 60
                        epochs=70,  # 70
                        validation_split=0.2  # Split part of training datasets for validation
                        )

'''
5. Evaluate the Model
'''


# Plot the trend of loss or metrics
def plot_metric(history, indicator):
    train_metrics = history.history[indicator]
    val_metrics = history.history['val_' + indicator]
    epochs = range(1, len(train_metrics) + 1)  # from 1 to epoch+1
    plt.plot(epochs, train_metrics, 'bo--')
    plt.plot(epochs, val_metrics, 'ro-')
    plt.title('Training and validation ' + indicator)
    plt.xlabel("Epochs")
    plt.ylabel(indicator)
    plt.legend(["train_" + indicator, 'val_' + indicator])
    plt.show()


plot_metric(history_fit, "loss")
plot_metric(history_fit, "AUC")
evaluation = model.evaluate(x=to_test, y=ans_test)
print("Loss: {loss}  Accuracy: {acc}".format(loss=evaluation[0], acc=evaluation[1]))

'''
6. Use the Model (scored 0.76555 in kaggle)
'''

# print(model.predict(to_test[0:10]))  # the probability of each person
# print(model.predict_classes(to_test[0:10]))  # the class they belongs to
answer = model.predict_classes(to_test)
test_answer_raw['Survived'] = answer
# test_answer_raw.to_csv(r".\ans.csv", index=False)

# '''
# 7. Save the Model
# '''
#
# '''1. Saves in a Keras-way'''
# # 1.1
# # Saved as model.h5
# model.save(r'titanic\keras_model.h5')  # the older Keras H5 format
# del model  # release the current model just for a test
#
# # Restore model.h5
# model = models.load_model(r'titanic\keras_model.h5')
# evaluation = model.evaluate(x=to_test, y=ans_test)
# print("Loss1: {loss}  Accuracy1: {acc}".format(loss=evaluation[0], acc=evaluation[1]))
#
# # 1.2
# # Save the model architecture
# json_str = model.to_json()
#
# # Save model weights
# model.save_weights(r'titanic\keras_model_weight.h5')
#
# # Restore model architecture
# model_json = models.model_from_json(json_str)
# model_json.compile(optimizer='adam',
#                    loss='binary_crossentropy',
#                    metrics=['AUC']
#                    )
#
# # Restore model weights
# model_json.load_weights(r'titanic\keras_model_weight.h5')
# evaluation = model.evaluate(x=to_test, y=ans_test)
# print("Loss2: {loss}  Accuracy2: {acc}".format(loss=evaluation[0], acc=evaluation[1]))
#
#
# '''2. (Recommend!)Save in a TF2-way'''
# # # Save model weights only
# # model.save_weights(r'titanic\tf_model_weights.ckpt', save_format="tf")
#
# # Save the whole model
# model.save(r'titanic\tf_model_savedmodel', save_format="tf")
#
# # Restore the model
# model_loaded = models.load_model(r'titanic\tf_model_savedmodel')
# evaluation = model.evaluate(x=to_test, y=ans_test)
# print("Loss3: {loss}  Accuracy3: {acc}".format(loss=evaluation[0], acc=evaluation[1]))


