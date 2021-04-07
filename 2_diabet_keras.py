# -*- coding: utf-8 -*-
"""

İU Data Klubu - Yapay Sinir Aglari Sunumu
07.04.2021
Author: Dr. Zeki Ozen

Dataset kaynagi: UCI Pima-Indian Diabet Dataset @ Kaggle
Link: https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data

Faydalanilan Kod Kaynağı:  https://www.kaggle.com/atulnet/pima-diabetes-keras-implementation/data#Pima-Indians-Diabetes-Database

"""
# Gerekli kutuphaneleri calisma ortamimiza dahil edelim
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split





df = pd.read_csv("diabetes.csv", delimiter=',')
print(df.shape)
df.describe().transpose()

"""
Input Variables (X):

Number of times pregnant
Plasma glucose concentration a 2 hours in an oral glucose tolerance test
Diastolic blood pressure (mm Hg)
Triceps skin fold thickness (mm)
2-Hour serum insulin (mu U/ml)
Body mass index (weight in kg/(height in m)^2)
Diabetes pedigree function
Age (years)
"""




#bagimli ve bagimsiz degiskenelri ayarlayalim

# y = a1x1 + a2x3 +a3x3  +++ b
# y = aX + b
X = df.iloc[:,:-1]
y = df.iloc[:,-1:]



# Veriyi 70-30 oraninda egitim ve test veri seti olarak ikiye ayiralim
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=40)
print(X_train.shape)
print(X_test.shape)


# Veriyi 0-1 araliginda normalize edelim
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.fit_transform(X_test)



# Keras kutuphane ve modulleri
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.utils import to_categorical


# one hot encode outputs
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)



# YSA Modelimizi kuralim
# 100-10-5-2 mimarimiz
model = Sequential()
model.add(Dense(100, activation='relu', input_dim=8))
model.add(Dense(5, activation='relu'))
model.add(Dense(2, activation='softmax'))

# Modelimizin parametrelerini verelim
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

# modelimizi egitmeye baslayalim
model.fit(X_train, y_train,  epochs=100, batch_size=16 )




from sklearn.metrics import classification_report,confusion_matrix
#egitim veri setindeki basarim
predict_train = model.predict(X_train)
scores_train = model.evaluate(X_train, y_train, verbose=0)
print('Accuracy on training data: {}% \n Error on training data: {}'.format(scores_train[1], 1 - scores_train[1]))


# test veri setindeki basarim
predict_test = model.predict(X_test)
scores_test = model.evaluate(X_test, y_test, verbose=0)
print('Accuracy on test data: {}% \n Error on test data: {}'.format(scores_test[1], 1 - scores_test[1]))


