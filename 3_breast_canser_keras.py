# -*- coding: utf-8 -*-
"""
İTF Yapay Zeka Klubu - Yapay Sinir Aglari Sunumu
28.11.2020
Author: Dr. Zeki Ozen

Dataset kaynagi: UCI Breast Cancer Dataset @ Scikit-learn

Faydalanilan Kod Kaynağı: https://medium.com/@tayyipgoren/classifying-breast-cancer-98-18-accurate-with-keras-106cf846cac0

"""

#gerekli kutuphaneleri import edelim
import numpy as np
import pandas as pd
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler



#veri setini yukleyelim
data_set = datasets.load_breast_cancer()

# bagimli ve bagimsiz degiskenleri ayarlayalim
X=data_set.data
y=data_set.target


# veri setini 0-1 araliginda olcekleyelim
scaler = MinMaxScaler()
X = scaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)


# ysa icin kullanacagimiz tenserflow ve keras kutuphanelerini 
# calisma ortamimiza dahil edelim
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout


# ysa modelimizi kuralim
# modelimiz uc katmanli ve katmanalrinda sirasiyla
# 30-10-1 adet noron bulunan mimarimiz 
model = Sequential()
model.add(Dense(30, activation='sigmoid', input_shape=(30,)))
model.add(Dense(10, activation='sigmoid'))
# dropout fonksiyonu
model.add(Dropout(0.25))
model.add(Dense(1, activation='sigmoid'))

# modelimizi derleyelim
model.compile(optimizer='adam', loss=tf.keras.losses.mean_squared_logarithmic_error)

# modelimizi egitelim
model.fit(X_train, y_train, batch_size=30, epochs=200, verbose=1)


# modelimizin dogrulugunu test veri seti ile sinayalim
#Test veri setinin basarimi
test_pred = model.predict(X_test)
# modelimizin ciktisi 0 ve 1 arasinda ondalikli sayilardir
# bunun anlami bir ornegin bir sinifa ait olma olasiligidir
print(test_pred)


#Orneklerin bir sinifa ait olma olasiliklarini >  0.5 ise 1, degilse 0 olarak kodluyoruz
test_pred = test_pred.round().astype(int)
print(test_pred)

# basit tablo uzerinden performansa bakalim
from sklearn.metrics import confusion_matrix 
conf_matrix = confusion_matrix(y_test, test_pred) 
print(conf_matrix)

# daha ayrintili degerlendirme icin confusion matrixi olusturalim
from sklearn.metrics import classification_report, accuracy_score
print(classification_report(y_test, test_pred))






