# -*- coding: utf-8 -*-
"""
İU DATACLUB - Yapay Sinir Aglari Sunumu
07.04.2021
Author: Dr. Zeki Ozen

Dataset kaynagi: UCI Pima-Indian Diabet Dataset @ Kaggle
Link: https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data

Faydalanilan Kod Kaynağı:  https://www.pluralsight.com/guides/machine-learning-neural-networks-scikit-learn

"""

# Gerekli kutuphaneleri calisma ortamimiza dahil edelim
import pandas as pd
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split


# veri setini calisma ortamimiza yukleyelim
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



# YSA Modelimizi kuralim
mlp = MLPClassifier(hidden_layer_sizes=(10,10,10), 
                    activation='relu', 
                    solver='adam', 
                    max_iter=50000)
# solver='adam'

"""
mlp = MLPClassifier(solver='sgd', 
                          alpha=1e-8, 
                          hidden_layer_sizes=(50, 5),
                          max_iter=100000, 
                          shuffle=True, 
                          learning_rate_init=0.001,
                          activation='relu', 
                          learning_rate='constant',
                          tol=1e-7,
                          random_state=0,
                          verbose=False)
"""




# modele egitim veri setini verelim ve egitmeye baslayalim
mlp.fit(X_train, y_train)

# modelin egitim ve test veri setlerindeki basarimini degerlendirelim
predict_train = mlp.predict(X_train)
predict_test = mlp.predict(X_test)


from sklearn.metrics import classification_report,confusion_matrix
#egitim veri setindeki basarim
print(confusion_matrix(y_train,predict_train))
print(classification_report(y_train,predict_train))


# test veri setindeki basarim
print(confusion_matrix(y_test,predict_test))
print(classification_report(y_test,predict_test))





# from sklearn.model_selection import GridSearchCV
"""
param_grid = [
        {
            'activation' : ['identity', 'logistic', 'tanh', 'relu'],
            'solver' : ['lbfgs', 'sgd', 'adam'],
            'max_iter':[50000],
            'hidden_layer_sizes': [
              (10,10,10), (25,5, 5), (78,3,14,2)
             ]
        }
       ]

clf = GridSearchCV(MLPClassifier(), param_grid, cv=3, scoring='accuracy')
clf.fit(X,y)


print('Best parameters set found on development set:')
print(clf.best_params_)
"""
