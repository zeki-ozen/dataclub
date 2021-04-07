# -*- coding: utf-8 -*-
"""
İU Data Klubu - Yapay Sinir Aglari Sunumu
07.04.2021
Author: Dr. Zeki Ozen

Dataset kaynagi: UCI Heart Disease-Cleveland @ Kaggle
Link: https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data

Faydalanilan Notebook:  https://towardsdatascience.com/heart-disease-uci-diagnosis-prediction-b1943ee835a7

"""



import pandas as pd 
import numpy as np 

# verisetini yukluyoruz
data = pd.read_csv('heart.csv',  sep=',') 

data.head() 
data.describe() 
data.head()
data.info()

"""
•    age: yas
•    sex: cinsiyet (1 = erkek, 0 = kadin)
•    cp: Göğüs ağrısı tipi
•    trestbps: Dinlenme durumunda kan basıncı (tansiyon) (mm Hg hastaneye kabulde)
•    chol: kolesterol  (mg/dl)
•    fbs: Tokluk şekeri düzeyi (> 120 mg/dl, 1 = true; 0 = false)
•    restecg: Dinlenme durumunda Elektrokardiyografı düzeyi
•    thalach: Maksimum kalp atış ritim tipi
•    exang: Egzersiz ile oluşan göğüs ağrısı (1 = yes; 0 = no)
•    oldpeak: Dinlenme durumunda ST değeri ('ST' relates to positions on the ECG plot. See more here)
•    slope: Pik egzersiz durumunda ST segmentinin eğimi
•    ca: Büyük damarların sayısı (0-3)
•    thal: thalassemia tipi
•    target: Kalp hastalığı (0 = no, 1 = yes)



cp: chest pain type
-- Value 0: asymptomatic
-- Value 1: atypical angina
-- Value 2: non-anginal pain
-- Value 3: typical angina
 
restecg: resting electrocardiographic results
-- Value 0: showing probable or definite left ventricular hypertrophy by Estes' criteria
-- Value 1: normal
-- Value 2: having ST-T wave abnormality (T wave inversions and/or ST elevation or depression of > 0.05 mV)
 
slope: the slope of the peak exercise ST segment
0: downsloping; 1: flat; 2: upsloping
 
thal
Results of the blood flow observed via the radioactive dye.
 
Value 0: NULL (dropped from the dataset previously)
Value 1: fixed defect (no blood flow in some part of the heart)
Value 2: normal blood flow
Value 3: reversible defect (a blood flow is observed but it is not normal)

"""




# Bagimli ve bagimsiz degiskenlerimizi ayiralim
X = data.iloc[:,:-1]
y = data.iloc[:,-1]



# hedef niteligik sinif sayilarina bakiliyor
print(y.value_counts())
print("Target Class Size:", y.value_counts("0"))



# Egitim ve test veri setlerimizi %70 egitim, %30 test olacak sekilde ayiralim
# Egitim ve test verisetinde sinif dagilimini korumasini stratify parametresi ile sagliyoruz
from sklearn.model_selection import train_test_split
X_train,X_test,y_train, y_test = train_test_split(X, y, test_size = 0.3,  
                                                 # stratify=y, 
                                                  random_state = 42) 


print(y_train.value_counts("0"))
print(y_test.value_counts("0"))


#Bagimli nitelikleri Standart normalizasyon yontemiyle normalize ediyoruz
#from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
#numerik degiskenler 
numerik_nit = ['age', 'trestbps', 'chol','thalach','oldpeak']
scaler=StandardScaler()
scaler.fit(X_train[numerik_nit])
X_train[numerik_nit] = scaler.transform(X_train[numerik_nit])
X_test[numerik_nit] = scaler.transform(X_test[numerik_nit])

#bagimli kategorik degiskenler one-hot-encoding yontemi ile kategorik formata donusturuluyor
X_train = pd.get_dummies( X_train, columns= ['sex', 'cp',  'fbs',  'restecg', 'exang', 'slope', 'ca', 'thal'], drop_first=True)
X_test = pd.get_dummies( X_test, columns= ['sex', 'cp',  'fbs',  'restecg', 'exang', 'slope', 'ca', 'thal'], drop_first=True)



import tensorflow as tf 
from tensorflow.keras.models import Sequential 
from tensorflow.keras.layers import Dense 
from sklearn.metrics import confusion_matrix

# Cok katmanli modelimizin mimarisini yapilandiriyoruz
# 22-15-10-1 katmandaki noron sayilarimiz
classifier = Sequential()

# girdi katmanimiz yapilandiriliyor
classifier.add(Dense(activation = "relu",
                     input_dim = X_train.shape[1],
                     units = 15, 
                     kernel_initializer = "uniform")) 

# gizli katmanimiz yapilandiriliyor
classifier.add(Dense(activation = "relu", 
                     units = 10,
                     kernel_initializer = "uniform"))  # kernel_initializer = "normal"

# cikti katmanimiz yapilandiriliyor
classifier.add(Dense(activation = "sigmoid", 
                     units = 1, # burasinin 1 olmasi muhim
                     kernel_initializer = "uniform")) 

# model olusturuluyor
classifier.compile(optimizer = 'adam' , 
                   loss = 'binary_crossentropy', 
                   metrics = ['accuracy'] ) 


#############################
# modelimizi egitelim       #
#############################

output = classifier.fit(X_train, y_train, 
               batch_size = 8,
               epochs = 100,
               validation_split=0.2,
               ) 


print(classifier.summary())



#############################
# Ve modelimizi test edelim #
#############################


#Egitim veri setinin basarimi
pred_train = classifier.predict(X_train)
scores_train_pred = classifier.evaluate(X_train, y_train, verbose=0)
print('Accuracy on training data: {}% \n Error on training data: {}'.format(scores_train_pred[1], 1 - scores_train_pred[1]))

#Test veri setinin basarimi
y_pred = classifier.predict(X_test)
print(y_pred)
scores_y_pred = classifier.evaluate(X_test, y_test, verbose=0)
print('Accuracy on test data: {}% \n Error on test data: {}'.format(scores_y_pred[1], 1 - scores_y_pred[1]))




#Orneklerin bulunma olasiliklarini >  0.5 ise 1, degilse 0 olarak kodluyoruz
#y_pred = y_pred.round().astype(int)
y_pred = (y_pred >= 0.5).astype(int)
print(y_pred)

conf_matrix = confusion_matrix(y_test, y_pred) 
print(conf_matrix)

from sklearn.metrics import classification_report, accuracy_score
print(classification_report(y_test, y_pred))
print(accuracy_score(y_test, y_pred))


"""
scores = classifier.evaluate(X_test, y_test)
print("\n%s: %.2f%%" % (classifier.metrics_names[1], scores[1]*100))
"""

# epoch iterasyonuna gore egitim ve test veri setlerinin dogruluk grafigi
import matplotlib.pyplot as plt
plt.plot(output.history['accuracy'])
plt.plot(output.history['val_accuracy'])
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Training', 'Validation'], loc='lower right')
#plt.savefig('Accuracy.png',dpi=100)
plt.show()

