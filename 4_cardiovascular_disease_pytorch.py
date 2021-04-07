# -*- coding: utf-8 -*-
"""
İU Data Klubu - Yapay Sinir Aglari Sunumu
07.04.2021
Author: Dr. Zeki Ozen

Dataset kaynagi: Cardiovascular Disease dataset @ Kaggle
Link: https://www.kaggle.com/sulianova/cardiovascular-disease-dataset


    PyTorch yeni baslayanlar icin guzel bir anlatim ve uygulama linki:
    https://www.freecodecamp.org/news/how-to-build-a-neural-network-with-pytorch/
    
Faydalanilan PyTorch Kod Kaynaklari:  
    https://curiousily.com/posts/build-your-first-neural-network-with-pytorch/
    https://www.kaggle.com/kanncaa1/pytorch-tutorial-for-deep-learning-lovers
    


"""


import numpy as np
import pandas as pd


data = pd.read_csv('cardio_train.csv', sep=';')
data.head()


#id sutunu kaldiriliyor
data.drop('id',axis=1,inplace=True)

#yas sutunu
#data.age = np.round(data.age/365.25,decimals=1)

#gender sutununda 2 ile temsil edilen veriyi 0 yapiyoruz
data.gender = data.gender.replace(2,0)



data.gender.value_counts()


# vucut kutle indeksini hesaplayip veri setimize ekliyoruz
data['bmi'] = data.weight / (data.height / 100) ** 2
# height ve weight kolonlarina artik gere kalmadigi icin kaldiriyoruz
data.drop('height', axis=1,inplace=True)
data.drop('weight', axis=1,inplace=True)

#bagimli ve bagimsiz degiskenelr ayarlaniyor
X = data[data.columns.difference(['cardio'])]
y = pd.DataFrame(data['cardio'])



# Egitim ve test veri setlerini 80-20 oraninda ayarliyoruz
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(X_train.info())
print(y_train.info())




#bagimli numerik degiskenler normalize ediliyor
#from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
#numerik degiskenler 
to_be_scaled_feat = ['age', 'ap_hi', 'ap_lo','bmi']
scaler=StandardScaler()
scaler.fit(X_train[to_be_scaled_feat])
X_train[to_be_scaled_feat] = scaler.transform(X_train[to_be_scaled_feat])
X_test[to_be_scaled_feat] = scaler.transform(X_test[to_be_scaled_feat])



#bagimli kategorik degiskenler one-hot-encoding yontemi ile kategorik formata donusturuluyor
X_train = pd.get_dummies( X_train, columns= ["gender", 'cholesterol',  'gluc',  'smoke', 'alco', 'active'], drop_first=True)
X_test = pd.get_dummies( X_test, columns= ["gender", 'cholesterol',  'gluc',  'smoke', 'alco', 'active'], drop_first=True)



# pytorch kutuphanesini calisma alanimiza dahil ediyoruz
import torch
from torch import nn, optim
import torch.nn.functional as F



# egitim ve test veri setlerimizi
# bagimli ve bagimsiz degiskenlerini torch yapisina donusturuyoruz
X_train = torch.from_numpy(X_train.to_numpy()).float()
y_train = torch.squeeze(torch.from_numpy(y_train.to_numpy()).float())
X_test = torch.from_numpy(X_test.to_numpy()).float()
y_test = torch.squeeze(torch.from_numpy(y_test.to_numpy()).float())
print(X_train.shape, y_train.shape)
print(X_test.shape, y_test.shape)



# modelin dogrulugunu hesaplayacak fonksiyon
def calculate_accuracy(y_true, y_pred):
  predicted = y_pred.ge(.5).view(-1)
  return (y_true == predicted).sum().float() / len(y_true)

# sonuclari yuvarlayacak fonksiyon
def round_tensor(t, decimal_places=3):
  return round(t.item(), decimal_places)


# Yapay sinir mimarisi bu sinif ile olusturuluyor
class Net(nn.Module):
    
  def __init__(self, n_features):
    super(Net, self).__init__()
    self.fc1 = nn.Linear(n_features, 5)
    self.fc2 = nn.Linear(5, 3)
    self.fc3 = nn.Linear(3, 1)
    
  def forward(self, x):
    x = F.relu(self.fc1(x))
    x = F.relu(self.fc2(x))
    return torch.sigmoid(self.fc3(x))



# Yapilandirdigimiz modelin girdi katmani boyutu, hata fonk, optimizasyon fonk. gibi 
# parametrelerini veriyoruz
net = Net(X_train.shape[1])

# hata hesaplama metriklerimiz
#criterion = nn.BCELoss()
criterion = nn.MSELoss()

# ogrenme fonksiyonumuz ve ogrenme orani
optimizer = optim.Adam(net.parameters(), lr=0.001)

# bu degiskenleri plot cizdirmek icin kullaniyoruz. cok gerekli degiller
count = 0
loss_list = []
iteration_list = []
accuracy_list = []

for epoch in range(1000):

    
    #egitim verisiyle model kuruluyor ve cikti tahmin ediliyor
    y_pred = net(X_train)
    y_pred = torch.squeeze(y_pred)
    #egitim verisindeki tahmin hatasi hesaplaniyor
    train_loss = criterion(y_pred, y_train)
    

    # her 100 iterasyonda bir hata hesapliyoruz      
    if epoch % 100 == 0:
      train_acc = calculate_accuracy(y_train, y_pred)
      y_test_pred = net(X_test)
      y_test_pred = torch.squeeze(y_test_pred)
      test_loss = criterion(y_test_pred, y_test)
      test_acc = calculate_accuracy(y_test, y_test_pred)

      loss_list.append(test_loss.data)
      iteration_list.append(count)
      accuracy_list.append(train_acc)
    
      print(
f'''epoch {epoch}
Train set - loss: {round_tensor(train_loss)}, accuracy: {round_tensor(train_acc)}
Test  set - loss: {round_tensor(test_loss)}, accuracy: {round_tensor(test_acc)}
''')

    # Clear gradients
    optimizer.zero_grad()
    
    # Calculating gradients
    train_loss.backward()
    # Update parameters
    optimizer.step()
    count += 1
    
    

# performans degerelndirme icin gerekli standart kutuphaneler
from sklearn.metrics import classification_report, accuracy_score
y_pred = net(X_test)
y_pred = y_pred.ge(.5).view(-1).cpu()
y_test = y_test.cpu()
print(classification_report(y_test, y_pred))
print(accuracy_score(y_test, y_pred))



import matplotlib.pyplot as plt
# visualization loss 
plt.plot(iteration_list,loss_list)
plt.xlabel("Number of iteration")
plt.ylabel("Loss")
plt.title("ANN: Loss vs Number of iteration")
plt.show()

# visualization accuracy 
plt.plot(iteration_list,accuracy_list,color = "red")
plt.xlabel("Number of iteration")
plt.ylabel("Accuracy")
plt.title("ANN: Accuracy vs Number of iteration")
plt.show()
            


