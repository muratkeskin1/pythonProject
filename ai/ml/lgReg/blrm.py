# -*- coding: utf-8 -*-
"""
Created on Thu Oct 13 23:09:20 2022

@author: SerkanSavas
"""

#Kullanacağımız kütüphaneleri tanımlayalım
import pandas as pd
import numpy as np
import random
from matplotlib import pyplot as plt
from matplotlib.pyplot import figure
import matplotlib.colors as clrs

class BLRM:
  # Model sınıfımıza ait kurucu methodu tanımlıyoruz
  def __init__(self, x, y):
    # Mode-li oluşturduğumuzda bazı parametreleri girdi (X) boyutuna göre varsayılan değerlerle oluşturmalıyız
    self.Y = y # Etiket değerleri
    self.X = x # Giriş verileri matrisi (Örnek sayısı X Her örnekte bulunan özniteli sayısı)
    self.b = np.ones((x.shape[0])) # Ör-nek sayımız kadar tüm değerleri 1 olan b vektörü oluşturalım
    self.w = np.zeros((x.shape[1])) # Öznite-lik sayımız kadar tüm değerleri 0 olan w vektörü oluşturalım
    self.hatalar=[] # Bu dizi içerisine her iterasyonda elde ettiğimiz hataları yükleyeceğiz
    self.dogruOranlari=[] # Bu dizi içerisine her iterasyonda elde ettiğimiz doğru yüzdesini yükleyeceğiz


  def lineer_fonksiyon(self,x):
    sonuc = np.dot(x, self.w)+self.b[:x.shape[0]]
    return sonuc

  def sigmoid(self,z):
    return 1/(1+np.exp(-1*z))  # Sigmoid fonksiyonunu oluşturalım

  def logit(self,p):
    return np.log(p/(1-p))    # Logit fonksiyonunu oluşturalım

  def hata_fonksiyonu(self, p):
    if((p==0).sum()>0):
      p +=0.00001

    if((p==1).sum()>0):
      p -=0.00001
    #BCE hata fonksiyonunu oluşturalım
    return np.nan_to_num(-1*( self.Y * np.log(p) +
                              (1 - self.Y) * np.log(1 - p)).mean()
                         )

  def optimizasyon_fonksiyonu(self, p):
    fark =  p - self.Y
    #BCE hata fonksiyonununa göre b değerlerini oluşturalım
    delta_b = np.mean(fark)
    #BCE hata fonksiyonununa göre w değerlerini oluşturalım
    delta_w = np.dot(self.X.T,fark) / self.Y.shape[0]
    return delta_w, delta_b

  def agirlik_guncelleme_fonksiyonu(self, dw,db,ogrenmeOrani=0.001):
    # Optimizasyon fonksiyonununa göre b değerlerini güncelleyelim
    self.b = self.b - ogrenmeOrani * db
    # Optimizasyon fonksiyonununa göre w değerlerini güncelleyelim
    self.w = self.w - ogrenmeOrani * dw

  def basari_hesaplama(self, tahmin, etiketler, esikDegeri=0.5):
    # Tahmin değerlerini eşik değerine göre etiketlere çevirelim
    tahminiEtiketler = np.asarray(
                           (tahmin >= esikDegeri),dtype="int"
                                )
    # Tah-min edilen etiketlerle gerçeklerini karşılaştırıp doğru oranını hesaplayalın
    basariOrani = np.asarray(
                   (tahminiEtiketler == etiketler),
                    dtype="int").sum()/len(etiketler)
    return basariOrani

  def modeli_egit(self, ogrenmeOrani, iterasyonSayisi):

    for i in range(iterasyonSayisi):
      lnr=self.lineer_fonksiyon(self.X)
      tahmin = self.sigmoid(lnr)
      hata = self.hata_fonksiyonu(tahmin)
      dw,db = self.optimizasyon_fonksiyonu(tahmin)
      self.agirlik_guncelleme_fonksiyonu(dw,db,ogrenmeOrani)
      self.hatalar.append(hata)
      bs=self.basari_hesaplama(tahmin, self.Y)
      self.dogruOranlari.append(bs)
      print("{:0}. İterasyon ------> Hatası : {:0.3f} Başarısı : {:0.3f}".format(i,hata,bs))

    print("\nTüm Eğitim Başarıyla Tammalandı Ortalama Hata : {:0.3f} Ortalama Başarı : {:0.3f}".format(np.mean(
                       self.hatalar),np.mean(self.dogruOranlari)))

  def tahmin_fonksiyonu(self, x, esikDegeri=0.5):
    lnr=self.lineer_fonksiyon(x)
    tahmin = self.sigmoid(lnr)
    siniflar = np.asarray((tahmin >= esikDegeri),
                           dtype="int")
    return siniflar


from sklearn.datasets import load_breast_cancer
from sklearn import metrics
# Hazır göğüs kanseri verilerini yükleyelim
data = load_breast_cancer()

# Verilerin 469 tanesini eğitim için ayıralım
x_egit = data.data[100:]
y_egit = data.target[100:]

# Verilerin kalan 100 tanesini test için ayıralım
x_test = data.data[0:100]
y_test = data.target[0:100]

# Modeli oluşturalım
model=BLRM(x=x_egit ,y=y_egit)

# Oluşturduğumuz modeli eğitelim
model.modeli_egit(ogrenmeOrani=0.1,iterasyonSayisi=500)

# Eğitilen model üzerinde test verilerimizi tahmit etmeye çalışalım
tahminet=model.tahmin_fonksiyonu(x_test)# Test verilerine göre tahminler
# Başarıyla tahmin ettiğimiz veri sayısı
basari=model.basari_hesaplama(tahminet, y_test)

# Hata Matrisi (Confusion Matrix) oluşturalım
cnf_matrix = metrics.confusion_matrix(y_test, tahminet)


# Modelin Her İterasyon İçin Eğitim Başarısı Grafiğini Oluşturalım
plt.plot(model.dogruOranlari, label='Eğitim Başarısı')
plt.legend( loc="lower right")
plt.title('Lojistik Regresyon Modeli Başarı Grafiği')
plt.xlabel('İterasyon Sayısı')
plt.ylabel('Başarı Yüzdesi')
plt.show()

# Modelin Her İterasyon İçin Eğitim Hata Değeri Grafiğini Oluşturalım
plt.plot(model.hatalar, label='Eğitim Hatası',color='red')
plt.legend( loc="upper right")
plt.title('Lojistik Regresyon Modeli Hata Grafiği')
plt.xlabel('İterasyon Sayısı')
plt.ylabel('Hata Değeri')
plt.show()

# Tahmin Ettiğimiz Test Verilerinin Hata Matrisi Grafiğini Oluşturalım
fig, ax = plt.subplots(figsize=(3, 3))
ax.imshow(cnf_matrix)
ax.grid(False)
ax.set_xlabel('Tahmin Edilenler', fontsize=14, color='black')
ax.set_ylabel('Gerçek Sınıflar', fontsize=14, color='black')
ax.xaxis.set(ticks=range(2))
ax.yaxis.set(ticks=range(2))
for i in range(2):
    for j in range(2):
        ax.text(j, i, cnf_matrix[i, j], ha='center', va='center',fontsize=18, color='white')
plt.show()


#HAZIR KÜTÜPHANELER
# Kullanacağımız kütüphaneyi ve modeli yükleyelim
from sklearn.linear_model import LogisticRegression

# Varsayılan ayarlarıyla modeli oluşturalım
hazırModel = LogisticRegression(solver='lbfgs', max_iter=3000)
# Hazır modeli eğitelim
hazırModel.fit(x_egit,y_egit)
# Hazır modele göre tahminde bulunalım
y_pred=hazırModel.predict(x_test)
# Hazır modelin test başarısını yazdıralım
hazırModelbasari=hazırModel.score(x_test,y_test)
print("Hazır Modelin Başarısı : {:0.3f}".format(hazırModelbasari))
