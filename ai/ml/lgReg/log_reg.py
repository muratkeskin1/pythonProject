# -*- coding: utf-8 -*-
"""
Created on Wed Oct 12 23:08:56 2022

@author: SerkanSavas
"""

import pandas as pd 
import numpy as np 
import random 
from matplotlib import pyplot as plt 
from matplotlib.pyplot import figure 
import matplotlib.colors as clrs 

#%%
# Rastgele 'random' 100 adet veri oluşturalım 
np.random.seed(0) # Verileri rastgele üretirken sabitledik
# Numpy kütüphanesini kullanarak 1.5 ortalama ve 2.5 standart sapmaya sahip rastgele 100 veri oluşturuyoruz 
kilo = 100 * np.random.sample(100)+ 15 
# 100 adet kilo verisinden boy verisini üretmekte kullanacağımız hata terimini oluşturalım 
eps = 5 * np.random.randn(100) 
boy = 140 + 0.4 * kilo + eps # Boy verilerimizi oluşturalım 
# Pandas kütüphanesinin dataframe fonksiyonu ile boy ve kilo verilerimize ait tablo oluşturuyoruz 
df = pd.DataFrame( {'Boy (cm)': boy, 'Kilo (kg)': kilo} ) 
print("Not : Verilerin tamamı çok uzun olduğundan sadece bir kısmını ekranda görüyoruz.") 
 
df.head() # İlk 5 veriyi gösterelim

# Kilo ve Boy Verileri Dağılım Grafiği 
plt.figure(figsize=(12, 6)) 
plt.scatter(kilo, boy, color = 'red') 
# Kilo ve Boy verilerini çizdirelim 
plt.title('Kilo Boy Oranı Grafiği') 
plt.xlabel('Kilo') 
plt.ylabel('Boy') 
plt.show()

#%%
#Lineer Regresyon
from sklearn.linear_model import LinearRegression

X = df.iloc[:, :-1].values
y = df.iloc[:, 1].values

reg = LinearRegression()
reg.fit(X, y)
y_pred = reg.predict(X)

plt.figure(figsize=(12, 6)) 
plt.scatter(kilo, boy, color = 'red') 
plt.scatter(y_pred, boy, color = 'blue')
plt.title('Kilo Boy Oranı Grafiği') 
plt.xlabel('Kilo') 
plt.ylabel('Boy') 
plt.show()

#%%
#Manuel hesap
kilo_Top = np.sum(kilo)       # Kilo verilerinin toplamı
boy_Top = np.sum(boy)         # Boy verilerinin toplamı
# Kilo ve boy verilerinin çarpımlarının toplamı
kiloBoy_Top= np.sum(kilo*boy)

# Kilo verilerinin karelerinin toplamı
kiloKare_Top = np.sum(np.power(kilo,2))     
n=100 # Toplam veri sayısı

# W ağırlıklarını tahmin edelim
w_tahmin=(n*kiloBoy_Top - kilo_Top*boy_Top) / (n*kiloKare_Top - kilo_Top**2)

# b değerlerini tahim edelim
b_tahmin= boy_Top/n - w_tahmin*(kilo_Top/n)

#Tahmin ettiğimiz ağırlık değerlerini kullanarak Lineer Regresyon Modelimize göre boy değerlerini kilolardan tahmin etmeye çalışalım 
boy_tahmin = w_tahmin * kilo + b_tahmin 

print("W değeri {:0.4f}  b değeri {:0.4f} ".format(w_tahmin,b_tahmin))


# Kilo ve Boy Verileri Dağılım Grafiği Lineer Regresyon Çizgisiyle Birlikte
plt.figure(figsize=(12, 6))
plt.plot(kilo, boy, 'ro')  # Kilo ve Boy verilerini çizdirelim

# Tahminlere göre regression çizgisini oluşturalım
plt.plot(kilo, boy_tahmin) 
plt.title('Kilo Boy Oranı Grafiği')
plt.xlabel('Kilo')
plt.ylabel('Boy')
plt.legend(labels=['Veri Noktaları', 'Lineer Regresyon Çizgisi'])
plt.show()


#%%
#Hata Hesaplama
n=100                             # Toplam veri sayısı
hata=boy - boy_tahmin             # Hata değerimiz (Gerçek boy değerleriyle tahmin ettiğimiz boy değerleri arasındaki fark)
SE=np.sqrt((np.sum(hata**2)/n))   # Standart hata (Standart Error – (SE))
MAE = np.sum(np.abs(hata))/n      # Hataların mutlak değerlerinin ortalaması (Mean Absolute Error – (MAE)) 
MSE = np.sum(hata**2)/n           # Hataların karelerinin ortalaması (Mean Squared Error – (MSE)) 

print("SE değeri {:0.4f}  MAE değeri {:0.4f}  MSE değeri {:0.4f} ".format(SE,MAE,MSE))

#%%
#Endeks
VKI=kilo/(boy/100)**2         # Vücut Kütle Endeksini hesaplayalım
# VKİ oranı 25'ten büyük olanlara True küçük olanlara False diyelim  
obezite_Etiketi=(VKI>25) 
     
# Kilo ve Obez Olma Olasılığı Verileri Dağılım Grafiği 
plt.figure(figsize=(12, 6))
cmap = clrs.ListedColormap(['green', 'red'])
# Kilo ve Obezite verilerini çizdirelim
plt.scatter(x = kilo, y= obezite_Etiketi, 
            c=(obezite_Etiketi == True).astype(float), 
            marker='d', cmap=cmap)  
plt.title('Kilo - Obez Olma Olasılığı Grafiği')
plt.xlabel('Kilo')
plt.ylabel('Obez Olma Olasılığı')
plt.show()

#%%
# Kilo ve Obez Olma Olasılığı Verileri Dağılım Grafiği Lineer Regresyon Çizgisiyle Birlikte
# Boy tahmini verilerimiz 100 ile 195 arasındaydı. Ancak Obez Olma Durumu verilerimiz 0 ile 1 arasında
bt=boy_tahmin-b_tahmin.min()   
# Lineer Regresyon Çizgisini verilere göre çizmek için 0 ile 1 arasında normalize ediyoruz                     
boy_tahmin_normalize=bt/bt.max()   
# Daha anlamlı bir görüntü elde etmek için çizgi boyutu ve açısını biraz değiştirdik                 
boy_tahmin_normalize=(boy_tahmin_normalize-0.3)*2   

plt.figure(figsize=(12, 6))
cmap = clrs.ListedColormap(['green', 'red'])

# Kilo ve Obezite verilerini çizdirelim
plt.scatter(x = kilo, y= obezite_Etiketi, 
            c=(obezite_Etiketi == True).astype(float), 
            marker='d', cmap=cmap) 

# Tahminlere göre regression çizgisini oluşturalım
plt.plot(kilo, boy_tahmin_normalize)          
plt.title('Kilo - Obez Olma Olasılığı Grafiği')
plt.xlabel('Kilo')
plt.ylabel('Obez Olma Olasılığı')
plt.show()

#%%
#Sigmoid
X=np.linspace(-7,7,100)# -7 ile 7 arasında 100 adet veri oluşturalım 
# Numpy kütüphanesini Sigmoid fonksiyonunu oluşturalım 
def sigmoid(x):
  return 1/(1+np.exp(-1*x))     

# Tüm X değerlerimizi sigmoid fonksiyonundan geçirerek Y değerlerini loluşturalım
Y = sigmoid(X)                  

#%%
### AYNI İŞLEMİ TORCH KÜTÜPHANESİNDEN SİGMOİD FONKSİYONUNU ÇAĞIRARAK DA KULLANABİLİRİZ
import torch 
import torch.nn as nn

# Torch sigmoid tensor türünden veri kabul ettiği için X değerlerini dönüştürelim
a=torch.tensor(X) 
# Torch sigmoid fonksiyonuna göre değerlerini hesaplayalım
Y_torch = nn.Sigmoid()(a) 

# İki değeri karşılaştırdığımızda eşit olduklarını görüyoruz
sonuc= np.array_equal(np.round(Y,4),np.round(Y_torch,4))  
if(sonuc==True):
  print("Sonuçlar Eşit")
else:
  print("Sonuçlar Eşit Değil")

#%%
# Sigmoid Fonksiyonuna ait grafik
plt.figure(figsize=(12, 6))
# X'e göre Y verilerini çizdirelim
plt.plot(X, Y,label = r"$\sigma(x) = \frac{1}{1+e^{-x}}$") 
plt.title('Sigmoid Fonksiyonu')
plt.xlabel('X')
plt.ylabel('Y')
plt.grid(True)
plt.axhline(y=0.5,color='grey',linestyle='--')
plt.axvline(color="grey")
plt.yticks([0.0,0.5,1.0])
plt.legend()
plt.show()

#%%
# X değerlerimiz -7 ile 7 arasındaydı ancak kilo değerlerimiz 0 ile 115 arasında. X'i yeniden oluşturalım
X=np.linspace(0,110,100)                                    

aa=['Sarı','Kırmızı']
# Sigmoid Fonksiyonuna ait grafik
plt.figure(figsize=(12, 6))
# X'e göre Y verilerini çizdirelim
plt.plot(X, Y,label = r"$\sigma(x) = \frac{1}{1+e^{-x}}$")  
plt.grid(True)
plt.axhline(y=0.5,color='grey',linestyle='--')
plt.axvline(color="grey")
plt.yticks([0.0,0.5,1.0])
plt.legend()

cmap = clrs.ListedColormap(['green', 'red'])
# Kilo ve Obezite verilerini çizdirelim
plt.scatter(x = kilo, y= obezite_Etiketi, 
            c=(obezite_Etiketi == True).astype(float), 
            marker='d', cmap=cmap)  
plt.title('Kilo - Obez Olma Olasılığı ve Sigmoid Grafiği')
plt.xlabel('Kilo')
plt.ylabel('Obez Olma Olasılığı')
plt.show()

#%%
#BCE Fonksiyonu
X=np.linspace(0,1,100)# X değerlerimizi 0 ile 1 arasında oluşturalım

# BCE fonksiyonunda Y =1 için ve Y = 0 için olmak üzere iki bölüm vardı. Bunların İkisini ayrı ayrı oluşturalım
# X değerlerini Y=1 için logaritmasını larak -1 ile çarpıp oluşturalım
Y1=-1* np.log(X)    
# X değerlerini Y=0 için logaritmasını larak -1 ile çarpıp oluşturalım
Y0=-1* np.log(1-X)              

# Sigmoid Fonksiyonuna ait grafik
plt.figure(figsize=(12, 6))
# X'e göre Y1 verilerini çizdirelim
plt.plot(X, Y1,label = "- y log(p)") 
# X'e göre Y0 verilerini çizdirelim
plt.plot(X, Y0,label = "- (y-1) log(1-p)")   
plt.grid(True)
plt.legend(loc="upper center")
plt.title('BCE Fonksiyonunun Sigmoid Değerlerine Göre Grafiği')
plt.xlabel('Sigmoid Tahmin Değerleri')
plt.ylabel('BCE Hata Değerleri')
plt.show()

