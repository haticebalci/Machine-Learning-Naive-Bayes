'''Python içine gerekli kütüphaneler import edilir.Diğer kütüphaneler ilgili işlem yapılmadan önce aşağıda import edilecektir. '''

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

'''Kullanacağımız veri seti Iris veri setidir.Python'ın pandas kütüphanesinin read_csv methodu ile veri setini import ediyoruz.'''

data=pd.read_csv('Iris.csv')
print(data)

'''Iris veri seti toplamda 5 kolondan oluşmaktadır.Kolonlardan biri bağımlı değişken diğerleri ise bağımsız değişkenlerdir.Bağımsız değişken kolonlarda verilen 
ölçüm özelliklerine species kolonu için sınıflandırma yapacağız.Öncesinde bağımsız değişkenlerdeki nitelikler için bir x matrisi,bağımlı değişken için ise bir y vektörü 
oluşturacağız.'''

X=data.iloc[:,1:-1]
Y=data.iloc[:,5:] 

'''Bağımlı ve bağımsız değişkenlerimizi belirledikten sonra Iris veri seti 4 bölüme ayrılır.Bu bölümlerden %67'lik kısım olan X_train ve Y_train eğitim için kullanılırken
%33'lük kısım olan XX_test ve Y_test ise makineye tahmin ettirilmeye çalışılacaktır.'''

from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X,Y,test_size=0.33,random_state=0)

'''Makine öğrenmesi sınıflandırma problemlerinde kullanılan algoritmalardan bir tanesi de Naive Bayes'tir.Bu algoritmanın temeli olasılık kuramına dayanır.Bir veriyi sınıflandırmak 
için olasılık kullanır.Bir verinin her sınıfa ait olabilme olasılığı hesaplanarak en yüksek olasılık değerine sahip sınıf o verinin sınıfı olarak tahmin edilir.Bağımsız değişkenler 
birbirinden bağımsız olarak değerlendirilir.Bu modelin avantajı ise basit,hızlı ve yüksek doğruluk seviyeleri olup dengesiz veri kümelerinde de çalışabilir.Naive Bayes genellikle 3 farklı
dağılım üzerinden çalışır.
 1)GaussianNB:Tahmin edilecek veri sürekli bir değerse(ondalıklı sayı)
 2)MultinominalNB:Tahmin edilecek veri nominal bir değerse(integer sayı)
 3)BernoulliNB:Tahmin edilecek veri ikili dağılıma sahipse(evet/hayır vs.)
 Bu çalışmada GaussianNB import edildi.'''

from sklearn.naive_bayes import GaussianNB
gnb = GaussianNB()
gnb.fit(X_train, Y_train.values.ravel())
result = gnb.predict(X_test)

'''Confusion matrix,sınıflandırma problemlerinde kullanılan bir performans ölçümüdür. Karışıklık matrisi, gerçek sınıfı ve tahmin edilen sınıfı içeren bir tablodur. 
Bu tablo, dört farklı değere sahip olabilir: true positive (TP), false positive (FP), true negative (TN) ve false negative (FN).TP, modelin doğru bir şekilde bir sınıfı
 belirlediği durumlarda oluşurken, FP modelin yanlış bir şekilde bir sınıfı belirlediği durumlarda oluşur.TN, modelin bir sınıfı doğru bir şekilde olmadığını belirlediği 
 durumlarda, FN ise modelin bir sınıfı yanlış bir şekilde olmadığını belirlediği durumlarda oluşur.Karmaşıklık matrisi, bu dört sonucu bir matris içinde gösterir.''' 

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(Y_test,result)
print(cm)
#Confusion Matrix:50 veri içinden 48 tanesi doğru tahmin edilmiştir.
#[[16  0  0] 
# [ 0 19  0]   
# [ 0  2 13]]

from sklearn.metrics import accuracy_score
accuracy = accuracy_score(Y_test, result)
print(accuracy)
#Başarı oranı:0.96