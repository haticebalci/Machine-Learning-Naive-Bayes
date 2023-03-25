# Machine-Learning-Naive-Bayes
# Naive Bayes
Bu kodlar bir sınıflandırma problemi olan Iris veri seti üzerinde Naive Bayes algoritması kullanarak sınıflandırma gerçekleştirir.

# Kodların açıklaması şu şekildedir:

Öncelikle pandas, numpy ve matplotlib.pyplot kütüphaneleri import edilir.
Iris veri seti pandas kütüphanesi ile okunur ve veriler X (bağımsız değişkenler) ve Y (bağımlı değişken) olarak ayrılır.
Veriler eğitim ve test verilerine bölünür. Burada, verilerin %67'si eğitim, %33'ü test için kullanılır.
Naive Bayes algoritması kullanarak eğitim verileri üzerinde bir model oluşturulur.
Oluşturulan model test verileri üzerinde kullanılarak tahminler yapılır.

# Sonuç:

Confusion matrix ve accuracy_score gibi metrikler kullanılarak modelin performansı değerlendirilir.
Kodların sonuçlarına göre, test verileri üzerinde oluşturulan model %96 doğruluk oranıyla sınıflandırma yapmaktadır ve confusion matrix'e göre 50 verinin 48 tanesi doğru tahmin edilmiştir.
