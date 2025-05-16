# Meme Kanseri Tespiti Projesi

Bu proje, makine öğrenmesi teknikleri kullanarak meme kanseri tespiti yapmayı amaçlamaktadır. Wisconsin Breast Cancer veri seti üzerinde çeşitli algoritmalar kullanılarak kötü huylu (malignant) ve iyi huylu (benign) tümörleri yüksek doğrulukla sınıflandırmayı hedefler.

## Proje İçeriği

Proje aşağıdaki bileşenlerden oluşmaktadır:

1. **Veri Analizi ve Görselleştirme**: `veri_analizi.py` - Veri setinin detaylı analizi ve görselleştirilmesi.
2. **Model Karşılaştırma**: `model_karsilastirma.py` - Farklı makine öğrenmesi modellerinin performans karşılaştırması.
3. **Hiperparametre Optimizasyonu**: `hiper_optimizasyon.py` - Modellerin hiperparametre optimizasyonu.
4. **Ana Uygulama**: `main.py` - Temel veri yükleme ve model eğitimi örneği.

## Kullanılan Modeller

- Lojistik Regresyon
- Destek Vektör Makineleri (SVM)
- Karar Ağaçları
- Rastgele Orman
- Gradient Boosting
- K-En Yakın Komşu (KNN)
- Yapay Sinir Ağları

## Proje Yapısı

```
├── main.py                          # Ana uygulama
├── veri_analizi.py                  # Veri analizi ve görselleştirme
├── model_karsilastirma.py           # Model karşılaştırma
├── hiper_optimizasyon.py            # Hiperparametre optimizasyonu
├── README.md                        # Proje açıklaması
├── roadmap.md                       # Proje geliştirme yol haritası
├── output.txt                       # Ana uygulama çıktıları
├── goruntuler/                      # Görselleştirmeler
│   ├── sinif_dagilimi.png
│   ├── korelasyon_matrisi.png
│   ├── pca_analizi.png
│   ├── oznitelik_onem.png
│   ├── model_karsilastirma/
│   └── hiper_optimizasyon/
└── modeller/                        # Eğitilmiş modeller
    ├── lojistik_regresyon.pkl
    ├── svm_linear.pkl
    └── optimized/
```

## Kurulum ve Kullanım

1. Gerekli kütüphaneleri yükleyin:
```bash
pip install pandas numpy seaborn matplotlib scikit-learn joblib
```

2. Veri analizi ve görselleştirme:
```bash
python veri_analizi.py
```

3. Model karşılaştırması:
```bash
python model_karsilastirma.py
```

4. Hiperparametre optimizasyonu:
```bash
python hiper_optimizasyon.py
```

## Veri Seti Hakkında

Bu projede scikit-learn kütüphanesinde bulunan Wisconsin Breast Cancer veri seti kullanılmıştır. Veri seti 569 örnek ve 30 öznitelik içermektedir.

- **Veri Kaynağı**: scikit-learn datasets
- **Öznitelikler**: Hücre çekirdeği özelliklerini içeren 30 farklı ölçüm
- **Hedef Değişken**: Tümörün tanısı (0: Kötü Huylu, 1: İyi Huylu)

## Geliştirme Planı

Projenin gelecekteki geliştirilme planları için `roadmap.md` dosyasına bakınız.

## Yazarlar

- [İsminiz] - Başlangıç Çalışması

## Lisans

Bu proje MIT lisansı altında lisanslanmıştır - ayrıntılar için LICENSE dosyasına bakın. 