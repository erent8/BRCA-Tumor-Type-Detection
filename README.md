#  Meme Kanseri Tespiti Projesi - Dokümantasyon

##  İçindekiler
- [Proje Hakkında](#proje-hakkında)
- [Proje Yapısı](#proje-yapısı)
- [Teknolojiler ve Kütüphaneler](#teknolojiler-ve-kütüphaneler)
- [Kullanılan Makine Öğrenmesi Algoritmaları](#kullanılan-makine-öğrenmesi-algoritmaları)
- [Özellikler](#özellikler)
- [Kurulum ve Çalıştırma](#kurulum-ve-çalıştırma)
- [Modül Detayları](#modül-detayları)
- [Web Uygulaması](#web-uygulaması)
- [Veri Seti Hakkında](#veri-seti-hakkında)
- [Model Performansı](#model-performansı)
- [Görselleştirmeler](#görselleştirmeler)
- [Geliştirme Yol Haritası](#geliştirme-yol-haritası)
- [Katkıda Bulunma](#katkıda-bulunma)

##  Proje Hakkında

Bu proje, **Wisconsin Breast Cancer** veri seti kullanılarak meme kanseri teşhisinde makine öğrenmesi tekniklerinin uygulandığı kapsamlı bir çalışmadır. Proje, meme tümörlerini **kötü huylu (malignant)** ve **iyi huylu (benign)** olarak yüksek doğrulukla sınıflandırmayı hedeflemektedir.

###  Projenin Öne Çıkan Özellikleri
-  **8 farklı makine öğrenmesi algoritması** karşılaştırması
-  **Streamlit tabanlı interaktif web uygulaması**
-  **Kapsamlı veri analizi ve görselleştirme**
-  **Hiperparametre optimizasyonu**
-  **Model yorumlama ve açıklanabilirlik (SHAP)**
-  **Ensemble öğrenme teknikleri**
-  **Öznitelik seçimi ve boyut azaltma**
-  **Detaylı performans analizi**

## Proje Yapısı

```
BRCA-Tumor-Type-Detection-main/
├──  main.py                          # Ana başlangıç uygulaması
├──  app.py                           # Streamlit web uygulaması (802 satır)
├──  veri_analizi.py                  # Veri keşfi ve görselleştirme
├──  model_karsilastirma.py           # Çoklu model karşılaştırması
├──   hiper_optimizasyon.py            # Hiperparametre optimizasyonu
├──  ensemble_ogrenme.py              # Topluluk öğrenme teknikleri
├──  oznitelik_secimi.py              # Öznitelik seçimi teknikleri
├──  shap_analizi.py                  # SHAP model yorumlama
├──  oznitelik_onem_analizi.py        # Öznitelik önem analizi
├──  precision_recall_analizi.py      # Precision-Recall curve analizi
├──  error_features_gorsellestir.py   # Hata analizi ve görselleştirme
├──  requirements.txt                 # Proje bağımlılıkları
├──   roadmap.md                      # Proje geliştirme yol haritası
└──  README.md                        # Temel proje açıklaması
```

###  Oluşturulan Klasörler
- ` modeller/` - Eğitilmiş modellerin saklandığı klasör
- ` goruntuler/` - Tüm görselleştirmelerin saklandığı klasör
  - ` veri_analizi/` - Veri analizi grafikleri
  - ` model_karsilastirma/` - Model karşılaştırma grafikleri
  - ` model_yorumlama/` - Model yorumlama grafikleri
  - ` ensemble/` - Ensemble model grafikleri
  - ` oznitelik_secimi/` - Öznitelik seçimi grafikleri

## Teknolojiler ve Kütüphaneler

### Temel Kütüphaneler
- ** Python 3.x** - Ana programlama dili
- ** NumPy** - Sayısal hesaplamalar
- ** Pandas** - Veri manipülasyonu ve analizi
- ** Matplotlib & Seaborn** - Veri görselleştirme
- ** Plotly** - Interaktif görselleştirmeler

### Makine Öğrenmesi
- ** Scikit-learn** - Makine öğrenmesi algoritmaları
- ** SHAP** - Model yorumlama ve açıklanabilirlik
- ** Joblib** - Model kaydetme ve yükleme

### Web Uygulaması
- ** Streamlit** - Web uygulaması framework'ü
- ** Pillow (PIL)** - Görüntü işleme

##  Kullanılan Makine Öğrenmesi Algoritmaları

### 1. **Lojistik Regresyon (Logistic Regression)**
- Linear sınıflandırma algoritması
- İkili sınıflandırma için ideal
- Yüksek yorumlanabilirlik

### 2. **Destek Vektör Makineleri (SVM)**
- **Linear Kernel**: Doğrusal ayrılabilir veriler için
- **RBF Kernel**: Non-linear veri için

### 3. **Karar Ağaçları (Decision Trees)**
- Yüksek yorumlanabilirlik
- Öznitelik önem analizi

### 4. **Rastgele Orman (Random Forest)**
- Ensemble yöntemi
- Overfitting'e karşı dirençli
- Öznitelik önem skorları

### 5. **Gradient Boosting**
- Güçlü ensemble yöntemi
- Yüksek performans
- Hiperparametre optimizasyonu

### 6. **K-En Yakın Komşu (KNN)**
- Instance-based öğrenme
- Non-parametrik yöntem

### 7. **Yapay Sinir Ağları (MLP)**
- Multi-layer perceptron
- Non-linear pattern recognition

### 8. **Ensemble Yöntemleri**
- **Voting Classifier**: Çoğunluk oylaması
- **Stacking Classifier**: Meta-learning

##  Özellikler

###  Veri Analizi ve Görselleştirme
- Sınıf dağılımı analizi
- Öznitelik korelasyon matrisi
- PCA analizi ile boyut azaltma
- Box plot ve violin plot analizleri
- Öznitelik dağılım grafikleri

###  Model Geliştirme
- 8 farklı algoritma ile kapsamlı karşılaştırma
- Cross-validation ile model değerlendirme
- ROC Curve ve AUC analizi
- Confusion Matrix görselleştirmesi
- Precision-Recall curve analizi

###  Hiperparametre Optimizasyonu
- **GridSearchCV**: Kapsamlı parametre araması
- **RandomizedSearchCV**: Rastgele parametre araması
- En iyi parametrelerin otomatik seçimi
- Optimized modellerin kaydedilmesi

###  Model Yorumlama
- **SHAP değerleri** ile model açıklanabilirliği
- Öznitelik önem analizi
- Permutation importance
- Global ve local açıklamalar

###  Ensemble Öğrenme
- Voting Classifier (Hard/Soft voting)
- Stacking Classifier
- Model kombinasyonları
- Ensemble performans analizi

###  Öznitelik Seçimi
- **SelectKBest**: K en iyi öznitelik seçimi
- **RFE**: Recursive Feature Elimination
- **SelectFromModel**: Model-based öznitelik seçimi
- **PCA**: Principal Component Analysis

###  Web Uygulaması
- Streamlit tabanlı kullanıcı dostu arayüz
- Interaktif model karşılaştırması
- Gerçek zamanlı tahmin yapma
- Görselleştirmelerin web üzerinde gösterimi

##  Kurulum ve Çalıştırma

### 1. Gerekli Kütüphaneleri Yükleme
```bash
pip install -r requirements.txt
```

### 2. Temel Analiz Çalıştırma
```bash
# Basit başlangıç
python main.py

# Detaylı veri analizi
python veri_analizi.py

# Model karşılaştırması
python model_karsilastirma.py

# Hiperparametre optimizasyonu
python hiper_optimizasyon.py
```

### 3. İleri Düzey Analizler
```bash
# Ensemble öğrenme
python ensemble_ogrenme.py

# Öznitelik seçimi
python oznitelik_secimi.py

# SHAP analizi
python shap_analizi.py

# Precision-Recall analizi
python precision_recall_analizi.py
```

### 4. Web Uygulamasını Başlatma
```bash
streamlit run app.py
```

##  Modül Detayları

###  main.py
**Amaç**: Projeye hızlı başlangıç  
**İçerik**:
- Veri yükleme ve ön işleme
- Lojistik Regresyon ve SVM modelleri
- Temel performans değerlendirmesi
- Sonuçları `output.txt` dosyasına kaydetme

###  veri_analizi.py (131 satır)
**Amaç**: Kapsamlı veri keşfi ve görselleştirme  
**İçerik**:
- Sınıf dağılımı analizi
- Öznitelik korelasyon matrisi
- PCA analizi ile boyut azaltma
- Öznitelik grup analizi (mean, error, worst)
- Box plot ve histogram görselleştirmeleri

###  model_karsilastirma.py (167 satır)
**Amaç**: Çoklu model performans karşılaştırması  
**İçerik**:
- 8 farklı ML algoritması
- Cross-validation ile model değerlendirme
- ROC Curve ve AUC hesaplama
- Confusion Matrix görselleştirmesi
- Model kaydetme ve yükleme

###  hiper_optimizasyon.py (267 satır)
**Amaç**: Model hiperparametre optimizasyonu  
**İçerik**:
- GridSearchCV ile kapsamlı parametre araması
- RandomizedSearchCV ile hızlı optimizasyon
- En iyi parametrelerin görselleştirilmesi
- Optimize edilmiş modellerin kaydedilmesi

###  ensemble_ogrenme.py (461 satır)
**Amaç**: Topluluk öğrenme teknikleri  
**İçerik**:
- Voting Classifier (Hard/Soft voting)
- Stacking Classifier implementasyonu
- Ensemble model performans analizi
- Çoklu model kombinasyonu testleri

###  oznitelik_secimi.py (444 satır)
**Amaç**: Öznitelik seçimi ve boyut azaltma  
**İçerik**:
- SelectKBest ile en iyi k öznitelik seçimi
- RFE (Recursive Feature Elimination)
- SelectFromModel ile model-based seçim
- PCA ile boyut azaltma
- Öznitelik seçim yöntemlerinin karşılaştırması

###  shap_analizi.py (241 satır)
**Amaç**: Model yorumlama ve açıklanabilirlik  
**İçerik**:
- SHAP değerleri hesaplama
- Waterfall plot'lar
- Summary plot'lar
- Global ve local açıklamalar
- Model-agnostic interpretability

###  oznitelik_onem_analizi.py (223 satır)
**Amaç**: Öznitelik önem analizi  
**İçerik**:
- Built-in feature importance
- Permutation importance
- Model-specific önem skorları
- Öznitelik önem görselleştirmeleri

###  precision_recall_analizi.py (215 satır)
**Amaç**: Precision-Recall curve analizi  
**İçerik**:
- Precision-Recall curve'leri
- Average Precision Score
- Model performans karşılaştırması
- Threshold optimizasyonu

###  app.py (802 satır)
**Amaç**: Comprehensive Streamlit web uygulaması  
**İçerik**:
- Multi-page web uygulaması
- Interaktif veri görselleştirme
- Model karşılaştırma arayüzü
- Gerçek zamanlı tahmin yapma
- Tüm analiz sonuçlarının web görünümü

##  Web Uygulaması

Streamlit tabanlı web uygulaması şu sayfaları içerir:

###  Ana Sayfa
- Proje hakkında genel bilgiler
- Kullanılan teknolojiler
- Model performans özetleri

###  Veri Analizi
- Interaktif veri keşfi
- Korelasyon matrisi heatmap'leri
- Dağılım grafikleri
- PCA analizi sonuçları

###  Model Karşılaştırma
- Model performans karşılaştırması
- ROC Curve'leri
- Confusion Matrix'ler
- Cross-validation sonuçları

###  Model Yorumlama
- SHAP analizi sonuçları
- Öznitelik önem skorları
- Model açıklanabilirlik grafikleri

###  Tahmin
- Gerçek zamanlı tahmin yapma
- Kullanıcı girişi ile test
- Tahmin sonuçlarının görselleştirilmesi

###  Ensemble Modeller
- Ensemble model sonuçları
- Voting ve Stacking performansları
- Model kombinasyon analizleri

###  Öznitelik Seçimi
- Öznitelik seçim sonuçları
- Boyut azaltma analizleri
- Seçim yöntemlerinin karşılaştırması

## Veri Seti Hakkında

### Wisconsin Breast Cancer Dataset
- **Kaynak**: Scikit-learn datasets
- **Örneklem Sayısı**: 569 hasta
- **Öznitelik Sayısı**: 30 nümerik öznitelik
- **Hedef Değişken**: Binary (0: Malignant, 1: Benign)

### Öznitelik Grupları
1. **Mean Features** (10 öznitelik)
   - radius, texture, perimeter, area, smoothness
   - compactness, concavity, concave points, symmetry, fractal dimension

2. **Standard Error Features** (10 öznitelik)
   - Yukarıdaki özelliklerin standart hata değerleri

3. **Worst Features** (10 öznitelik)
   - Yukarıdaki özelliklerin en kötü (en büyük) değerleri

### Sınıf Dağılımı
- **Benign (İyi Huylu)**: 357 örnek (%62.7)
- **Malignant (Kötü Huylu)**: 212 örnek (%37.3)

## Model Performansı

### En İyi Performans Gösteren Modeller
1. **Gradient Boosting**: ~97-98% doğruluk
2. **Random Forest**: ~96-97% doğruluk
3. **SVM (RBF)**: ~96-97% doğruluk
4. **Lojistik Regresyon**: ~95-96% doğruluk

### Değerlendirme Metrikleri
- **Accuracy**: Genel doğruluk
- **Precision**: Pozitif tahmin doğruluğu
- **Recall**: Gerçek pozitifleri yakalama oranı
- **F1-Score**: Precision ve Recall'ın harmonik ortalaması
- **AUC-ROC**: ROC eğrisi altındaki alan
- **AUC-PR**: Precision-Recall eğrisi altındaki alan

## Görselleştirmeler

### Veri Analizi Grafikleri
- Sınıf dağılımı bar grafikleri
- Korelasyon matrisi heatmap'leri
- PCA scatter plot'ları
- Öznitelik dağılım histogramları
- Box plot ve violin plot'lar

### Model Karşılaştırma Grafikleri
- Model doğruluk karşılaştırması
- ROC Curve'leri
- Precision-Recall Curve'leri
- Confusion Matrix'ler
- Cross-validation sonuçları

### Model Yorumlama Grafikleri
- SHAP summary plot'ları
- SHAP waterfall plot'ları
- Öznitelik önem bar grafikleri
- Permutation importance grafikleri

### Ensemble ve Öznitelik Seçimi
- Ensemble model performans karşılaştırması
- Öznitelik seçim sonuçları
- Boyut azaltma analizi grafikleri

## Geliştirme Yol Haritası

### ✅ Tamamlanan Özellikler
- [x] Kapsamlı veri analizi ve görselleştirme
- [x] 8 farklı ML algoritması implementasyonu
- [x] Hiperparametre optimizasyonu
- [x] Model karşılaştırma ve değerlendirme
- [x] SHAP analizi ile model yorumlama
- [x] Ensemble öğrenme teknikleri
- [x] Precision-Recall curve analizi
- [x] Streamlit web uygulaması
- [x] Öznitelik önem analizi

### 🔄 Devam Eden İşler
- [ ] Öznitelik seçimi teknikleri tamamlanması
- [ ] Pipeline oluşturma (preprocessing + model training)
- [ ] Veri dengesizliği için teknikler (SMOTE, undersampling)

### 🚀 Gelecek Özellikler
- [ ] Doktor/hasta kullanımı için gelişmiş UI
- [ ] Bulut platformunda deployment
- [ ] Gerçek zamanlı veri güncellemesi
- [ ] Transfer öğrenme yaklaşımları
- [ ] Görüntü verisi entegrasyonu

## Katkıda Bulunma

Bu proje açık kaynak kodludur ve katkılara açıktır. Katkıda bulunmak için:

1. Repository'yi fork edin
2. Feature branch oluşturun (`git checkout -b feature/YeniOzellik`)
3. Değişikliklerinizi commit edin (`git commit -m 'Yeni özellik eklendi'`)
4. Branch'inizi push edin (`git push origin feature/YeniOzellik`)
5. Pull Request oluşturun

### Katkı Alanları
- 🐛 **Bug fixes**: Hata düzeltmeleri
- ✨ **New features**: Yeni özellik ekleme
- 📖 **Documentation**: Dokümantasyon iyileştirme
- 🎨 **UI/UX**: Kullanıcı arayüzü geliştirme
- 🧪 **Testing**: Test kapsamını artırma
- ⚡ **Performance**: Performans optimizasyonu

## 📄 Lisans

Bu proje MIT lisansı altında lisanslanmıştır. Detaylar için [LICENSE](LICENSE) dosyasına bakınız.

## 📧 İletişim

Proje hakkında sorularınız için:
- 📧 E-posta: [your-email@example.com]
- 🐙 GitHub: [github.com/your-username]
- 💼 LinkedIn: [linkedin.com/in/your-profile]

---

**⚡ Not**: Bu proje eğitim amaçlıdır ve gerçek tıbbi teşhis için kullanılmamalıdır. Her zaman bir sağlık profesyonelinden tavsiye alınmalıdır.

**🎯 Hedef**: Bu proje, makine öğrenmesi tekniklerinin sağlık alanındaki potansiyelini göstermek ve öğrencilere kapsamlı bir ML projesi örneği sunmak amacıyla geliştirilmiştir. 
