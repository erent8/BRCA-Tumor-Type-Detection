import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import joblib
import shap
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Görsel klasörünü oluştur
os.makedirs('goruntuler/model_yorumlama', exist_ok=True)

# Veri setini yükle
print("Veri seti yükleniyor...")
data = load_breast_cancer()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = data.target

# Veriyi eğitim ve test olarak ayır
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Öznitelikleri ölçeklendir
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# X_test_scaled'ı DataFrame'e dönüştür ve kolonları ayarla
X_test_df = pd.DataFrame(X_test_scaled, columns=X.columns)

# Modelleri yükle
print("Modeller yükleniyor...")
model_klasoru = 'modeller'
modeller = {}

if os.path.exists(model_klasoru):
    for model_dosya in os.listdir(model_klasoru):
        if model_dosya.endswith('.pkl'):
            model_adi = os.path.splitext(model_dosya)[0]
            model_yolu = os.path.join(model_klasoru, model_dosya)
            modeller[model_adi] = joblib.load(model_yolu)
else:
    print(f"Hata: '{model_klasoru}' klasörü bulunamadı!")
    print("Lütfen önce modelleri eğitip kaydedin.")
    exit(1)

# Model görsel adlarını ayarla
model_goruntuleyici_isimler = {
    'logistic_regression_optimized': 'Lojistik Regresyon',
    'svm_optimized': 'SVM',
    'random_forest_optimized': 'Rastgele Orman',
    'gradient_boosting_optimized': 'Gradient Boosting',
    'neural_network_optimized': 'Yapay Sinir Ağı'
}

# SHAP ile Model Yorumlama
print("SHAP analizi yapılıyor...")

# Sadece tree tabanlı modelleri ve Lojistik Regresyonu ele alalım (hız ve uyumluluk için)
tree_modeller = {
    'random_forest_optimized': 'Rastgele Orman',
    'gradient_boosting_optimized': 'Gradient Boosting'
}

lineer_modeller = {
    'logistic_regression_optimized': 'Lojistik Regresyon'
}

# Örneklem büyüklüğü (hesaplama zamanı için sınırlandırılabilir)
ornek_sayisi = 100
shap_ornek = X_test_df.iloc[:ornek_sayisi]

# Her model için SHAP değerlerini hesapla ve görselleştir
# Tree modelleri için
for model_adi, model in {**tree_modeller}.items():
    try:
        # Model adını görüntülenebilir formata çevir
        goruntuleyici_isim = model_goruntuleyici_isimler.get(model_adi, model_adi)
        print(f"{goruntuleyici_isim} için SHAP değerleri hesaplanıyor...")
        
        # SHAP hesaplayıcı oluştur
        explainer = shap.TreeExplainer(modeller[model_adi])
        
        # SHAP değerlerini hesapla
        shap_values = explainer.shap_values(shap_ornek)
        
        # Birden fazla sınıf varsa pozitif sınıf için SHAP değerlerini al
        if isinstance(shap_values, list):
            shap_values = shap_values[1]  # İkinci sınıf (malign)
        
        # SHAP özet grafik (özniteliklerin önem sıralaması)
        plt.figure(figsize=(12, 8))
        shap.summary_plot(shap_values, shap_ornek, plot_type="bar", show=False)
        plt.title(f"{goruntuleyici_isim} - Öznitelik Önem Sıralaması (SHAP)", fontsize=16)
        plt.tight_layout()
        plt.savefig(f'goruntuler/model_yorumlama/{model_adi}_shap_ozet.png', bbox_inches='tight', dpi=300)
        plt.close()
        
        # SHAP beeswarm plot (öznitelik katkı dağılımı)
        plt.figure(figsize=(14, 10))
        shap.summary_plot(shap_values, shap_ornek, show=False)
        plt.title(f"{goruntuleyici_isim} - SHAP Değerleri", fontsize=16)
        plt.tight_layout()
        plt.savefig(f'goruntuler/model_yorumlama/{model_adi}_shap_beeswarm.png', bbox_inches='tight', dpi=300)
        plt.close()
        
        # En önemli 10 öznitelik
        en_onemli_oznitelikler = pd.DataFrame({
            'öznitelik': X_test_df.columns,
            'önem': np.abs(shap_values).mean(axis=0)
        }).sort_values('önem', ascending=False).head(10)
        
        print(f"{goruntuleyici_isim} için en önemli 10 öznitelik:")
        print(en_onemli_oznitelikler)
        
        # En önemli 10 öznitelik için detaylı SHAP analizi
        top_idx = en_onemli_oznitelikler['öznitelik'].head(5).values
        
        # SHAP bağımlılık grafikleri
        for oznitelik in top_idx:
            plt.figure(figsize=(10, 7))
            shap.dependence_plot(oznitelik, shap_values, shap_ornek, show=False)
            plt.title(f"{goruntuleyici_isim} - '{oznitelik}' SHAP Bağımlılık Grafiği", fontsize=14)
            plt.tight_layout()
            plt.savefig(f'goruntuler/model_yorumlama/{model_adi}_{oznitelik.replace(" ", "_")}_shap_dependency.png', 
                        bbox_inches='tight', dpi=300)
            plt.close()
        
    except Exception as e:
        print(f"Hata: {model_adi} için SHAP hesaplaması yapılamadı: {str(e)}")

# Lineer modeller için
for model_adi, model in {**lineer_modeller}.items():
    try:
        # Model adını görüntülenebilir formata çevir
        goruntuleyici_isim = model_goruntuleyici_isimler.get(model_adi, model_adi)
        print(f"{goruntuleyici_isim} için SHAP değerleri hesaplanıyor...")
        
        # Lojistik Regresyon için SHAP hesaplayıcı
        if 'logistic' in model_adi.lower():
            explainer = shap.LinearExplainer(modeller[model_adi], X_train_scaled)
            shap_values = explainer.shap_values(X_test_scaled[:ornek_sayisi])
            
            # SHAP özet grafik
            plt.figure(figsize=(12, 8))
            shap.summary_plot(shap_values, shap_ornek, plot_type="bar", show=False)
            plt.title(f"{goruntuleyici_isim} - Öznitelik Önem Sıralaması (SHAP)", fontsize=16)
            plt.tight_layout()
            plt.savefig(f'goruntuler/model_yorumlama/{model_adi}_shap_ozet.png', bbox_inches='tight', dpi=300)
            plt.close()
            
            # SHAP beeswarm plot
            plt.figure(figsize=(14, 10))
            shap.summary_plot(shap_values, shap_ornek, show=False)
            plt.title(f"{goruntuleyici_isim} - SHAP Değerleri", fontsize=16)
            plt.tight_layout()
            plt.savefig(f'goruntuler/model_yorumlama/{model_adi}_shap_beeswarm.png', bbox_inches='tight', dpi=300)
            plt.close()
            
            # En önemli 10 öznitelik
            en_onemli_oznitelikler = pd.DataFrame({
                'öznitelik': X_test_df.columns,
                'önem': np.abs(shap_values).mean(axis=0)
            }).sort_values('önem', ascending=False).head(10)
            
            print(f"{goruntuleyici_isim} için en önemli 10 öznitelik:")
            print(en_onemli_oznitelikler)
            
            # En önemli 5 öznitelik için detaylı SHAP analizi
            top_idx = en_onemli_oznitelikler['öznitelik'].head(5).values
            
            # SHAP bağımlılık grafikleri
            for oznitelik in top_idx:
                plt.figure(figsize=(10, 7))
                shap.dependence_plot(oznitelik, shap_values, shap_ornek, show=False)
                plt.title(f"{goruntuleyici_isim} - '{oznitelik}' SHAP Bağımlılık Grafiği", fontsize=14)
                plt.tight_layout()
                plt.savefig(f'goruntuler/model_yorumlama/{model_adi}_{oznitelik.replace(" ", "_")}_shap_dependency.png', 
                            bbox_inches='tight', dpi=300)
                plt.close()
    
    except Exception as e:
        print(f"Hata: {model_adi} için SHAP hesaplaması yapılamadı: {str(e)}")

# SHAP değerlerinin modeller arası karşılaştırması
# Tüm modellerin en önemli özniteliklerini bir araya getir
try:
    print("\nModeller arası öznitelik önem karşılaştırması yapılıyor...")
    
    # Analize dahil edilecek modeller
    karsilastirma_modelleri = {**tree_modeller, **lineer_modeller}
    
    # Her model için en önemli öznitelikleri topla
    karsilastirma_data = []
    
    for model_adi in karsilastirma_modelleri:
        if model_adi in tree_modeller:
            explainer = shap.TreeExplainer(modeller[model_adi])
            shap_values = explainer.shap_values(shap_ornek)
            if isinstance(shap_values, list):
                shap_values = shap_values[1]  # İkinci sınıf (malign)
        else:  # lineer modeller
            explainer = shap.LinearExplainer(modeller[model_adi], X_train_scaled)
            shap_values = explainer.shap_values(X_test_scaled[:ornek_sayisi])
        
        goruntuleyici_isim = model_goruntuleyici_isimler.get(model_adi, model_adi)
        
        en_onemli_oznitelikler = pd.DataFrame({
            'öznitelik': X_test_df.columns,
            'önem': np.abs(shap_values).mean(axis=0),
            'model': goruntuleyici_isim
        })
        
        karsilastirma_data.append(en_onemli_oznitelikler)
    
    # Tüm modelleri birleştir
    karsilastirma_df = pd.concat(karsilastirma_data)
    
    # Top 10 özniteliği seç
    top_10_oznitelikler = karsilastirma_df.groupby('öznitelik')['önem'].mean().nlargest(10).index.tolist()
    
    # Sadece top 10 öznitelikleri filtrele
    karsilastirma_df = karsilastirma_df[karsilastirma_df['öznitelik'].isin(top_10_oznitelikler)]
    
    # Görselleştir
    plt.figure(figsize=(15, 10))
    sns.barplot(x='önem', y='öznitelik', hue='model', data=karsilastirma_df)
    plt.title('Modeller Arası Öznitelik Önem Karşılaştırması (SHAP)', fontsize=16)
    plt.xlabel('SHAP Değeri (Mutlak Ortalama)', fontsize=14)
    plt.ylabel('Öznitelik', fontsize=14)
    plt.legend(title='Model', fontsize=12, title_fontsize=14)
    plt.tight_layout()
    plt.savefig('goruntuler/model_yorumlama/model_karsilastirma_shap.png', bbox_inches='tight', dpi=300)
    plt.close()
    
except Exception as e:
    print(f"Hata: Modeller arası karşılaştırma yapılamadı: {str(e)}")

print("\nSHAP analizi tamamlandı.")
print("Tüm SHAP görselleri 'goruntuler/model_yorumlama/' klasörüne kaydedildi.") 