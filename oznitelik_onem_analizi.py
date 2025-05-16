import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import joblib
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.inspection import permutation_importance

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

# Öznitelik önem analizi
print("Öznitelik önem analizi yapılıyor...")

# Tree-based modelleri analiz et (Rastgele Orman ve Gradient Boosting)
tree_modeller = {
    key: value for key, value in modeller.items() 
    if 'random_forest' in key.lower() or 'gradient_boosting' in key.lower()
}

# Her tree-based model için öznitelik önemini görselleştir
for model_adi, model in tree_modeller.items():
    goruntuleyici_isim = model_goruntuleyici_isimler.get(model_adi, model_adi)
    print(f"{goruntuleyici_isim} için öznitelik önemleri hesaplanıyor...")
    
    try:
        # Dahili feature_importances_ değerlerini al
        if hasattr(model, 'feature_importances_'):
            # Öznitelik önem değerlerini al
            importances = model.feature_importances_
            
            # Öznitelik önemlerini DataFrame'e dönüştür
            onem_df = pd.DataFrame({
                'öznitelik': X.columns,
                'önem': importances
            }).sort_values('önem', ascending=False)
            
            # İlk 20 özniteliği görselleştir
            plt.figure(figsize=(12, 10))
            sns.barplot(x='önem', y='öznitelik', data=onem_df.head(20), palette='viridis')
            plt.title(f'{goruntuleyici_isim} - Öznitelik Önem Sıralaması', fontsize=16)
            plt.xlabel('Önem Derecesi', fontsize=14)
            plt.ylabel('Öznitelik', fontsize=14)
            plt.tight_layout()
            plt.savefig(f'goruntuler/model_yorumlama/{model_adi}_feature_importance.png', bbox_inches='tight', dpi=300)
            plt.close()
            
            # Permütasyon önemini hesapla
            print(f"{goruntuleyici_isim} için permütasyon önemleri hesaplanıyor...")
            
            # Permütasyon önemini hesapla (daha güvenilir bir ölçü)
            perm_importance = permutation_importance(model, X_test_scaled, y_test, n_repeats=10, random_state=42)
            
            # Permütasyon önem sonuçlarını DataFrame'e dönüştür
            perm_df = pd.DataFrame({
                'öznitelik': X.columns,
                'önem': perm_importance.importances_mean,
                'std': perm_importance.importances_std
            }).sort_values('önem', ascending=False)
            
            # İlk 20 özniteliği görselleştir
            plt.figure(figsize=(12, 10))
            ax = sns.barplot(x='önem', y='öznitelik', data=perm_df.head(20), palette='viridis')
            
            # Hata çubuklarını ekle
            for i, (_, row) in enumerate(perm_df.head(20).iterrows()):
                ax.errorbar(row['önem'], i, xerr=row['std'], color='black', capsize=3)
                
            plt.title(f'{goruntuleyici_isim} - Permütasyon Öznitelik Önem Sıralaması', fontsize=16)
            plt.xlabel('Permütasyon Önem Derecesi', fontsize=14)
            plt.ylabel('Öznitelik', fontsize=14)
            plt.tight_layout()
            plt.savefig(f'goruntuler/model_yorumlama/{model_adi}_permutation_importance.png', bbox_inches='tight', dpi=300)
            plt.close()
            
            # En önemli öznitelikleri yazdır
            print(f"\n{goruntuleyici_isim} için en önemli 10 öznitelik (Dahili Önem):")
            print(onem_df.head(10))
            
            print(f"\n{goruntuleyici_isim} için en önemli 10 öznitelik (Permütasyon Önem):")
            print(perm_df.head(10))
        else:
            print(f"Hata: {model_adi} için öznitelik önemleri bulunamadı.")
    
    except Exception as e:
        print(f"Hata: {model_adi} için öznitelik önem analizi yapılamadı: {str(e)}")

# Tüm tree-based modellerin öznitelik önemlerini karşılaştır
try:
    print("\nModeller arası öznitelik önem karşılaştırması yapılıyor...")
    
    # Her model için öznitelik önemlerini topla
    karsilastirma_data = []
    
    for model_adi, model in tree_modeller.items():
        if hasattr(model, 'feature_importances_'):
            goruntuleyici_isim = model_goruntuleyici_isimler.get(model_adi, model_adi)
            
            onem_df = pd.DataFrame({
                'öznitelik': X.columns,
                'önem': model.feature_importances_,
                'model': goruntuleyici_isim
            })
            
            karsilastirma_data.append(onem_df)
    
    # Tüm modelleri birleştir
    if karsilastirma_data:
        karsilastirma_df = pd.concat(karsilastirma_data)
        
        # Top 15 özniteliği seç
        top_15_oznitelikler = karsilastirma_df.groupby('öznitelik')['önem'].mean().nlargest(15).index.tolist()
        
        # Sadece top 15 öznitelikleri filtrele
        karsilastirma_df = karsilastirma_df[karsilastirma_df['öznitelik'].isin(top_15_oznitelikler)]
        
        # Görselleştir
        plt.figure(figsize=(15, 12))
        sns.barplot(x='önem', y='öznitelik', hue='model', data=karsilastirma_df)
        plt.title('Modeller Arası Öznitelik Önem Karşılaştırması', fontsize=16)
        plt.xlabel('Önem Derecesi', fontsize=14)
        plt.ylabel('Öznitelik', fontsize=14)
        plt.legend(title='Model', fontsize=12, title_fontsize=14)
        plt.tight_layout()
        plt.savefig('goruntuler/model_yorumlama/model_karsilastirma_onem.png', bbox_inches='tight', dpi=300)
        plt.close()
    
except Exception as e:
    print(f"Hata: Modeller arası karşılaştırma yapılamadı: {str(e)}")

# Permütasyon önemlerine göre karşılaştırma
try:
    print("\nModeller arası permütasyon önem karşılaştırması yapılıyor...")
    
    # Her model için permütasyon önemlerini topla
    perm_karsilastirma_data = []
    
    for model_adi, model in tree_modeller.items():
        goruntuleyici_isim = model_goruntuleyici_isimler.get(model_adi, model_adi)
        
        # Permütasyon önemini hesapla
        perm_importance = permutation_importance(model, X_test_scaled, y_test, n_repeats=5, random_state=42)
        
        perm_df = pd.DataFrame({
            'öznitelik': X.columns,
            'önem': perm_importance.importances_mean,
            'model': goruntuleyici_isim
        })
        
        perm_karsilastirma_data.append(perm_df)
    
    # Tüm modelleri birleştir
    if perm_karsilastirma_data:
        perm_karsilastirma_df = pd.concat(perm_karsilastirma_data)
        
        # Top 15 özniteliği seç
        top_15_perm_oznitelikler = perm_karsilastirma_df.groupby('öznitelik')['önem'].mean().nlargest(15).index.tolist()
        
        # Sadece top 15 öznitelikleri filtrele
        perm_karsilastirma_df = perm_karsilastirma_df[perm_karsilastirma_df['öznitelik'].isin(top_15_perm_oznitelikler)]
        
        # Görselleştir
        plt.figure(figsize=(15, 12))
        sns.barplot(x='önem', y='öznitelik', hue='model', data=perm_karsilastirma_df)
        plt.title('Modeller Arası Permütasyon Öznitelik Önem Karşılaştırması', fontsize=16)
        plt.xlabel('Permütasyon Önem Derecesi', fontsize=14)
        plt.ylabel('Öznitelik', fontsize=14)
        plt.legend(title='Model', fontsize=12, title_fontsize=14)
        plt.tight_layout()
        plt.savefig('goruntuler/model_yorumlama/model_karsilastirma_permutasyon_onem.png', bbox_inches='tight', dpi=300)
        plt.close()
    
except Exception as e:
    print(f"Hata: Modeller arası permütasyon karşılaştırması yapılamadı: {str(e)}")

print("\nÖznitelik önem analizi tamamlandı.")
print("Tüm görseller 'goruntuler/model_yorumlama/' klasörüne kaydedildi.") 