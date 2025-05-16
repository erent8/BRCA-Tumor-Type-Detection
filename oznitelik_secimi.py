import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import joblib
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, precision_score, recall_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif, RFE
from sklearn.feature_selection import SelectFromModel
from sklearn.decomposition import PCA
import warnings
warnings.filterwarnings('ignore')

# Görsel klasörünü oluştur
os.makedirs('goruntuler/oznitelik_secimi', exist_ok=True)

# Veri setini yükleme
print("Veri seti yükleniyor...")
data = load_breast_cancer()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = data.target

# Öznitelik isimlerini düzenleme (kısaltma)
kisa_isimler = {
    'mean radius': 'radius_mean',
    'mean texture': 'texture_mean',
    'mean perimeter': 'perimeter_mean',
    'mean area': 'area_mean',
    'mean smoothness': 'smoothness_mean',
    'mean compactness': 'compactness_mean',
    'mean concavity': 'concavity_mean',
    'mean concave points': 'concave_pts_mean',
    'mean symmetry': 'symmetry_mean',
    'mean fractal dimension': 'fractal_dim_mean',
    'radius error': 'radius_se',
    'texture error': 'texture_se',
    'perimeter error': 'perimeter_se',
    'area error': 'area_se',
    'smoothness error': 'smoothness_se',
    'compactness error': 'compactness_se',
    'concavity error': 'concavity_se',
    'concave points error': 'concave_pts_se',
    'symmetry error': 'symmetry_se',
    'fractal dimension error': 'fractal_dim_se',
    'worst radius': 'radius_worst',
    'worst texture': 'texture_worst',
    'worst perimeter': 'perimeter_worst',
    'worst area': 'area_worst',
    'worst smoothness': 'smoothness_worst',
    'worst compactness': 'compactness_worst',
    'worst concavity': 'concavity_worst',
    'worst concave points': 'concave_pts_worst',
    'worst symmetry': 'symmetry_worst',
    'worst fractal dimension': 'fractal_dim_worst'
}

X.columns = [kisa_isimler[col] for col in X.columns]

# Veriyi eğitim ve test olarak ayır
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Öznitelikleri ölçeklendir
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Öznitelik seçim yöntemlerinin sonuçlarını saklayacak sözlük
secim_sonuclari = {}

print("Öznitelik seçim teknikleri uygulanıyor...")

# 1. Univariate Feature Selection (ANOVA F-değeri)
print("\n1. Univariate Feature Selection (ANOVA F-değeri)")
anova_selector = SelectKBest(f_classif, k=10)
X_train_anova = anova_selector.fit_transform(X_train_scaled, y_train)
X_test_anova = anova_selector.transform(X_test_scaled)

# Öznitelik skorlarını al
anova_scores = anova_selector.scores_
anova_mask = anova_selector.get_support()
anova_secilen = X.columns[anova_mask]

print("ANOVA F-değeri ile seçilen 10 öznitelik:")
for i, feature in enumerate(anova_secilen):
    print(f"{feature}: {anova_scores[anova_mask][i]}")

secim_sonuclari['ANOVA F-değeri'] = {
    'secilen_oznitelikler': list(anova_secilen),
    'skorlar': list(anova_scores[anova_mask]),
    'X_train': X_train_anova,
    'X_test': X_test_anova
}

# 2. Mutual Information
print("\n2. Mutual Information Feature Selection")
mi_selector = SelectKBest(mutual_info_classif, k=10)
X_train_mi = mi_selector.fit_transform(X_train_scaled, y_train)
X_test_mi = mi_selector.transform(X_test_scaled)

# Öznitelik skorlarını al
mi_scores = mi_selector.scores_
mi_mask = mi_selector.get_support()
mi_secilen = X.columns[mi_mask]

print("Mutual Information ile seçilen 10 öznitelik:")
for i, feature in enumerate(mi_secilen):
    print(f"{feature}: {mi_scores[mi_mask][i]}")

secim_sonuclari['Mutual Information'] = {
    'secilen_oznitelikler': list(mi_secilen),
    'skorlar': list(mi_scores[mi_mask]),
    'X_train': X_train_mi,
    'X_test': X_test_mi
}

# 3. Recursive Feature Elimination (RFE) - Lojistik Regresyon temelli
print("\n3. Recursive Feature Elimination (Lojistik Regresyon)")
lr_model = LogisticRegression(max_iter=1000)
rfe_selector = RFE(estimator=lr_model, n_features_to_select=10, step=1)
X_train_rfe = rfe_selector.fit_transform(X_train_scaled, y_train)
X_test_rfe = rfe_selector.transform(X_test_scaled)

# Seçilen öznitelikleri al
rfe_mask = rfe_selector.get_support()
rfe_secilen = X.columns[rfe_mask]
rfe_rankings = rfe_selector.ranking_

print("RFE ile seçilen 10 öznitelik:")
for feature in rfe_secilen:
    print(f"{feature}")

secim_sonuclari['RFE (Lojistik Regresyon)'] = {
    'secilen_oznitelikler': list(rfe_secilen),
    'X_train': X_train_rfe,
    'X_test': X_test_rfe
}

# 4. Önem tabanlı seçim (Random Forest)
print("\n4. Önem Tabanlı Seçim (Random Forest)")
rf_selector = RandomForestClassifier(n_estimators=100, random_state=42)
rf_selector.fit(X_train_scaled, y_train)

# Öznitelik önemlerini al
rf_importances = rf_selector.feature_importances_
rf_indices = np.argsort(rf_importances)[::-1][:10]  # En önemli 10 öznitelik
rf_secilen = X.columns[rf_indices]

# Model tabanlı öznitelik seçimi
rf_select_model = SelectFromModel(rf_selector, threshold="mean", prefit=True)
X_train_rf = rf_select_model.transform(X_train_scaled)
X_test_rf = rf_select_model.transform(X_test_scaled)

rf_model_mask = rf_select_model.get_support()
rf_model_secilen = X.columns[rf_model_mask]

print("Random Forest önem skorlarına göre en önemli 10 öznitelik:")
for i, idx in enumerate(rf_indices):
    print(f"{X.columns[idx]}: {rf_importances[idx]}")

print("\nRandom Forest model tabanlı seçilen öznitelikler:")
print(rf_model_secilen)

secim_sonuclari['Random Forest Önem'] = {
    'secilen_oznitelikler': list(rf_secilen),
    'skorlar': [rf_importances[idx] for idx in rf_indices],
    'X_train': X_train_rf,
    'X_test': X_test_rf,
    'model_secilen': list(rf_model_secilen)
}

# 5. Gradient Boosting Feature Importance
print("\n5. Gradient Boosting Önem Tabanlı Seçim")
gb_selector = GradientBoostingClassifier(n_estimators=100, random_state=42)
gb_selector.fit(X_train_scaled, y_train)

# Öznitelik önemlerini al
gb_importances = gb_selector.feature_importances_
gb_indices = np.argsort(gb_importances)[::-1][:10]  # En önemli 10 öznitelik
gb_secilen = X.columns[gb_indices]

# Model tabanlı öznitelik seçimi
gb_select_model = SelectFromModel(gb_selector, threshold="mean", prefit=True)
X_train_gb = gb_select_model.transform(X_train_scaled)
X_test_gb = gb_select_model.transform(X_test_scaled)

gb_model_mask = gb_select_model.get_support()
gb_model_secilen = X.columns[gb_model_mask]

print("Gradient Boosting önem skorlarına göre en önemli 10 öznitelik:")
for i, idx in enumerate(gb_indices):
    print(f"{X.columns[idx]}: {gb_importances[idx]}")

print("\nGradient Boosting model tabanlı seçilen öznitelikler:")
print(gb_model_secilen)

secim_sonuclari['Gradient Boosting Önem'] = {
    'secilen_oznitelikler': list(gb_secilen),
    'skorlar': [gb_importances[idx] for idx in gb_indices],
    'X_train': X_train_gb,
    'X_test': X_test_gb,
    'model_secilen': list(gb_model_secilen)
}

# 6. PCA (Boyut azaltma)
print("\n6. PCA (Temel Bileşen Analizi)")
pca = PCA(n_components=10)
X_train_pca = pca.fit_transform(X_train_scaled)
X_test_pca = pca.transform(X_test_scaled)

# Açıklanan varyansı göster
explained_variance = pca.explained_variance_ratio_
cumulative_variance = np.cumsum(explained_variance)

print(f"İlk 10 bileşenin açıkladığı toplam varyans: {cumulative_variance[-1]:.4f}")
print("Bileşen başına açıklanan varyans:")
for i, var in enumerate(explained_variance):
    print(f"Bileşen {i+1}: {var:.4f}")

secim_sonuclari['PCA'] = {
    'explained_variance': explained_variance,
    'cumulative_variance': cumulative_variance,
    'X_train': X_train_pca,
    'X_test': X_test_pca
}

# Her yöntem için model performansını ölçme
print("\n\nSeçilen özniteliklerle model performansı değerlendiriliyor...")
print(f"{'Yöntem':<25} {'Accuracy':<10} {'AUC':<10} {'F1':<10} {'Precision':<10} {'Recall':<10}")
print("-" * 75)

performans_sonuclari = {}

for yontem, veri in secim_sonuclari.items():
    # Lojistik Regresyon modeli ile değerlendir
    lr = LogisticRegression(max_iter=1000)
    
    # PCA için özel durum (öznitelik isimleri yok)
    if yontem == 'PCA':
        X_train_selected = veri['X_train']
        X_test_selected = veri['X_test']
    else:
        X_train_selected = veri['X_train']
        X_test_selected = veri['X_test']
    
    # Modeli eğit
    lr.fit(X_train_selected, y_train)
    
    # Tahminler
    y_pred = lr.predict(X_test_selected)
    y_proba = lr.predict_proba(X_test_selected)[:, 1]
    
    # Metrikler
    accuracy = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_proba)
    f1 = f1_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    
    performans_sonuclari[yontem] = {
        'accuracy': accuracy,
        'auc': auc,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }
    
    print(f"{yontem:<25} {accuracy:<10.4f} {auc:<10.4f} {f1:<10.4f} {precision:<10.4f} {recall:<10.4f}")

# Tüm özniteliklerle model performansı - baseline için
print("\nTüm öznitelikler ile model performansı (baseline):")
lr_full = LogisticRegression(max_iter=1000)
lr_full.fit(X_train_scaled, y_train)
y_pred_full = lr_full.predict(X_test_scaled)
y_proba_full = lr_full.predict_proba(X_test_scaled)[:, 1]

accuracy_full = accuracy_score(y_test, y_pred_full)
auc_full = roc_auc_score(y_test, y_proba_full)
f1_full = f1_score(y_test, y_pred_full)
precision_full = precision_score(y_test, y_pred_full)
recall_full = recall_score(y_test, y_pred_full)

performans_sonuclari['Tüm Öznitelikler'] = {
    'accuracy': accuracy_full,
    'auc': auc_full,
    'f1': f1_full,
    'precision': precision_full,
    'recall': recall_full
}

print(f"{'Tüm Öznitelikler':<25} {accuracy_full:<10.4f} {auc_full:<10.4f} {f1_full:<10.4f} {precision_full:<10.4f} {recall_full:<10.4f}")

# Görselleştirmeler
print("\nÖznitelik seçim sonuçları görselleştiriliyor...")

# 1. Farklı yöntemlerle seçilen özniteliklerin karşılaştırması
plt.figure(figsize=(20, 10))

# Tüm yöntemlerdeki öznitelikleri bir araya getir
tum_secilen_oznitelikler = []
for yontem, veri in secim_sonuclari.items():
    if yontem != 'PCA':  # PCA için öznitelik seçimi yok
        tum_secilen_oznitelikler.extend(veri['secilen_oznitelikler'])
    
# Benzersiz öznitelikleri al
benzersiz_oznitelikler = sorted(set(tum_secilen_oznitelikler))

# Her yöntem için seçilen özniteliklerin matrisini oluştur
secim_matrisi = np.zeros((len(secim_sonuclari) - 1, len(benzersiz_oznitelikler)))  # PCA hariç

for i, (yontem, veri) in enumerate([item for item in secim_sonuclari.items() if item[0] != 'PCA']):
    for j, oznitelik in enumerate(benzersiz_oznitelikler):
        if oznitelik in veri['secilen_oznitelikler']:
            secim_matrisi[i, j] = 1

# Heatmap olarak göster
plt.figure(figsize=(18, 8))
yontemler = [yontem for yontem in secim_sonuclari.keys() if yontem != 'PCA']
sns.heatmap(secim_matrisi, annot=False, cmap='YlGnBu', 
            xticklabels=benzersiz_oznitelikler, yticklabels=yontemler)
plt.title('Farklı Yöntemlerle Seçilen Öznitelikler', fontsize=16)
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig('goruntuler/oznitelik_secimi/secilen_oznitelikler_karsilastirma.png', dpi=300, bbox_inches='tight')
plt.close()

# 2. Her bir seçim yöntemi için öznitelik skorlarını gösteren bar plotlar
for yontem, veri in secim_sonuclari.items():
    if yontem != 'PCA' and 'skorlar' in veri:  # PCA ve RFE için skorlar mevcut değil
        plt.figure(figsize=(14, 8))
        
        oznitelikler = veri['secilen_oznitelikler']
        skorlar = veri['skorlar']
        
        # Skorları büyükten küçüğe sırala
        sira = np.argsort(skorlar)[::-1]
        oznitelikler = [oznitelikler[i] for i in sira]
        skorlar = [skorlar[i] for i in sira]
        
        # Bar plot
        plt.bar(range(len(oznitelikler)), skorlar, color='skyblue')
        plt.xticks(range(len(oznitelikler)), oznitelikler, rotation=45, ha='right')
        plt.title(f'{yontem} Öznitelik Skorları', fontsize=16)
        plt.xlabel('Öznitelikler')
        plt.ylabel('Skor')
        plt.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        plt.savefig(f'goruntuler/oznitelik_secimi/{yontem.replace(" ", "_").lower()}_skorlar.png', 
                   dpi=300, bbox_inches='tight')
        plt.close()

# 3. PCA için açıklanan varyans grafiği
plt.figure(figsize=(12, 8))
plt.bar(range(1, len(secim_sonuclari['PCA']['explained_variance']) + 1), 
       secim_sonuclari['PCA']['explained_variance'], alpha=0.7, color='skyblue')
plt.plot(range(1, len(secim_sonuclari['PCA']['cumulative_variance']) + 1), 
         secim_sonuclari['PCA']['cumulative_variance'], 'r-o', linewidth=2)
plt.grid(axis='y', alpha=0.3)
plt.xlabel('Bileşen Sayısı', fontsize=12)
plt.ylabel('Açıklanan Varyans Oranı', fontsize=12)
plt.title('PCA: Açıklanan Varyans', fontsize=16)
plt.savefig('goruntuler/oznitelik_secimi/pca_variance.png', dpi=300, bbox_inches='tight')
plt.close()

# 4. Model performans karşılaştırması
metrikler = ['accuracy', 'auc', 'f1', 'precision', 'recall']
metrik_gorunumler = {'accuracy': 'Accuracy', 'auc': 'AUC', 'f1': 'F1', 
                    'precision': 'Precision', 'recall': 'Recall'}

# Her bir metrik için performans karşılaştırması
for metrik in metrikler:
    plt.figure(figsize=(14, 8))
    
    yontemler = list(performans_sonuclari.keys())
    degerler = [performans_sonuclari[y][metrik] for y in yontemler]
    
    # Bar plot
    renkler = ['#3498db' if y != 'Tüm Öznitelikler' else '#e74c3c' for y in yontemler]
    plt.bar(range(len(yontemler)), degerler, color=renkler)
    plt.xticks(range(len(yontemler)), yontemler, rotation=45, ha='right')
    plt.title(f'{metrik_gorunumler[metrik]} Karşılaştırması', fontsize=16)
    plt.ylabel(metrik_gorunumler[metrik])
    
    # Değerleri çubukların üzerine ekle
    for i, v in enumerate(degerler):
        plt.text(i, v + 0.005, f'{v:.4f}', ha='center', fontsize=9)
        
    plt.ylim(min(degerler) - 0.02, max(degerler) + 0.02)
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'goruntuler/oznitelik_secimi/{metrik}_karsilastirma.png', dpi=300, bbox_inches='tight')
    plt.close()

# 5. Tüm metriklerin yöntem bazında karşılaştırması
plt.figure(figsize=(16, 10))

# Veriyi hazırla
df_perf = pd.DataFrame(columns=['Yöntem', 'Metrik', 'Değer'])

for yontem, metrikler in performans_sonuclari.items():
    for metrik, deger in metrikler.items():
        df_perf = df_perf._append({'Yöntem': yontem, 'Metrik': metrik_gorunumler[metrik], 'Değer': deger}, 
                                 ignore_index=True)

# Bar plot
sns.barplot(x='Yöntem', y='Değer', hue='Metrik', data=df_perf)
plt.title('Öznitelik Seçim Yöntemlerinin Performans Karşılaştırması', fontsize=16)
plt.xlabel('')
plt.ylabel('Değer')
plt.ylim(0.94, 1.01)
plt.xticks(rotation=45, ha='right')
plt.grid(axis='y', alpha=0.3)
plt.legend(title='Metrik')
plt.tight_layout()
plt.savefig('goruntuler/oznitelik_secimi/tum_metrikler_karsilastirma.png', dpi=300, bbox_inches='tight')
plt.close()

# En iyi performans gösteren yöntemin özniteliklerini kaydet
en_iyi_yontem = max(performans_sonuclari.items(), key=lambda x: x[1]['auc'])[0]
print(f"\nEn iyi performans gösteren yöntem: {en_iyi_yontem}")

if en_iyi_yontem != 'PCA' and en_iyi_yontem != 'Tüm Öznitelikler':
    en_iyi_oznitelikler = secim_sonuclari[en_iyi_yontem]['secilen_oznitelikler']
    print(f"Seçilen öznitelikler: {en_iyi_oznitelikler}")
    
    # En iyi özellikleri kaydet
    with open('goruntuler/oznitelik_secimi/en_iyi_oznitelikler.txt', 'w') as f:
        f.write(f"En iyi performans gösteren yöntem: {en_iyi_yontem}\n")
        f.write("Seçilen öznitelikler:\n")
        for oznitelik in en_iyi_oznitelikler:
            f.write(f"- {oznitelik}\n")
        
        f.write("\nPerformans metrikleri:\n")
        for metrik, deger in performans_sonuclari[en_iyi_yontem].items():
            f.write(f"- {metrik_gorunumler[metrik]}: {deger:.4f}\n")
else:
    print("En iyi performans tüm öznitelikler veya PCA ile elde edildi.")

print("\nÖznitelik seçimi analizi tamamlandı.")
print("Tüm görsel ve sonuçlar 'goruntuler/oznitelik_secimi/' klasörüne kaydedildi.") 