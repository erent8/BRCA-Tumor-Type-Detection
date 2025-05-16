import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import joblib
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_curve, average_precision_score, PrecisionRecallDisplay
from sklearn.metrics import auc, precision_score, recall_score, f1_score
from sklearn.preprocessing import StandardScaler

# Görsel klasörünü oluştur
os.makedirs('goruntuler/model_karsilastirma', exist_ok=True)

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

# Tüm modeller için Precision-Recall analizi
print("Precision-Recall analizi yapılıyor...")

# Her model için precision ve recall değerlerini hesapla
precision_dict = {}
recall_dict = {}
average_precision_dict = {}

plt.figure(figsize=(12, 10))

# Her model için precision-recall eğrisi çiz
for model_adi, model in modeller.items():
    # Modelin olasılık tahminlerini al
    y_score = model.predict_proba(X_test_scaled)[:, 1]
    
    # Precision-Recall değerlerini hesapla
    precision, recall, _ = precision_recall_curve(y_test, y_score)
    precision_dict[model_adi] = precision
    recall_dict[model_adi] = recall
    
    # Average precision score hesapla
    average_precision = average_precision_score(y_test, y_score)
    average_precision_dict[model_adi] = average_precision
    
    # Precision-Recall eğrisinin altındaki alan
    pr_auc = auc(recall, precision)
    
    # Görsel isim
    goruntuleyici_isim = model_goruntuleyici_isimler.get(model_adi, model_adi)
    
    # PR eğrisini çiz
    plt.plot(recall, precision, lw=2, 
             label=f'{goruntuleyici_isim} (AP = {average_precision:.3f}, AUC = {pr_auc:.3f})')

# Rastgele sınıflandırıcı çizgisi (baseline)
plt.plot([0, 1], [sum(y_test)/len(y_test), sum(y_test)/len(y_test)], linestyle='--', 
         lw=2, color='gray', label='Rastgele Sınıflandırıcı')

# Grafiği biçimlendir
plt.xlabel('Recall (Duyarlılık)', fontsize=12)
plt.ylabel('Precision (Kesinlik)', fontsize=12)
plt.title('Precision-Recall Eğrileri - Tüm Modeller', fontsize=14)
plt.legend(loc='best', fontsize=10)
plt.grid(True, linestyle='--', alpha=0.7)

# Grafiği kaydet
plt.savefig('goruntuler/model_karsilastirma/tum_modeller_precision_recall.png', bbox_inches='tight', dpi=300)
print("Tüm modellerin Precision-Recall eğrisi kaydedildi.")

# Her model için ayrı Precision-Recall eğrisi
for model_adi, model in modeller.items():
    goruntuleyici_isim = model_goruntuleyici_isimler.get(model_adi, model_adi)
    
    # Modelin olasılık tahminlerini al
    y_score = model.predict_proba(X_test_scaled)[:, 1]
    
    # Precision-Recall görselleştirici
    disp = PrecisionRecallDisplay.from_predictions(
        y_test, y_score, name=goruntuleyici_isim, plot_chance_level=True
    )
    
    # Grafiği biçimlendir
    fig, ax = plt.subplots(figsize=(8, 8))
    disp.plot(ax=ax)
    ax.set_title(f'{goruntuleyici_isim} - Precision-Recall Eğrisi', fontsize=14)
    ax.grid(True, linestyle='--', alpha=0.7)
    
    # Ekstra bilgiler ekle
    average_precision = average_precision_dict[model_adi]
    ax.annotate(f'Average Precision: {average_precision:.3f}', 
                xy=(0.05, 0.05), xycoords='axes fraction', 
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8))
    
    # F1 score, Precision ve Recall değerlerini hesapla
    y_pred = model.predict(X_test_scaled)
    f1 = f1_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    
    # Metrikleri ekle
    metrics_text = f'F1: {f1:.3f}\nPrecision: {precision:.3f}\nRecall: {recall:.3f}'
    ax.annotate(metrics_text, xy=(0.05, 0.15), xycoords='axes fraction',
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8))
    
    # Grafiği kaydet
    plt.savefig(f'goruntuler/model_karsilastirma/{model_adi}_precision_recall.png', 
                bbox_inches='tight', dpi=300)
    plt.close()
    
    print(f"{goruntuleyici_isim} için Precision-Recall eğrisi kaydedildi.")

# Tüm modellerin Precision ve Recall değerlerini göster
print("\nModel performansları:")
print(f"{'Model':<25} {'F1 Score':<10} {'Precision':<10} {'Recall':<10} {'Avg Precision':<15}")
print("-" * 70)

for model_adi, model in modeller.items():
    y_pred = model.predict(X_test_scaled)
    f1 = f1_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    avg_precision = average_precision_dict[model_adi]
    
    goruntuleyici_isim = model_goruntuleyici_isimler.get(model_adi, model_adi)
    print(f"{goruntuleyici_isim:<25} {f1:<10.3f} {precision:<10.3f} {recall:<10.3f} {avg_precision:<15.3f}")

# Precision-Recall-F1 karşılaştırma barplot
plt.figure(figsize=(12, 8))

model_isimleri = [model_goruntuleyici_isimler.get(model, model) for model in modeller.keys()]
precision_values = []
recall_values = []
f1_values = []
ap_values = []

for model_adi, model in modeller.items():
    y_pred = model.predict(X_test_scaled)
    
    precision_values.append(precision_score(y_test, y_pred))
    recall_values.append(recall_score(y_test, y_pred))
    f1_values.append(f1_score(y_test, y_pred))
    ap_values.append(average_precision_dict[model_adi])

# Bar plot için veri hazırla
df_metrics = pd.DataFrame({
    'Model': model_isimleri,
    'Precision': precision_values,
    'Recall': recall_values,
    'F1 Score': f1_values,
    'Average Precision': ap_values
})

# Uzun formata dönüştür
df_melted = pd.melt(df_metrics, id_vars=['Model'], 
                   value_vars=['Precision', 'Recall', 'F1 Score', 'Average Precision'],
                   var_name='Metrik', value_name='Değer')

# Bar plot çiz
plt.figure(figsize=(14, 10))
ax = sns.barplot(x='Model', y='Değer', hue='Metrik', data=df_melted)

# Grafiği biçimlendir
plt.title('Modellerin Precision, Recall ve F1 Karşılaştırması', fontsize=16)
plt.xlabel('Model', fontsize=14)
plt.ylabel('Değer', fontsize=14)
plt.ylim(0.8, 1.01)  # Y eksenini 0.8-1.0 arasında sınırla
plt.legend(title='Metrik', fontsize=12, title_fontsize=14)
plt.xticks(rotation=45, ha='right')
plt.grid(axis='y', linestyle='--', alpha=0.7)

# Bar değerlerini göster
for container in ax.containers:
    ax.bar_label(container, fmt='%.3f', fontsize=8)

plt.tight_layout()
plt.savefig('goruntuler/model_karsilastirma/precision_recall_karsilastirma.png', 
            bbox_inches='tight', dpi=300)
print("Precision-Recall karşılaştırma grafiği kaydedildi.")

print("\nPrecision-Recall analizi tamamlandı.")
print("Tüm grafikler 'goruntuler/model_karsilastirma/' klasörüne kaydedildi.") 