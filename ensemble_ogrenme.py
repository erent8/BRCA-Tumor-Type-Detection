import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import joblib
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.metrics import roc_curve, precision_recall_curve, auc
from sklearn.ensemble import VotingClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
import warnings
warnings.filterwarnings('ignore')

# Görsel klasörünü oluştur
os.makedirs('goruntuler/ensemble', exist_ok=True)

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

# Daha önce kaydedilmiş optimize edilmiş modelleri yükle
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

# Optimizasyon sonuçlarına göre modelleri yeniden oluştur
print("En iyi hiperparametrelere sahip modeller oluşturuluyor...")
lr_model = LogisticRegression(C=0.1, max_iter=1000, penalty='l2', solver='liblinear')
svm_model = SVC(C=1, gamma='scale', kernel='rbf', probability=True)
rf_model = RandomForestClassifier(n_estimators=50, min_samples_split=5, min_samples_leaf=1, max_features='log2', max_depth=20, random_state=42)
gb_model = GradientBoostingClassifier(n_estimators=100, min_samples_split=5, min_samples_leaf=2, max_depth=3, learning_rate=0.1, random_state=42)
nn_model = MLPClassifier(solver='adam', learning_rate='adaptive', hidden_layer_sizes=(50,), alpha=0.001, activation='relu', random_state=42, max_iter=1000)

# Model adlandırma sözlüğü
model_isimler = {
    'lr': 'Lojistik Regresyon',
    'svm': 'SVM',
    'rf': 'Rastgele Orman',
    'gb': 'Gradient Boosting',
    'nn': 'Yapay Sinir Ağı'
}

# Bireysel modelleri eğit
base_models = {
    'lr': lr_model,
    'svm': svm_model,
    'rf': rf_model,
    'gb': gb_model,
    'nn': nn_model
}

# Her bireysel modeli eğit ve performansını değerlendir
print("\nBireysel Model Performansları:")
print(f"{'Model':<20} {'Accuracy':<10} {'AUC':<10} {'F1':<10} {'Precision':<10} {'Recall':<10}")
print("-" * 70)

bireysel_model_sonuclari = {}

for model_adi, model in base_models.items():
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)
    y_prob = model.predict_proba(X_test_scaled)[:, 1]
    
    accuracy = accuracy_score(y_test, y_pred)
    auc_score = roc_auc_score(y_test, y_prob)
    f1 = f1_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    
    bireysel_model_sonuclari[model_adi] = {
        'accuracy': accuracy,
        'auc': auc_score,
        'f1': f1,
        'precision': precision,
        'recall': recall,
        'model': model
    }
    
    print(f"{model_isimler[model_adi]:<20} {accuracy:<10.4f} {auc_score:<10.4f} {f1:<10.4f} {precision:<10.4f} {recall:<10.4f}")

# Ensemble Modelleri Oluştur
print("\nEnsemble modelleri oluşturuluyor...")

# 1. Voting Classifier (Hard Voting)
print("1. Hard Voting Classifier oluşturuluyor...")
hard_voting = VotingClassifier(
    estimators=[
        ('lr', lr_model), 
        ('svm', svm_model), 
        ('rf', rf_model), 
        ('gb', gb_model), 
        ('nn', nn_model)
    ],
    voting='hard'
)

# 2. Voting Classifier (Soft Voting)
print("2. Soft Voting Classifier oluşturuluyor...")
soft_voting = VotingClassifier(
    estimators=[
        ('lr', lr_model), 
        ('svm', svm_model), 
        ('rf', rf_model), 
        ('gb', gb_model), 
        ('nn', nn_model)
    ],
    voting='soft'
)

# 3. Stacking Classifier
print("3. Stacking Classifier oluşturuluyor...")
stacking = StackingClassifier(
    estimators=[
        ('lr', lr_model), 
        ('svm', svm_model), 
        ('rf', rf_model), 
        ('gb', gb_model)
    ],
    final_estimator=LogisticRegression(),
    cv=5
)

# 4. Advanced Stacking (özel stacking)
print("4. Advanced Stacking Classifier oluşturuluyor...")
# Birinci seviye modeller
level1_models = [
    ('lr', lr_model),
    ('svm', svm_model),
    ('rf', rf_model),
    ('gb', gb_model)
]

# İkinci seviye model (meta-model)
meta_model = LogisticRegression()

advanced_stacking = StackingClassifier(
    estimators=level1_models,
    final_estimator=meta_model,
    cv=5,
    stack_method='predict_proba'
)

# Ensemble modellerini eğit ve değerlendir
ensemble_modeller = {
    'hard_voting': hard_voting,
    'soft_voting': soft_voting,
    'stacking': stacking,
    'advanced_stacking': advanced_stacking
}

ensemble_isimler = {
    'hard_voting': 'Hard Voting',
    'soft_voting': 'Soft Voting',
    'stacking': 'Stacking',
    'advanced_stacking': 'Advanced Stacking'
}

# Model performanslarını saklamak için
tum_model_sonuclari = bireysel_model_sonuclari.copy()

# Her ensemble modeli eğit ve performansını değerlendir
print("\nEnsemble Model Performansları:")
print(f"{'Model':<20} {'Accuracy':<10} {'AUC':<10} {'F1':<10} {'Precision':<10} {'Recall':<10}")
print("-" * 70)

for model_adi, model in ensemble_modeller.items():
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)
    
    # AUC hesaplamak için
    if hasattr(model, "predict_proba"):
        y_prob = model.predict_proba(X_test_scaled)[:, 1]
        auc_score = roc_auc_score(y_test, y_prob)
    else:
        # Bu durumda direkt olarak prediction'ları kullan
        auc_score = roc_auc_score(y_test, y_pred)
    
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    
    tum_model_sonuclari[model_adi] = {
        'accuracy': accuracy,
        'auc': auc_score,
        'f1': f1,
        'precision': precision,
        'recall': recall,
        'model': model
    }
    
    print(f"{ensemble_isimler[model_adi]:<20} {accuracy:<10.4f} {auc_score:<10.4f} {f1:<10.4f} {precision:<10.4f} {recall:<10.4f}")

# Cross-validation sonuçları
print("\nCross-validation sonuçları (Accuracy):")
print(f"{'Model':<20} {'Mean':<10} {'Std':<10}")
print("-" * 40)

cv_results = {}
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

for model_adi, model_info in tum_model_sonuclari.items():
    if model_adi in base_models:
        model_display_name = model_isimler[model_adi]
    else:
        model_display_name = ensemble_isimler[model_adi]
    
    model = model_info['model']
    
    # Cross-validation
    cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=cv, scoring='accuracy')
    cv_results[model_adi] = {
        'mean': cv_scores.mean(),
        'std': cv_scores.std()
    }
    
    print(f"{model_display_name:<20} {cv_scores.mean():<10.4f} {cv_scores.std():<10.4f}")

# En iyi modelleri kaydet
print("\nEn iyi modeller kaydediliyor...")
os.makedirs('modeller/ensemble', exist_ok=True)

for model_adi, model_info in tum_model_sonuclari.items():
    if model_adi in ensemble_modeller:
        model = model_info['model']
        model_dosya = os.path.join('modeller/ensemble', f"{model_adi}.pkl")
        joblib.dump(model, model_dosya)
        print(f"{ensemble_isimler[model_adi]} kaydedildi: {model_dosya}")

# Görselleştirmeler
print("\nModel performans karşılaştırma görselleri oluşturuluyor...")

# 1. Accuracy karşılaştırması (bar plot)
plt.figure(figsize=(14, 8))
model_isimleri = []
accuracy_degerleri = []

for model_adi, model_info in tum_model_sonuclari.items():
    if model_adi in base_models:
        model_isimleri.append(model_isimler[model_adi])
    else:
        model_isimleri.append(ensemble_isimler[model_adi])
    
    accuracy_degerleri.append(model_info['accuracy'])

# Bar plot
indices = np.arange(len(model_isimleri))
bar_width = 0.7

# Renkleri ayarla (ensemble modelleri vurgula)
colors = ['#3498db', '#3498db', '#3498db', '#3498db', '#3498db', '#e74c3c', '#e74c3c', '#e74c3c', '#e74c3c']

plt.bar(indices, accuracy_degerleri, bar_width, color=colors)
plt.xlabel('Modeller', fontsize=14)
plt.ylabel('Accuracy', fontsize=14)
plt.title('Model Accuracy Karşılaştırması', fontsize=16)
plt.xticks(indices, model_isimleri, rotation=45, ha='right')
plt.ylim(0.94, 1.0)  # Değerleri daha iyi görmek için
plt.grid(axis='y', alpha=0.3)

# Bar'ların üzerine değerleri yaz
for i, v in enumerate(accuracy_degerleri):
    plt.text(i, v + 0.001, f"{v:.4f}", ha='center', fontsize=10)

plt.tight_layout()
plt.savefig('goruntuler/ensemble/accuracy_karsilastirma.png', dpi=300, bbox_inches='tight')
plt.close()

# 2. ROC eğrileri
plt.figure(figsize=(12, 10))

linestyles = ['-', '--', '-.', ':', '-', '--', '-.', ':']
colors = ['#3498db', '#2ecc71', '#9b59b6', '#e74c3c', '#f39c12', '#1abc9c', '#d35400', '#c0392b', '#27ae60']

for i, (model_adi, model_info) in enumerate(tum_model_sonuclari.items()):
    model = model_info['model']
    
    if hasattr(model, "predict_proba"):
        y_prob = model.predict_proba(X_test_scaled)[:, 1]
    else:
        y_prob = model.predict(X_test_scaled)
    
    # ROC eğrisini hesapla
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    auc_score = auc(fpr, tpr)
    
    # Display name
    if model_adi in base_models:
        model_display_name = model_isimler[model_adi]
    else:
        model_display_name = ensemble_isimler[model_adi]
    
    # ROC eğrisini çiz
    plt.plot(fpr, tpr, 
             label=f'{model_display_name} (AUC = {auc_score:.4f})',
             linewidth=2, 
             linestyle=linestyles[i % len(linestyles)],
             color=colors[i % len(colors)])

# Rastgele tahmin çizgisi
plt.plot([0, 1], [0, 1], 'k--', label='Rastgele Tahmin')

plt.xlabel('False Positive Rate (1 - Specificity)', fontsize=12)
plt.ylabel('True Positive Rate (Sensitivity)', fontsize=12)
plt.title('ROC Eğrileri Karşılaştırması', fontsize=16)
plt.legend(loc='lower right')
plt.grid(alpha=0.3)
plt.savefig('goruntuler/ensemble/roc_karsilastirma.png', dpi=300, bbox_inches='tight')
plt.close()

# 3. Precision-Recall eğrileri
plt.figure(figsize=(12, 10))

for i, (model_adi, model_info) in enumerate(tum_model_sonuclari.items()):
    model = model_info['model']
    
    if hasattr(model, "predict_proba"):
        y_prob = model.predict_proba(X_test_scaled)[:, 1]
    else:
        y_prob = model.predict(X_test_scaled)
    
    # Precision-Recall eğrisini hesapla
    precision, recall, _ = precision_recall_curve(y_test, y_prob)
    pr_auc = auc(recall, precision)
    
    # Display name
    if model_adi in base_models:
        model_display_name = model_isimler[model_adi]
    else:
        model_display_name = ensemble_isimler[model_adi]
    
    # Precision-Recall eğrisini çiz
    plt.plot(recall, precision, 
             label=f'{model_display_name} (AUC = {pr_auc:.4f})',
             linewidth=2, 
             linestyle=linestyles[i % len(linestyles)],
             color=colors[i % len(colors)])

# Rastgele tahmin çizgisi
plt.axhline(y=sum(y_test)/len(y_test), color='k', linestyle='--', 
            label=f'Rastgele Tahmin (Precision = {sum(y_test)/len(y_test):.4f})')

plt.xlabel('Recall', fontsize=12)
plt.ylabel('Precision', fontsize=12)
plt.title('Precision-Recall Eğrileri Karşılaştırması', fontsize=16)
plt.legend(loc='lower left')
plt.grid(alpha=0.3)
plt.savefig('goruntuler/ensemble/precision_recall_karsilastirma.png', dpi=300, bbox_inches='tight')
plt.close()

# 4. Detaylı metrikler karşılaştırması
metrics = ['accuracy', 'auc', 'f1', 'precision', 'recall']
metric_display = {
    'accuracy': 'Accuracy',
    'auc': 'AUC', 
    'f1': 'F1 Score',
    'precision': 'Precision',
    'recall': 'Recall'
}

# Veriyi hazırla
df_metrics = pd.DataFrame()

for model_adi, model_info in tum_model_sonuclari.items():
    if model_adi in base_models:
        model_display_name = model_isimler[model_adi]
    else:
        model_display_name = ensemble_isimler[model_adi]
    
    model_metrics = {metric: model_info[metric] for metric in metrics}
    model_metrics['model'] = model_display_name
    
    df_model = pd.DataFrame([model_metrics])
    df_metrics = pd.concat([df_metrics, df_model], ignore_index=True)

# Veriyi uzun formata dönüştür
df_melted = pd.melt(df_metrics, id_vars=['model'], value_vars=metrics, 
                   var_name='Metric', value_name='Value')

# Display name'leri düzelt
df_melted['Metric'] = df_melted['Metric'].map(metric_display)

# Görselleştir
plt.figure(figsize=(15, 10))
sns.barplot(x='model', y='Value', hue='Metric', data=df_melted)

plt.title('Model Performans Metrikleri Karşılaştırması', fontsize=16)
plt.xlabel('Model', fontsize=14)
plt.ylabel('Değer', fontsize=14)
plt.ylim(0.8, 1.02)  # Y eksenini değerlere göre düzenle
plt.xticks(rotation=45, ha='right')
plt.legend(title='Metrikler')
plt.grid(axis='y', alpha=0.3)
plt.tight_layout()

plt.savefig('goruntuler/ensemble/tum_metrikler_karsilastirma.png', dpi=300, bbox_inches='tight')
plt.close()

# 5. Confusion Matrix
best_models = ['soft_voting', 'stacking', 'gb', 'lr']
fig, axes = plt.subplots(2, 2, figsize=(15, 12))
axes = axes.flatten()

for i, model_adi in enumerate(best_models):
    model_info = tum_model_sonuclari[model_adi]
    model = model_info['model']
    y_pred = model.predict(X_test_scaled)
    
    # Display name
    if model_adi in base_models:
        model_display_name = model_isimler[model_adi]
    else:
        model_display_name = ensemble_isimler[model_adi]
    
    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[i])
    axes[i].set_title(f'Confusion Matrix - {model_display_name}', fontsize=12)
    axes[i].set_xlabel('Tahmin Edilen Sınıf')
    axes[i].set_ylabel('Gerçek Sınıf')
    
    # Doğruluk, hassasiyet ve duyarlılık değerlerini ekle
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    axes[i].text(0.5, -0.15, f'Accuracy: {acc:.4f}, Precision: {prec:.4f}, Recall: {rec:.4f}', 
              ha='center', transform=axes[i].transAxes)

plt.tight_layout()
plt.savefig('goruntuler/ensemble/confusion_matrices.png', dpi=300, bbox_inches='tight')
plt.close()

print("\nEnsemble analizi tamamlandı ve sonuçlar kaydedildi.")
print("Tüm görseller 'goruntuler/ensemble/' klasörüne kaydedildi.") 