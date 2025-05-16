import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc, precision_recall_curve
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
import os
import pickle
import warnings

warnings.filterwarnings('ignore')

# Çıktı klasörünü oluştur
os.makedirs('modeller', exist_ok=True)
os.makedirs('goruntuler/model_karsilastirma', exist_ok=True)

# Veri setini yükle
cancer = load_breast_cancer()
X = pd.DataFrame(cancer.data, columns=cancer.feature_names)
y = pd.Series(cancer.target, name='target')

# Veri setini eğitim ve test kümelerine ayır
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Özellikleri ölçeklendir
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Modelleri tanımla
models = {
    'Lojistik Regresyon': LogisticRegression(max_iter=1000, random_state=42),
    'SVM (Linear)': SVC(kernel='linear', random_state=42, probability=True),
    'SVM (RBF)': SVC(kernel='rbf', random_state=42, probability=True),
    'Karar Ağacı': DecisionTreeClassifier(random_state=42),
    'Rastgele Orman': RandomForestClassifier(random_state=42),
    'Gradient Boosting': GradientBoostingClassifier(random_state=42),
    'K-En Yakın Komşu': KNeighborsClassifier(),
    'Yapay Sinir Ağı': MLPClassifier(max_iter=1000, random_state=42)
}

# Sonuçları saklayacağımız veri yapıları
model_scores = {}
model_roc_auc = {}
model_cv_scores = {}
all_predictions = {}

# Her modeli eğit, değerlendir ve kaydet
for name, model in models.items():
    print(f"\n{name} modeli eğitiliyor...")
    
    # Modeli eğit
    model.fit(X_train_scaled, y_train)
    
    # Test seti üzerinde tahmin yap
    y_pred = model.predict(X_test_scaled)
    y_pred_proba = model.predict_proba(X_test_scaled)[:, 1] if hasattr(model, "predict_proba") else None
    
    # Tahminleri sakla
    all_predictions[name] = {"y_pred": y_pred, "y_pred_proba": y_pred_proba}
    
    # Model skoru (accuracy)
    score = model.score(X_test_scaled, y_test)
    model_scores[name] = score
    
    # 5-fold cross-validation
    cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5)
    model_cv_scores[name] = cv_scores
    
    print(f"Test seti doğruluğu: {score:.4f}")
    print(f"5-fold CV ortalama doğruluğu: {cv_scores.mean():.4f} (±{cv_scores.std():.4f})")
    
    # Modeli kaydet
    with open(f'modeller/{name.replace(" ", "_").lower()}.pkl', 'wb') as f:
        pickle.dump(model, f)
    
    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.title(f'{name} - Confusion Matrix', fontsize=15)
    plt.xlabel('Tahmin Edilen Sınıf')
    plt.ylabel('Gerçek Sınıf')
    plt.savefig(f'goruntuler/model_karsilastirma/{name.replace(" ", "_").lower()}_cm.png', bbox_inches='tight')
    plt.close()
    
    # Sınıflandırma raporu
    print(f"\n{name} Sınıflandırma Raporu:")
    print(classification_report(y_test, y_pred))
    
    # ROC curve (sadece predict_proba metodu varsa)
    if y_pred_proba is not None:
        fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
        roc_auc = auc(fpr, tpr)
        model_roc_auc[name] = roc_auc
        
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], 'k--', lw=2)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('Yanlış Pozitif Oranı')
        plt.ylabel('Doğru Pozitif Oranı')
        plt.title(f'{name} - ROC Eğrisi', fontsize=15)
        plt.legend(loc="lower right")
        plt.savefig(f'goruntuler/model_karsilastirma/{name.replace(" ", "_").lower()}_roc.png', bbox_inches='tight')
        plt.close()

# Tüm modellerin karşılaştırmalı sonuçlarını göster
results_df = pd.DataFrame({
    'Model': list(model_scores.keys()),
    'Test Doğruluğu': list(model_scores.values()),
    'CV Ortalama Doğruluğu': [cv.mean() for cv in model_cv_scores.values()],
    'CV Standart Sapma': [cv.std() for cv in model_cv_scores.values()],
    'ROC AUC (Eğer Varsa)': [model_roc_auc.get(model, np.nan) for model in model_scores.keys()]
})

results_df = results_df.sort_values('Test Doğruluğu', ascending=False).reset_index(drop=True)
print("\nModel Karşılaştırma Sonuçları:")
print(results_df)

# Sonuçları kaydet
results_df.to_csv('modeller/model_karsilastirma_sonuclari.csv', index=False)

# Karşılaştırmalı model doğruluk grafiği
plt.figure(figsize=(12, 8))
ax = sns.barplot(x='Model', y='Test Doğruluğu', data=results_df, palette='viridis')
plt.title('Model Doğruluk Karşılaştırması', fontsize=15)
plt.xlabel('Model')
plt.ylabel('Test Doğruluğu')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()

# Doğruluk değerlerini çubukların üzerine ekle
for i, v in enumerate(results_df['Test Doğruluğu']):
    ax.text(i, v + 0.01, f'{v:.4f}', ha='center', fontsize=9)

plt.savefig('goruntuler/model_karsilastirma/dogruluk_karsilastirma.png', bbox_inches='tight')
plt.close()

# ROC eğrilerini aynı grafikte karşılaştır
plt.figure(figsize=(12, 8))
for name, preds in all_predictions.items():
    if preds["y_pred_proba"] is not None:
        fpr, tpr, _ = roc_curve(y_test, preds["y_pred_proba"])
        plt.plot(fpr, tpr, lw=2, label=f'{name} (AUC = {model_roc_auc[name]:.2f})')

plt.plot([0, 1], [0, 1], 'k--', lw=2)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('Yanlış Pozitif Oranı')
plt.ylabel('Doğru Pozitif Oranı')
plt.title('Tüm Modeller - ROC Eğrisi Karşılaştırması', fontsize=15)
plt.legend(loc="lower right")
plt.grid(True, linestyle='--', alpha=0.7)
plt.savefig('goruntuler/model_karsilastirma/tum_modeller_roc.png', bbox_inches='tight')
plt.close()

print("\nModel karşılaştırma tamamlandı. Sonuçlar 'modeller' ve 'goruntuler/model_karsilastirma' klasörlerine kaydedildi.") 