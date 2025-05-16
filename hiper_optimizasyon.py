import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc, precision_recall_curve
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
import os
import pickle
import warnings
import joblib
from datetime import datetime

warnings.filterwarnings('ignore')

# Çıktı klasörünü oluştur
os.makedirs('modeller/optimized', exist_ok=True)
os.makedirs('goruntuler/hiper_optimizasyon', exist_ok=True)

# Başlangıç zamanını kaydet
start_time = datetime.now()
print(f"Optimizasyon başlangıç zamanı: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")

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

# Hiperparametre uzayları
param_grids = {
    'Lojistik Regresyon': {
        'estimator': LogisticRegression(random_state=42),
        'param_grid': {
            'C': [0.001, 0.01, 0.1, 1, 10, 100],
            'penalty': ['l1', 'l2', 'elasticnet', None],
            'solver': ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga'],
            'max_iter': [1000]
        }
    },
    'SVM': {
        'estimator': SVC(probability=True, random_state=42),
        'param_grid': {
            'C': [0.1, 1, 10, 100],
            'gamma': [0.001, 0.01, 0.1, 1, 'scale', 'auto'],
            'kernel': ['linear', 'rbf', 'poly', 'sigmoid']
        }
    },
    'Rastgele Orman': {
        'estimator': RandomForestClassifier(random_state=42),
        'param_grid': {
            'n_estimators': [50, 100, 200],
            'max_depth': [None, 10, 20, 30],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
            'max_features': ['sqrt', 'log2']
        }
    },
    'Gradient Boosting': {
        'estimator': GradientBoostingClassifier(random_state=42),
        'param_grid': {
            'n_estimators': [50, 100, 200],
            'learning_rate': [0.01, 0.05, 0.1, 0.2],
            'max_depth': [3, 5, 7],
            'min_samples_split': [2, 5],
            'min_samples_leaf': [1, 2]
        }
    },
    'Yapay Sinir Ağı': {
        'estimator': MLPClassifier(random_state=42, max_iter=1000),
        'param_grid': {
            'hidden_layer_sizes': [(50,), (100,), (50, 50), (100, 50)],
            'activation': ['relu', 'tanh', 'logistic'],
            'alpha': [0.0001, 0.001, 0.01],
            'learning_rate': ['constant', 'adaptive'],
            'solver': ['adam', 'sgd']
        }
    }
}

# Optimize edilmiş modelleri ve sonuçları saklayacağımız sözlükler
optimized_models = {}
optimization_results = {}

# Her model için GridSearchCV veya RandomizedSearchCV uygula
for model_name, model_info in param_grids.items():
    print(f"\n{model_name} modeli için hiperparametre optimizasyonu başlıyor...")
    
    # Büyük parametre uzayları için RandomizedSearchCV, küçük olanlar için GridSearchCV
    if model_name in ['Rastgele Orman', 'Gradient Boosting', 'Yapay Sinir Ağı']:
        print(f"{model_name} için RandomizedSearchCV kullanılıyor...")
        search = RandomizedSearchCV(
            estimator=model_info['estimator'],
            param_distributions=model_info['param_grid'],
            n_iter=20,  # Rastgele 20 kombinasyon dene
            cv=5,
            scoring='accuracy',
            n_jobs=-1,
            verbose=1,
            random_state=42
        )
    else:
        print(f"{model_name} için GridSearchCV kullanılıyor...")
        search = GridSearchCV(
            estimator=model_info['estimator'],
            param_grid=model_info['param_grid'],
            cv=5,
            scoring='accuracy',
            n_jobs=-1,
            verbose=1
        )
    
    # Arama işlemini başlat
    search.fit(X_train_scaled, y_train)
    
    # En iyi sonuçları kaydet
    best_params = search.best_params_
    best_score = search.best_score_
    best_model = search.best_estimator_
    
    # Test seti üzerinde değerlendir
    y_pred = best_model.predict(X_test_scaled)
    y_pred_proba = best_model.predict_proba(X_test_scaled)[:, 1]
    test_score = best_model.score(X_test_scaled, y_test)
    
    # Sonuçları sakla
    optimization_results[model_name] = {
        'best_params': best_params,
        'best_cv_score': best_score,
        'test_score': test_score
    }
    
    # Modeli kaydet
    optimized_models[model_name] = best_model
    joblib.dump(best_model, f'modeller/optimized/{model_name.replace(" ", "_").lower()}_optimized.pkl')
    
    # Optimizasyon sonuçlarını yazdır
    print(f"\n{model_name} optimizasyon sonuçları:")
    print(f"En iyi parametreler: {best_params}")
    print(f"En iyi CV skoru: {best_score:.4f}")
    print(f"Test seti doğruluğu: {test_score:.4f}")
    
    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.title(f'{model_name} (Optimized) - Confusion Matrix', fontsize=15)
    plt.xlabel('Tahmin Edilen Sınıf')
    plt.ylabel('Gerçek Sınıf')
    plt.savefig(f'goruntuler/hiper_optimizasyon/{model_name.replace(" ", "_").lower()}_cm.png', bbox_inches='tight')
    plt.close()
    
    # ROC curve
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Yanlış Pozitif Oranı')
    plt.ylabel('Doğru Pozitif Oranı')
    plt.title(f'{model_name} (Optimized) - ROC Eğrisi', fontsize=15)
    plt.legend(loc="lower right")
    plt.savefig(f'goruntuler/hiper_optimizasyon/{model_name.replace(" ", "_").lower()}_roc.png', bbox_inches='tight')
    plt.close()
    
    # Optimizasyon sürecindeki iyileştirmeyi görmek için CV sonuçlarını görselleştir
    cv_results = pd.DataFrame(search.cv_results_)
    
    plt.figure(figsize=(12, 6))
    plt.scatter(
        cv_results.index, 
        cv_results['mean_test_score'], 
        c=cv_results.index, 
        cmap='viridis',
        alpha=0.8,
        s=50
    )
    plt.axhline(y=best_score, color='r', linestyle='--', label=f'En iyi skor: {best_score:.4f}')
    plt.title(f'{model_name} Hiperparametre Optimizasyonu Sonuçları', fontsize=15)
    plt.xlabel('Deneme İndeksi')
    plt.ylabel('Ortalama CV Skoru')
    plt.colorbar(label='Deneme İndeksi')
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'goruntuler/hiper_optimizasyon/{model_name.replace(" ", "_").lower()}_optimization_process.png', bbox_inches='tight')
    plt.close()

# Tüm optimizasyon sonuçlarını karşılaştır
results_df = pd.DataFrame({
    'Model': list(optimization_results.keys()),
    'En İyi CV Skoru': [results['best_cv_score'] for results in optimization_results.values()],
    'Test Doğruluğu': [results['test_score'] for results in optimization_results.values()]
})

results_df = results_df.sort_values('Test Doğruluğu', ascending=False).reset_index(drop=True)
print("\nOptimize Edilmiş Model Karşılaştırması:")
print(results_df)

# Sonuçları kaydet
results_df.to_csv('modeller/optimized/optimization_results.csv', index=False)

# Karşılaştırmalı model doğruluk grafiği
plt.figure(figsize=(12, 8))
ax = sns.barplot(
    data=pd.melt(
        results_df, 
        id_vars=['Model'], 
        value_vars=['En İyi CV Skoru', 'Test Doğruluğu']
    ),
    x='Model', 
    y='value', 
    hue='variable',
    palette=['#5cb85c', '#337ab7']
)
plt.title('Optimize Edilmiş Modellerin Performans Karşılaştırması', fontsize=15)
plt.xlabel('Model')
plt.ylabel('Doğruluk')
plt.xticks(rotation=45, ha='right')
plt.legend(title='Metrik')
plt.tight_layout()

# Doğruluk değerlerini çubukların üzerine ekle
for container in ax.containers:
    ax.bar_label(container, fmt='%.4f', fontsize=8)

plt.savefig('goruntuler/hiper_optimizasyon/optimized_model_comparison.png', bbox_inches='tight')
plt.close()

# Optimizasyon sonuçlarını ayrıntılı olarak JSON formatında kaydet
import json

with open('modeller/optimized/detailed_results.json', 'w', encoding='utf-8') as f:
    # Best parameters JSON serileştirme için uyumlu hale getirme
    serializable_results = {}
    for model_name, results in optimization_results.items():
        serializable_results[model_name] = {
            'best_cv_score': results['best_cv_score'],
            'test_score': results['test_score'],
            'best_params': {k: (v if isinstance(v, (str, int, float, bool, list, dict)) or v is None else str(v)) 
                           for k, v in results['best_params'].items()}
        }
    
    json.dump(serializable_results, f, indent=4)

# Bitiş zamanını kaydet ve toplam süreyi hesapla
end_time = datetime.now()
duration = end_time - start_time
print(f"\nOptimizasyon tamamlandı!")
print(f"Başlangıç zamanı: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
print(f"Bitiş zamanı: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
print(f"Toplam süre: {duration}")
print(f"\nSonuçlar 'modeller/optimized' ve 'goruntuler/hiper_optimizasyon' klasörlerine kaydedildi.") 