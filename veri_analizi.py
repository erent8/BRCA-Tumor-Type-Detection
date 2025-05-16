import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_breast_cancer
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import os

# Çıktı klasörünü oluştur
os.makedirs('goruntuler', exist_ok=True)

# Veri setini yükle
cancer = load_breast_cancer()
X = pd.DataFrame(cancer.data, columns=cancer.feature_names)
y = pd.Series(cancer.target, name='target')

# Sınıf etiketlerini ekleyelim (0: malignant, 1: benign)
df = X.copy()
df['diagnosis'] = y
df['diagnosis_text'] = df['diagnosis'].map({0: 'Malignant (Kötü Huylu)', 1: 'Benign (İyi Huylu)'})

print(f"Veri seti boyutu: {df.shape}")
print(f"Hedef dağılımı: \n{df['diagnosis'].value_counts()}")
print(f"Öznitelikler: {cancer.feature_names}")

# 1. Sınıf Dağılımı Görselleştirmesi
plt.figure(figsize=(10, 6))
sns.countplot(x='diagnosis_text', data=df)
plt.title('Kanser Türü Dağılımı', fontsize=15)
plt.xlabel('Tanı')
plt.ylabel('Hasta Sayısı')
plt.savefig('goruntuler/sinif_dagilimi.png', bbox_inches='tight')
plt.close()

# 2. Öznitelik Korelasyon Analizi
plt.figure(figsize=(20, 16))
correlation_matrix = df.drop(['diagnosis_text'], axis=1).corr()
sns.heatmap(correlation_matrix, annot=False, cmap='coolwarm', linewidths=0.5)
plt.title('Öznitelikler Arasındaki Korelasyon Matrisi', fontsize=15)
plt.savefig('goruntuler/korelasyon_matrisi.png', bbox_inches='tight')
plt.close()

# Özniteliklerin alt gruplarını tanımlayalım (daha organize görselleştirme için)
feature_groups = {
    'mean_features': [col for col in X.columns if col.startswith('mean')],
    'error_features': [col for col in X.columns if col.startswith('error')],
    'worst_features': [col for col in X.columns if col.startswith('worst')]
}

# 3. Her öznitelik grubu için dağılım plotları
for group_name, features in feature_groups.items():
    # Grup içindeki öznitelikler için dağılım grafikleri
    plt.figure(figsize=(20, 15))
    for i, feature in enumerate(features, 1):
        plt.subplot(3, 4, i)
        sns.histplot(data=df, x=feature, hue='diagnosis_text', kde=True, palette=['red', 'green'])
        plt.title(f'{feature}')
        plt.tight_layout()
    plt.savefig(f'goruntuler/{group_name}_dagilim.png', bbox_inches='tight')
    plt.close()

# 4. PCA analizi ve görselleştirme
# Verileri standartlaştırma
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# PCA uygulama
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)
pca_df = pd.DataFrame(X_pca, columns=['PC1', 'PC2'])
pca_df['diagnosis'] = y
pca_df['diagnosis_text'] = pca_df['diagnosis'].map({0: 'Malignant (Kötü Huylu)', 1: 'Benign (İyi Huylu)'})

# PCA sonuçlarını görselleştirme
plt.figure(figsize=(12, 8))
sns.scatterplot(x='PC1', y='PC2', hue='diagnosis_text', data=pca_df, palette=['red', 'green'], s=100, alpha=0.7)
plt.title('Meme Kanseri PCA Analizi (2 Bileşen)', fontsize=15)
plt.xlabel(f'Birinci Ana Bileşen (Varyans: {pca.explained_variance_ratio_[0]:.2%})')
plt.ylabel(f'İkinci Ana Bileşen (Varyans: {pca.explained_variance_ratio_[1]:.2%})')
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend(title='Tanı')
plt.savefig('goruntuler/pca_analizi.png', bbox_inches='tight')
plt.close()

# 5. Öznitelik önemliliği (Basit bir yaklaşım: sınıflar arası ortalama farkı)
feature_importance = pd.DataFrame(columns=['feature', 'importance'])

for column in X.columns:
    malignant_mean = df.loc[df['diagnosis'] == 0, column].mean()
    benign_mean = df.loc[df['diagnosis'] == 1, column].mean()
    importance = abs(malignant_mean - benign_mean)
    feature_importance = pd.concat([feature_importance, pd.DataFrame({'feature': [column], 'importance': [importance]})])

feature_importance = feature_importance.sort_values('importance', ascending=False).reset_index(drop=True)

# Top 10 öznitelikleri görselleştirme
plt.figure(figsize=(12, 8))
sns.barplot(x='importance', y='feature', data=feature_importance.head(10))
plt.title('En Önemli 10 Öznitelik (Sınıf Ortalamaları Arasındaki Farka Göre)', fontsize=15)
plt.xlabel('Önem Derecesi (Mutlak Fark)')
plt.ylabel('Öznitelik')
plt.tight_layout()
plt.savefig('goruntuler/oznitelik_onem.png', bbox_inches='tight')
plt.close()

# 6. Box plot - En önemli 5 özellik için sınıflara göre dağılım
top_5_features = feature_importance.head(5)['feature'].values

plt.figure(figsize=(15, 10))
for i, feature in enumerate(top_5_features, 1):
    plt.subplot(2, 3, i)
    sns.boxplot(x='diagnosis_text', y=feature, data=df, palette=['red', 'green'])
    plt.title(f'{feature}')
    plt.xticks(rotation=45)
    plt.tight_layout()
plt.savefig('goruntuler/top5_boxplot.png', bbox_inches='tight')
plt.close()

# 7. Violin plot - En önemli 5 özellik için sınıflara göre dağılım
plt.figure(figsize=(15, 10))
for i, feature in enumerate(top_5_features, 1):
    plt.subplot(2, 3, i)
    sns.violinplot(x='diagnosis_text', y=feature, data=df, palette=['red', 'green'], inner='quartile')
    plt.title(f'{feature}')
    plt.xticks(rotation=45)
    plt.tight_layout()
plt.savefig('goruntuler/top5_violinplot.png', bbox_inches='tight')
plt.close()

print("Veri analizi tamamlandı. Görselleştirmeler 'goruntuler' klasörüne kaydedildi.") 