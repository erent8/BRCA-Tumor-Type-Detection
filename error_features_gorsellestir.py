import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import os
from sklearn.datasets import load_breast_cancer

# Veri setini yükle
data = load_breast_cancer()
df = pd.DataFrame(data.data, columns=data.feature_names)
df['diagnosis'] = data.target

# Error features'ları seç
error_features = [col for col in df.columns if 'error' in col]

# Görsel klasörünü kontrol et/oluştur
os.makedirs('goruntuler', exist_ok=True)

# Kaç sütun ve satır kullanılacak
n_cols = 3
n_rows = int(np.ceil(len(error_features) / n_cols))

# Figürü oluştur
plt.figure(figsize=(20, 15))

# Her bir "error" özniteliği için histogram çiz
for i, feature in enumerate(error_features):
    plt.subplot(n_rows, n_cols, i + 1)
    
    # İyi huylu (benign) ve kötü huylu (malign) ayrı ayrı çiz
    sns.histplot(df[df['diagnosis'] == 1][feature], color='green', alpha=0.5, label='Benign (İyi Huylu)')
    sns.histplot(df[df['diagnosis'] == 0][feature], color='red', alpha=0.5, label='Malign (Kötü Huylu)')
    
    plt.title(f'{feature} Dağılımı')
    plt.xlabel(feature)
    plt.ylabel('Frekans')
    plt.legend()

plt.tight_layout()
plt.suptitle('Error Öznitelikleri Dağılımları', fontsize=20)
plt.subplots_adjust(top=0.95)

# Görüntüyü kaydet
plt.savefig('goruntuler/error_features_dagilim.png', dpi=100, bbox_inches='tight')
print(f"Görsel kaydedildi: goruntuler/error_features_dagilim.png")

# Violin Plot - Alternatif görselleştirme
plt.figure(figsize=(20, 15))

for i, feature in enumerate(error_features):
    plt.subplot(n_rows, n_cols, i + 1)
    
    # Violin plot çiz - güncel seaborn sürümü için uyarlanmış
    ax = sns.violinplot(x='diagnosis', y=feature, data=df, 
                hue='diagnosis',  # hue parametresi eklendi
                palette=['red', 'green'],  # sıralı palette formatı
                inner='quartile',
                legend=False)  # legend'ı kapatıyoruz
    
    # x eksenindeki etiketleri değiştir
    ax.set_xticklabels(['Malign (Kötü Huylu)', 'Benign (İyi Huylu)'])
    
    plt.title(f'{feature} Violin Plot')
    
plt.tight_layout()
plt.suptitle('Error Öznitelikleri Violin Plot', fontsize=20)
plt.subplots_adjust(top=0.95)

# Görüntüyü kaydet
plt.savefig('goruntuler/error_features_violin.png', dpi=100, bbox_inches='tight')
print(f"Görsel kaydedildi: goruntuler/error_features_violin.png")

print("Tamamlandı!") 