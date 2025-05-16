import warnings 
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler as SC
from sklearn.model_selection import train_test_split
from sklearn.metrics import ConfusionMatrixDisplay, classification_report, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.datasets import load_breast_cancer

# Çıktıları bir dosyaya yazdıralım - UTF-8 kodlaması ekledik
with open('output.txt', 'w', encoding='utf-8') as f:
    # Scikit-learn'den meme kanseri veri setini yükle
    cancer = load_breast_cancer()
    X = pd.DataFrame(cancer.data, columns=cancer.feature_names)
    y = pd.Series(cancer.target, name='target')

    # Veri setini inceleyelim
    f.write(f"Veri seti şekli: {X.shape}\n")
    f.write(f"Hedef değişken:\n{y.value_counts()}\n")
    f.write(f"Sınıf etiketleri: {cancer.target_names}\n")

    # Veri setinin ilk 5 satırını yazdır
    f.write("\nİlk 5 satır:\n")
    f.write(f"{X.head().to_string()}\n")

    # Veri setini eğitim ve test setlerine ayır
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    f.write(f"\nEğitim seti boyutu: {X_train.shape}\n")
    f.write(f"Test seti boyutu: {X_test.shape}\n")

    # Verileri ölçeklendir
    scaler = SC()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Lojistik Regresyon modeli
    lr = LogisticRegression(max_iter=1000)
    lr.fit(X_train_scaled, y_train)
    lr_pred = lr.predict(X_test_scaled)
    
    # Modelin performansı
    f.write("\nLojistik Regresyon Model Performansı:\n")
    f.write(f"Doğruluk: {lr.score(X_test_scaled, y_test):.4f}\n")
    f.write(f"Sınıflandırma Raporu:\n{classification_report(y_test, lr_pred)}\n")

    # SVM modeli
    svm = SVC(kernel='linear')
    svm.fit(X_train_scaled, y_train)
    svm_pred = svm.predict(X_test_scaled)
    
    # Modelin performansı
    f.write("\nSVM Model Performansı:\n")
    f.write(f"Doğruluk: {svm.score(X_test_scaled, y_test):.4f}\n")
    f.write(f"Sınıflandırma Raporu:\n{classification_report(y_test, svm_pred)}\n")

print("İşlem tamamlandı! Sonuçlar 'output.txt' dosyasına kaydedildi.")

