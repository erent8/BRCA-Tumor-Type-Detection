import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import plotly.express as px
import plotly.graph_objects as go
from PIL import Image

# Sayfa yapılandırması
st.set_page_config(
    page_title="Meme Kanseri Tespiti",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Başlık ve açıklama
st.title("Meme Kanseri Tespiti Projesi")
st.markdown("""
Bu uygulama, makine öğrenmesi modelleri kullanarak meme kanseri teşhisine yardımcı olmak için geliştirilmiştir.
Wisconsin Meme Kanseri veri seti üzerinde eğitilmiş modeller kullanılarak yüksek doğrulukta tahminler yapabilmektedir.
""")

# Sidebar oluşturma
st.sidebar.title("Navigasyon")
sayfalar = ["Ana Sayfa", "Veri Analizi", "Model Karşılaştırma", 
            "Model Yorumlama", "Tahmin", "Ensemble Modeller", "Öznitelik Seçimi"]
sayfa = st.sidebar.selectbox("Sayfa Seçin", sayfalar)

# Modelleri yükleme
@st.cache_resource
def modelleri_yukle():
    modeller = {}
    model_klasoru = "modeller"
    
    if os.path.exists(model_klasoru):
        for model_dosyasi in os.listdir(model_klasoru):
            if model_dosyasi.endswith(".pkl"):
                model_adi = os.path.splitext(model_dosyasi)[0]
                model_yolu = os.path.join(model_klasoru, model_dosyasi)
                modeller[model_adi] = joblib.load(model_yolu)
    
    return modeller

# Veri setini yükleme
@st.cache_data
def veri_setini_yukle():
    from sklearn.datasets import load_breast_cancer
    data = load_breast_cancer()
    df = pd.DataFrame(data.data, columns=data.feature_names)
    df['diagnosis'] = data.target
    return df, data

# Görüntüleri yükleme
@st.cache_data
def goruntuleri_yukle():
    goruntu_klasoru = "goruntuler"
    goruntu_tipleri = {
        "veri_analizi": [],
        "model_karsilastirma": [],
        "model_yorumlama": [],
        "ensemble": [],
        "oznitelik_secimi": []
    }
    
    if os.path.exists(goruntu_klasoru):
        # Klasördeki görüntüleri tara ve kategorize et
        for klasor, _, dosyalar in os.walk(goruntu_klasoru):
            for dosya in dosyalar:
                if dosya.endswith(('.png', '.jpg', '.jpeg')):
                    dosya_yolu = os.path.join(klasor, dosya)
                    
                    # Görüntü tipine göre sınıflandır
                    if "veri_analizi" in dosya_yolu or "korelasyon" in dosya_yolu:
                        goruntu_tipleri["veri_analizi"].append(dosya_yolu)
                    elif "model_karsilastirma" in dosya_yolu or "roc_" in dosya_yolu.lower() or "confusion_matrix" in dosya_yolu:
                        goruntu_tipleri["model_karsilastirma"].append(dosya_yolu)
                    elif "model_yorumlama" in dosya_yolu or "oznitelik_onem" in dosya_yolu:
                        goruntu_tipleri["model_yorumlama"].append(dosya_yolu)
                    elif "ensemble" in dosya_yolu:
                        goruntu_tipleri["ensemble"].append(dosya_yolu)
                    elif "oznitelik_secimi" in dosya_yolu:
                        goruntu_tipleri["oznitelik_secimi"].append(dosya_yolu)
    
    return goruntu_tipleri

# Veri ve modelleri yükleme
df, data = veri_setini_yukle()
modeller = modelleri_yukle()
goruntu_tipleri = goruntuleri_yukle()

# Ana Sayfa
if sayfa == "Ana Sayfa":
    st.header("Proje Hakkında")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("""
        ### Proje Amacı
        Bu projede, makine öğrenmesi teknikleri kullanarak meme kanseri teşhisi için bir model geliştirilmiştir. 
        Proje, kanser teşhisinde destek amaçlı bir araç olarak kullanılabilir.
        
        ### Kullanılan Veri Seti
        Wisconsin Meme Kanseri Veri Seti kullanılmıştır. Bu veri seti, meme kanseri hücrelerine ait 30 farklı öznitelik içermektedir.
        
        ### Kullanılan Modeller
        - Lojistik Regresyon
        - Destek Vektör Makineleri (SVM)
        - Karar Ağaçları
        - Rastgele Orman
        - Gradient Boosting
        - K-En Yakın Komşu (KNN)
        - Yapay Sinir Ağları
        """)
    
    with col2:
        st.markdown("""
        ### Nasıl Kullanılır?
        1. **Veri Analizi** sayfasında veri seti hakkında detaylı bilgi edinebilirsiniz.
        2. **Model Karşılaştırma** sayfasında farklı modellerin performanslarını inceleyebilirsiniz.
        3. **Tahmin** sayfasında kendi girdiğiniz değerlerle tahmin yapabilirsiniz.
        
        ### En İyi Performans
        Hiperparametre optimizasyonu sonucunda:
        - **Lojistik Regresyon**: Test doğruluğu: 0.9912, En iyi CV skoru: 0.978
        - **SVM**: Test doğruluğu: 0.9825, En iyi CV skoru: 0.9758
        - **Rastgele Orman**: Test doğruluğu: 0.9737, En iyi CV skoru: 0.9648
        """)
    
    st.markdown("---")
    st.subheader("Veri Seti Detayları")
    
    # Veri seti görüntüleme seçenekleri
    if st.checkbox("Veri setini göster"):
        # Tüm veri setini göstermek için seçenekler
        gosterim_secenekleri = st.radio(
            "Görüntüleme Seçenekleri:", 
            ["İlk 5 satır", "Tüm Veri Seti", "Özet İstatistikler"],
            horizontal=True
        )
        
        if gosterim_secenekleri == "İlk 5 satır":
            st.write(df.head())
        elif gosterim_secenekleri == "Tüm Veri Seti":
            # Veri setinin boyutunu göster
            st.write(f"Satır sayısı: {df.shape[0]}, Sütun sayısı: {df.shape[1]}")
            
            # Tüm veri setini göstermek için
            st.dataframe(df, height=600, use_container_width=True)
            
            # Sayfalama için alternatif
            sayfa_boyutu = st.slider("Sayfa başına satır sayısı", 10, 100, 50)
            sayfa_numarasi = st.number_input("Sayfa numarası", 1, int(np.ceil(df.shape[0]/sayfa_boyutu)), 1)
            
            baslangic = (sayfa_numarasi - 1) * sayfa_boyutu
            bitis = min(baslangic + sayfa_boyutu, df.shape[0])
            
            st.write(f"Gösterilen satırlar: {baslangic+1} - {bitis} / {df.shape[0]}")
            st.dataframe(df.iloc[baslangic:bitis], use_container_width=True)
            
        elif gosterim_secenekleri == "Özet İstatistikler":
            st.write("Öznitelik istatistikleri:")
            st.write(df.describe())

# Veri Analizi Sayfası
elif sayfa == "Veri Analizi":
    st.header("Veri Analizi ve Görselleştirme")
    
    tab1, tab2, tab3 = st.tabs(["Veri Dağılımları", "Korelasyon Analizi", "PCA Analizi"])
    
    with tab1:
        st.subheader("Öznitelik Dağılımları")
        goruntuleri = [img for img in goruntu_tipleri["veri_analizi"] if any(k in img.lower() for k in ["dagilim", "distribution", "histogram", "sinif", "boxplot", "violinplot"])]
        
        if goruntuleri:
            for goruntu in goruntuleri:
                st.image(goruntu, use_container_width=True, caption=os.path.basename(goruntu))
        else:
            st.warning("Veri dağılım görüntüleri bulunamadı. Lütfen 'veri_analizi.py' scriptini çalıştırarak görüntüleri oluşturun.")
    
    with tab2:
        st.subheader("Korelasyon Analizi")
        goruntuleri = [img for img in goruntu_tipleri["veri_analizi"] if any(k in img.lower() for k in ["korelasyon", "correlation", "heatmap"])]
        
        if goruntuleri:
            for goruntu in goruntuleri:
                st.image(goruntu, use_container_width=True, caption=os.path.basename(goruntu))
        
        # İnteraktif korelasyon matrisi
        st.subheader("İnteraktif Korelasyon Matrisi")
        correlation = df.corr()
        fig = px.imshow(correlation, text_auto=True, aspect="auto")
        st.plotly_chart(fig, use_container_width=True)
    
    with tab3:
        st.subheader("PCA Analizi")
        goruntuleri = [img for img in goruntu_tipleri["veri_analizi"] if "pca" in img.lower()]
        
        if goruntuleri:
            for goruntu in goruntuleri:
                st.image(goruntu, use_container_width=True, caption=os.path.basename(goruntu))
            
        # İnteraktif PCA
        st.subheader("İnteraktif PCA Grafiği")
        
        X = df.drop('diagnosis', axis=1)
        y = df['diagnosis']
        
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        pca = PCA(n_components=3)
        X_pca = pca.fit_transform(X_scaled)
        
        pca_df = pd.DataFrame(
            data=X_pca, 
            columns=['PC1', 'PC2', 'PC3']
        )
        pca_df['diagnosis'] = y.map({0: 'Malign (Kötü Huylu)', 1: 'Benign (İyi Huylu)'})
        
        fig = px.scatter_3d(
            pca_df, x='PC1', y='PC2', z='PC3', color='diagnosis',
            opacity=0.7, color_discrete_map={'Malign (Kötü Huylu)': 'red', 'Benign (İyi Huylu)': 'blue'}
        )
        st.plotly_chart(fig, use_container_width=True)
        
        st.write(f"Açıklanan Varyans Oranı: {pca.explained_variance_ratio_}")
        
        # Açıklanan varyans oranı grafiği
        explained_var = np.append(pca.explained_variance_ratio_, 
                                 np.sum(pca.explained_variance_ratio_[3:]))
        labels = ['PC1', 'PC2', 'PC3', 'Diğer PC\'ler']
        
        fig = px.pie(values=explained_var, names=labels, title='PCA Bileşenlerinin Açıkladığı Varyans')
        st.plotly_chart(fig, use_container_width=True)

# Model Karşılaştırma Sayfası
elif sayfa == "Model Karşılaştırma":
    st.header("Model Karşılaştırma")
    
    tab1, tab2, tab3, tab4 = st.tabs(["Doğruluk Karşılaştırması", "ROC Eğrileri", "Confusion Matrix", "Precision-Recall Eğrileri"])
    
    with tab1:
        st.subheader("Model Doğruluk Karşılaştırması")
        goruntuleri = [img for img in goruntu_tipleri["model_karsilastirma"] if any(k in img.lower() for k in ["accuracy", "comparison", "dogruluk", "karsilastirma"])]
        
        if goruntuleri:
            for goruntu in goruntuleri:
                st.image(goruntu, use_container_width=True, caption=os.path.basename(goruntu))
        else:
            st.warning("Doğruluk karşılaştırma görselleri bulunamadı.")
            
        # Model karşılaştırma tablosu
        model_sonuclari = {
            "Lojistik Regresyon": {"Test Doğruluğu": 0.9912, "CV Skoru": 0.9780},
            "SVM": {"Test Doğruluğu": 0.9825, "CV Skoru": 0.9758},
            "Rastgele Orman": {"Test Doğruluğu": 0.9737, "CV Skoru": 0.9648},
            "Gradient Boosting": {"Test Doğruluğu": 0.9561, "CV Skoru": 0.9626},
            "Yapay Sinir Ağı": {"Test Doğruluğu": 0.9737, "CV Skoru": 0.9780}
        }
        
        sonuc_df = pd.DataFrame.from_dict(model_sonuclari, orient='index')
        st.table(sonuc_df)
        
        # İnteraktif model karşılaştırma grafiği
        fig = go.Figure()
        
        fig.add_trace(go.Bar(
            x=list(model_sonuclari.keys()),
            y=[model["Test Doğruluğu"] for model in model_sonuclari.values()],
            name='Test Doğruluğu',
            marker_color='indianred'
        ))
        
        fig.add_trace(go.Bar(
            x=list(model_sonuclari.keys()),
            y=[model["CV Skoru"] for model in model_sonuclari.values()],
            name='CV Skoru',
            marker_color='lightsalmon'
        ))
        
        fig.update_layout(
            title='Model Performans Karşılaştırması',
            xaxis_tickfont_size=14,
            yaxis=dict(
                title=dict(
                    text='Skor',
                    font=dict(size=16)
                ),
                tickfont=dict(size=14),
                range=[0.90, 1.0]
            ),
            legend=dict(
                x=0,
                y=1.0,
                bgcolor='rgba(255, 255, 255, 0)',
                bordercolor='rgba(255, 255, 255, 0)'
            ),
            barmode='group',
            bargap=0.15,
            bargroupgap=0.1
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        st.subheader("ROC Eğrileri")
        goruntuleri = [img for img in goruntu_tipleri["model_karsilastirma"] if "roc" in img.lower()]
        
        if goruntuleri:
            st.subheader("Tüm Modeller ROC Eğrisi")
            tum_modeller_roc = [img for img in goruntuleri if "tum" in img.lower() or "all" in img.lower()]
            if tum_modeller_roc:
                st.image(tum_modeller_roc[0], use_container_width=True, caption=os.path.basename(tum_modeller_roc[0]))
            
            st.subheader("Modellere Göre ROC Eğrileri")
            bireysel_roc = [img for img in goruntuleri if not ("tum" in img.lower() or "all" in img.lower())]
            
            # ROC eğrilerini iki sütunda göster
            cols = st.columns(2)
            for i, goruntu in enumerate(bireysel_roc):
                with cols[i % 2]:
                    st.image(goruntu, use_container_width=True, caption=os.path.basename(goruntu))
        else:
            st.warning("ROC eğrisi görselleri bulunamadı.")
    
    with tab3:
        st.subheader("Confusion Matrix")
        goruntuleri = [img for img in goruntu_tipleri["model_karsilastirma"] if "cm" in img.lower() or "confusion" in img.lower()]
        
        if goruntuleri:
            # Confusion Matrix'leri iki sütunda göster
            cols = st.columns(2)
            for i, goruntu in enumerate(goruntuleri):
                with cols[i % 2]:
                    st.image(goruntu, use_container_width=True, caption=os.path.basename(goruntu))
        else:
            st.warning("Confusion Matrix görselleri bulunamadı.")
            
    with tab4:
        st.subheader("Precision-Recall Eğrileri")
        pr_goruntuleri = [img for img in goruntu_tipleri["model_karsilastirma"] if "precision" in img.lower() or "recall" in img.lower()]
        
        if pr_goruntuleri:
            st.subheader("Tüm Modeller Precision-Recall Eğrisi")
            tum_modeller_pr = [img for img in pr_goruntuleri if "tum_modeller" in img.lower()]
            if tum_modeller_pr:
                st.image(tum_modeller_pr[0], use_container_width=True, caption=os.path.basename(tum_modeller_pr[0]))
            
            karsilastirma_pr = [img for img in pr_goruntuleri if "karsilastirma" in img.lower()]
            if karsilastirma_pr:
                st.image(karsilastirma_pr[0], use_container_width=True, caption="Precision, Recall ve F1 Karşılaştırması")
            
            st.subheader("Modellere Göre Precision-Recall Eğrileri")
            bireysel_pr = [img for img in pr_goruntuleri if not ("tum_modeller" in img.lower() or "karsilastirma" in img.lower())]
            
            # Precision-Recall eğrilerini iki sütunda göster
            cols = st.columns(2)
            for i, goruntu in enumerate(bireysel_pr):
                with cols[i % 2]:
                    st.image(goruntu, use_container_width=True, caption=os.path.basename(goruntu))
        else:
            st.warning("Precision-Recall eğrisi görselleri bulunamadı.")
            
    st.subheader("Önemli Öznitelikler")
    onem_goruntuleri = [img for img in goruntu_tipleri["veri_analizi"] if any(k in img.lower() for k in ["feature_importance", "importance", "öznitelik", "onem"])]
    
    if onem_goruntuleri:
        for goruntu in onem_goruntuleri:
            st.image(goruntu, use_container_width=True, caption=os.path.basename(goruntu))
    else:
        st.warning("Öznitelik önem analizi görselleri bulunamadı.")

# Model Yorumlama Sayfası
elif sayfa == "Model Yorumlama":
    st.header("Model Yorumlama")
    
    st.subheader("Öznitelik Önem Analizi")
    
    # Görsel ve sekmeler
    tab1, tab2 = st.tabs(["Dahili Önem", "Permütasyon Önem"])
    
    with tab1:
        st.info("Dahili önem, modelin eğitim sürecinde özniteliklere atanan ağırlıkları gösterir. Bu değerler, modelin her özniteliğe verdiği önemi yansıtır.")
        
        # Dahili önem görselleri
        dahili_onem_goruntuleri = [img for img in goruntu_tipleri["model_yorumlama"] if "feature_importance" in img.lower() and not "permutation" in img.lower()]
        
        if dahili_onem_goruntuleri:
            # İlk olarak karşılaştırma görselini göster
            karsilastirma_goruntu = [img for img in dahili_onem_goruntuleri if "karsilastirma_onem" in img.lower()]
            if karsilastirma_goruntu:
                st.image(karsilastirma_goruntu[0], use_container_width=True, caption="Modeller Arası Öznitelik Önem Karşılaştırması")
            
            # Sonra her modelin kendi görselini göster
            bireysel_goruntuleri = [img for img in dahili_onem_goruntuleri if not "karsilastirma" in img.lower()]
            
            for goruntu in bireysel_goruntuleri:
                st.image(goruntu, use_container_width=True, caption=os.path.basename(goruntu))
        else:
            st.warning("Dahili önem görselleri bulunamadı.")
    
    with tab2:
        st.info("Permütasyon önem analizi, bir özniteliğin değerlerini karıştırarak modelin performans değişimini ölçer. Eğer bir özniteliğin değerleri karıştırıldığında performans çok düşerse, o öznitelik önemlidir.")
        
        # Permütasyon önem görselleri
        permutasyon_onem_goruntuleri = [img for img in goruntu_tipleri["model_yorumlama"] if "permutation" in img.lower()]
        
        if permutasyon_onem_goruntuleri:
            # İlk olarak karşılaştırma görselini göster
            perm_karsilastirma = [img for img in permutasyon_onem_goruntuleri if "karsilastirma" in img.lower()]
            if perm_karsilastirma:
                st.image(perm_karsilastirma[0], use_container_width=True, caption="Modeller Arası Permütasyon Önem Karşılaştırması")
            
            # Sonra her modelin kendi görselini göster
            perm_bireysel = [img for img in permutasyon_onem_goruntuleri if not "karsilastirma" in img.lower()]
            
            for goruntu in perm_bireysel:
                st.image(goruntu, use_container_width=True, caption=os.path.basename(goruntu))
        else:
            st.warning("Permütasyon önem görselleri bulunamadı.")
            
    st.subheader("Öznitelik Analizi Yorumu")
    st.markdown("""
    ### Öznitelik Önemlerinin Analizi:
    
    **En Önemli Öznitelikler:**
    
    1. **mean concave points (Ortalama konkav noktalar)**: Hücre çekirdeğinin konturu üzerindeki içbükey bölgelerin ortalama sayısı. Bu öznitelik, kötü huylu tümörlerin teşhisinde en ayırt edici faktörlerden biridir.
    
    2. **worst concave points (En kötü konkav noktalar)**: En kötü hücre çekirdeğindeki içbükey bölgelerin sayısı. Bu da kötü huylu tümörlerin belirlenmesinde kritik öneme sahip.
    
    3. **worst radius (En kötü yarıçap)**: En kötü hücre çekirdeğinin yarıçapı, tümör hücrelerinin boyutundaki anormallikleri işaret eder.
    
    4. **worst perimeter (En kötü çevre)**: En kötü hücre çekirdeğinin çevresi, tümör hücrelerinin şekil bozukluklarını gösterir.
    
    5. **mean texture (Ortalama doku)**: Hücre çekirdeğinin gri seviye değerlerindeki standart sapma, doku düzensizliğini gösterir.
    
    **Klinik Önemi:**
    
    Bu analiz, meme kanseri tespitinde en kritik faktörlerin hücre çekirdeğinin şekil özellikleri olduğunu göstermektedir. Özellikle içbükey noktalar (concave points), hücre zarının içe doğru girinti yapan bölümleri, kötü huylu tümörlerin karakteristik bir özelliğidir.
    
    Modeller arasındaki tutarlılık, bu özniteliklerin gerçekten önemli olduğunu doğrulamaktadır. Doktorlar, bu faktörlere özellikle dikkat ederek teşhis sürecini geliştirebilirler.
    """)

# Tahmin Sayfası
elif sayfa == "Tahmin":
    st.header("Kanser Tahmini")
    
    st.markdown("""
    Aşağıdaki formu doldurarak yeni bir örnek için kanser tahmini yapabilirsiniz.
    **Not:** Gerçek tıbbi teşhis yerine geçmez, sadece bilgi amaçlıdır.
    """)
    
    # Kategorilere göre öznitelikleri gruplayalım
    ozellik_gruplari = {
        "Radius (Yarıçap)": ["mean radius", "radius error", "worst radius"],
        "Texture (Doku)": ["mean texture", "texture error", "worst texture"],
        "Perimeter (Çevre)": ["mean perimeter", "perimeter error", "worst perimeter"],
        "Area (Alan)": ["mean area", "area error", "worst area"],
        "Smoothness (Pürüzsüzlük)": ["mean smoothness", "smoothness error", "worst smoothness"],
        "Compactness (Yoğunluk)": ["mean compactness", "compactness error", "worst compactness"],
        "Concavity (İçbükeylik)": ["mean concavity", "concavity error", "worst concavity"],
        "Concave Points (İçbükey Noktalar)": ["mean concave points", "concave points error", "worst concave points"],
        "Symmetry (Simetri)": ["mean symmetry", "symmetry error", "worst symmetry"],
        "Fractal Dimension (Fraktal Boyut)": ["mean fractal dimension", "fractal dimension error", "worst fractal dimension"]
    }
    
    # Varsayılan değerler (ortalama değerler)
    default_values = df.drop('diagnosis', axis=1).mean().to_dict()
    
    # Tahmin için kullanılacak değerler
    girilen_degerler = {}
    
    # Modeli seçme
    model_secimi = st.selectbox(
        "Tahmin için model seçin:",
        ["Lojistik Regresyon", "SVM", "Rastgele Orman", "Gradient Boosting", "Yapay Sinir Ağı"]
    )
    
    # Grupları sekmelerde gösterelim
    grup_sekmeler = st.tabs(list(ozellik_gruplari.keys()))
    
    for i, (grup_adi, ozellikler) in enumerate(ozellik_gruplari.items()):
        with grup_sekmeler[i]:
            st.subheader(f"{grup_adi} Özellikleri")
            
            for ozellik in ozellikler:
                girilen_degerler[ozellik] = st.slider(
                    f"{ozellik}",
                    float(df[ozellik].min()),
                    float(df[ozellik].max()),
                    float(default_values[ozellik]),
                    step=0.1,
                    format="%.2f"
                )
    
    # Tahmin butonu
    if st.button("Tahmin Yap"):
        # Girilen değerleri vektör haline getirme
        girdi_vektor = pd.DataFrame([girilen_degerler])
        
        # Doğru model ismini map etme
        model_map = {
            "Lojistik Regresyon": "logistic_regression_optimized",
            "SVM": "svm_optimized",
            "Rastgele Orman": "random_forest_optimized",
            "Gradient Boosting": "gradient_boosting_optimized",
            "Yapay Sinir Ağı": "neural_network_optimized"
        }
        
        # Varsayılan model (eğer seçilen model bulunamazsa)
        varsayilan_model = next(iter(modeller.values()))
        
        # Seçilen modeli al
        secilen_model_ismi = model_map.get(model_secimi)
        model = modeller.get(secilen_model_ismi, varsayilan_model)
        
        # Tahmin yap
        tahmin = model.predict(girdi_vektor)
        tahmin_olasiligi = model.predict_proba(girdi_vektor)
        
        # Sonuçları göster
        sonuc_container = st.container()
        with sonuc_container:
            if tahmin[0] == 1:
                st.success("**Tahmin: Benign (İyi Huylu)**")
                st.markdown(f"Olasılık: **{tahmin_olasiligi[0][1]:.2%}**")
                st.markdown("Bu tahmin, incelenen hücre örneğinin iyi huylu (benign) olduğunu göstermektedir.")
            else:
                st.error("**Tahmin: Malign (Kötü Huylu)**")
                st.markdown(f"Olasılık: **{tahmin_olasiligi[0][0]:.2%}**")
                st.markdown("Bu tahmin, incelenen hücre örneğinin kötü huylu (malign) olduğunu göstermektedir.")
            
            st.warning("**Not:** Bu sonuç sadece bilgi amaçlıdır ve gerçek tıbbi teşhis yerine geçmez. Lütfen gerçek teşhis için bir doktora başvurun.")
            
            # İnteraktif donut chart
            labels = ["Malign (Kötü Huylu)", "Benign (İyi Huylu)"]
            values = [tahmin_olasiligi[0][0], tahmin_olasiligi[0][1]]
            
            fig = go.Figure(data=[go.Pie(
                labels=labels,
                values=values,
                hole=.4,
                marker_colors=['#FF6B6B', '#4CAF50']
            )])
            
            fig.update_layout(
                title_text="Tahmin Olasılığı",
                annotations=[dict(text=f"{max(values):.1%}", x=0.5, y=0.5, font_size=20, showarrow=False)]
            )
            
            st.plotly_chart(fig, use_container_width=True)

# Ensemble Modeller Sayfası
elif sayfa == "Ensemble Modeller":
    st.title("Ensemble Öğrenme Teknikleri")
    
    st.markdown("""
    ## Ensemble (Topluluk) Modelleri Nedir?
    
    Ensemble (topluluk) öğrenme, birden fazla modelin bir arada kullanılarak daha iyi performans elde edilmesini sağlayan bir tekniktir. 
    Bu projede aşağıdaki ensemble teknikleri uygulanmıştır:
    
    1. **Hard Voting:** Her model bir sınıf tahmini yapar ve en çok oy alan sınıf seçilir.
    2. **Soft Voting:** Her model olasılık değerleri üretir ve bu olasılıkların ortalaması alınarak final tahmin yapılır.
    3. **Stacking:** Birinci seviye modellerin tahminleri, ikinci seviye bir meta-model tarafından birleştirilir.
    4. **Advanced Stacking:** Stacking'in daha gelişmiş bir versiyonu, çapraz doğrulama ile tahminlerin kalitesini artırır.
    """)
    
    # Görsel seçimi
    ensemble_gorselleri = goruntu_tipleri["ensemble"]
    
    if len(ensemble_gorselleri) == 0:
        st.warning("Ensemble model görselleri bulunamadı. Lütfen önce 'ensemble_ogrenme.py' scriptini çalıştırın.")
    else:
        # Görsel kategorileri
        gorsel_kategorileri = {
            "Doğruluk Karşılaştırması": "accuracy_karsilastirma.png",
            "ROC Eğrileri": "roc_karsilastirma.png",
            "Precision-Recall Eğrileri": "precision_recall_karsilastirma.png",
            "Tüm Metrikler": "tum_metrikler_karsilastirma.png",
            "Confusion Matrisleri": "confusion_matrices.png"
        }
        
        # Sekmeler oluştur
        tabs = st.tabs(list(gorsel_kategorileri.keys()))
        
        for i, (tab_isim, gorsel_isim) in enumerate(gorsel_kategorileri.items()):
            with tabs[i]:
                # İlgili görseli bul
                for gorsel in ensemble_gorselleri:
                    if gorsel_isim in gorsel:
                        st.image(gorsel, caption=tab_isim, use_column_width=True)
                        break
        
        # Sonuçların açıklaması
        st.markdown("""
        ## Analiz Sonuçları
        
        Ensemble modellerin karşılaştırmalı analizi sonucunda:
        
        1. **Doğruluk (Accuracy):** Lojistik Regresyon bireysel model olarak en yüksek doğruluk değerine sahiptir (%99.12). Hard Voting ensemble modeli (%98.25) de iyi bir performans göstermiştir.
        
        2. **AUC Değeri:** Bireysel modeller içinde Lojistik Regresyon en yüksek AUC değerine sahipken (0.9987), ensemble modellerinden Soft Voting, Stacking ve Advanced Stacking yüksek AUC performansı (0.9971) göstermiştir.
        
        3. **F1 Skoru:** Lojistik Regresyon en yüksek F1 skoruna sahiptir (0.9930). Ensemble modellerinden Hard Voting ikinci sırada yer almaktadır (0.9861).
        
        4. **Kararlılık (Stability):** Cross-validation sonuçlarına göre Hard Voting ensemble modeli en kararlı performansa sahip modellerden biri olarak öne çıkmaktadır (standart sapma: 0.0082).
        
        5. **Genel Değerlendirme:** Bu veri seti için ensemble modeller, bireysel bazı modelleri geçemese de, genellikle daha kararlı ve güvenilir sonuçlar verme eğilimindedir. Özellikle daha karmaşık ve büyük veri setlerinde, ensemble modellerin avantajları daha belirgin olabilir.
        """)
        
        # Model sonuçları tablosu
        st.subheader("Model Performans Metrikleri")
        
        metrikler_df = pd.DataFrame({
            "Model": ["Lojistik Regresyon", "SVM", "Rastgele Orman", "Gradient Boosting", "Yapay Sinir Ağı", 
                     "Hard Voting", "Soft Voting", "Stacking", "Advanced Stacking"],
            "Accuracy": [0.9912, 0.9825, 0.9737, 0.9561, 0.9737, 0.9825, 0.9737, 0.9737, 0.9737],
            "AUC": [0.9987, 0.9974, 0.9974, 0.9938, 0.9954, 0.9767, 0.9971, 0.9971, 0.9971],
            "F1 Score": [0.9930, 0.9861, 0.9790, 0.9650, 0.9790, 0.9861, 0.9790, 0.9790, 0.9790],
            "Precision": [0.9861, 0.9726, 0.9722, 0.9583, 0.9722, 0.9726, 0.9722, 0.9722, 0.9722],
            "Recall": [1.0000, 1.0000, 0.9859, 0.9718, 0.9859, 1.0000, 0.9859, 0.9859, 0.9859],
        })
        
        st.dataframe(metrikler_df, use_container_width=True)
        
        st.markdown("""
        ## Sonuç ve Öneriler
        
        Ensemble modeller, bu özel veri seti için bireysel bazı modelleri geçemese de, genellikle daha kararlı ve güvenilir sonuçlar verme eğilimindedir. Özellikle daha karmaşık ve büyük veri setlerinde, ensemble modellerin avantajları daha belirgin olabilir.
        
        **Öneriler:**
        
        1. Daha büyük ve karmaşık veri setleri üzerinde ensemble teknikleri daha büyük bir avantaj sağlayabilir.
        2. Farklı hiperparametre kombinasyonları ile ensemble modellerin performansı daha da artırılabilir.
        3. Ağırlıklı oylama (weighted voting) gibi daha sofistike ensemble tekniklerinin denenmesi faydalı olabilir.
        4. Ensemble modellerin eğitim süresi daha uzun olduğundan, gerçek zamanlı uygulamalar için performans-doğruluk dengesi gözetilmelidir.
        """)

# Öznitelik Seçimi Sayfası
elif sayfa == "Öznitelik Seçimi":
    st.title("Öznitelik Seçimi Teknikleri")
    
    st.markdown("""
    ## Öznitelik Seçimi Nedir?
    
    Öznitelik seçimi, makine öğrenmesi modellerinin eğitiminde kullanılan özniteliklerin (değişkenlerin) 
    en önemlilerini belirleyerek boyutsallığı azaltan bir tekniktir. Bu teknik şu avantajları sağlar:
    
    1. **Daha hızlı eğitim**: Daha az sayıda değişkenle eğitim süresi kısalır.
    2. **Daha iyi genelleme**: Gürültü içeren veya alakasız öznitelikler çıkarılır.
    3. **Daha yorumlanabilir modeller**: Hangi özelliklerin gerçekten önemli olduğunu görmek, sonuçları anlamayı kolaylaştırır.
    4. **Bellek kullanımını azaltma**: Özellikle büyük veri setlerinde kaynakları daha verimli kullanmayı sağlar.
    
    Bu projede aşağıdaki öznitelik seçim teknikleri uygulanmıştır:
    """)
    
    # Teknik açıklamaları
    teknoloji_aciklamalari = {
        "ANOVA F-değeri": "İstatistiksel bir test olan ANOVA F-testi kullanarak, her bir özniteliğin hedef değişkenle olan ilişkisini değerlendirir.",
        "Mutual Information": "Öznitelik ve hedef arasındaki karşılıklı bilgiyi ölçer. Doğrusal olmayan ilişkileri de yakalayabilir.",
        "RFE (Lojistik Regresyon)": "Recursive Feature Elimination, özyinelemeli öznitelik eleme tekniğidir. Tüm özniteliklerle başlayıp en önemsizleri adım adım çıkarır.",
        "Random Forest Önem": "Random Forest modelinin öznitelik önem skorlarını kullanarak en önemli öznitelikleri seçer.",
        "Gradient Boosting Önem": "Gradient Boosting modelinin belirlediği öznitelik önem skorlarına göre seçim yapar.",
        "PCA": "Temel Bileşen Analizi, veriyi daha düşük boyutlu bir uzaya dönüştüren bir boyut indirgeme tekniğidir."
    }
    
    # Öznitelik seçim yöntemleri
    col1, col2 = st.columns([1, 1])
    
    with col1:
        for teknik, aciklama in list(teknoloji_aciklamalari.items())[:3]:
            st.markdown(f"### {teknik}")
            st.markdown(aciklama)
            
    with col2:
        for teknik, aciklama in list(teknoloji_aciklamalari.items())[3:]:
            st.markdown(f"### {teknik}")
            st.markdown(aciklama)
    
    # Görsel sonuçları
    oznitelik_secimi_gorselleri = goruntu_tipleri["oznitelik_secimi"]
    
    if len(oznitelik_secimi_gorselleri) == 0:
        st.warning("Öznitelik seçimi görselleri bulunamadı. Lütfen önce 'oznitelik_secimi.py' scriptini çalıştırın.")
    else:
        # Seçilen öznitelikler karşılaştırması
        st.header("Seçilen Öznitelikler Karşılaştırması")
        karsilastirma_gorseli = [img for img in oznitelik_secimi_gorselleri if "secilen_oznitelikler_karsilastirma" in img]
        if karsilastirma_gorseli:
            st.image(karsilastirma_gorseli[0], use_container_width=True, 
                     caption="Farklı Yöntemlerle Seçilen Özniteliklerin Karşılaştırması")
            
            st.markdown("""
            **Gözlem**: Yukarıdaki ısı haritası (heatmap) farklı öznitelik seçim yöntemlerinin hangi özniteliği seçtiğini göstermektedir.
            Mavi renkli hücreler ilgili yöntemin o özniteliği seçtiğini gösterir. Birçok yöntem tarafından ortak seçilen öznitelikler,
            genel olarak daha önemli olma eğilimindedir.
            """)
        
        # Performans karşılaştırması
        st.header("Performans Karşılaştırması")
        
        # Tüm metrikler karşılaştırması
        tum_metrikler_gorseli = [img for img in oznitelik_secimi_gorselleri if "tum_metrikler_karsilastirma" in img]
        if tum_metrikler_gorseli:
            st.image(tum_metrikler_gorseli[0], use_container_width=True,
                    caption="Öznitelik Seçim Yöntemlerinin Performans Karşılaştırması")
            
            st.markdown("""
            **Gözlem**: Yukarıdaki grafik, her bir öznitelik seçim yönteminin farklı metrikler açısından performansını göstermektedir.
            Öznitelik sayısını azaltmamıza rağmen, tüm yöntemler yüksek performans sergiliyor ve tam öznitelik setine yakın sonuçlar veriyor.
            """)
        
        # Bireysel metrik karşılaştırmaları
        col1, col2 = st.columns([1, 1])
        metrikler = ["accuracy", "auc", "f1", "precision", "recall"]
        
        for i, metrik in enumerate(metrikler):
            metrik_gorseli = [img for img in oznitelik_secimi_gorselleri if f"{metrik}_karsilastirma" in img]
            if metrik_gorseli:
                with col1 if i % 2 == 0 else col2:
                    st.image(metrik_gorseli[0], use_container_width=True,
                            caption=f"{metrik.title()} Metriği Karşılaştırması")
        
        # PCA Açıklanan Varyans
        st.header("PCA: Açıklanan Varyans")
        pca_gorseli = [img for img in oznitelik_secimi_gorselleri if "pca_variance" in img]
        if pca_gorseli:
            st.image(pca_gorseli[0], use_container_width=True,
                    caption="PCA: Açıklanan Varyans Oranı")
            
            st.markdown("""
            **Gözlem**: Bu grafik, Temel Bileşenler Analizinde (PCA) her bir bileşenin ne kadar varyans açıkladığını göstermektedir.
            İlk birkaç bileşen, veri setindeki varyansın büyük bir kısmını açıklamaktadır. 
            İlk 10 bileşen, toplam varyansın yaklaşık %95'ini açıklamaktadır. Bu, 30 öznitelikten 10'una düşüşle bile
            veri setindeki bilgilerin çoğunu koruyabildiğimizi gösterir.
            """)
        
        # Öznitelik Skorları
        st.header("Öznitelik Önem Skorları")
        
        # Sekmelerde yöntemleri göster
        skor_gorsel_kategorileri = {
            "ANOVA F-değeri": "anova_f-degeri_skorlar.png",
            "Mutual Information": "mutual_information_skorlar.png",
            "Random Forest Önem": "random_forest_onem_skorlar.png",
            "Gradient Boosting Önem": "gradient_boosting_onem_skorlar.png"
        }
        
        tabs = st.tabs(list(skor_gorsel_kategorileri.keys()))
        
        for i, (tab_isim, gorsel_isim) in enumerate(skor_gorsel_kategorileri.items()):
            with tabs[i]:
                for gorsel in oznitelik_secimi_gorselleri:
                    if gorsel_isim in gorsel.lower():
                        st.image(gorsel, use_container_width=True, 
                                caption=f"{tab_isim} ile Seçilen Özniteliklerin Skorları")
                        break
        
        # En iyi yöntem ve öznitelikler
        st.header("En İyi Öznitelik Alt Kümesi")
        
        # En iyi öznitelikler dosyasını oku
        en_iyi_dosya = "goruntuler/oznitelik_secimi/en_iyi_oznitelikler.txt"
        if os.path.exists(en_iyi_dosya):
            with open(en_iyi_dosya, 'r') as f:
                en_iyi_icerik = f.read()
            
            st.code(en_iyi_icerik, language=None)
            
            st.markdown("""
            ### Sonuç ve Değerlendirme
            
            Yapılan öznitelik seçimi analizi sonucunda, daha az sayıda öznitelik kullanarak tüm özniteliklerle 
            benzer veya daha iyi performans elde edebildiğimizi görüyoruz. Bu, modellerimizin daha verimli ve yorumlanabilir 
            olmasını sağlar.
            
            **Önemli Gözlemler:**
            
            1. Öznitelik sayısını 30'dan 10'a düşürmemize rağmen, model performansında düşüş yaşanmamıştır.
            2. "concave points" (içbükey noktalar) ile ilgili öznitelikler, birçok seçim yönteminde en önemli öznitelikler arasında yer almaktadır.
            3. "worst" (en kötü) değerler kategorisindeki özelliklerin çoğu, en önemli öznitelikler arasında seçilmiştir.
            4. PCA kullanarak boyut indirgeme yapıldığında, ilk 10 bileşen toplam varyansın %95'ini açıklamaktadır.
            
            Bu sonuçlar, meme kanseri tespitinde belirli morfolojik özelliklerin daha belirleyici olduğunu doğrulamaktadır.
            İleriki çalışmalarda, yalnızca seçilen önemli öznitelikleri kullanarak modelleri yeniden eğitebilir ve 
            gerçek zamanlı uygulamalar için daha hızlı bir sistem geliştirebiliriz.
            """)
        else:
            st.warning("En iyi öznitelikler dosyası bulunamadı. Lütfen 'oznitelik_secimi.py' scriptini çalıştırın.")

# Footer
st.markdown("---")
st.markdown("© 2023 Meme Kanseri Tespiti Projesi | Yapay Zeka ve Makine Öğrenmesi Uygulaması") 