import streamlit as st
import cv2
import numpy as np
import os
import pandas as pd
from datetime import datetime, timedelta
from fpdf import FPDF

# Sayfa Ayarları
st.set_page_config(page_title="Tekulaş Profesyonel Denetim", layout="wide")

# Klasör ve Dosya Kontrolleri
if not os.path.exists("data"): os.makedirs("data")
DB_FILE = "data/denetim_gecmisi.csv"

# --- YAN MENÜ (ARŞİV VE İSTATİSTİK) ---
st.sidebar.title("📊 Filo Yönetim Paneli")

if os.path.exists(DB_FILE):
    df = pd.read_csv(DB_FILE)
    st.sidebar.metric("Toplam Denetim", len(df))
    st.sidebar.subheader("Hızlı Sorgulama")
    plaka_sorgu = st.sidebar.selectbox("Araç Geçmişi:", ["Seçiniz..."] + sorted(df['Plaka'].unique().tolist()))
    
    if plaka_sorgu != "Seçiniz...":
        araç_df = df[df['Plaka'] == plaka_sorgu].sort_values(by='Tarih', ascending=False)
        st.sidebar.write(araç_df[['Tarih', 'Denetçi', 'Genel_Durum']].head(5))

# --- FONKSİYONLAR ---
def rapor_olustur(plaka, tarih, denetci, sonuclar):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", 'B', 16)
    pdf.cell(0, 10, txt=f"PROFESYONEL HASAR RAPORU - {plaka}", ln=True, align='C')
    pdf.set_font("Arial", size=12)
    pdf.cell(0, 10, txt=f"Tarih: {tarih} | Denetçi: {denetci}", ln=True)
    pdf.ln(10)
    
    for aci, veri in sonuclar.items():
        pdf.set_font("Arial", 'B', 12)
        pdf.cell(0, 10, txt=f"Bölge: {aci} | Tespit: {veri['durum']}", ln=True)
        pdf.set_font("Arial", size=10)
        pdf.cell(0, 10, txt=f"Not: {veri['not']}", ln=True)
        if veri['resim_yolu']:
            pdf.image(veri['resim_yolu'], w=90)
            pdf.ln(5)
    
    rapor_adi = f"Rapor_{plaka}_{tarih}.pdf"
    pdf.output(rapor_adi)
    return rapor_adi

def analiz_et(eski_yol, yeni_img, aci, plaka):
    if not os.path.exists(eski_yol):
        return "İlk Kayıt", None, 0
    img_eski = cv2.imread(eski_yol)
    img_yeni = cv2.resize(yeni_img, (img_eski.shape[1], img_eski.shape[0]))
    g1 = cv2.cvtColor(img_eski, cv2.COLOR_BGR2GRAY)
    g2 = cv2.cvtColor(img_yeni, cv2.COLOR_BGR2GRAY)
    fark = cv2.absdiff(cv2.GaussianBlur(g1, (5,5), 0), cv2.GaussianBlur(g2, (5,5), 0))
    _, esik = cv2.threshold(fark, 35, 255, cv2.THRESH_BINARY)
    konturlar, _ = cv2.findContours(esik, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    hasar_vurgulu = img_yeni.copy()
    sayi = sum(1 for c in konturlar if cv2.contourArea(c) > 600)
    
    if sayi > 0:
        resim_yolu = f"data/temp_{aci}_{plaka}.jpg"
        for c in konturlar:
            if cv2.contourArea(c) > 600:
                (x, y, w, h) = cv2.boundingRect(c)
                cv2.rectangle(hasar_vurgulu, (x, y), (x + w, y + h), (0, 0, 255), 3)
        cv2.imwrite(resim_yolu, hasar_vurgulu)
        return f"Değişim ({sayi} nokta)", resim_yolu, sayi
    return "Temiz", None, 0

# --- ANA EKRAN ---
st.title("🚌 Tekulaş Akıllı Denetim Sistemi")

with st.expander("📝 Denetim Bilgileri", expanded=True):
    c1, c2 = st.columns(2)
    plaka = c1.text_input("ARAÇ PLAKASI:", placeholder="59 ...").upper().strip()
    denetci = c2.text_input("DENETÇİ / ŞOFÖR ADI:", placeholder="Ad Soyad")

if plaka and denetci:
    st.divider()
    acilar = ["Ön", "Arka", "Sağ Yan", "Sol Yan"]
    veriler = {}
    
    cols = st.columns(2)
    for i, aci in enumerate(acilar):
        with cols[i % 2]:
            st.subheader(f"📍 {aci}")
            img_file = st.file_uploader(f"{aci} Çek/Yükle", type=['jpg','png','jpeg'], key=f"img_{aci}")
            notu = st.text_input(f"{aci} İçin Not (Opsiyonel):", key=f"not_{aci}")
            seviye = st.select_slider(f"{aci} Hasar Seviyesi:", options=["Yok", "Düşük", "Orta", "Yüksek"], key=f"sev_{aci}")
            veriler[aci] = {"dosya": img_file, "not": notu, "seviye": seviye}

    if st.button("🚀 DENETİMİ KAYDET VE ANALİZ ET"):
        if all(v["dosya"] for v in veriler.values()):
            bugun = datetime.now().strftime("%Y-%m-%d")
            dun = (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")
            rapor_detay = {}
            toplam_hasar_skoru = 0
            
            with st.spinner("Yapay zeka hasar taraması yapıyor..."):
                for aci, icerik in veriler.items():
                    file_bytes = np.asarray(bytearray(icerik["dosya"].read()), dtype=np.uint8)
                    img = cv2.imdecode(file_bytes, 1)
                    
                    yeni_yol = f"data/{plaka}_{aci}_{bugun}.jpg"
                    cv2.imwrite(yeni_yol, img)
                    
                    eski_yol = f"data/{plaka}_{aci}_{dun}.jpg"
                    durum, resim, skor = analiz_et(eski_yol, img, aci, plaka)
                    toplam_hasar_skoru += skor
                    rapor_detay[aci] = {"durum": durum, "resim_yolu": resim, "not": icerik["not"]}
            
            # Excel/CSV Kaydı
            yeni_kayit = {
                "Tarih": bugun, "Plaka": plaka, "Denetçi": denetci, 
                "Hasar_Noktasi": toplam_hasar_skoru, 
                "Genel_Durum": "Hasarlı" if toplam_hasar_skoru > 0 else "Temiz"
            }
            df_yeni = pd.DataFrame([yeni_kayit])
            if os.path.exists(DB_FILE):
                df_old = pd.read_csv(DB_FILE)
                pd.concat([df_old, df_yeni]).to_csv(DB_FILE, index=False)
            else:
                df_yeni.to_csv(DB_FILE, index=False)
            
            # Rapor Sunumu
            pdf_yolu = rapor_olustur(plaka, bugun, denetci, rapor_detay)
            st.success(f"✅ İşlem Başarılı! Toplam {toplam_hasar_skoru} değişim noktası bulundu.")
            with open(pdf_yolu, "rb") as f:
                st.download_button("📥 Profesyonel PDF Raporu İndir", f, file_name=pdf_yolu)
        else:
            st.error("Lütfen 4 açının da fotoğrafını yükleyiniz.")
