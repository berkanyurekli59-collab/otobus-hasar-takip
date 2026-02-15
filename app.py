import streamlit as st
import cv2
import numpy as np
import os
import shutil  # Daha güvenli dosya kopyalama için eklendi
from datetime import datetime
from fpdf import FPDF

# --- KLASÖR AYARLARI ---
dirs = ["raporlar", "video_arsivi", "temp"]
for d in dirs:
    if not os.path.exists(d):
        os.makedirs(d)

# --- GÖRÜNTÜ İŞLEME FONKSİYONLARI ---
def goruntu_normallestir(frame):
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)
    limg = cv2.merge((cl, a, b))
    return cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)

def rapor_olustur(plaka, skor, hasar_tipi, frame_path):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", 'B', 16)
    pdf.cell(0, 10, txt="OTOBUS HASAR ANALIZ RAPORU", ln=True, align='C')
    pdf.ln(10)
    pdf.set_font("Arial", size=12)
    pdf.cell(0, 10, txt=f"Arac Plakasi: {plaka}", ln=True)
    pdf.cell(0, 10, txt=f"Tarih: {datetime.now().strftime('%d/%m/%Y %H:%M')}", ln=True)
    pdf.cell(0, 10, txt=f"Hasar Skoru: {skor}", ln=True)
    pdf.cell(0, 10, txt=f"Tespit: {hasar_tipi}", ln=True)
    
    if frame_path and os.path.exists(frame_path):
        pdf.ln(10)
        pdf.image(frame_path, x=10, w=180)
    
    rapor_adi = f"raporlar/{plaka}_{datetime.now().strftime('%Y%m%d_%H%M')}.pdf"
    pdf.output(rapor_adi)
    return rapor_adi

def kiyaslamali_analiz(eski_video_yolu, yeni_video_yolu, esik=15000):
    cap_old = cv2.VideoCapture(eski_video_yolu)
    cap_new = cv2.VideoCapture(yeni_video_yolu)
    
    max_fark = 0
    en_iyi_kare = None
    maske_kare = None

    while True:
        ret1, frame_old = cap_old.read()
        ret2, frame_new = cap_new.read()
        if not ret1 or not ret2: break

        n_old = goruntu_normallestir(frame_old)
        n_new = goruntu_normallestir(frame_new)

        diff = cv2.absdiff(cv2.cvtColor(n_old, cv2.COLOR_BGR2GRAY), 
                           cv2.cvtColor(n_new, cv2.COLOR_BGR2GRAY))
        _, mask = cv2.threshold(diff, 40, 255, cv2.THRESH_BINARY)
        skor = np.sum(mask == 255)

        if skor > max_fark:
            max_fark = skor
            en_iyi_kare = frame_new
            maske_kare = mask

    cap_old.release()
    cap_new.release()
    
    if max_fark > esik:
        return True, max_fark, en_iyi_kare, maske_kare
    return False, max_fark, None, None

# --- STREAMLIT ARAYÜZÜ ---
st.set_page_config(page_title="Otobüs Hasar Takip", layout="wide")

st.sidebar.title("🚌 Araç Yönetimi")
yeni_plaka = st.sidebar.text_input("Yeni Plaka Kaydet/Seç:", "").upper()
kayitli_videolar = [f.replace("_kayit.mp4", "") for f in os.listdir("video_arsivi") if f.endswith(".mp4")]
secilen_plaka = st.sidebar.selectbox("Kayıtlı Plakalar:", [""] + kayitli_videolar)
aktif_plaka = yeni_plaka if yeni_plaka else secilen_plaka

st.sidebar.markdown("---")
st.sidebar.subheader("🗄️ Rapor Arşivi")
arama = st.sidebar.text_input("Raporlarda Ara:")
raporlar = sorted([f for f in os.listdir("raporlar") if f.endswith(".pdf")], reverse=True)

for r in raporlar:
    if arama.upper() in r.upper():
        with st.sidebar.expander(f"📄 {r.split('_')[0]}"):
            with open(f"raporlar/{r}", "rb") as f:
                st.download_button("İndir", f, file_name=r, key=f"dl_{r}")

st.title(f"📊 Hasar Analiz Paneli: {aktif_plaka if aktif_plaka else 'Araç Seçiniz'}")

if aktif_plaka:
    uploaded_video = st.file_uploader("Kontrol Videosunu Yükle", type=["mp4", "mov"])
    
    if uploaded_video:
        # Geçici dosyayı kaydet
        temp_yolu = os.path.join("temp", f"{aktif_plaka}_temp.mp4")
        with open(temp_yolu, "wb") as f:
            f.write(uploaded_video.getbuffer())
        
        st.video(temp_yolu) # Videoyu ekranda göster (Yüklendiğini teyit et)
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("💾 Referans Olarak Kaydet"):
                hedef_yol = os.path.join("video_arsivi", f"{aktif_plaka}_kayit.mp4")
                shutil.copy(temp_yolu, hedef_yol) # Rename yerine Copy daha güvenlidir
                st.success(f"✅ {aktif_plaka} plakalı araç için referans video başarıyla kaydedildi.")
        
        with col2:
            if st.button("🔍 Hasar Analizi Yap"):
                eski_yol = os.path.join("video_arsivi", f"{aktif_plaka}_kayit.mp4")
                if os.path.exists(eski_yol):
                    with st.spinner("İki video karşılaştırılıyor, lütfen bekleyin..."):
                        hasar_var, skor, kare, maske = kiyaslamali_analiz(eski_yol, temp_yolu)
                        
                        if hasar_var:
                            st.error(f"⚠️ Yeni Hasar Tespit Edildi! (Fark Skoru: {skor})")
                            img_path = f"temp/{aktif_plaka}_hasar.jpg"
                            cv2.imwrite(img_path, kare)
                            
                            c1, c2 = st.columns(2)
                            c1.image(kare, caption="Tespit Edilen Hasarlı Bölge", use_column_width=True)
                            c2.image(maske, caption="Hasar Maskesi (Piksel Farkı)", use_column_width=True)
                            
                            pdf_yolu = rapor_olustur(aktif_plaka, skor, "Yeni Hasar", img_path)
                            with open(pdf_yolu, "rb") as f:
                                st.download_button("📥 PDF Raporunu İndir", f, file_name=os.path.basename(pdf_yolu))
                        else:
                            st.success("✅ Karşılaştırma Tamamlandı: İki video arasında anlamlı bir fark bulunamadı.")
                else:
                    st.warning("⚠️ Bu aracın geçmiş (temiz) kaydı bulunamadı. Lütfen önce 'Referans Olarak Kaydet' butonuna basın.")
else:
    st.info("İşleme başlamak için sol panelden plaka girişi yapın.")
