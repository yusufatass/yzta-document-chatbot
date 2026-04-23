import streamlit as st
import os

import sys

# Backend modüllerine erişim sağlayabilmek için root'u path'e ekle
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.backend.memory import dokumani_hafizaya_al
from src.backend.chat import soru_sor_sync
from src.config import UPLOAD_DIR

def process_documents(files):
    for dosya in files:
        dosya_yolu = os.path.join(UPLOAD_DIR, dosya.name)
        with open(dosya_yolu, "wb") as f:
            f.write(dosya.getbuffer())
            
        try:
            # Backend process: chunking and embedding
            dokumani_hafizaya_al(dosya_yolu)
        except Exception as e:
            return False, f"Hata: {str(e)}"
            
    return True, f"{len(files)} doküman başarıyla işlendi ve vektör veritabanına kaydedildi."

def generate_response(query):
    try:
        cevap, kaynaklar = soru_sor_sync(query)
        sources_list = []
        if kaynaklar:
            for doc in kaynaklar:
                sources_list.append({
                    "source": os.path.basename(doc.metadata.get("source", "Bilinmiyor")), 
                    "page": doc.metadata.get("page", "Bilinmiyor")
                })
        return cevap, sources_list
    except Exception as e:
        return f"Cevap üretilirken backend kaynaklı hata oluştu: {str(e)}", []


# --- SİSTEM / SAYFA AYARLARI ---
st.set_page_config(
    page_title="Dokümanlarınla Sohbet Et",
    page_icon="📚",
    layout="wide"
)

def init_session_state():
    """Mesaj geçmişini ve doküman işleme durumunu st.session_state ile hafızada tutar."""
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    if "docs_processed" not in st.session_state:
        st.session_state.docs_processed = False

def main():
    init_session_state()

    # --- YAN PANEL (SIDEBAR): Dosya Yükleme Paneli ---
    with st.sidebar:
        st.title("📂 Doküman Yükleme")
        st.markdown("Sohbet etmek istediğiniz PDF, DOCX ve TXT dosyalarını yükleyin.")
        
        # Çoklu dosya yükleme özelliği
        uploaded_files = st.file_uploader(
            "Dosyalarınızı seçin",
            type=["pdf", "docx", "txt"],
            accept_multiple_files=True
        )
        
        if st.button("🚀 İşle ve Vektörize Et", use_container_width=True):
            if uploaded_files:
                # İşlem spinner'ı
                with st.spinner("Dokümanlar parçalanıyor (chunking) ve embedding işlemi yapılıyor... Lütfen bekleyin. ⏳"):
                    success, message = process_documents(uploaded_files)
                    
                    if success:
                        st.session_state.docs_processed = True
                        st.success(message)
                    else:
                        st.error(f"Hata oluştu: {message}")
            else:
                st.warning("Lütfe işlenecek dosya yükleyin!")
                
        st.divider()
        st.markdown("### Sistem Durumu")
        if st.session_state.docs_processed:
            st.success("✅ RAG Sistemi Hazır - Soru Sorabilirsiniz")
        else:
            st.info("ℹ️ Henüz doküman yüklenmedi / işlenmedi.")

    # --- ANA EKRAN (CHAT ARAYÜZÜ) ---
    st.title("📚 Kendi Dokümanların ile Sohbet Et")
    st.markdown("RAG mimarisi kullanılarak yüklediğiniz belgelerden bilgi edinin.")
    
    # 1. Eski mesajları ekrana bas (Chat geçmişi)
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            # Bonus Özellik: Kaynakların gösterimi (Eğer asistan mesajıysa ve kaynağı varsa)
            if msg["role"] == "assistant" and msg.get("sources"):
                with st.expander("🔗 Kaynaklar (Source Tracking)"):
                    for source in msg["sources"]:
                        sayfa_bilgisi = source.get('page', 'Bilinmiyor')
                        st.caption(f"- Belge: **{source['source']}** (Sayfa/Bölüm: {sayfa_bilgisi})")

    # 2. Kullanıcıdan yeni mesaj al
    if prompt := st.chat_input("Dokümanlarınızla ilgili bir soru sorun..."):
        # Kullanıcı mesajını state'e kaydet ve ekrana bas
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
            
        # 3. Asistan (LLM) cevabını üret ve ekrana bas
        with st.chat_message("assistant"):
            if not st.session_state.docs_processed:
                # Kullanıcı doküman yüklemeden soru sormaya çalışırsa uyar
                uyari_mesaji = "Lütfen soru sormadan önce sol paneli kullanarak dokümanlarınızı yükleyip **'İşle ve Vektörize Et'** butonuna basın."
                st.warning(uyari_mesaji)
                st.session_state.messages.append({"role": "assistant", "content": uyari_mesaji, "sources": []})
            else:
                with st.spinner("Cevap oluşturuluyor ve ilgili bilgi aranıyor..."):
                    # Backend LLM üretim fonksiyonu çağrılıyor
                    response, sources = generate_response(prompt)
                    st.markdown(response)
                    
                    # Bonus Özellik kısmının asistan cevabı sırasında render edilmesi
                    if sources:
                        with st.expander("🔗 Kaynaklar (Source Tracking)"):
                            for source in sources:
                                sayfa_bilgisi = source.get('page', 'Bilinmiyor')
                                st.caption(f"- Belge: **{source['source']}** (Sayfa/Bölüm: {sayfa_bilgisi})")
                                
                    # Asistan cevabını state'e ekle
                    st.session_state.messages.append({
                        "role": "assistant", 
                        "content": response, 
                        "sources": sources
                    })

if __name__ == "__main__":
    main()
