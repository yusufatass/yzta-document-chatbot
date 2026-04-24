import streamlit as st
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.backend.memory import dokumani_hafizaya_al
from src.backend.chat import soru_sor_sync, get_embeddings
from src.config import UPLOAD_DIR, DB_DIR

def process_documents(files):
    for dosya in files:
        dosya_yolu = os.path.join(UPLOAD_DIR, dosya.name)
        with open(dosya_yolu, "wb") as f:
            f.write(dosya.getbuffer())
        try:
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

st.set_page_config(
    page_title="Dokümanlarınla Sohbet Et",
    page_icon="📚",
    layout="wide"
)

def init_session_state():
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # BUG 6: Sayfa yenilenince DB'yi kontrol ederek gerçek durumu yansıt
    if "docs_processed" not in st.session_state:
        try:
            from langchain_community.vectorstores import Chroma
            db = Chroma(persist_directory=DB_DIR, embedding_function=get_embeddings())
            st.session_state.docs_processed = len(db.get()["ids"]) > 0
        except Exception:
            st.session_state.docs_processed = False

def main():
    init_session_state()

    with st.sidebar:
        st.title("📂 Doküman Yükleme")
        st.markdown("Sohbet etmek istediğiniz PDF, DOCX ve TXT dosyalarını yükleyin.")
        
        uploaded_files = st.file_uploader(
            "Dosyalarınızı seçin",
            type=["pdf", "docx", "txt"],
            accept_multiple_files=True
        )
        
        if st.button("🚀 İşle ve Vektörize Et", use_container_width=True):
            if uploaded_files:
                with st.spinner("Dokümanlar parçalanıyor ve embedding işlemi yapılıyor... ⏳"):
                    success, message = process_documents(uploaded_files)
                    if success:
                        st.session_state.docs_processed = True
                        st.success(message)
                    else:
                        st.error(f"Hata oluştu: {message}")
            else:
                st.warning("Lütfen işlenecek dosya yükleyin!")
                
        st.divider()
        st.markdown("### Sistem Durumu")
        if st.session_state.docs_processed:
            st.success("✅ RAG Sistemi Hazır - Soru Sorabilirsiniz")
        else:
            st.info("ℹ️ Henüz doküman yüklenmedi / işlenmedi.")

    st.title("📚 Kendi Dokümanların ile Sohbet Et")
    st.markdown("RAG mimarisi kullanılarak yüklediğiniz belgelerden bilgi edinin.")
    
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg["role"] == "assistant" and msg.get("sources"):
                with st.expander("🔗 Kaynaklar (Source Tracking)"):
                    for source in msg["sources"]:
                        sayfa_bilgisi = source.get('page', 'Bilinmiyor')
                        st.caption(f"- Belge: **{source['source']}** (Sayfa/Bölüm: {sayfa_bilgisi})")

    if prompt := st.chat_input("Dokümanlarınızla ilgili bir soru sorun..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
            
        with st.chat_message("assistant"):
            if not st.session_state.docs_processed:
                uyari_mesaji = "Lütfen soru sormadan önce sol paneli kullanarak dokümanlarınızı yükleyip **'İşle ve Vektörize Et'** butonuna basın."
                st.warning(uyari_mesaji)
                st.session_state.messages.append({"role": "assistant", "content": uyari_mesaji, "sources": []})
            else:
                with st.spinner("Cevap oluşturuluyor ve ilgili bilgi aranıyor..."):
                    response, sources = generate_response(prompt)
                    st.markdown(response)
                    if sources:
                        with st.expander("🔗 Kaynaklar (Source Tracking)"):
                            for source in sources:
                                sayfa_bilgisi = source.get('page', 'Bilinmiyor')
                                st.caption(f"- Belge: **{source['source']}** (Sayfa/Bölüm: {sayfa_bilgisi})")
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": response,
                        "sources": sources
                    })

if __name__ == "__main__":
    main()