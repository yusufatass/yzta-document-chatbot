import streamlit as st
import os
import sys
import logging
import uuid

# Backend modüllerine erişim sağlayabilmek için root'u path'e ekle
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# config.py'ı ilk import et — telemetry kapatma ve loglama orada yapılıyor
from src.config import UPLOAD_DIR, DEFAULT_PROVIDER
from src.backend.memory import dokumani_hafizaya_al, veritabanini_sifirla
from src.backend.chat import soru_sor_sync

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────
# 📄 Doküman İşleme
# ──────────────────────────────────────────────
def process_documents(files, session_id):
    """
    Yüklenen dosyaları sırayla işle.
    Bir dosyada hata olsa bile diğerlerine devam eder.
    """
    sonuclar = []       # Her dosyanın sonuç bilgisi
    hatalar = []        # Hata oluşan dosyalar

    for dosya in files:
        # Dosya isimlerinin çakışmasını önlemek için session_id prefix'i ekle
        guvenli_dosya_adi = f"{session_id}_{dosya.name}"
        dosya_yolu = os.path.join(UPLOAD_DIR, guvenli_dosya_adi)

        # Dosyayı diske yaz
        with open(dosya_yolu, "wb") as f:
            f.write(dosya.getbuffer())

        try:
            sonuc = dokumani_hafizaya_al(dosya_yolu, session_id=session_id)
            sonuclar.append(sonuc)
        except Exception as e:
            logger.error("Dosya işleme hatası (%s): %s", dosya.name, e)
            hatalar.append({"dosya": dosya.name, "hata": str(e)})
        finally:
            # Geçici dosyayı her zaman temizle
            if os.path.exists(dosya_yolu):
                try:
                    os.remove(dosya_yolu)
                except OSError:
                    pass

    return sonuclar, hatalar


def generate_response(query, provider, session_id):
    """LLM'den cevap al ve kaynakları düzenle."""
    try:
        cevap, kaynaklar = soru_sor_sync(query, provider=provider, session_id=session_id)
        sources_list = []
        if kaynaklar:
            for doc in kaynaklar:
                sources_list.append({
                    "source": os.path.basename(doc.metadata.get("source", "Bilinmiyor")),
                    "page": doc.metadata.get("page", "Bilinmiyor")
                })
        return cevap, sources_list
    except Exception as e:
        logger.error("Cevap üretme hatası: %s", e)
        return f"Cevap üretilirken bir hata oluştu: {str(e)}", []


# ──────────────────────────────────────────────
# ⚙️ Sayfa & Oturum (Session) Ayarları
# ──────────────────────────────────────────────
st.set_page_config(
    page_title="Dokümanlarınla Sohbet Et",
    page_icon="📚",
    layout="wide"
)


def init_session_state():
    """Mesaj geçmişini ve doküman işleme durumunu st.session_state ile hafızada tutar."""
    if "session_id" not in st.session_state:
        st.session_state.session_id = str(uuid.uuid4())
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "docs_processed" not in st.session_state:
        st.session_state.docs_processed = False
    if "provider" not in st.session_state:
        st.session_state.provider = DEFAULT_PROVIDER


# ──────────────────────────────────────────────
# 🚀 Ana Uygulama
# ──────────────────────────────────────────────
def main():
    init_session_state()

    # ─── YAN PANEL (SIDEBAR) ───
    with st.sidebar:
        st.title("📂 Doküman Yükleme")
        st.markdown("Sohbet etmek istediğiniz **PDF**, **DOCX** ve **TXT** dosyalarını yükleyin.")

        # Çoklu dosya yükleme
        uploaded_files = st.file_uploader(
            "Dosyalarınızı seçin",
            type=["pdf", "docx", "txt"],
            accept_multiple_files=True
        )

        if st.button("🚀 İşle ve Vektörize Et", use_container_width=True):
            if uploaded_files:
                with st.spinner("Dokümanlar parçalanıyor ve embedding işlemi yapılıyor... ⏳"):
                    sonuclar, hatalar = process_documents(uploaded_files, st.session_state.session_id)

                    # Başarılı dosyaları göster
                    if sonuclar:
                        st.session_state.docs_processed = True
                        for s in sonuclar:
                            if s["zaten_vardi"]:
                                st.info(f"ℹ️ **{s['dosya']}** zaten veritabanında mevcut, atlandı.")
                            else:
                                st.success(f"✅ **{s['dosya']}** → {s['parca_sayisi']} parça oluşturuldu.")

                    # Hataları göster
                    for h in hatalar:
                        st.error(f"❌ **{h['dosya']}**: {h['hata']}")

                    if not sonuclar and hatalar:
                        st.error("Hiçbir dosya işlenemedi.")
            else:
                st.warning("Lütfen işlenecek dosya yükleyin!")

        st.divider()

        # ─── LLM Provider Seçimi ───
        st.markdown("### 🤖 LLM Ayarları")
        provider_secimi = st.selectbox(
            "Model Sağlayıcı",
            options=["groq", "google"],
            format_func=lambda x: "⚡ Groq (Llama 3.3)" if x == "groq" else "🧠 Google Gemini",
            index=0 if st.session_state.provider == "groq" else 1,
        )
        st.session_state.provider = provider_secimi

        st.divider()

        # ─── Sistem Durumu ───
        st.markdown("### 📊 Sistem Durumu")
        if st.session_state.docs_processed:
            st.success("✅ RAG Sistemi Hazır — Soru Sorabilirsiniz")
        else:
            st.info("ℹ️ Henüz doküman yüklenmedi / işlenmedi.")

        st.divider()

        # ─── Yönetim Butonları ───
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🗑️ Sohbeti Temizle", use_container_width=True):
                st.session_state.messages = []
                st.rerun()
        with col2:
            if st.button("🔄 DB Sıfırla", use_container_width=True):
                if veritabanini_sifirla(session_id=st.session_state.session_id):
                    st.session_state.docs_processed = False
                    st.session_state.messages = []
                    st.success("Veritabanı sıfırlandı!")
                    st.rerun()
                else:
                    st.error("Veritabanı sıfırlanamadı!")

    # ─── ANA EKRAN (CHAT ARAYÜZÜ) ───
    st.title("📚 Kendi Dokümanların ile Sohbet Et")
    st.markdown("RAG mimarisi kullanılarak yüklediğiniz belgelerden bilgi edinin.")

    # 1. Eski mesajları ekrana bas (Chat geçmişi)
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            # Kaynak bilgilerini göster
            if msg["role"] == "assistant" and msg.get("sources"):
                with st.expander("🔗 Kaynaklar"):
                    for source in msg["sources"]:
                        sayfa_bilgisi = source.get('page', 'Bilinmiyor')
                        st.caption(f"📄 **{source['source']}** — Sayfa/Bölüm: {sayfa_bilgisi}")

    # 2. Kullanıcıdan yeni mesaj al
    if prompt := st.chat_input("Dokümanlarınızla ilgili bir soru sorun..."):
        # Kullanıcı mesajını kaydet ve göster
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # 3. Asistan cevabını üret
        with st.chat_message("assistant"):
            if not st.session_state.docs_processed:
                uyari = ("Lütfen soru sormadan önce sol panelden dokümanlarınızı yükleyip "
                         "**'İşle ve Vektörize Et'** butonuna basın.")
                st.warning(uyari)
                st.session_state.messages.append({
                    "role": "assistant", "content": uyari, "sources": []
                })
            else:
                with st.spinner("Cevap oluşturuluyor..."):
                    response, sources = generate_response(
                        prompt, 
                        provider=st.session_state.provider,
                        session_id=st.session_state.session_id
                    )
                    st.markdown(response)

                    if sources:
                        with st.expander("🔗 Kaynaklar"):
                            for source in sources:
                                sayfa_bilgisi = source.get('page', 'Bilinmiyor')
                                st.caption(f"📄 **{source['source']}** — Sayfa/Bölüm: {sayfa_bilgisi}")

                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": response,
                        "sources": sources
                    })


if __name__ == "__main__":
    main()
