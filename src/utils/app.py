import streamlit as st

st.set_page_config(page_title="YZTA Chatbot", layout="wide")

st.title("📄 Kendi Dokümanların ile Sohbet Et")

# Dosya Yükleme Alanı [cite: 18, 41]
uploaded_files = st.file_uploader("Dokümanları Yükle (PDF, DOCX, TXT)", accept_multiple_files=True)

# Chat Ekranı [cite: 38]
if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Dokümanların hakkında bir şey sor..."):
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Burası ileride backend/rag_chain.py ile bağlanacak
    with st.chat_message("assistant"):
        st.markdown("Şu an geliştirme aşamasındayım, yakında cevap verebileceğim!")
        