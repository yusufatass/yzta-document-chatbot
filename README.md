# 📚 YZTA Document Chatbot
> **Verilerinizle Konuşun:** Doküman analizi için optimize edilmiş, yüksek performanslı ve modern bir RAG (Retrieval-Augmented Generation) sistemi.

[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Frontend-Streamlit-red.svg)](https://streamlit.io/)
[![FastAPI](https://img.shields.io/badge/API-FastAPI-green.svg)](https://fastapi.tiangolo.com/)
[![LangChain](https://img.shields.io/badge/Framework-LangChain-white.svg)](https://www.langchain.com/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## 🔗 Bağlantılar ve Görseller

| 🚀 [Canlı Demo (Streamlit)](https://yzta-document-chatbot.streamlit.app/) |
| :--- |

### 📸 Ekran Görüntüsü

<p align="center">
  <img src="https://github.com/user-attachments/assets/40637309-5ddf-4145-8d77-aa0924a9501a" alt="YZTA Chatbot Arayüzü" width="100%">
  <br>
  <em>Uygulamanın ana arayüzü, doküman yükleme ve sohbet paneli.</em>
</p>
---

## ✨ Özellikler

- 📄 **Geniş Format Desteği:** PDF, DOCX ve TXT dosyalarını akıllı parçalama (chunking) algoritmasıyla işler.
- 🤖 **Dual LLM Entegrasyonu:** Hız için **Groq (Llama 3.3)**, karmaşık analizler için **Google Gemini** desteği.
- 🔍 **Semantik Arama:** ChromaDB ve HuggingFace embeddings (`all-MiniLM-L6-v2`) kullanarak bağlamsal erişim sağlar.
- 📍 **Kaynak Gösterimi:** Yapay zeka, cevabı hangi belgenin hangi sayfasından aldığını şeffafça belirtir.
- 🛡️ **Veri Bütünlüğü:** Mükerrer doküman kontrolü sayesinde aynı dosya sistemde iki kez işlenmez.
- 🌊 **Gerçek Zamanlı Yanıt:** FastAPI tabanlı streaming API ile kelime kelime (token-by-token) yanıt akışı.

---

## ⚙️ Teknik Mimari (RAG Pipeline)

YZTA Chatbot, veriyi sadece saklamaz; onu anlamlandırır. Süreç şu şekilde işler:

1.  **Ingestion:** Dokümanlar yüklenir ve `RecursiveCharacterTextSplitter` ile anlamlı parçalara ayrılır.
2.  **Embedding:** Metin parçaları, HuggingFace modelleri ile 384 boyutlu vektörlere dönüştürülür.
3.  **Indexing:** Vektörler, hızlı benzerlik araması için **ChromaDB** üzerinde indekslenir.
4.  **Retrieval:** Kullanıcı sorusu geldiğinde, vektör uzayında en yakın (benzer) metin parçaları getirilir.
5.  **Generation:** Getirilen bağlam (context) ve soru, LLM'e iletilerek kesin, tutarlı ve kaynaklı bir cevap üretilir.

---

## 🛠️ Tech Stack

| Stack | Teknoloji | Açıklama |
| :--- | :--- | :--- |
| **Frontend** | `Streamlit` | Kullanıcı dostu, interaktif web arayüzü. |
| **Backend API** | `FastAPI` | Asenkron, yüksek performanslı REST servisleri. |
| **Vektör Veritabanı**| `ChromaDB` | Yerel ve hızlı vektör depolama çözümü. |
| **Orchestration** | `LangChain` | RAG zincirleri ve LLM yönetimi. |
| **Embeddings** | `HuggingFace` | Cihaz üzerinde çalışan açık kaynak vektör modelleri. |
| **Modeller** | `Groq / Gemini` | SOTA (State-of-the-art) büyük dil modelleri. |

---

## 🚀 Setup

### 1. Create virtual environment
```bash
python -m venv venv
.\venv\Scripts\Activate.ps1  # Windows
source venv/bin/activate     # Linux/Mac
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Create `.env` file
```bash
cp .env.example .env
# Edit .env and add your API keys
```

### 4. Run Streamlit app
```bash
python -m streamlit run src/app.py
```

### 5. (Optional) Run FastAPI server
```bash
uvicorn src.backend.api:app --reload
```

## 📁 Project Structure

```
yzta-chatbot/
├── src/
│   ├── app.py              # Streamlit frontend
│   ├── config.py           # Central configuration & logging
│   └── backend/
│       ├── chat.py          # RAG logic, LLM chains, prompts
│       ├── memory.py        # Document processing, embedding, vector DB
│       └── api.py           # FastAPI REST endpoints
├── db/                      # ChromaDB vector database (auto-created)
├── uploads/                 # Temporary file uploads (auto-created)
├── requirements.txt
├── .env.example
└── .gitignore
```

## 📡 API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/upload` | Upload and process a document |
| `POST` | `/ask/stream` | Ask a question (streaming response) |
| `GET` | `/summarize` | Summarize all documents |

## 📝 License

Bu proje MIT Lisansı altında lisanslanmıştır. Eğitim amaçlı kullanıma uygundur.
