# 📚 YZTA Document Chatbot

AI-powered document analysis system using **Retrieval-Augmented Generation (RAG)**. Upload your PDF, DOCX, or TXT documents and chat with them using powerful LLMs.

## ✨ Features

- 📄 **Multi-format document upload** — PDF, DOCX, TXT
- 🤖 **Dual LLM support** — Groq (Llama 3.3) & Google Gemini
- 💬 **Interactive chat interface** — Streamlit-based conversational UI
- 🔍 **Semantic search** — Vector-based retrieval with ChromaDB
- 📎 **Source tracking** — See which document & page the answer came from
- 🔄 **Duplicate detection** — Same document won't be processed twice
- 🌊 **Streaming API** — FastAPI endpoint with real-time token streaming

## 🛠️ Tech Stack

| Component | Technology |
|-----------|-----------|
| Frontend | Streamlit |
| API | FastAPI |
| LLM | Groq (Llama 3.3), Google Gemini |
| Vector DB | ChromaDB |
| Embeddings | HuggingFace (all-MiniLM-L6-v2) |
| Orchestration | LangChain |

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

This project is for educational purposes.
