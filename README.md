# YZTA Document Chatbot - RAG Backend

AI-powered document analysis system using Retrieval-Augmented Generation (RAG).

## Features

- 📄 PDF document upload and processing
- 🤖 Streaming Q&A with AI (Groq + Google Gemini)
- 📊 Document summarization
- 🔍 Vector-based semantic search (Chroma)
- ⚡ FastAPI REST endpoints

## Tech Stack

- **Framework**: FastAPI
- **LLM**: Groq, Google Gemini
- **Vector DB**: Chroma
- **Embeddings**: HuggingFace
- **Text Processing**: LangChain

## Setup

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

### 3. Create .env file
```bash
cp .env.example .env
# Edit .env and add your API keys
```

### 4. Run server
```bash
uvicorn main:app --reload
```

## API Endpoints

- `POST /upload` - Upload PDF document
- `POST /ask/stream` - Ask question (streaming response)
- `GET /summarize` - Get document summary

## Project Structure

```
backend-ai/
├── main.py          # FastAPI app
├── chat.py          # RAG logic
├── memory.py        # Document processing
├── requirements.txt
├── .env.example
└── .gitignore
```
