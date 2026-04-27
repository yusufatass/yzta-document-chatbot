import os
import logging
import asyncio

from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA, load_summarize_chain
from langchain.callbacks import AsyncIteratorCallbackHandler
from langchain.prompts import PromptTemplate

from src.config import (
    DB_DIR, GROQ_MODEL, GEMINI_MODEL,
    DEFAULT_PROVIDER, RETRIEVER_K,
)
from src.backend.memory import get_embeddings

load_dotenv()

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────
# 📝 Türkçe RAG Prompt Şablonu
# ──────────────────────────────────────────────
_RAG_PROMPT = PromptTemplate(
    template="""Sen yardımsever bir Türkçe asistansın. Aşağıdaki bağlam bilgisini kullanarak kullanıcının sorusuna en doğru ve kapsamlı şekilde Türkçe cevap ver.

Kurallar:
- Yalnızca verilen bağlam bilgisine dayanarak cevap ver.
- Eğer bağlamda sorunun cevabı yoksa, "Bu bilgi yüklenen dokümanlarda bulunamadı." de.
- Cevabını açık, anlaşılır ve düzenli bir şekilde yaz.

Bağlam:
{context}

Soru: {question}

Cevap:""",
    input_variables=["context", "question"],
)


# ──────────────────────────────────────────────
# 🤖 LLM Sağlayıcı Seçimi
# ──────────────────────────────────────────────
def get_llm(provider: str = DEFAULT_PROVIDER, streaming: bool = False, callbacks=None):
    """Seçilen provider'a göre LLM nesnesini döndür."""
    if provider == "google":
        return ChatGoogleGenerativeAI(
            model=GEMINI_MODEL,
            google_api_key=os.getenv("GOOGLE_API_KEY"),
            streaming=streaming,
            callbacks=callbacks,
        )
    else:
        return ChatGroq(
            model=GROQ_MODEL,
            api_key=os.getenv("GROQ_API_KEY"),
            streaming=streaming,
            callbacks=callbacks,
        )


def _get_retriever(session_id: str = "default"):
    """Vektör veritabanından retriever oluştur."""
    embeddings = get_embeddings()
    db_path = os.path.join(DB_DIR, session_id)
    vector_db = Chroma(persist_directory=db_path, embedding_function=embeddings)
    return vector_db.as_retriever(search_kwargs={"k": RETRIEVER_K})


def _build_rag_chain(provider: str, session_id: str = "default", streaming: bool = False, callbacks=None):
    """RAG zincirini oluştur — retriever + LLM + prompt."""
    llm = get_llm(provider=provider, streaming=streaming, callbacks=callbacks)
    retriever = _get_retriever(session_id=session_id)

    return RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={"prompt": _RAG_PROMPT},
    )


# ──────────────────────────────────────────────
# 💬 Senkron Soru-Cevap (Streamlit için)
# ──────────────────────────────────────────────
def soru_sor_sync(kullanici_sorusu: str, provider: str = DEFAULT_PROVIDER, session_id: str = "default"):
    """
    Kullanıcının sorusuna senkron olarak cevap ver.

    Returns:
        tuple: (cevap_metni: str, kaynak_dokumanlar: list)
    """
    logger.info("Soru alındı (provider=%s): %s", provider, kullanici_sorusu[:80])

    # Streamlit'in ScriptRunner thread'inde event loop olmayabiliyor.
    # Google Gemini SDK arka planda asyncio kullandığı için bu gerekli.
    try:
        asyncio.get_event_loop()
    except RuntimeError:
        asyncio.set_event_loop(asyncio.new_event_loop())

    rag_zinciri = _build_rag_chain(provider=provider, session_id=session_id, streaming=False)
    response = rag_zinciri.invoke({"query": kullanici_sorusu})

    cevap = response['result']
    kaynaklar = response.get('source_documents', [])

    logger.info("Cevap üretildi — %d kaynak döndürüldü.", len(kaynaklar))
    return cevap, kaynaklar


# ──────────────────────────────────────────────
# 🌊 Streaming Soru-Cevap (API için)
# ──────────────────────────────────────────────
async def soru_sor_stream(kullanici_sorusu: str, provider: str = DEFAULT_PROVIDER, session_id: str = "default"):
    """Kullanıcının sorusuna streaming (akışlı) olarak cevap ver."""
    callback = AsyncIteratorCallbackHandler()
    rag_zinciri = _build_rag_chain(
        provider=provider, session_id=session_id, streaming=True, callbacks=[callback]
    )

    task = asyncio.create_task(rag_zinciri.ainvoke({"query": kullanici_sorusu}))

    async for token in callback.aiter():
        yield token

    await task


# ──────────────────────────────────────────────
# 📋 Doküman Özetleme
# ──────────────────────────────────────────────
def ozetle(provider: str = DEFAULT_PROVIDER, session_id: str = "default"):
    """Veritabanındaki tüm dokümanları özetle."""
    embeddings = get_embeddings()
    db_path = os.path.join(DB_DIR, session_id)
    vector_db = Chroma(persist_directory=db_path, embedding_function=embeddings)
    docs = vector_db.get()['documents']

    llm = get_llm(provider=provider)
    summarize_chain = load_summarize_chain(llm, chain_type="map_reduce")

    from langchain_core.documents import Document
    doc_objects = [Document(page_content=t) for t in docs]
    return summarize_chain.run(doc_objects)