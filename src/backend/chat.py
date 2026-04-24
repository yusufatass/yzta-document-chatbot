import os
from functools import lru_cache
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA, load_summarize_chain
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.callbacks import AsyncIteratorCallbackHandler
from langchain_core.documents import Document
import asyncio

from src.config import DB_DIR

load_dotenv()

@lru_cache(maxsize=1)
def get_embeddings():
    return HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

def get_llm(provider="groq", streaming=False, callbacks=None):
    if provider == "google":
        return ChatGoogleGenerativeAI(
            model="gemini-1.5-pro",
            google_api_key=os.getenv("GOOGLE_API_KEY"),
            streaming=streaming,
            callbacks=callbacks
        )
    else:
        return ChatGroq(
            model="llama-3.3-70b-versatile",
            api_key=os.getenv("GROQ_API_KEY"),
            streaming=streaming,
            callbacks=callbacks
        )

def _db_bos_mu(vector_db):
    """Veritabanında hiç doküman yoksa True döner."""
    return len(vector_db.get()["ids"]) == 0

async def soru_sor_stream(kullanici_sorusu, provider="groq"):
    callback = AsyncIteratorCallbackHandler()
    vector_db = Chroma(persist_directory=DB_DIR, embedding_function=get_embeddings())

    # BUG 7: Boş DB kontrolü
    if _db_bos_mu(vector_db):
        yield "Henüz hiç doküman yüklenmemiş. Lütfen önce bir dosya yükleyin."
        return

    llm = get_llm(provider=provider, streaming=True, callbacks=[callback])
    
    rag_zinciri = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vector_db.as_retriever(),
        return_source_documents=True
    )

    task = asyncio.create_task(rag_zinciri.ainvoke({"query": kullanici_sorusu}))

    async for token in callback.aiter():
        yield token

    # BUG 5: Kaynak dokümanları task sonucundan al ve yield et
    result = await task
    source_docs = result.get("source_documents", [])
    if source_docs:
        kaynaklar = [
            f"{os.path.basename(doc.metadata.get('source', 'Bilinmiyor'))} (Sayfa: {doc.metadata.get('page', '?')})"
            for doc in source_docs
        ]
        yield "\n\n---\n**Kaynaklar:** " + ", ".join(kaynaklar)

def ozetle(provider="groq"):
    vector_db = Chroma(persist_directory=DB_DIR, embedding_function=get_embeddings())

    # BUG 7: Boş DB kontrolü
    if _db_bos_mu(vector_db):
        raise ValueError("Özetlenecek doküman bulunamadı. Lütfen önce bir dosya yükleyin.")

    docs = vector_db.get()['documents']
    llm = get_llm(provider=provider)
    summarize_chain = load_summarize_chain(llm, chain_type="map_reduce")

    # BUG 4: Boş chunk'ları filtrele, .run() yerine .invoke() kullan
    doc_objects = [Document(page_content=t) for t in docs if t.strip()]
    result = summarize_chain.invoke(doc_objects)
    return result["output_text"]

def soru_sor_sync(kullanici_sorusu, provider="groq"):
    vector_db = Chroma(persist_directory=DB_DIR, embedding_function=get_embeddings())

    # BUG 7: Boş DB kontrolü
    if _db_bos_mu(vector_db):
        return "Henüz hiç doküman yüklenmemiş. Lütfen önce bir dosya yükleyin.", []

    llm = get_llm(provider=provider, streaming=False)
    
    rag_zinciri = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vector_db.as_retriever(),
        return_source_documents=True
    )
    
    response = rag_zinciri.invoke({"query": kullanici_sorusu})
    return response['result'], response.get('source_documents', [])