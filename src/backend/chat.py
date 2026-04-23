import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA, load_summarize_chain
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.callbacks import AsyncIteratorCallbackHandler
import asyncio

from src.config import DB_DIR

load_dotenv()

def get_llm(provider="groq", streaming=False, callbacks=None):
    # LLM Provider Selection (LLM Sağlayıcı Seçimi)
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

async def soru_sor_stream(kullanici_sorusu, provider="groq"):
    # Streaming Logic (Akış Mantığı)
    callback = AsyncIteratorCallbackHandler()
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vector_db = Chroma(persist_directory=DB_DIR, embedding_function=embeddings)
    
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

    await task

def ozetle(provider="groq"):
    # Summarization Logic (Özetleme Mantığı)
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vector_db = Chroma(persist_directory=DB_DIR, embedding_function=embeddings)
    docs = vector_db.get()['documents'] # Tüm doküman metinlerini çek
    
    llm = get_llm(provider=provider)
    summarize_chain = load_summarize_chain(llm, chain_type="map_reduce")
    
    # Doküman nesnelerine çevir ve özetle
    from langchain_core.documents import Document
    doc_objects = [Document(page_content=t) for t in docs]
    return summarize_chain.run(doc_objects)

def soru_sor_sync(kullanici_sorusu, provider="groq"):
    """Frontend (Streamlit) için Senkron Soru Sorma Fonksiyonu."""
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vector_db = Chroma(persist_directory=DB_DIR, embedding_function=embeddings)
    
    llm = get_llm(provider=provider, streaming=False)
    
    rag_zinciri = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vector_db.as_retriever(),
        return_source_documents=True
    )
    
    response = rag_zinciri.invoke({"query": kullanici_sorusu})
    return response['result'], response.get('source_documents', [])