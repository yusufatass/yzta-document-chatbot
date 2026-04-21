import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

# .env dosyasındaki anahtarı oku
load_dotenv()

def dokumani_hafizaya_al(pdf_yolu):
    # 1. Pdf i yükle (Load)
    loader = PyPDFLoader(pdf_yolu)
    dokumanlar = loader.load()

    # 2. Metni parçalara böl (Chunking)
    # chunk_overlap: parçaların birbiriyle çakışma miktarı
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    parcalar = text_splitter.split_documents(dokumanlar)

    # 3. Sayısallaştırma (Embedding)
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    # 4. Vektör Veritabanına Kaydet (Vector Store)
    vector_db = Chroma.from_documents(
        documents=parcalar, 
        embedding=embeddings, 
        persist_directory="./db"
    )
    
    print(f"Başarılı! {len(parcalar)} adet parça hafızaya alındı.")
    return vector_db

