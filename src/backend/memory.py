import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from src.config import DB_DIR

# .env dosyasındaki anahtarı oku
load_dotenv()

def dokumani_hafizaya_al(dosya_yolu):
    # Dosya formatına göre loader seç
    uzanti = os.path.splitext(dosya_yolu)[1].lower()
    
    if uzanti == '.pdf':
        loader = PyPDFLoader(dosya_yolu)
    elif uzanti == '.txt':
        loader = TextLoader(dosya_yolu)
    elif uzanti == '.docx':
        from langchain_community.document_loaders import Docx2txtLoader
        loader = Docx2txtLoader(dosya_yolu)
    else:
        raise ValueError(f"Desteklenmeyen dosya formatı: {uzanti}. Lütfen PDF, DOCX veya TXT kullanınız.")
    
    # 1. Dosyayı yükle (Load)
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
        persist_directory=DB_DIR
    )
    
    print(f"Başarılı! {len(parcalar)} adet parça hafızaya alındı.")
    return vector_db

