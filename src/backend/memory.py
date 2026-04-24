import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from src.config import DB_DIR
from src.backend.chat import get_embeddings  # ← chat.py'deki önbellekli fonksiyon

load_dotenv()

def dokumani_hafizaya_al(dosya_yolu):
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

    vector_db = Chroma(persist_directory=DB_DIR, embedding_function=get_embeddings())
    
    mevcut_kaynaklar = set()
    mevcut_veriler = vector_db.get()
    if mevcut_veriler and mevcut_veriler.get('metadatas'):
        for meta in mevcut_veriler['metadatas']:
            if meta and meta.get('source'):
                mevcut_kaynaklar.add(meta['source'])
    
    if dosya_yolu in mevcut_kaynaklar:
        print(f"Uyarı: '{os.path.basename(dosya_yolu)}' zaten veritabanında mevcut, tekrar eklenmedi.")
        return vector_db

    dokumanlar = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    parcalar = text_splitter.split_documents(dokumanlar)

    vector_db.add_documents(parcalar)
    
    print(f"Başarılı! {len(parcalar)} adet parça hafızaya alındı.")
    return vector_db