import os
import hashlib
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from src.config import DB_DIR

# .env dosyasındaki anahtarı oku
load_dotenv()

# ✅ Global embeddings cache - modeli bir kez yükle
_embeddings_cache = None

def get_embeddings():
    """Embeddings modelini cache ederek yükle (bir kez yüklenecek)"""
    global _embeddings_cache
    if _embeddings_cache is None:
        _embeddings_cache = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    return _embeddings_cache

def get_file_hash(dosya_yolu):
    """Dosyanın MD5 hash'ini hesapla (duplicate detection için)"""
    with open(dosya_yolu, 'rb') as f:
        return hashlib.md5(f.read()).hexdigest()

def dosya_zaten_var_mi(dosya_hash):
    """Veritabanında dosya zaten var mı kontrol et"""
    try:
        vector_db = Chroma(persist_directory=DB_DIR, embedding_function=get_embeddings())
        # Metadata'da dosya hash'i ara
        try:
            results = vector_db.get(where={"source_hash": dosya_hash})
            return len(results.get('ids', [])) > 0
        except:
            # Eğer where filtresi başarısız olursa, tüm dokümanları kontrol et
            all_docs = vector_db.get()
            for metadata in all_docs.get('metadatas', []):
                if metadata.get('source_hash') == dosya_hash:
                    return True
            return False
    except:
        # Veritabanı yoksa yeni başladığını varsay
        return False

def dokumani_hafizaya_al(dosya_yolu):
    # ✅ Dosya var mı kontrol et
    if not os.path.exists(dosya_yolu):
        raise FileNotFoundError(f"Dosya bulunamadı: {dosya_yolu}")
    
    # ✅ Dosya hash'i hesapla (duplicate detection)
    dosya_hash = get_file_hash(dosya_yolu)
    
    # ✅ Aynı dosya zaten var mı kontrol et
    if dosya_zaten_var_mi(dosya_hash):
        print(f"Uyarı: Bu dosya zaten veri tabanında mevcut. Tekrar embedding yapılmıyor.")
        return
    
    # Dosya formatına göre loader seç
    uzanti = os.path.splitext(dosya_yolu)[1].lower()
    
    dokumanlar = None
    
    try:
        if uzanti == '.pdf':
            loader = PyPDFLoader(dosya_yolu)
            dokumanlar = loader.load()
        elif uzanti == '.txt':
            # ✅ TextLoader'a encoding parametresi ekle
            loader = TextLoader(dosya_yolu, encoding='utf-8')
            try:
                dokumanlar = loader.load()
            except:
                # UTF-8 başarısız olursa, Windows-1252 dene
                try:
                    loader = TextLoader(dosya_yolu, encoding='windows-1252')
                    dokumanlar = loader.load()
                except:
                    # Son çare: Dosyayı manuel oku
                    with open(dosya_yolu, 'r', encoding='utf-8', errors='replace') as f:
                        icerik = f.read()
                    from langchain_core.documents import Document
                    dokumanlar = [Document(page_content=icerik, metadata={"source": dosya_yolu})]
        elif uzanti == '.docx':
            from langchain_community.document_loaders import Docx2txtLoader
            loader = Docx2txtLoader(dosya_yolu)
            dokumanlar = loader.load()
        else:
            raise ValueError(f"Desteklenmeyen dosya formatı: {uzanti}. Lütfen PDF, DOCX veya TXT kullanınız.")
    
        if dokumanlar is None or len(dokumanlar) == 0:
            raise ValueError(f"Dosyadan içerik yüklenemedi: {dosya_yolu}")
    
    except Exception as e:
        raise Exception(f"Dosya yükleme hatası ({uzanti}): {str(e)}")

    # 2. Metni parçalara böl (Chunking)
    # chunk_overlap: parçaların birbiriyle çakışma miktarı
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    parcalar = text_splitter.split_documents(dokumanlar)
    
    # ✅ Dosya hash'i metadata'ya ekle
    for doc in parcalar:
        doc.metadata["source_hash"] = dosya_hash

    # 3. Sayısallaştırma (Embedding) - Cache'den yükle
    embeddings = get_embeddings()

    # 4. Vektör Veritabanına Kaydet (Vector Store)
    vector_db = Chroma.from_documents(
        documents=parcalar, 
        embedding=embeddings, 
        persist_directory=DB_DIR
    )
    
    print(f"Başarılı! {len(parcalar)} adet parça hafızaya alındı. (Hash: {dosya_hash})")
    return vector_db

