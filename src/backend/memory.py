import os
import hashlib
import logging

from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

from src.config import DB_DIR, EMBEDDING_MODEL, CHUNK_SIZE, CHUNK_OVERLAP

# .env dosyasındaki anahtarları oku
load_dotenv()

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────
# 🧠 Embeddings Cache — modeli bir kez yükle, tekrar kullan
# ──────────────────────────────────────────────
_embeddings_cache = None


def get_embeddings():
    """Embeddings modelini cache ederek yükle (uygulama boyunca bir kez yüklenecek)."""
    global _embeddings_cache
    if _embeddings_cache is None:
        logger.info("Embedding modeli yükleniyor: %s", EMBEDDING_MODEL)
        _embeddings_cache = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
        logger.info("Embedding modeli başarıyla yüklendi.")
    return _embeddings_cache


# ──────────────────────────────────────────────
# 🔍 Duplikasyon Tespiti (Duplicate Detection)
# ──────────────────────────────────────────────
def get_file_hash(dosya_yolu: str) -> str:
    """Dosyanın MD5 hash'ini hesapla — aynı dosyanın tekrar yüklenmesini önlemek için."""
    with open(dosya_yolu, 'rb') as f:
        return hashlib.md5(f.read()).hexdigest()


def dosya_zaten_var_mi(dosya_hash: str) -> bool:
    """Veritabanında bu hash'e sahip bir doküman var mı kontrol et."""
    try:
        vector_db = Chroma(persist_directory=DB_DIR, embedding_function=get_embeddings())
        results = vector_db.get(where={"source_hash": dosya_hash})
        return len(results.get('ids', [])) > 0
    except Exception:
        # where filtresi başarısız olursa fallback: tüm dokümanları tara
        try:
            vector_db = Chroma(persist_directory=DB_DIR, embedding_function=get_embeddings())
            all_docs = vector_db.get()
            for metadata in all_docs.get('metadatas', []):
                if metadata.get('source_hash') == dosya_hash:
                    return True
        except Exception:
            pass  # Veritabanı henüz yoksa — yeni başlıyor demektir
        return False


# ──────────────────────────────────────────────
# 📄 Doküman Yükleme & Embedding (Ana İş Mantığı)
# ──────────────────────────────────────────────
def dokumani_hafizaya_al(dosya_yolu: str) -> dict:
    """
    Bir dokümanı yükle, parçala, embedding yap ve vektör veritabanına kaydet.

    Returns:
        dict: İşlem sonucu bilgisi — {"dosya": str, "parca_sayisi": int, "zaten_vardi": bool}
    """
    # 1. Dosya var mı kontrol et
    if not os.path.exists(dosya_yolu):
        raise FileNotFoundError(f"Dosya bulunamadı: {dosya_yolu}")

    dosya_adi = os.path.basename(dosya_yolu)

    # 2. Duplikasyon kontrolü
    dosya_hash = get_file_hash(dosya_yolu)
    if dosya_zaten_var_mi(dosya_hash):
        logger.info("Dosya zaten mevcut, atlanıyor: %s", dosya_adi)
        return {"dosya": dosya_adi, "parca_sayisi": 0, "zaten_vardi": True}

    # 3. Dosya formatına göre loader seç
    uzanti = os.path.splitext(dosya_yolu)[1].lower()
    dokumanlar = _dosyayi_yukle(dosya_yolu, uzanti)

    if not dokumanlar:
        raise ValueError(f"Dosyadan içerik yüklenemedi: {dosya_adi}")

    # 4. Metni parçalara böl (Chunking)
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP
    )
    parcalar = text_splitter.split_documents(dokumanlar)

    # 5. Her parçanın metadata'sına dosya hash'ini ekle (ileride duplicate detection için)
    for doc in parcalar:
        doc.metadata["source_hash"] = dosya_hash

    # 6. Vektör veritabanına kaydet
    embeddings = get_embeddings()
    Chroma.from_documents(
        documents=parcalar,
        embedding=embeddings,
        persist_directory=DB_DIR
    )

    logger.info("✅ %s → %d parça hafızaya alındı.", dosya_adi, len(parcalar))
    return {"dosya": dosya_adi, "parca_sayisi": len(parcalar), "zaten_vardi": False}


def _dosyayi_yukle(dosya_yolu: str, uzanti: str):
    """Dosya uzantısına göre uygun loader ile dokümanı yükle."""
    try:
        if uzanti == '.pdf':
            return PyPDFLoader(dosya_yolu).load()

        elif uzanti == '.txt':
            return _txt_yukle(dosya_yolu)

        elif uzanti == '.docx':
            from langchain_community.document_loaders import Docx2txtLoader
            return Docx2txtLoader(dosya_yolu).load()

        else:
            raise ValueError(
                f"Desteklenmeyen dosya formatı: {uzanti}. "
                "Lütfen PDF, DOCX veya TXT kullanınız."
            )
    except ValueError:
        raise  # ValueError'ları olduğu gibi yukarı ilet
    except Exception as e:
        raise RuntimeError(f"Dosya yükleme hatası ({uzanti}): {e}") from e


def _txt_yukle(dosya_yolu: str):
    """TXT dosyasını birden fazla encoding denemesiyle yükle."""
    for encoding in ('utf-8', 'windows-1252', 'iso-8859-9'):
        try:
            return TextLoader(dosya_yolu, encoding=encoding).load()
        except Exception:
            continue

    # Son çare: hataya toleranslı okuma
    logger.warning("Standart encoding'ler başarısız oldu, hataya toleranslı okuma yapılıyor: %s", dosya_yolu)
    with open(dosya_yolu, 'r', encoding='utf-8', errors='replace') as f:
        icerik = f.read()
    from langchain_core.documents import Document
    return [Document(page_content=icerik, metadata={"source": dosya_yolu})]


# ──────────────────────────────────────────────
# 🗑️ Veritabanı Yönetimi
# ──────────────────────────────────────────────
def veritabanini_sifirla() -> bool:
    """
    Vektör veritabanındaki TÜM verileri sil.
    ChromaDB'nin kendi API'sini kullanır — dosya kilidi sorunu olmaz.
    """
    try:
        import chromadb
        client = chromadb.PersistentClient(path=DB_DIR)

        # Tüm koleksiyonları sil
        koleksiyonlar = client.list_collections()
        for kol in koleksiyonlar:
            client.delete_collection(kol.name)
            logger.info("Koleksiyon silindi: %s", kol.name)

        if not koleksiyonlar:
            logger.info("Silinecek koleksiyon bulunamadı — veritabanı zaten boş.")
        else:
            logger.info("Vektör veritabanı sıfırlandı — %d koleksiyon silindi.", len(koleksiyonlar))

        return True
    except Exception as e:
        logger.error("Veritabanı sıfırlanamadı: %s", e)
        return False
