import os
import logging

# ──────────────────────────────────────────────
# 🔇 Üçüncü parti telemetry/uyarı ayarları
#    (Diğer modüllerden ÖNCE import edildiği için burada yapılmalı)
# ──────────────────────────────────────────────
os.environ["ANONYMIZED_TELEMETRY"] = "False"          # ChromaDB telemetry kapat
os.environ["TOKENIZERS_PARALLELISM"] = "false"         # HuggingFace tokenizer uyarısını kapat
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"   # HuggingFace symlink uyarısını kapat

# ──────────────────────────────────────────────
# 📁 Proje Dizin Yapısı
# ──────────────────────────────────────────────
# Kök dizin (src/config.py olduğu için bir üst klasör proje köküdür)
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

# Sabit yollar
DB_DIR = os.path.join(BASE_DIR, 'db')
UPLOAD_DIR = os.path.join(BASE_DIR, 'uploads')

# Klasörleri otomatik oluştur (Varsa dokunmaz)
os.makedirs(DB_DIR, exist_ok=True)
os.makedirs(UPLOAD_DIR, exist_ok=True)

# ──────────────────────────────────────────────
# 🤖 Model & RAG Ayarları
# ──────────────────────────────────────────────
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

# Chunking (Metin parçalama) ayarları
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 100

# Retriever — sorguya en benzer kaç parçanın getirileceği
RETRIEVER_K = 5

# LLM modelleri
GROQ_MODEL = "llama-3.3-70b-versatile"
GEMINI_MODEL = "gemini-2.0-flash"

# Varsayılan LLM provider
DEFAULT_PROVIDER = "groq"

# ──────────────────────────────────────────────
# 📝 Loglama Ayarları
# ──────────────────────────────────────────────
LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
LOG_LEVEL = logging.INFO

logging.basicConfig(format=LOG_FORMAT, level=LOG_LEVEL)

# Gürültülü üçüncü parti loglarını sustur
logging.getLogger("chromadb").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("urllib3").setLevel(logging.WARNING)
logging.getLogger("sentence_transformers").setLevel(logging.WARNING)
