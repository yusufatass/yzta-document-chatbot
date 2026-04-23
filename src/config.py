import os

# Kök dizin (src/config.py olduğu için bir üst klasör proje köküdür)
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

# Sabit yollar
DB_DIR = os.path.join(BASE_DIR, 'db')
UPLOAD_DIR = os.path.join(BASE_DIR, 'uploads')

# Klasörleri otomatik oluştur (Varsa dokunmaz)
os.makedirs(DB_DIR, exist_ok=True)
os.makedirs(UPLOAD_DIR, exist_ok=True)
