import os
import logging

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from src.backend.chat import soru_sor_stream, ozetle
from src.backend.memory import dokumani_hafizaya_al
from src.config import UPLOAD_DIR

logger = logging.getLogger(__name__)

app = FastAPI(title="RAG Backend API")

# Maksimum dosya boyutu: 50 MB
MAX_FILE_SIZE_MB = 50


class SoruIstegi(BaseModel):
    soru: str
    provider: str = "groq"  # Varsayılan sağlayıcı


@app.post("/upload")
async def dosya_yukle(dosya: UploadFile = File(...)):
    """Doküman yükle, parçala ve vektör veritabanına kaydet."""
    # Dosya uzantı kontrolü
    uzanti = os.path.splitext(dosya.filename)[1].lower()
    if uzanti not in ('.pdf', '.docx', '.txt'):
        raise HTTPException(
            status_code=400,
            detail=f"Desteklenmeyen dosya formatı: {uzanti}. PDF, DOCX veya TXT kullanın."
        )

    icerik = await dosya.read()

    # Dosya boyutu kontrolü
    boyut_mb = len(icerik) / (1024 * 1024)
    if boyut_mb > MAX_FILE_SIZE_MB:
        raise HTTPException(
            status_code=400,
            detail=f"Dosya çok büyük ({boyut_mb:.1f} MB). Maksimum: {MAX_FILE_SIZE_MB} MB."
        )

    dosya_yolu = os.path.join(UPLOAD_DIR, dosya.filename)
    with open(dosya_yolu, "wb") as f:
        f.write(icerik)

    try:
        sonuc = dokumani_hafizaya_al(dosya_yolu)
        return {"mesaj": f"{dosya.filename} başarıyla yüklendi.", "detay": sonuc}
    except Exception as e:
        logger.error("Dosya işleme hatası: %s", e)
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        # Geçici dosyayı her zaman temizle
        if os.path.exists(dosya_yolu):
            try:
                os.remove(dosya_yolu)
            except OSError:
                logger.warning("Geçici dosya silinemedi: %s", dosya_yolu)


@app.post("/ask/stream")
async def cevap_ver_stream(istek: SoruIstegi):
    """Soruya akışlı (streaming) yanıt döndür."""
    return StreamingResponse(
        soru_sor_stream(istek.soru, istek.provider),
        media_type="text/event-stream"
    )


@app.get("/summarize")
async def dokuman_ozetle(provider: str = "groq"):
    """Veritabanındaki dokümanları özetle."""
    try:
        summary = ozetle(provider)
        return {"ozet": summary}
    except Exception as e:
        logger.error("Özetleme hatası: %s", e)
        raise HTTPException(status_code=500, detail=str(e))