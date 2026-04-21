import os
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from chat import soru_sor_stream, ozetle
from memory import dokumani_hafizaya_al

app = FastAPI(title="RAG Backend API")

# Yüklenen dosyalar için geçici klasör oluştur (Ensure upload directory exists)
UPLOAD_DIR = "uploads"
if not os.path.exists(UPLOAD_DIR):
    os.makedirs(UPLOAD_DIR)

class SoruIstegi(BaseModel):
    soru: str
    provider: str = "groq" # Varsayılan sağlayıcı

@app.post("/upload")
async def dosya_yukle(dosya: UploadFile = File(...)):
    dosya_yolu = os.path.join(UPLOAD_DIR, dosya.filename)
    with open(dosya_yolu, "wb") as f:
        f.write(await dosya.read())
    
    dokumani_hafizaya_al(dosya_yolu)
    return {"mesaj": f"{dosya.filename} başarılı yüklendi"}

@app.post("/ask/stream")
async def cevap_ver_stream(istek: SoruIstegi):
    # Streaming Response (Akışlı Yanıt)
    return StreamingResponse(
        soru_sor_stream(istek.soru, istek.provider), 
        media_type="text/event-stream"
    )

@app.get("/summarize")
async def dokuman_ozetle(provider: str = "groq"):
    # Summary Endpoint (Özet Uç Noktası)
    try:
        summary = ozetle(provider)
        return {"ozet": summary}
    except Exception as e:
        return {"hata": str(e)}