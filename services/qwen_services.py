from fastapi import FastAPI, UploadFile
from src.SuryaOCR import SuryaOCR
import tempfile, os

app = FastAPI()
surya = SuryaOCR()

@app.post("/ocr")
async def ocr(file: UploadFile):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name
    surya.load_image(tmp_path)
    result = surya.inference()
    os.unlink(tmp_path)
    return {"result": str(result)}