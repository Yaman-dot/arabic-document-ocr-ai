from fastapi import FastAPI, UploadFile
from src.QwenOCR import QwenOCR
import tempfile
import os

app = FastAPI()
qwen = QwenOCR()


@app.post("/ocr")
async def ocr(file: UploadFile):
    """Full page OCR — used when Surya fails to detect any lines."""
    with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name

    qwen.load_image(tmp_path)
    result = qwen.inference()
    os.unlink(tmp_path)

    full_text = result or ""
    return {
        "text": full_text,
        "text_lines": [{"text": full_text, "confidence": 1.0, "bbox": None}],
        "page_conf": 1.0,
    }


@app.post("/ocr_crop")
async def ocr_crop(file: UploadFile):
    """
    Single line crop OCR — receives a cropped bbox region from Surya
    and returns the corrected text for that line only.
    """
    with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name

    qwen.load_image(tmp_path)
    result = qwen.inference(
        prompt="اقرأ النص العربي الموجود في هذه الصورة بدقة. أعد النص فقط بدون أي تعليقات أو إضافات."
    )
    os.unlink(tmp_path)

    return {"text": (result or "").strip()}