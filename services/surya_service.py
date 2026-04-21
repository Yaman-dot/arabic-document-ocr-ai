from fastapi import FastAPI, UploadFile
from src.SuryaOCR import SuryaOCR
import tempfile
import os

app = FastAPI()
surya = SuryaOCR()

@app.post("/ocr")
async def ocr(file: UploadFile):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name

    surya.load_image(tmp_path)
    predictions = surya.inference()
    os.unlink(tmp_path)

    if not predictions:
        return {"text": "", "text_lines": [], "page_conf": 0.0}

    pred = predictions[0]
    text_lines = []
    for line in pred.text_lines:
        conf = getattr(line, "confidence", getattr(line, "score", 1.0))
        # bbox: [x1, y1, x2, y2]
        bbox = list(line.bbox) if hasattr(line, "bbox") else None
        text_lines.append({
            "text": line.text,
            "confidence": conf,
            "bbox": bbox,
        })

    all_confs = [l["confidence"] for l in text_lines]
    page_conf = sum(all_confs) / len(all_confs) if all_confs else 0.0
    full_text = "\n".join(l["text"] for l in text_lines)

    return {
        "text": full_text,
        "text_lines": text_lines,
        "page_conf": page_conf,
    }