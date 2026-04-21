# Byanat OCR

A Windows-based Arabic document digitalization pipeline that combines [Surya OCR](https://github.com/VikParuchuri/surya) and [Qwen2.5-VL](https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct) into a dual-model, confidence-routed batch processing system. Scanned Arabic documents go in — clean, searchable, layout-aware PDFs come out.

---

## How It Works

Byanat runs in three phases per batch:

**Phase 1 — Surya on everything**
Surya OCR processes every page in the batch, producing per-line bounding boxes and confidence scores. This gives us layout (positions, line order) for the entire document.

**Phase 2 — Global threshold computation**
All line confidences across the entire batch are collected. A dynamic threshold is computed as `mean - 0.5 × std`. This batch-level calibration means the threshold adapts to the difficulty of your specific document set rather than being a fixed value.

**Phase 3 — Line-level routing to Qwen**
For each text line, if Surya's confidence is below the threshold, the bounding box region is cropped from the original image and sent to Qwen2.5-VL for re-recognition. Qwen's accurate text replaces Surya's output for that line — but Surya's bounding box is always preserved. This means:
- **Layout always comes from Surya** (fast, structured, bbox-accurate)
- **Text accuracy comes from Qwen** (context-aware, semantically correct)
- The output PDF positions every line at its correct spatial location

```
Input (image / PDF / folder)
        ↓
BatchProcessor — resolves inputs, converts PDF pages to images
        ↓
OCRRouter
  Phase 1 → Surya full page → bboxes + confidences
  Phase 2 → compute batch-wide dynamic threshold
  Phase 3 → low-confidence lines → crop → Qwen
        ↓
PDFWriter
  Layer 1 → faded original page (visual context)
  Layer 2 → OCR text at Surya bbox coordinates (RTL, Arabic-reshaped)
        ↓
Output PDF (_digitalized.pdf)
```

---

## Project Structure

```
Byanat/ocr/
├── src/
│   ├── SuryaOCR.py           # Surya OCR wrapper
│   ├── QwenOCR.py            # Qwen2.5-VL wrapper
│   ├── BatchProcessor.py     # Input resolver (image / PDF / folder)
│   ├── OCRRouter.py          # Confidence-based dual-model routing
│   └── PDFWriter.py          # RTL Arabic PDF renderer (reportlab)
├── services/
│   ├── surya_service.py      # FastAPI service — port 8001
│   └── qwen_service.py       # FastAPI service — port 8002
├── venv-surya/               # Surya virtual environment
├── venv-qwen/                # Qwen virtual environment
├── launch_services.py        # Starts both services, health checks, auto-restart
├── main_batch.py             # CLI entry point for batch processing
├── requirements-surya.txt
└── requirements-qwen.txt
```

---

## Requirements

- Windows 10/11 (tested), Python 3.10+
- NVIDIA GPU with CUDA 12.x drivers (13.x driver works with cu124 PyTorch build)
- ~6 GB VRAM minimum (Surya ~1 GB + Qwen 4-bit ~3 GB)
- [Amiri-Regular.ttf](https://github.com/aliftype/amiri/releases) placed in `src/`

---

## Setup

### 1. Clone the repo



### 2. Create the Surya venv

```powershell
python -m venv venv-surya
.\venv-surya\Scripts\activate
pip install -r requirements-surya.txt
deactivate
```

### 3. Create the Qwen venv

```powershell
python -m venv venv-qwen
.\venv-qwen\Scripts\activate
pip install -r requirements-qwen.txt
deactivate
```

### 4. Install PyTorch with CUDA in both venvs

```powershell
# Run in each venv separately
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
```

### 5. Download the Qwen model

```powershell
.\venv-qwen\Scripts\activate
python -c "
from huggingface_hub import snapshot_download
snapshot_download(repo_id='sherif1313/Arabic-handwritten-OCR-4bit-Qwen2.5-VL-3B-v2')
"
```

### 6. Download the Arabic font

Download [Amiri-Regular.ttf](https://github.com/aliftype/amiri/releases) and place it in `src/`.

---

## Usage

### Start services + run batch pipeline

```powershell
# Activate either venv (only needs `requests`)
.\venv-surya\Scripts\activate

# Single image
python launch_services.py --run-batch "docs\scan.png"

# Single PDF
python launch_services.py --run-batch "docs\document.pdf"

# Folder of PDFs and images
python launch_services.py --run-batch "docs\" --output "output\"
```

### Options

| Flag | Default | Description |
|---|---|---|
| `--output` | `output` | Output directory for digitalized PDFs |
| `--dynamic` | `True` | Use batch-level dynamic confidence threshold |
| `--no-dynamic` | — | Use static threshold instead |
| `--static-threshold` | `0.85` | Threshold value when `--no-dynamic` is set |
| `--bg-opacity` | `0.15` | Original page background opacity (0 = none, 1 = full) |
| `--dpi` | `150` | DPI for PDF page rendering |
| `--font` | auto | Path to Arabic TTF font |

### Keep services running (for repeated use)

```powershell
# Terminal 1 — start and keep alive
python launch_services.py

# Terminal 2 — run batch against already-running services
python main_batch.py "docs\" --output "output\"
```

### Service endpoints

| Service | Port | Endpoints |
|---|---|---|
| Surya | 8001 | `POST /ocr` — full page OCR |
| Qwen | 8002 | `POST /ocr` — full page OCR |
| Qwen | 8002 | `POST /ocr_crop` — single line crop OCR |

Swagger UI available at `http://localhost:8001/docs` and `http://localhost:8002/docs`.

---

## Models

| Model | Role | VRAM |
|---|---|---|
| [Surya OCR](https://github.com/VikParuchuri/surya) | Layout detection, bounding boxes, initial text, confidence scoring | ~1 GB |
| [sherif1313/Arabic-handwritten-OCR-4bit-Qwen2.5-VL-3B-v2](https://huggingface.co/sherif1313/Arabic-handwritten-OCR-4bit-Qwen2.5-VL-3B-v2) | High-accuracy Arabic text recognition on low-confidence line crops | ~3 GB |

---

## Why Two Venvs?

Surya and Qwen have conflicting dependencies (different `transformers` versions, model-specific requirements). Running them as separate FastAPI microservices on different ports lets each use its own isolated environment while communicating cleanly over HTTP.

---

## Output

Each input file produces a `<filename>_digitalized.pdf` in the output directory containing:
- A faded version of the original page as a background layer (configurable opacity)
- Clean Arabic text rendered at the correct spatial positions using Surya's bounding boxes
- Fully RTL-correct text using `arabic-reshaper` and `python-bidi`

---

## License

MIT