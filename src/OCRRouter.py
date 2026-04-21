import numpy as np
import requests
from dataclasses import dataclass, field
from typing import List, Optional
from pathlib import Path
from PIL import Image
import io

import sys
sys.path.append(str(Path(__file__).parent))
from BatchProcessor import PageRecord

SURYA_URL     = "http://localhost:8001/ocr"
QWEN_URL      = "http://localhost:8002/ocr"
QWEN_CROP_URL = "http://localhost:8002/ocr_crop"

# Minimum crop size — tiny slivers confuse Qwen
MIN_CROP_W = 20
MIN_CROP_H = 10


@dataclass
class TextLine:
    text: str
    confidence: float
    bbox: Optional[List[float]]   # [x1, y1, x2, y2] in image pixels


@dataclass
class OCRResult:
    text: str                     # full page text joined from lines
    text_lines: List[TextLine]    # per-line results with bbox for PDFWriter
    page_conf: float              # mean line confidence (Surya-sourced)
    model_used: str               # "surya" | "surya+qwen" | "qwen"
    page_record: PageRecord = field(repr=False)


class OCRRouter:
    def __init__(
        self,
        use_dynamic_threshold: bool = True,
        static_threshold: float = 0.85,
        surya_url: str = SURYA_URL,
        qwen_url: str = QWEN_URL,
        qwen_crop_url: str = QWEN_CROP_URL,
    ):
        self.use_dynamic_threshold = use_dynamic_threshold
        self.static_threshold = static_threshold
        self.surya_url     = surya_url
        self.qwen_url      = qwen_url
        self.qwen_crop_url = qwen_crop_url

        self.global_threshold: Optional[float] = None
        self._all_confidences: List[float] = []

    # ------------------------------------------------------------------ #
    #  Public API                                                          #
    # ------------------------------------------------------------------ #

    def run(self, pages: List[PageRecord]) -> List[OCRResult]:
        """
        Three-phase pipeline:
          Phase 1 — Surya on all pages → bboxes + confidences
          Phase 2 — Compute global threshold across all line confidences
          Phase 3 — Per line: keep Surya text OR re-OCR crop with Qwen
        """
        # Phase 1 — Surya on all pages
        print(f"\n[OCRRouter] Phase 1: Running Surya on {len(pages)} page(s)...")
        surya_results = [self._call_surya(p) for p in pages]

        # Phase 2 — Compute threshold
        if self.use_dynamic_threshold:
            self._compute_global_threshold(surya_results)
            threshold = self.global_threshold
            print(f"[OCRRouter] Dynamic threshold (batch-level): {threshold:.4f}")
        else:
            threshold = self.static_threshold
            print(f"[OCRRouter] Static threshold: {threshold:.4f}")

        # Phase 3 — Line-level routing
        print(f"\n[OCRRouter] Phase 3: Re-OCR'ing low-confidence lines with Qwen...")
        final_results = []
        total_lines = qwen_lines = 0

        for surya_res in surya_results:
            result = self._route_lines(surya_res, threshold)
            final_results.append(result)
            total_lines += len(result.text_lines)
            qwen_lines  += sum(1 for l in result.text_lines if l.confidence == 1.0)

        surya_lines = total_lines - qwen_lines
        print(f"\n[OCRRouter] Done. {surya_lines}/{total_lines} lines via Surya, "
              f"{qwen_lines}/{total_lines} lines via Qwen.")
        return final_results

    # ------------------------------------------------------------------ #
    #  Line-level routing                                                  #
    # ------------------------------------------------------------------ #

    def _route_lines(self, surya_res: OCRResult, threshold: float) -> OCRResult:
        """
        For each Surya text line:
          - confidence >= threshold → keep Surya text
          - confidence <  threshold → crop bbox from image → Qwen crop
        Bboxes always come from Surya, text content is decided per line.
        """
        page = surya_res.page_record
        img  = page.image
        corrected_lines = []
        any_qwen = False

        for line in surya_res.text_lines:
            if line.confidence >= threshold or line.bbox is None:
                corrected_lines.append(line)
                continue

            # Crop the line region out of the full page image
            crop = self._crop_line(img, line.bbox)
            if crop is None:
                corrected_lines.append(line)
                continue

            qwen_text = self._call_qwen_crop(crop)
            if qwen_text:
                corrected_lines.append(TextLine(
                    text=qwen_text,
                    confidence=1.0,   # Qwen result — no score available
                    bbox=line.bbox,   # keep Surya's layout position
                ))
                any_qwen = True
            else:
                corrected_lines.append(line)  # fallback to Surya

        full_text = "\n".join(l.text for l in corrected_lines)
        model = "surya+qwen" if any_qwen else "surya"

        return OCRResult(
            text=full_text,
            text_lines=corrected_lines,
            page_conf=surya_res.page_conf,
            model_used=model,
            page_record=page,
        )

    def _crop_line(self, img: Image.Image, bbox: List[float]) -> Optional[Image.Image]:
        """Crop a text line region from the full page image."""
        x1, y1, x2, y2 = [int(v) for v in bbox]
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(img.width,  x2)
        y2 = min(img.height, y2)

        if (x2 - x1) < MIN_CROP_W or (y2 - y1) < MIN_CROP_H:
            return None

        return img.crop((x1, y1, x2, y2))

    # ------------------------------------------------------------------ #
    #  Threshold computation                                               #
    # ------------------------------------------------------------------ #

    def _compute_global_threshold(self, results: List[OCRResult]):
        """
        Collect every line confidence across ALL pages,
        then threshold = mean - 0.5 * std.
        """
        self._all_confidences = []
        for res in results:
            for line in res.text_lines:
                self._all_confidences.append(line.confidence)

        if not self._all_confidences:
            self.global_threshold = self.static_threshold
            return

        arr = np.array(self._all_confidences)
        self.global_threshold = float(np.mean(arr) - 0.5 * np.std(arr))

    # ------------------------------------------------------------------ #
    #  Service calls                                                       #
    # ------------------------------------------------------------------ #

    def _image_to_bytes(self, image: Image.Image) -> bytes:
        buf = io.BytesIO()
        image.save(buf, format="PNG")
        buf.seek(0)
        return buf.getvalue()

    def _call_surya(self, page: PageRecord) -> OCRResult:
        img_bytes = self._image_to_bytes(page.image)
        try:
            response = requests.post(
                self.surya_url,
                files={"file": ("page.png", img_bytes, "image/png")},
                timeout=120,
            )
            response.raise_for_status()
            data = response.json()
        except Exception as e:
            print(f"  [Surya] Error on page {page.page_num}: {e}")
            return self._empty_result(page, "surya")

        text_lines = [
            TextLine(
                text=line["text"],
                confidence=line["confidence"],
                bbox=line["bbox"],
            )
            for line in data.get("text_lines", [])
        ]
        page_conf = (
            float(np.mean([l.confidence for l in text_lines]))
            if text_lines else 0.0
        )
        return OCRResult(
            text=data.get("text", ""),
            text_lines=text_lines,
            page_conf=page_conf,
            model_used="surya",
            page_record=page,
        )

    def _call_qwen_crop(self, crop: Image.Image) -> Optional[str]:
        """Send a single line crop to Qwen and get back corrected text."""
        img_bytes = self._image_to_bytes(crop)
        try:
            response = requests.post(
                self.qwen_crop_url,
                files={"file": ("crop.png", img_bytes, "image/png")},
                timeout=60,
            )
            response.raise_for_status()
            return response.json().get("text", "").strip() or None
        except Exception as e:
            print(f"  [Qwen/crop] Error: {e}")
            return None

    def _empty_result(self, page: PageRecord, model: str) -> OCRResult:
        return OCRResult(
            text="",
            text_lines=[],
            page_conf=0.0,
            model_used=model,
            page_record=page,
        )