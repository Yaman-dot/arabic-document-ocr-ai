from pathlib import Path
from typing import List, Dict, Tuple, Optional
from collections import defaultdict
from PIL import Image
import io
import numpy as np

import arabic_reshaper
from bidi.algorithm import get_display

from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import A4
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.lib.utils import ImageReader

import sys
sys.path.append(str(Path(__file__).parent))
from BatchProcessor import BatchProcessor, PageRecord
from OCRRouter import OCRResult

# ------------------------------------------------------------------ #
#  Arabic font — download Amiri if missing:                           #
#  https://github.com/aliftype/amiri/releases                        #
#  Place Amiri-Regular.ttf next to PDFWriter.py                      #
# ------------------------------------------------------------------ #
DEFAULT_FONT_CANDIDATES = [
    Path(__file__).parent / "Amiri-Regular.ttf",
    Path("C:/Windows/Fonts/arial.ttf"),
    Path("C:/Windows/Fonts/tahoma.ttf"),
]


class PDFWriter:
    def __init__(
        self,
        font_path: Optional[str] = None,
        bg_opacity: float = 0.15,   # 0.0 = invisible, 1.0 = full original
        font_size_bbox: float = 0.65,  # fraction of bbox height used as font size
        fallback_font_size: int = 12,  # font size for Qwen flowing layout
        min_font_size: int = 6,
        max_font_size: int = 32,
    ):
        self.bg_opacity = bg_opacity
        self.font_size_bbox = font_size_bbox
        self.fallback_font_size = fallback_font_size
        self.min_font_size = min_font_size
        self.max_font_size = max_font_size

        self.font_name = self._register_font(font_path)
        print(f"[PDFWriter] Using font: {self.font_name}")

    # ------------------------------------------------------------------ #
    #  Public API                                                          #
    # ------------------------------------------------------------------ #

    def write(
        self,
        results: List[OCRResult],
        batch: BatchProcessor,
        output_dir: Optional[str] = None,
    ) -> Dict[str, Path]:
        """
        Group results by source file and write one output PDF per source.
        Returns {source_path: output_pdf_path}.
        """
        # Group by source path, preserving page order
        grouped: Dict[str, List[OCRResult]] = defaultdict(list)
        for res in results:
            grouped[res.page_record.source_path].append(res)

        output_paths = {}
        for source_path, page_results in grouped.items():
            page_results.sort(key=lambda r: r.page_record.page_num)
            out_path = (
                Path(output_dir) / (Path(source_path).stem + "_digitalized.pdf")
                if output_dir
                else batch.output_path_for(source_path)
            )
            self._write_file(source_path, page_results, out_path)
            output_paths[source_path] = out_path

        return output_paths

    # ------------------------------------------------------------------ #
    #  Per-file rendering                                                  #
    # ------------------------------------------------------------------ #

    def _write_file(
        self,
        source_path: str,
        page_results: List[OCRResult],
        out_path: Path,
    ):
        out_path.parent.mkdir(parents=True, exist_ok=True)
        c = canvas.Canvas(str(out_path))

        for res in page_results:
            page_w_pts, page_h_pts = self._page_size_pts(res.page_record)
            c.setPageSize((page_w_pts, page_h_pts))

            # Layer 1: faded original page background
            self._draw_background(c, res.page_record, page_w_pts, page_h_pts)

            # Layer 2: OCR text
            if res.model_used == "surya" and any(
                l.bbox is not None for l in res.text_lines
            ):
                self._draw_bbox_text(c, res, page_w_pts, page_h_pts)
            else:
                self._draw_flowing_text(c, res, page_w_pts, page_h_pts)

            c.showPage()

        c.save()
        print(f"[PDFWriter] Saved → {out_path}")

    # ------------------------------------------------------------------ #
    #  Background layer                                                    #
    # ------------------------------------------------------------------ #

    def _draw_background(
        self,
        c: canvas.Canvas,
        page: PageRecord,
        page_w_pts: float,
        page_h_pts: float,
    ):
        if self.bg_opacity <= 0.0:
            return

        # Blend original image with white to achieve opacity effect
        img = page.image.convert("RGBA")
        white = Image.new("RGBA", img.size, (255, 255, 255, 255))
        alpha = int(self.bg_opacity * 255)
        overlay = Image.new("RGBA", img.size, (255, 255, 255, 255 - alpha))
        blended = Image.alpha_composite(img, overlay).convert("RGB")

        buf = io.BytesIO()
        blended.save(buf, format="PNG")
        buf.seek(0)

        c.drawImage(
            ImageReader(buf),
            0, 0,
            width=page_w_pts,
            height=page_h_pts,
            preserveAspectRatio=False,
        )

    # ------------------------------------------------------------------ #
    #  Surya: bbox-positioned text                                         #
    # ------------------------------------------------------------------ #

    def _draw_bbox_text(
        self,
        c: canvas.Canvas,
        res: OCRResult,
        page_w_pts: float,
        page_h_pts: float,
    ):
        dpi = self._get_dpi(res.page_record)
        scale = 72.0 / dpi  # pixel → PDF points

        for line in res.text_lines:
            if not line.text.strip() or line.bbox is None:
                continue

            x1, y1, x2, y2 = line.bbox
            # Convert to PDF points; flip Y (PDF origin = bottom-left)
            pdf_x1 = x1 * scale
            pdf_x2 = x2 * scale
            pdf_y1 = page_h_pts - y2 * scale  # top of bbox in PDF coords
            pdf_y2 = page_h_pts - y1 * scale  # bottom of bbox in PDF coords

            bbox_h = pdf_y2 - pdf_y1
            bbox_w = pdf_x2 - pdf_x1

            font_size = max(
                self.min_font_size,
                min(self.max_font_size, bbox_h * self.font_size_bbox),
            )

            shaped = self._shape_arabic(line.text)
            c.setFont(self.font_name, font_size)
            c.setFillColorRGB(0, 0, 0)

            # RTL: anchor text to right edge of bbox
            text_y = pdf_y1 + (bbox_h - font_size) / 2  # vertically centre in bbox
            c.drawRightString(pdf_x2, text_y, shaped)

    # ------------------------------------------------------------------ #
    #  Qwen: flowing RTL text                                              #
    # ------------------------------------------------------------------ #

    def _draw_flowing_text(
        self,
        c: canvas.Canvas,
        res: OCRResult,
        page_w_pts: float,
        page_h_pts: float,
    ):
        margin = 40
        x_right  = page_w_pts - margin
        y_cursor = page_h_pts - margin
        line_height = self.fallback_font_size * 1.6

        c.setFont(self.font_name, self.fallback_font_size)
        c.setFillColorRGB(0, 0, 0)

        for line in res.text_lines:
            for text_line in line.text.split("\n"):
                if not text_line.strip():
                    y_cursor -= line_height * 0.5
                    continue

                shaped = self._shape_arabic(text_line)
                c.drawRightString(x_right, y_cursor, shaped)
                y_cursor -= line_height

                if y_cursor < margin:
                    c.showPage()
                    c.setPageSize((page_w_pts, page_h_pts))
                    self._draw_background(
                        c, res.page_record, page_w_pts, page_h_pts
                    )
                    c.setFont(self.font_name, self.fallback_font_size)
                    y_cursor = page_h_pts - margin

    # ------------------------------------------------------------------ #
    #  Helpers                                                             #
    # ------------------------------------------------------------------ #

    def _shape_arabic(self, text: str) -> str:
        """Reshape Arabic and apply BiDi for correct RTL visual order."""
        reshaped = arabic_reshaper.reshape(text)
        return get_display(reshaped)

    def _page_size_pts(self, page: PageRecord) -> Tuple[float, float]:
        """Return page dimensions in PDF points."""
        if page.is_pdf_page:
            # original_size is already in PDF points from fitz
            return page.original_size
        else:
            # original_size is in pixels — convert using the render DPI
            dpi = self._get_dpi(page)
            w_pts = page.original_size[0] * 72.0 / dpi
            h_pts = page.original_size[1] * 72.0 / dpi
            return (w_pts, h_pts)

    def _get_dpi(self, page: PageRecord) -> float:
        """Infer the DPI used to render the PIL image stored in the PageRecord."""
        if page.is_pdf_page and page.original_size:
            # original_size = PDF points, image.size = rendered pixels
            # dpi = pixels / (points / 72)
            scale_x = page.image.size[0] / page.original_size[0]
            return scale_x * 72.0
        return 150.0  # default BatchProcessor DPI

    def _register_font(self, font_path: Optional[str]) -> str:
        candidates = (
            [Path(font_path)] if font_path else DEFAULT_FONT_CANDIDATES
        )
        for path in candidates:
            if path.exists():
                try:
                    pdfmetrics.registerFont(TTFont("ArabicFont", str(path)))
                    return "ArabicFont"
                except Exception as e:
                    print(f"[PDFWriter] Could not register {path}: {e}")

        print(
            "[PDFWriter] WARNING: No Arabic font found. "
            "Download Amiri-Regular.ttf from https://github.com/aliftype/amiri/releases "
            "and place it next to PDFWriter.py"
        )
        return "Helvetica"  # ASCII fallback — Arabic will not render correctly