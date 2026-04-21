import os
from pathlib import Path
from PIL import Image
from typing import List, Tuple, Dict
from dataclasses import dataclass, field
import fitz  # pymupdf

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp"}
PDF_EXTENSION = ".pdf"


@dataclass
class PageRecord:
    """Represents one page/image unit to be OCR'd."""
    image: Image.Image          # PIL image ready for OCR
    source_path: str            # original file path
    page_num: int               # 0-indexed (always 0 for standalone images)
    original_size: Tuple[int, int]  # (width, height) in points/pixels
    is_pdf_page: bool           # True if extracted from a PDF
    pdf_doc: fitz.Document = field(default=None, repr=False)  # kept open for visual layer


class BatchProcessor:
    def __init__(self, output_dir: str = "output", dpi: int = 150):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.dpi = dpi
        self._pdf_docs: Dict[str, fitz.Document] = {}  # cache open PDF docs

    # ------------------------------------------------------------------ #
    #  Public API                                                          #
    # ------------------------------------------------------------------ #

    def resolve(self, input_path: str) -> List[PageRecord]:
        """
        Accept a single image, single PDF, or a folder containing either.
        Returns a flat ordered list of PageRecords ready for OCR.
        """
        p = Path(input_path)

        if not p.exists():
            raise FileNotFoundError(f"Input path does not exist: {input_path}")

        if p.is_dir():
            return self._process_folder(p)
        elif p.suffix.lower() == PDF_EXTENSION:
            return self._process_pdf(p)
        elif p.suffix.lower() in IMAGE_EXTENSIONS:
            return [self._process_image(p)]
        else:
            raise ValueError(f"Unsupported file type: {p.suffix}")

    def output_path_for(self, source_path: str) -> Path:
        """Returns the output PDF path for a given source file."""
        stem = Path(source_path).stem
        return self.output_dir / f"{stem}_digitalized.pdf"

    def close(self):
        """Release all open PDF documents."""
        for doc in self._pdf_docs.values():
            doc.close()
        self._pdf_docs.clear()

    # ------------------------------------------------------------------ #
    #  Internal helpers                                                    #
    # ------------------------------------------------------------------ #

    def _process_folder(self, folder: Path) -> List[PageRecord]:
        records = []
        files = sorted(folder.iterdir())

        for f in files:
            if f.suffix.lower() == PDF_EXTENSION:
                records.extend(self._process_pdf(f))
            elif f.suffix.lower() in IMAGE_EXTENSIONS:
                records.append(self._process_image(f))

        if not records:
            raise ValueError(f"No supported files found in folder: {folder}")

        print(f"[BatchProcessor] Found {len(records)} page(s) across {folder}")
        return records

    def _process_pdf(self, pdf_path: Path) -> List[PageRecord]:
        doc = fitz.open(str(pdf_path))
        self._pdf_docs[str(pdf_path)] = doc  # keep open for PDFWriter visual layer

        records = []
        for page_num in range(len(doc)):
            page = doc[page_num]
            pil_image = self._fitz_page_to_pil(page)
            original_size = (page.rect.width, page.rect.height)  # in PDF points

            records.append(PageRecord(
                image=pil_image,
                source_path=str(pdf_path),
                page_num=page_num,
                original_size=original_size,
                is_pdf_page=True,
                pdf_doc=doc,
            ))

        print(f"[BatchProcessor] {pdf_path.name}: {len(records)} page(s)")
        return records

    def _process_image(self, image_path: Path) -> PageRecord:
        img = Image.open(str(image_path)).convert("RGB")
        print(f"[BatchProcessor] {image_path.name}: single image")
        return PageRecord(
            image=img,
            source_path=str(image_path),
            page_num=0,
            original_size=img.size,  # (width, height) in pixels
            is_pdf_page=False,
            pdf_doc=None,
        )

    def _fitz_page_to_pil(self, page: fitz.Page) -> Image.Image:
        """Render a PDF page to a PIL Image at self.dpi."""
        zoom = self.dpi / 72  # PDF default is 72 DPI
        mat = fitz.Matrix(zoom, zoom)
        pix = page.get_pixmap(matrix=mat, alpha=False)
        return Image.frombytes("RGB", [pix.width, pix.height], pix.samples)