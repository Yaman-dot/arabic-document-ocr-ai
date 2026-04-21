import sys
from src.BatchProcessor import BatchProcessor
from src.OCRRouter import OCRRouter
from src.PDFWriter import PDFWriter

def run(
    input_path: str,
    output_dir: str = "output",
    use_dynamic_threshold: bool = True,
    static_threshold: float = 0.85,
    bg_opacity: float = 0.15,
    dpi: int = 150,
    font_path: str = None,
):
    print("=" * 60)
    print(f"  Lumina OCR Batch Pipeline")
    print(f"  Input:     {input_path}")
    print(f"  Output:    {output_dir}")
    print(f"  Threshold: {'dynamic (batch-level)' if use_dynamic_threshold else f'static={static_threshold}'}")
    print("=" * 60)

    # Stage 1: resolve inputs
    batch = BatchProcessor(output_dir=output_dir, dpi=dpi)
    pages = batch.resolve(input_path)

    # Stage 2: OCR with confidence-based routing
    router = OCRRouter(
        use_dynamic_threshold=use_dynamic_threshold,
        static_threshold=static_threshold,
    )
    results = router.run(pages)

    # Stage 3: render output PDFs
    writer = PDFWriter(
        font_path=font_path,
        bg_opacity=bg_opacity,
    )
    output_paths = writer.write(results, batch, output_dir=output_dir)

    batch.close()

    print("\n" + "=" * 60)
    print("  Output PDFs:")
    for src, out in output_paths.items():
        print(f"    {src} → {out}")
    print("=" * 60)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Lumina OCR Batch Pipeline")
    parser.add_argument("input", help="Image file, PDF file, or folder")
    parser.add_argument("--output", default="output", help="Output directory")
    parser.add_argument("--static-threshold", type=float, default=0.85)
    parser.add_argument("--dynamic", action="store_true", default=True,
                        help="Use dynamic batch-level threshold (default)")
    parser.add_argument("--no-dynamic", dest="dynamic", action="store_false",
                        help="Use static threshold instead")
    parser.add_argument("--bg-opacity", type=float, default=0.15,
                        help="Original page background opacity (0=none, 1=full)")
    parser.add_argument("--dpi", type=int, default=150,
                        help="DPI for PDF page rendering")
    parser.add_argument("--font", default=None,
                        help="Path to Arabic TTF font (default: Amiri-Regular.ttf)")
    args = parser.parse_args()

    run(
        input_path=args.input,
        output_dir=args.output,
        use_dynamic_threshold=args.dynamic,
        static_threshold=args.static_threshold,
        bg_opacity=args.bg_opacity,
        dpi=args.dpi,
        font_path=args.font,
    )