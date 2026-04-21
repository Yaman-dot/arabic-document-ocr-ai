from src.MultiOCR import MultiOCR

ocr = MultiOCR()
results = ocr.both(r"data\input\Screenshot 2026-03-27 193051.png")
print(results["surya"])
print(results["qwen"])