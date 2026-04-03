from src.QwenOCR import QwenOCR

qwen = QwenOCR()
qwen.load_image(r"data\input\Screenshot 2026-03-27 193051.png")
result = qwen.inference()
print(result)