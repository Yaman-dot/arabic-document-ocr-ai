import requests

SURYA_URL = "http://localhost:8001/ocr"
QWEN_URL  = "http://localhost:8002/ocr"

class MultiOCR:
    def __init__(self):
        self.surya_url = SURYA_URL
        self.qwen_url  = QWEN_URL

    def _call(self, url, image_path):
        with open(image_path, "rb") as f:
            response = requests.post(url, files={"file": f})
        return response.json()["result"]

    def surya(self, image_path):
        return self._call(self.surya_url, image_path)

    def qwen(self, image_path):
        return self._call(self.qwen_url, image_path)

    def both(self, image_path):
        return {
            "surya": self.surya(image_path),
            "qwen":  self.qwen(image_path),
        }