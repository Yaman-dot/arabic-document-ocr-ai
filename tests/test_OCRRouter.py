import unittest
from unittest.mock import patch, MagicMock
from PIL import Image
import numpy as np

from src.OCRRouter import OCRRouter, OCRResult, TextLine

SURYA_URL = "http://127.0.0.1:8001/ocr"
QWEN_URL  = "http://127.0.0.1:8002/ocr"
class TestOCRRouter(unittest.TestCase):
    def setUp(self):
        self.router = OCRRouter(
            use_dynamic_threshold=False,
            static_threshold=0.85,
            surya_url=SURYA_URL,
            qwen_url=QWEN_URL
        )
        
        # Create a mock PageRecord so we don't depend on BatchProcessor internals
        self.mock_page = MagicMock()
        self.mock_page.image = Image.new('RGB', (100, 100), color='white')
        self.mock_page.page_num = 1
        self.mock_page.source_path = "dummy/path/doc.pdf"

    def test_image_to_bytes(self):
        img_bytes = self.router._image_to_bytes(self.mock_page.image)
        self.assertIsInstance(img_bytes, bytes)
        self.assertTrue(img_bytes.startswith(b'\x89PNG\r\n\x1a\n'))  # Valid PNG signature

    @patch('src.OCRRouter.requests.post')
    def test_call_surya_success(self, mock_post):
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "text": "Hello Surya",
            "text_lines": [
                {"text": "Hello", "confidence": 0.9, "bbox": [0, 0, 10, 10]},
                {"text": "Surya", "confidence": 0.8, "bbox": [10, 0, 20, 10]}
            ]
        }
        mock_post.return_value = mock_response

        result = self.router._call_surya(self.mock_page)

        self.assertEqual(result.model_used, "surya")
        self.assertEqual(result.text, "Hello Surya")
        self.assertEqual(len(result.text_lines), 2)
        self.assertAlmostEqual(result.page_conf, 0.85)  # mean of [0.9, 0.8]
        mock_post.assert_called_once()

    @patch('src.OCRRouter.requests.post')
    def test_call_surya_failure(self, mock_post):
        # Simulate a timeout or connection error
        mock_post.side_effect = Exception("Network Error")
        
        result = self.router._call_surya(self.mock_page)
        
        self.assertEqual(result.model_used, "surya")
        self.assertEqual(result.text, "")
        self.assertEqual(result.page_conf, 0.0)
        self.assertEqual(len(result.text_lines), 0)

    @patch('src.OCRRouter.requests.post')
    def test_call_qwen_success(self, mock_post):
        mock_response = MagicMock()
        mock_response.json.return_value = {"text": "Hello Qwen"}
        mock_post.return_value = mock_response

        result = self.router._call_qwen(self.mock_page)

        self.assertEqual(result.model_used, "qwen")
        self.assertEqual(result.text, "Hello Qwen")
        self.assertEqual(len(result.text_lines), 1)
        self.assertEqual(result.page_conf, 1.0)
        self.assertIsNone(result.text_lines[0].bbox)

    def test_compute_global_threshold(self):
        # Create fake text line confidences across two pages
        res1 = OCRResult(
            text="", 
            text_lines=[TextLine("", 0.9, None), TextLine("", 0.8, None)], 
            page_conf=0.85, 
            model_used="surya", 
            page_record=self.mock_page
        )
        res2 = OCRResult(
            text="", 
            text_lines=[TextLine("", 0.7, None)], 
            page_conf=0.7, 
            model_used="surya", 
            page_record=self.mock_page
        )
        
        self.router._compute_global_threshold([res1, res2])
        
        # [0.9, 0.8, 0.7] -> mean = 0.8, std = ~0.08165
        expected_mean = np.mean([0.9, 0.8, 0.7])
        expected_std = np.std([0.9, 0.8, 0.7])
        expected_thresh = float(expected_mean - 0.5 * expected_std)
        
        self.assertAlmostEqual(self.router.global_threshold, expected_thresh)

    @patch.object(OCRRouter, '_call_qwen')
    @patch.object(OCRRouter, '_call_surya')
    def test_run_routing(self, mock_call_surya, mock_call_qwen):
        self.router.use_dynamic_threshold = False
        self.router.static_threshold = 0.85

        page1 = MagicMock(page_num=1, source_path="p1.pdf")
        page2 = MagicMock(page_num=2, source_path="p2.pdf")

        # Mock: Surya returns high conf for page1, low conf for page2
        mock_call_surya.side_effect = [
            OCRResult("P1", [], 0.9, "surya", page1),
            OCRResult("P2", [], 0.7, "surya", page2)
        ]

        # Mock: Qwen handles page2 successfully
        mock_call_qwen.return_value = OCRResult("P2 Qwen", [], 1.0, "qwen", page2)

        results = self.router.run([page1, page2])

        self.assertEqual(len(results), 2)
        
        # Verify Page 1 was handled by Surya and Page 2 was re-routed to Qwen
        self.assertEqual(results[0].model_used, "surya")
        self.assertEqual(results[1].model_used, "qwen")
        
        self.assertEqual(mock_call_surya.call_count, 2)
        mock_call_qwen.assert_called_once_with(page2)

if __name__ == '__main__':
    unittest.main()