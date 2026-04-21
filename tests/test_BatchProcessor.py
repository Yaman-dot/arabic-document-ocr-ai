import unittest
import tempfile
import os
from pathlib import Path
from PIL import Image
import fitz

from src.BatchProcessor import BatchProcessor, PageRecord

class TestBatchProcessor(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        self.test_dir = Path(self.temp_dir.name)
        
        # Create a dummy image
        self.img_path = self.test_dir / "test_image.png"
        img = Image.new('RGB', (100, 100), color='red')
        img.save(self.img_path)
        
        # Create a dummy PDF
        self.pdf_path = self.test_dir / "test_doc.pdf"
        doc = fitz.open()
        doc.new_page(width=200, height=200)
        doc.save(str(self.pdf_path))
        doc.close()

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_resolve_single_image(self):
        processor = BatchProcessor(output_dir=str(self.test_dir / "output"))
        records = processor.resolve(str(self.img_path))
        
        self.assertEqual(len(records), 1)
        self.assertFalse(records[0].is_pdf_page)
        self.assertEqual(records[0].source_path, str(self.img_path))
        self.assertEqual(records[0].page_num, 0)
        self.assertEqual(records[0].original_size, (100, 100))

    def test_resolve_single_pdf(self):
        processor = BatchProcessor(output_dir=str(self.test_dir / "output"))
        records = processor.resolve(str(self.pdf_path))
        
        self.assertEqual(len(records), 1)
        self.assertTrue(records[0].is_pdf_page)
        self.assertEqual(records[0].source_path, str(self.pdf_path))
        self.assertEqual(records[0].page_num, 0)
        processor.close()

    def test_resolve_folder(self):
        processor = BatchProcessor(output_dir=str(self.test_dir / "output"))
        records = processor.resolve(str(self.test_dir))
        
        self.assertEqual(len(records), 2)
        processor.close()

    def test_output_path_for(self):
        processor = BatchProcessor(output_dir=str(self.test_dir / "output"))
        out_path = processor.output_path_for(str(self.img_path))
        self.assertEqual(out_path.name, "test_image_digitalized.pdf")

if __name__ == '__main__':
    unittest.main()