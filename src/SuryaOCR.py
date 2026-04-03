import os
os.environ["TORCH_DEVICE"] = "cuda" 

import numpy as np
from PIL import Image
import torch
from surya.recognition import RecognitionPredictor
from surya.detection import DetectionPredictor
from surya.foundation import FoundationPredictor
class SuryaOCR:
    def __init__(self):
        self.foundation_predictor = FoundationPredictor()
        self.recognition_predictor = RecognitionPredictor(self.foundation_predictor)
        self.detection_predictor = DetectionPredictor()
        self.image = None
        self.IMAGE_PATH = None
        self.predictions = None  
        self.dyn_threshold = None

        print("Successfully loaded Surya OCR")
        print(f"VRAM after Surya: {torch.cuda.memory_allocated()/1e9:.2f} GB")

    def load_image(self, image_path: str):
        self.IMAGE_PATH = image_path
        self.image = Image.open(image_path).convert("RGB")
    
    '''def inference(self, langs=["en", "ar"]):
        if self.image is None:
            print("No image loaded.")
            return None

        images = [self.image]
        
        self.predictions = self.recognition_predictor(
            images=images,
            langs=[langs],
            det_predictor=self.detection_predictor,
            task_names=["ocr"] * len(images)
        )
        return self.predictions'''
    def inference(self):
        if self.image is None:
            print("No image loaded.")
            return None

        images = [self.image]

        # Call with detection predictor
        self.predictions = self.recognition_predictor(
            images,
            det_predictor=self.detection_predictor
        )

        return self.predictions
    def calculate_dynamic_threshold(self):
        if not self.predictions:
            raise ValueError("Run inference() first.")

        pred = self.predictions[0]
        confidences = [getattr(line, "confidence", getattr(line, "score", 1.0)) for line in pred.text_lines]

        mean_conf = np.mean(confidences)
        std_conf = np.std(confidences)

        # Dynamic threshold = mean - 0.5 * std
        self.dyn_threshold = mean_conf - 0.5 * std_conf
        return self.dyn_threshold

    def flag_low_conf_paper(self, static_conf=None, use_static_threshold=False):
        if not self.predictions:
            raise ValueError("Run inference() first.")

        threshold = static_conf if use_static_threshold else self.dyn_threshold
        if threshold is None:
            threshold = self.calculate_dynamic_threshold()

        pred = self.predictions[0]
        confidences = [getattr(line, "confidence", getattr(line, "score", 1.0)) for line in pred.text_lines]

        # Page-level confidence = mean of line confidences
        page_conf = np.mean(confidences)

        flag = page_conf < threshold
        return flag, page_conf