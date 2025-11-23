"""
Face Detection Module - Detects faces in images using RetinaFace.

This module is responsible for detecting faces in images and returning
bounding boxes, landmarks, and confidence scores. It does NOT extract
face regions or embeddings - those are handled by other modules.

Responsibilities:
- Detect faces in images
- Return detection information (bbox, landmarks, confidence)
- Support both file paths and numpy arrays
"""

import cv2
import numpy as np
from typing import List, Dict
import insightface


class FaceDetector:
    """
    Face detection using RetinaFace model from InsightFace.
    
    This class handles face detection only. It detects faces in images
    and returns their locations, landmarks, and confidence scores.
    
    Example:
        detector = FaceDetector()
        detections = detector.detect("image.jpg")
        for det in detections:
            print(f"Face at {det['bbox']} with confidence {det['det_score']}")
    """
    
    def __init__(self, model_name: str = 'buffalo_l'):
        """
        Initialize RetinaFace detector.
        
        Args:
            model_name: InsightFace model variant
                - 'buffalo_l': Large model (most accurate, slower)
                - 'buffalo_m': Medium model (balanced)
                - 'buffalo_s': Small model (faster, less accurate)
        """
        # Initialize InsightFace FaceAnalysis app
        # Uses CPU by default; change to ['CUDAExecutionProvider'] for GPU
        self.app = insightface.app.FaceAnalysis(name=model_name, providers=['CPUExecutionProvider'])
        self.app.prepare(ctx_id=-1, det_size=(640, 640))
    
    def detect(self, image: np.ndarray) -> List[Dict]:
        """
        Detect faces in an image.
        
        Args:
            image: Image as numpy array (BGR format) or file path (str)
            
        Returns:
            List of detection dictionaries, each containing:
            - 'bbox': bounding box [x1, y1, x2, y2]
            - 'landmark': facial landmarks (if available)
            - 'det_score': detection confidence score (0.0 to 1.0)
        """
        # Load image if path is provided
        if isinstance(image, str):
            img = cv2.imread(image)
            if img is None:
                raise ValueError(f"Could not load image from {image}")
        else:
            img = image.copy()
        
        # Detect faces
        faces = self.app.get(img)
        
        # Format detections
        detections = []
        for face in faces:
            detection = {
                'bbox': face.bbox.astype(int).tolist(),  # [x1, y1, x2, y2]
                'landmark': face.landmark_2d_106.astype(int).tolist() if hasattr(face, 'landmark_2d_106') else None,
                'det_score': float(face.det_score)
            }
            detections.append(detection)
        
        return detections
    
    def detect_from_path(self, image_path: str) -> List[Dict]:
        """
        Detect faces from an image file path.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            List of detection dictionaries
        """
        return self.detect(image_path)

