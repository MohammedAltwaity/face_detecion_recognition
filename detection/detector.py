"""
Face Detection Module - Detects faces in images using RetinaFace.

This module is responsible for detecting faces in images and returning
bounding boxes, landmarks, and confidence scores. It can also save
detected faces to a folder.

Responsibilities:
- Detect faces in images
- Return detection information (bbox, landmarks, confidence)
- Optionally save detected faces to folder with timestamps
- Support both file paths and numpy arrays
"""

import cv2
import numpy as np
from typing import List, Dict, Optional
import insightface
import os
from datetime import datetime


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
    
    def __init__(self, model_name: str = 'buffalo_l', save_detected_faces: bool = True):
        """
        Initialize RetinaFace detector.
        
        Args:
            model_name: InsightFace model variant
                - 'buffalo_l': Large model (most accurate, slower)
                - 'buffalo_m': Medium model (balanced)
                - 'buffalo_s': Small model (faster, less accurate)
            save_detected_faces: If True, saves detected faces to folder
        """
        # Initialize InsightFace FaceAnalysis app
        # Uses CPU by default; change to ['CUDAExecutionProvider'] for GPU
        self.app = insightface.app.FaceAnalysis(name=model_name, providers=['CPUExecutionProvider'])
        self.app.prepare(ctx_id=-1, det_size=(640, 640))
        
        # Create detected_faces folder if saving is enabled
        self.save_detected_faces = save_detected_faces
        self.detected_faces_dir = os.path.join(os.path.dirname(__file__), "detected_faces")
        if self.save_detected_faces:
            os.makedirs(self.detected_faces_dir, exist_ok=True)
    
    def _extract_and_save_face(self, image: np.ndarray, bbox: List[int], 
                               face_number: int, timestamp: str) -> Optional[str]:
        """
        Extract face region and save to detected_faces folder.
        
        Args:
            image: Full image
            bbox: Bounding box [x1, y1, x2, y2]
            face_number: Face number for filename (1, 2, 3, ...)
            timestamp: Timestamp string
            
        Returns:
            Path to saved face image, or None if failed
        """
        x1, y1, x2, y2 = bbox
        
        # Ensure coordinates are within image bounds
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(image.shape[1], x2)
        y2 = min(image.shape[0], y2)
        
        # Extract face region
        face_region = image[y1:y2, x1:x2]
        
        if face_region.size == 0:
            return None
        
        # Create filename: face_001_20240101_120000.jpg
        filename = f"face_{face_number:03d}_{timestamp}.jpg"
        filepath = os.path.join(self.detected_faces_dir, filename)
        
        # Save face image
        cv2.imwrite(filepath, face_region)
        
        return filepath
    
    def detect(self, image) -> List[Dict]:
        """
        Detect faces in an image and optionally save them to folder.
        
        Args:
            image: Image as numpy array (BGR format) or file path (str)
            
        Returns:
            List of detection dictionaries, each containing:
            - 'bbox': bounding box [x1, y1, x2, y2]
            - 'landmark': facial landmarks (if available)
            - 'det_score': detection confidence score (0.0 to 1.0)
            - 'saved_path': Path to saved face image (if saved)
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
        
        # Get timestamp for this detection session
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Format detections and save faces if enabled
        detections = []
        for idx, face in enumerate(faces, start=1):
            bbox = face.bbox.astype(int).tolist()  # [x1, y1, x2, y2]
            
            detection = {
                'bbox': bbox,
                'landmark': face.landmark_2d_106.astype(int).tolist() if hasattr(face, 'landmark_2d_106') else None,
                'det_score': float(face.det_score)
            }
            
            # Save detected face if enabled
            if self.save_detected_faces:
                saved_path = self._extract_and_save_face(img, bbox, idx, timestamp)
                detection['saved_path'] = saved_path
                if saved_path:
                    print(f"Saved detected face {idx} to: {saved_path}")
            
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
    
    def detect_and_save_faces(self, image_path: str) -> List[Dict]:
        """
        Detect all faces in an image and save them to folder.
        This is a convenience method that extracts all faces and saves them in one call.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            List of detection dictionaries, each containing:
            - 'bbox': bounding box [x1, y1, x2, y2]
            - 'landmark': facial landmarks (if available)
            - 'det_score': detection confidence score (0.0 to 1.0)
            - 'saved_path': Path to saved face image
            
        Example:
            detector = FaceDetector()
            detections = detector.detect_and_save_faces("group.jpg")
            # All faces are now extracted and saved in detection/detected_faces/ folder
            # Files: face_001_timestamp.jpg, face_002_timestamp.jpg, etc.
        """
        # Ensure saving is enabled for this call
        original_save_setting = self.save_detected_faces
        self.save_detected_faces = True
        
        try:
            detections = self.detect(image_path)
            print(f"\n✓ Successfully extracted and saved {len(detections)} face(s) to {self.detected_faces_dir}")
            return detections
        finally:
            # Restore original setting
            self.save_detected_faces = original_save_setting
    
    def detect_and_save_faces(self, image_path: str) -> List[Dict]:
        """
        Detect all faces in an image and save them to folder.
        This is a convenience method that detects faces and saves them in one call.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            List of detection dictionaries, each containing:
            - 'bbox': bounding box [x1, y1, x2, y2]
            - 'landmark': facial landmarks (if available)
            - 'det_score': detection confidence score (0.0 to 1.0)
            - 'saved_path': Path to saved face image
            
        Example:
            detector = FaceDetector()
            detections = detector.detect_and_save_faces("group.jpg")
            # All faces are now saved in detection/detected_faces/ folder
        """
        return self.detect(image_path)

