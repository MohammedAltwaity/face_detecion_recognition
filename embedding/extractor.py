"""
Embedding Extraction Module - Extracts face embeddings using ArcFace.

This module is responsible for extracting 512-dimensional face embeddings
from face images. These embeddings are used for face recognition and matching.

Responsibilities:
- Extract embeddings from face images
- Support single face operations only
- Return normalized embeddings for similarity comparison
"""

import cv2
import numpy as np
from typing import List, Union, Optional, Tuple
import insightface
import os
from datetime import datetime

# ============================================================================
# CONFIGURATION: Control whether extracted faces are saved to folder
# ============================================================================
# Set to True only if you want to save extracted face images to embedding/extracted_faces/
# By default, faces are NOT saved (set to False)
# ============================================================================
DEFAULT_SAVE_EXTRACTED_FACES = False


class EmbeddingExtractor:
    """
    Face embedding extraction using ArcFace model from InsightFace.
    
    This class extracts 512-dimensional face embeddings from face images.
    The embeddings are L2-normalized, making cosine similarity an ideal
    distance metric for face matching.
    
    Example:
        extractor = EmbeddingExtractor()
        embedding = extractor.extract("face.jpg")
        # embedding is a 512-dim numpy array (float32, normalized)
    """
    
    def __init__(self, model_name: str = 'buffalo_l', save_extracted_faces: bool = None):
        """
        Initialize ArcFace extractor.
        
        Args:
            model_name: InsightFace model variant
                - 'buffalo_l': Large model (most accurate, 512-dim embeddings)
                - 'buffalo_m': Medium model (balanced)
                - 'buffalo_s': Small model (faster)
            save_extracted_faces: If True, saves extracted faces to embedding/extracted_faces/ folder.
                                 If None, uses DEFAULT_SAVE_EXTRACTED_FACES from module config.
                                 Default: False (faces are NOT saved)
        """
        # Initialize InsightFace app which includes both detection and embedding
        self.app = insightface.app.FaceAnalysis(name=model_name, providers=['CPUExecutionProvider'])
        self.app.prepare(ctx_id=-1, det_size=(640, 640))
        
        # Use module default if not explicitly provided
        if save_extracted_faces is None:
            save_extracted_faces = DEFAULT_SAVE_EXTRACTED_FACES
        
        # Set up extracted_faces folder ONLY if saving is enabled
        self.save_extracted_faces = save_extracted_faces
        self.extracted_faces_dir = None
        self.face_counter = 1
        
        # Only create folder and initialize counter if saving is enabled
        if self.save_extracted_faces:
            self.extracted_faces_dir = os.path.join(os.path.dirname(__file__), "extracted_faces")
            # Create folder only when needed (not in __init__)
            self.face_counter = self._get_next_face_number()
    
    def _get_next_face_number(self) -> int:
        """Get the next face number for naming extracted faces."""
        if not self.extracted_faces_dir or not os.path.exists(self.extracted_faces_dir):
            return 1
        
        existing_files = [f for f in os.listdir(self.extracted_faces_dir) if f.endswith('.jpg')]
        if not existing_files:
            return 1
        
        # Extract numbers from filenames (format: face_001_timestamp.jpg)
        numbers = []
        for f in existing_files:
            try:
                parts = f.split('_')
                if len(parts) >= 2 and parts[0] == 'face':
                    num = int(parts[1])
                    numbers.append(num)
            except:
                continue
        
        return max(numbers) + 1 if numbers else 1
    
    
    def _extract_and_save_face(self, image: np.ndarray, bbox: List[int], 
                               face_number: int, timestamp: str) -> str:
        """
        Extract face region and save to extracted_faces folder.
        Creates folder if it doesn't exist.
        
        Args:
            image: Full image
            bbox: Bounding box [x1, y1, x2, y2]
            face_number: Face number for filename
            timestamp: Timestamp string
            
        Returns:
            Path to saved face image
        """
        # Ensure folder exists (create if it doesn't)
        os.makedirs(self.extracted_faces_dir, exist_ok=True)
        
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
        filepath = os.path.join(self.extracted_faces_dir, filename)
        
        # Save face image
        cv2.imwrite(filepath, face_region)
        
        return filepath
    
    def _extract_embedding_from_cropped_face(self, face_img: np.ndarray) -> np.ndarray:
        """
        Extract embedding directly from a pre-cropped face image without detection.
        Uses InsightFace's recognition model directly.
        
        Args:
            face_img: Cropped face image as numpy array (BGR format)
            
        Returns:
            512-dimensional face embedding vector (normalized, float32)
        """
        # Get the recognition model
        rec_model = self.app.models.get('recognition')
        if rec_model is None:
            raise ValueError("Recognition model not available")
        
        # Resize to 112x112 (required by recognition model)
        face_resized = cv2.resize(face_img, (112, 112))
        
        # InsightFace recognition model expects input in specific format
        # Normalize: (BGR - 127.5) / 128.0
        face_normalized = (face_resized.astype(np.float32) - 127.5) / 128.0
        
        # Convert to RGB (InsightFace models typically expect RGB)
        face_rgb = cv2.cvtColor(face_normalized, cv2.COLOR_BGR2RGB)
        
        # Transpose to CHW format: (3, 112, 112)
        face_transposed = np.transpose(face_rgb, (2, 0, 1))
        
        # Add batch dimension: (1, 3, 112, 112)
        face_batch = np.expand_dims(face_transposed, axis=0)
        
        # Get the ONNX session from the recognition model
        # InsightFace uses ONNX models, so we access the session
        if hasattr(rec_model, 'session'):
            # Run inference using ONNX session
            input_name = rec_model.session.get_inputs()[0].name
            output = rec_model.session.run(None, {input_name: face_batch})
            embedding = output[0]
        elif hasattr(rec_model, 'forward'):
            # Try forward method if available
            embedding = rec_model.forward(face_batch)
        else:
            # Fallback: try calling the model directly
            embedding = rec_model(face_batch)
        
        # Flatten if needed
        embedding = embedding.flatten()
        
        # Normalize the embedding (L2 normalization)
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm
        
        return embedding.astype(np.float32)
    
    def extract(self, face_image: Union[str, np.ndarray], is_cropped_face: bool = False) -> np.ndarray:
        """
        Extract face embedding from a SINGLE face image.
        Optionally saves extracted face to folder.
        
        Args:
            face_image: Path to face image file or numpy array (BGR format)
            is_cropped_face: If True, assumes image is already a cropped face and extracts
                           embedding directly without detection. If False, detects face first.
            
        Returns:
            512-dimensional face embedding vector (normalized, float32)
            
        Raises:
            ValueError: If no face or multiple faces are detected (when is_cropped_face=False)
        """
        # Load image if path is provided
        if isinstance(face_image, str):
            img = cv2.imread(face_image)
            if img is None:
                raise ValueError(f"Could not load image from {face_image}")
        else:
            img = face_image.copy()
        
        # If it's a pre-cropped face, extract embedding directly
        if is_cropped_face:
            embedding = self._extract_embedding_from_cropped_face(img)
            
            # Save extracted face if enabled (save the entire image as it's already cropped)
            if self.save_extracted_faces and self.extracted_faces_dir:
                # Create folder only when actually saving
                os.makedirs(self.extracted_faces_dir, exist_ok=True)
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"face_{self.face_counter:03d}_{timestamp}.jpg"
                filepath = os.path.join(self.extracted_faces_dir, filename)
                success = cv2.imwrite(filepath, img)
                if success and os.path.exists(filepath):
                    print(f"  ✓ Saved extracted face to: {filepath}")
                else:
                    print(f"  ✗ Failed to save extracted face to: {filepath}")
                self.face_counter += 1
            
            return embedding
        
        # Otherwise, detect face first
        # Get face detection and embedding
        faces = self.app.get(img)
        
        if len(faces) == 0:
            # If no face detected, assume it's a pre-cropped face and try direct extraction
            print("  ℹ No face detected, assuming pre-cropped face image...")
            embedding = self._extract_embedding_from_cropped_face(img)
            
            # Save extracted face if enabled
            if self.save_extracted_faces and self.extracted_faces_dir:
                # Create folder only when actually saving
                os.makedirs(self.extracted_faces_dir, exist_ok=True)
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"face_{self.face_counter:03d}_{timestamp}.jpg"
                filepath = os.path.join(self.extracted_faces_dir, filename)
                success = cv2.imwrite(filepath, img)
                if success and os.path.exists(filepath):
                    print(f"  ✓ Saved extracted face to: {filepath}")
                else:
                    print(f"  ✗ Failed to save extracted face to: {filepath}")
                self.face_counter += 1
            
            return embedding
        
        if len(faces) > 1:
            raise ValueError(f"Multiple faces detected ({len(faces)}). Please provide an image with exactly ONE face.")
        
        # Get the single face
        face = faces[0]
        
        # Get bounding box
        bbox = face.bbox.astype(int).tolist()
        
        # Save extracted face if enabled
        if self.save_extracted_faces and self.extracted_faces_dir:
            # Create folder only when actually saving
            os.makedirs(self.extracted_faces_dir, exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            saved_path = self._extract_and_save_face(img, bbox, self.face_counter, timestamp)
            if saved_path and os.path.exists(saved_path):
                print(f"  ✓ Saved extracted face to: {saved_path}")
            else:
                print(f"  ✗ Failed to save extracted face")
            self.face_counter += 1
        
        # Extract embedding
        embedding = face.normed_embedding  # Already normalized
        
        return embedding
    
    def extract_batch(self, face_images: List[Union[str, np.ndarray]]) -> List[np.ndarray]:
        """
        Extract embeddings from multiple face images.
        Each image must contain exactly ONE face.
        
        Args:
            face_images: List of face image paths or numpy arrays
            
        Returns:
            List of 512-dimensional face embedding vectors
        """
        embeddings = []
        for face_image in face_images:
            try:
                embedding = self.extract(face_image)
                embeddings.append(embedding)
            except ValueError as e:
                print(f"Warning: {e}")
                continue
            except Exception as e:
                print(f"Warning: Failed to extract embedding from image: {e}")
                continue
        
        return embeddings
    
    def extract_single_from_image(self, image: np.ndarray) -> np.ndarray:
        """
        Extract embedding for a SINGLE face from an image.
        Raises error if multiple faces or no faces detected.
        
        Args:
            image: Full image as numpy array (BGR format)
            
        Returns:
            512-dimensional face embedding vector (normalized, float32)
            
        Raises:
            ValueError: If no face or multiple faces are detected
        """
        faces = self.app.get(image)
        
        if len(faces) == 0:
            raise ValueError("No face detected in the image. Please provide an image with exactly one face.")
        
        if len(faces) > 1:
            raise ValueError(f"Multiple faces detected ({len(faces)}). Please provide an image with exactly ONE face.")
        
        # Get the single face
        face = faces[0]
        
        # Get bounding box
        bbox = face.bbox.astype(int).tolist()
        
        # Save extracted face if enabled
        if self.save_extracted_faces and self.extracted_faces_dir:
            # Create folder only when actually saving
            os.makedirs(self.extracted_faces_dir, exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            saved_path = self._extract_and_save_face(image, bbox, self.face_counter, timestamp)
            if saved_path and os.path.exists(saved_path):
                print(f"  ✓ Saved extracted face to: {saved_path}")
            else:
                print(f"  ✗ Failed to save extracted face")
            self.face_counter += 1
        
        embedding = face.normed_embedding
        return embedding

