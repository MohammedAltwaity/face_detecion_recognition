"""
Embedding Extraction Module - Extracts face embeddings using ArcFace.

This module is responsible for extracting 512-dimensional face embeddings
from face images. These embeddings are used for face recognition and matching.

Responsibilities:
- Extract embeddings from face images
- Support single and batch processing
- Return normalized embeddings for similarity comparison
"""

import cv2
import numpy as np
from typing import List, Union
import insightface


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
    
    def __init__(self, model_name: str = 'buffalo_l'):
        """
        Initialize ArcFace extractor.
        
        Args:
            model_name: InsightFace model variant
                - 'buffalo_l': Large model (most accurate, 512-dim embeddings)
                - 'buffalo_m': Medium model (balanced)
                - 'buffalo_s': Small model (faster)
        """
        # Initialize InsightFace app which includes both detection and embedding
        self.app = insightface.app.FaceAnalysis(name=model_name, providers=['CPUExecutionProvider'])
        self.app.prepare(ctx_id=-1, det_size=(640, 640))
    
    def extract(self, face_image: Union[str, np.ndarray]) -> np.ndarray:
        """
        Extract face embedding from a face image.
        
        Args:
            face_image: Path to face image file or numpy array (BGR format)
            
        Returns:
            512-dimensional face embedding vector (normalized, float32)
            
        Raises:
            ValueError: If no face is detected in the image
        """
        # Load image if path is provided
        if isinstance(face_image, str):
            img = cv2.imread(face_image)
            if img is None:
                raise ValueError(f"Could not load image from {face_image}")
        else:
            img = face_image.copy()
        
        # Get face detection and embedding
        faces = self.app.get(img)
        
        if len(faces) == 0:
            raise ValueError("No face detected in the image")
        
        # Use the first detected face
        face = faces[0]
        embedding = face.normed_embedding  # Already normalized
        
        return embedding
    
    def extract_batch(self, face_images: List[Union[str, np.ndarray]]) -> List[np.ndarray]:
        """
        Extract embeddings from multiple face images.
        
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
            except Exception as e:
                print(f"Warning: Failed to extract embedding from image: {e}")
                continue
        
        return embeddings
    
    def extract_from_image(self, image: np.ndarray) -> List[np.ndarray]:
        """
        Extract embeddings for all faces in an image.
        
        Args:
            image: Full image as numpy array (BGR format)
            
        Returns:
            List of 512-dimensional face embedding vectors (one per face)
        """
        faces = self.app.get(image)
        
        if len(faces) == 0:
            return []
        
        embeddings = []
        for face in faces:
            embedding = face.normed_embedding
            embeddings.append(embedding)
        
        return embeddings

