"""
Face Recognition Pipeline - Main orchestrator for the complete system.

This module orchestrates all components to provide a high-level API for
face recognition tasks. It combines detection, extraction, embedding, and
storage into a unified interface.

The pipeline workflow:
1. Detect faces in images
2. Extract face regions (optional)
3. Extract embeddings from faces
4. Store or search in database
"""

import cv2
import numpy as np
from typing import List, Tuple, Optional, Dict
from detection import FaceDetector
from embedding import EmbeddingExtractor
from storage import FaceInsert, FaceSearch


class FaceRecognitionPipeline:
    """
    Complete face recognition pipeline orchestrating all components.
    
    This is the main class that provides a high-level API for face recognition.
    It combines detection, embedding extraction, and storage into a unified
    interface.
    
    Example:
        pipeline = FaceRecognitionPipeline()
        
        # Add a person
        person_id = pipeline.add_person("alice.jpg", "Alice")
        
        # Search for a person
        results = pipeline.search("query.jpg", threshold=0.6)
        
        pipeline.close()
    """
    
    def __init__(self, db_path: str = "faces.db", model_name: str = 'buffalo_l'):
        """
        Initialize the face recognition pipeline.
        
        Args:
            db_path: Path to SQLite database file. Defaults to "faces.db"
            model_name: InsightFace model variant ('buffalo_l', 'buffalo_m', 'buffalo_s')
        """
        # Initialize components
        self.detector = FaceDetector(model_name=model_name)
        self.extractor = EmbeddingExtractor(model_name=model_name)
        self.inserter = FaceInsert(db_path=db_path)
        self.searcher = FaceSearch(db_path=db_path)
    
    def add_person(self, image_path: str, name: str, metadata: Optional[Dict] = None) -> int:
        """
        Add a new person to the database from an image.
        
        The image must contain exactly one face. The face is detected,
        its embedding is extracted, and stored in the database.
        
        Args:
            image_path: Path to image file containing exactly one face
            name: Name of the person
            metadata: Optional dictionary with additional information
            
        Returns:
            int: The database ID of the newly added person
            
        Raises:
            ValueError: If no face or multiple faces are detected
        """
        # Load image
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not load image from {image_path}")
        
        # Detect faces
        detections = self.detector.detect(image)
        
        # Validate exactly one face
        if len(detections) == 0:
            raise ValueError("No face detected in the image")
        
        if len(detections) > 1:
            raise ValueError(f"Multiple faces detected ({len(detections)}). "
                           f"Please provide an image with exactly one face.")
        
        # Extract embedding (using extractor which handles detection internally)
        embedding = self.extractor.extract(image)
        
        # Store in database using inserter
        person_id = self.inserter.add(name, embedding, metadata)
        
        return person_id
    
    def search(self, image_path: str, threshold: float = 0.6) -> List[Dict]:
        """
        Search for persons in the database based on faces in an image.
        
        Can handle multiple faces in a single image. For each detected face,
        searches the database for matching persons.
        
        Args:
            image_path: Path to image file (can contain multiple faces)
            threshold: Similarity threshold for matching (0.0 to 1.0)
        
        Returns:
            List of dictionaries, one per detected face. Each contains:
            - 'face_index': Index of the face
            - 'bbox': Bounding box coordinates [x1, y1, x2, y2]
            - 'det_score': Detection confidence score
            - 'matches': List of matching persons (person_id, name, similarity, metadata)
        """
        # Load image
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not load image from {image_path}")
        
        # Detect faces
        detections = self.detector.detect(image)
        
        if len(detections) == 0:
            return []
        
        # Extract embeddings for all faces
        embeddings = self.extractor.extract_from_image(image)
        
        # Search database for each face using searcher
        results = []
        for i, (detection, embedding) in enumerate(zip(detections, embeddings)):
            # Search in database
            matches = self.searcher.search(embedding, threshold=threshold)
            
            result = {
                'face_index': i,
                'bbox': detection['bbox'],
                'det_score': detection['det_score'],
                'matches': matches
            }
            results.append(result)
        
        return results
    
    def process_image(self, image_path: str, threshold: float = 0.6) -> List[Dict]:
        """
        Full pipeline: detect faces → extract embeddings → search database.
        
        Alias for search() method. Processes an image and searches for matches.
        
        Args:
            image_path: Path to image file
            threshold: Similarity threshold for matching
        
        Returns:
            List of dictionaries with detection and match information
        """
        return self.search(image_path, threshold=threshold)
    
    def get_all_persons(self) -> List[Tuple[int, str, Optional[Dict], str]]:
        """
        Retrieve all persons from the database.
        
        Returns:
            List of tuples: (person_id, name, metadata, created_at)
        """
        return self.searcher.get_all()
    
    def delete_person(self, person_id: int) -> bool:
        """
        Delete a person from the database.
        
        Args:
            person_id: ID of the person to delete
            
        Returns:
            True if deleted, False if not found
        """
        return self.searcher.delete(person_id)
    
    def close(self):
        """Close all database connections."""
        self.inserter.close()
        self.searcher.close()
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()

