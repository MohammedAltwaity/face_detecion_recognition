"""
Search Module - Search for faces in the database.

This module handles searching for persons in the database based on face embeddings.
It uses cosine similarity to find matching faces.

Responsibilities:
- Search for similar faces using cosine similarity
- Retrieve person information based on embeddings
- Filter results by similarity threshold
"""

import sqlite3
import numpy as np
import json
from typing import List, Tuple, Optional, Dict


class FaceSearch:
    """
    Handles searching for faces in the database.
    
    This class is responsible for searching the database to find persons
    whose face embeddings match a query embedding. It uses cosine similarity
    to determine matches.
    
    Example:
        searcher = FaceSearch("faces.db")
        matches = searcher.search(query_embedding, threshold=0.6)
        for person_id, name, similarity, metadata in matches:
            print(f"Found {name} with {similarity:.2%} similarity")
        searcher.close()
    """
    
    def __init__(self, db_path: str = "faces.db"):
        """
        Initialize the face search database connection.
        
        Args:
            db_path: Path to the SQLite database file. Defaults to "faces.db"
        """
        self.db_path = db_path
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
    
    def search(self, embedding: np.ndarray, threshold: float = 0.6) -> List[Tuple[int, str, float, Optional[Dict]]]:
        """
        Search for a person in the database based on face embedding similarity.
        
        This method performs a linear search through all stored embeddings,
        calculating cosine similarity between the query embedding and each
        stored embedding. Results are filtered by the threshold and sorted
        by similarity score (highest first).
        
        Args:
            embedding: Query face embedding vector (512 dimensions, float32, normalized)
                      This should be extracted using the embedding module
            threshold: Minimum cosine similarity threshold (0.0 to 1.0)
                      - 0.5-0.6: More lenient (may have false positives)
                      - 0.6-0.7: Balanced (recommended default)
                      - 0.7-0.8: Strict (high accuracy)
                      - 0.8+: Very strict (may miss valid matches)
            
        Returns:
            List of tuples, each containing:
            - person_id (int): Database ID of the matched person
            - name (str): Name of the matched person
            - similarity (float): Cosine similarity score (0.0 to 1.0)
            - metadata (dict or None): Person's metadata dictionary
            
            Results are sorted by similarity score in descending order.
            Empty list if no matches found above the threshold.
            
        Example:
            # After embedding extraction:
            query_embedding = extractor.extract(query_image)
            matches = searcher.search(query_embedding, threshold=0.65)
            
            if matches:
                person_id, name, similarity, metadata = matches[0]
                print(f"Found {name} with {similarity:.2%} similarity")
        """
        cursor = self.conn.cursor()
        # Fetch all persons from database
        cursor.execute("SELECT id, name, embedding, metadata FROM persons")
        rows = cursor.fetchall()
        
        results = []
        for person_id, name, embedding_bytes, metadata_str in rows:
            # Deserialize embedding from binary format
            stored_embedding = np.frombuffer(embedding_bytes, dtype=np.float32)
            
            # Calculate cosine similarity
            similarity = self._cosine_similarity(embedding, stored_embedding)
            
            # Only include results above the threshold
            if similarity >= threshold:
                # Deserialize metadata JSON string back to dictionary
                metadata = json.loads(metadata_str) if metadata_str else None
                results.append((person_id, name, similarity, metadata))
        
        # Sort by similarity score (highest first)
        results.sort(key=lambda x: x[2], reverse=True)
        return results
    
    def get_all(self) -> List[Tuple[int, str, Optional[Dict], str]]:
        """
        Retrieve all persons from the database.
        
        Returns:
            List of tuples: (person_id, name, metadata, created_at)
        """
        cursor = self.conn.cursor()
        cursor.execute("SELECT id, name, metadata, created_at FROM persons")
        rows = cursor.fetchall()
        
        results = []
        for person_id, name, metadata_str, created_at in rows:
            metadata = json.loads(metadata_str) if metadata_str else None
            results.append((person_id, name, metadata, created_at))
        
        return results
    
    def get_by_id(self, person_id: int) -> Optional[Tuple[int, str, Optional[Dict], str]]:
        """
        Retrieve a specific person by ID with all information.
        
        Args:
            person_id: ID of the person to retrieve
            
        Returns:
            Tuple of (person_id, name, metadata, created_at) or None if not found
        """
        cursor = self.conn.cursor()
        cursor.execute("SELECT id, name, metadata, created_at FROM persons WHERE id = ?", (person_id,))
        row = cursor.fetchone()
        
        if row is None:
            return None
        
        person_id, name, metadata_str, created_at = row
        metadata = json.loads(metadata_str) if metadata_str else None
        return (person_id, name, metadata, created_at)
    
    def delete(self, person_id: int) -> bool:
        """
        Delete a person from the database.
        
        Args:
            person_id: ID of the person to delete
            
        Returns:
            True if person was deleted, False if not found
        """
        cursor = self.conn.cursor()
        cursor.execute("DELETE FROM persons WHERE id = ?", (person_id,))
        self.conn.commit()
        return cursor.rowcount > 0
    
    def _cosine_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """
        Calculate cosine similarity between two vectors.
        
        Args:
            vec1: First embedding vector (should be normalized)
            vec2: Second embedding vector (should be normalized)
            
        Returns:
            float: Cosine similarity score (0.0 to 1.0)
        """
        # Normalize vectors
        vec1_norm = vec1 / (np.linalg.norm(vec1) + 1e-8)
        vec2_norm = vec2 / (np.linalg.norm(vec2) + 1e-8)
        
        # Calculate cosine similarity
        similarity = np.dot(vec1_norm, vec2_norm)
        
        # Clip to [0, 1] range
        return float(np.clip(similarity, 0.0, 1.0))
    
    def close(self):
        """Close the database connection."""
        self.conn.close()
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()

