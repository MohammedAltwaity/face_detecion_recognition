"""
Insert Module - Insert face data into database after detection and embedding extraction.

This module handles inserting person information and face embeddings into the database.
It is used after face detection and embedding extraction are complete.

Responsibilities:
- Insert person data with face embeddings
- Store metadata along with embeddings
- Manage database connections for insert operations
"""

import sqlite3
import numpy as np
import json
from typing import Optional, Dict


class FaceInsert:
    """
    Handles insertion of face data into the database.
    
    This class is responsible for storing person information along with
    their face embeddings in the database. It is used after face detection
    and embedding extraction are complete.
    
    Example:
        inserter = FaceInsert("faces.db")
        person_id = inserter.add("Alice", embedding, {"age": 25})
        inserter.close()
    """
    
    def __init__(self, db_path: str = "faces.db"):
        """
        Initialize the face insert database connection.
        
        Args:
            db_path: Path to the SQLite database file. Defaults to "faces.db"
        """
        self.db_path = db_path
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self._create_table()
    
    def _create_table(self):
        """Create the persons table if it doesn't exist."""
        cursor = self.conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS persons (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                embedding BLOB NOT NULL,
                metadata TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        self.conn.commit()
    
    def add(self, name: str, embedding: np.ndarray, metadata: Optional[Dict] = None) -> int:
        """
        Add a new person with their face embedding to the database.
        
        This method is called after face detection and embedding extraction.
        It stores the person's information along with their face embedding.
        
        Args:
            name: Name of the person (required)
            embedding: Face embedding vector (512 dimensions, float32, normalized)
                      This should be extracted using the embedding module
            metadata: Optional dictionary with additional information
                     Example: {"age": 30, "department": "Engineering"}
            
        Returns:
            int: The auto-generated ID of the newly added person
            
        Example:
            # After detection and embedding extraction:
            embedding = extractor.extract(image)
            person_id = inserter.add("John Doe", embedding, {"age": 30})
        """
        cursor = self.conn.cursor()
        
        # Serialize embedding to bytes (512 float32 values = 2048 bytes)
        embedding_bytes = embedding.tobytes()
        
        # Serialize metadata to JSON string
        metadata_str = json.dumps(metadata) if metadata else None
        
        # Insert into database
        cursor.execute("""
            INSERT INTO persons (name, embedding, metadata)
            VALUES (?, ?, ?)
        """, (name, embedding_bytes, metadata_str))
        
        self.conn.commit()
        return cursor.lastrowid
    
    def update_metadata(self, person_id: int, metadata: Dict) -> bool:
        """
        Update metadata for an existing person.
        
        Args:
            person_id: ID of the person to update
            metadata: Dictionary with updated metadata
            
        Returns:
            True if person was updated, False if not found
        """
        cursor = self.conn.cursor()
        metadata_str = json.dumps(metadata) if metadata else None
        cursor.execute("""
            UPDATE persons 
            SET metadata = ?
            WHERE id = ?
        """, (metadata_str, person_id))
        self.conn.commit()
        return cursor.rowcount > 0
    
    def close(self):
        """Close the database connection."""
        self.conn.close()
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()

