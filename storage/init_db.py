"""
Database Initialization Module - Creates and initializes the face recognition database.

This module creates the database with the correct schema matching how we insert data.
Run this before using the database for the first time.

Usage:
    from storage.init_db import init_database
    init_database("faces.db")
"""

import sqlite3
import os
from typing import Optional


def init_database(db_path: str = None) -> None:
    """
    Initialize the face recognition database with the correct schema.
    
    This creates the 'persons' table with the same structure used by FaceInsert.
    The table structure matches exactly how data is inserted in storage/insert.py.
    
    Args:
        db_path: Path to the SQLite database file. 
                 If None, defaults to "storage/faces.db" (relative to this file)
        
    Example:
        init_database("storage/faces.db")
        print("Database initialized successfully!")
    """
    if db_path is None:
        # Default to storage/faces.db relative to this file
        script_dir = os.path.dirname(os.path.abspath(__file__))
        db_path = os.path.join(script_dir, "faces.db")
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Create the persons table (same structure as in FaceInsert._create_table)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS persons (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            embedding BLOB NOT NULL,
            metadata TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    
    conn.commit()
    conn.close()
    print(f"✓ Database initialized at: {db_path}")


if __name__ == "__main__":
    # Initialize database when run directly (uses default: storage/faces.db)
    init_database()
    print("Database ready to use!")

