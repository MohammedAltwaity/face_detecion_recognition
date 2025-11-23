"""
Simple and reliable face insertion utility.

This module provides a clean way to insert faces into the database without
configuration files or caching issues. Just provide the image path, name, and metadata.

Usage:
    from storage.insert_face import insert_face
    
    # Simple usage
    person_id = insert_face(
        image_path='detection/detected_faces/face_001.jpg',
        name='John Doe',
        metadata={'age': 30, 'department': 'Engineering'}
    )
    
    # Or run directly
    python storage/insert_face.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2
import numpy as np
import shutil
import glob
from datetime import datetime
from typing import Optional, Dict
from embedding import EmbeddingExtractor
from storage import FaceInsert
from storage.init_db import init_database


def insert_face(
    image_path: str,
    name: str,
    metadata: Optional[Dict] = None,
    db_path: Optional[str] = None
) -> int:
    """
    Insert a face into the database with all necessary processing.
    
    This function handles everything:
    - Path resolution and pattern matching
    - Embedding extraction
    - Image saving to storage/images/
    - Database insertion with metadata
    - Error handling
    
    Args:
        image_path: Path to the face image (can be relative or absolute)
                   Supports pattern matching (e.g., 'face_001*.jpg')
        name: Name of the person (required)
        metadata: Optional dictionary with additional information
                 Example: {'age': 30, 'department': 'Engineering'}
        db_path: Optional database path (defaults to 'storage/faces.db')
    
    Returns:
        int: The person ID of the newly inserted record
    
    Raises:
        FileNotFoundError: If the image file cannot be found
        ValueError: If name is empty or image processing fails
    
    Example:
        person_id = insert_face(
            image_path='detection/detected_faces/face_001_20251123_152805.jpg',
            name='Mohammed Ali',
            metadata={'age': 25, 'department': 'Sales'}
        )
        print(f"Inserted person with ID: {person_id}")
    """
    # Validate inputs
    if not name or not name.strip():
        raise ValueError("Name cannot be empty!")
    
    if not image_path:
        raise ValueError("Image path cannot be empty!")
    
    # Set up database path
    if db_path is None:
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        db_path = os.path.join(project_root, "storage", "faces.db")
    
    # Ensure database exists
    storage_dir = os.path.dirname(db_path)
    os.makedirs(storage_dir, exist_ok=True)
    if not os.path.exists(db_path):
        init_database(db_path)
    
    # Resolve image path with pattern matching
    resolved_image_path = _resolve_image_path(image_path)
    
    print(f"Processing face insertion:")
    print(f"  Image: {resolved_image_path}")
    print(f"  Name: {name}")
    print(f"  Metadata: {metadata}")
    print("-" * 70)
    
    # Initialize extractor and inserter
    extractor = EmbeddingExtractor(save_extracted_faces=False)  # Don't save to extracted_faces
    inserter = FaceInsert(db_path=db_path)
    
    try:
        # Step 1: Extract embedding
        print("Step 1: Extracting face embedding...")
        embedding = extractor.extract(resolved_image_path, is_cropped_face=True)
        print(f"  ✓ Embedding extracted: shape={embedding.shape}, norm={np.linalg.norm(embedding):.6f}")
        
        # Step 2: Save person's image to storage/images/
        print("\nStep 2: Saving person's image...")
        images_dir = os.path.join(storage_dir, "images")
        os.makedirs(images_dir, exist_ok=True)
        
        # Create safe filename from name
        safe_name = "".join(c for c in name if c.isalnum() or c in (' ', '-', '_')).strip()
        safe_name = safe_name.replace(' ', '_')
        if not safe_name:
            raise ValueError(f"Name '{name}' resulted in empty filename after sanitization")
        
        # Get file extension
        _, ext = os.path.splitext(resolved_image_path)
        if not ext:
            ext = '.jpg'
        
        # Load image
        img = cv2.imread(resolved_image_path)
        if img is None:
            raise ValueError(f"Could not load image from: {resolved_image_path}")
        
        # Create temporary file with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        temp_image_path = os.path.join(images_dir, f"{safe_name}_temp_{timestamp}{ext}")
        cv2.imwrite(temp_image_path, img)
        if not os.path.exists(temp_image_path):
            raise IOError(f"Failed to save temporary image")
        
        # Step 3: Prepare metadata
        if metadata is None:
            metadata = {}
        else:
            metadata = metadata.copy()
        
        metadata["original_path"] = os.path.abspath(resolved_image_path)
        
        # Step 4: Insert into database
        print("\nStep 3: Inserting into database...")
        person_id = inserter.add(name, embedding, metadata)
        print(f"  ✓ Record created with Person ID: {person_id}")
        
        # Step 5: Rename image file with person_id
        final_image_path = os.path.join(images_dir, f"{safe_name}_{person_id}{ext}")
        if os.path.exists(final_image_path):
            os.remove(final_image_path)
        os.rename(temp_image_path, final_image_path)
        
        # Update metadata with final image path
        relative_path = os.path.relpath(final_image_path, os.path.dirname(os.path.dirname(__file__)))
        metadata["image_path"] = relative_path
        inserter.update_metadata(person_id, metadata)
        
        print(f"\n✓ Successfully inserted person!")
        print(f"  Person ID: {person_id}")
        print(f"  Name: {name}")
        print(f"  Image saved: {final_image_path}")
        print(f"  Database: {db_path}")
        
        return person_id
        
    except Exception as e:
        # Clean up temp file on error
        if 'temp_image_path' in locals() and os.path.exists(temp_image_path):
            try:
                os.remove(temp_image_path)
            except:
                pass
        raise
    finally:
        inserter.close()


def _resolve_image_path(image_path: str) -> str:
    """
    Resolve image path with pattern matching support.
    
    Handles:
    - Relative paths
    - Absolute paths
    - Pattern matching (e.g., 'face_001*.jpg' finds 'face_001_20251123_152805.jpg')
    """
    original_path = image_path
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    # If absolute path exists, use it
    if os.path.isabs(image_path) and os.path.exists(image_path):
        return image_path
    
    # Try relative to project root
    possible_path = os.path.join(project_root, image_path)
    if os.path.exists(possible_path):
        return possible_path
    
    # Try relative to current directory
    current_dir = os.getcwd()
    possible_path = os.path.normpath(os.path.join(current_dir, image_path))
    if os.path.exists(possible_path):
        return possible_path
    
    # Try pattern matching
    dir_part = os.path.dirname(image_path)
    file_part = os.path.basename(image_path)
    
    if '*' not in file_part and '.' in file_part:
        file_name, file_ext = os.path.splitext(file_part)
        
        # Create pattern from base name (e.g., face_001 from face_001_20251123_114613)
        if '_' in file_name:
            parts = file_name.split('_')
            if len(parts) >= 2:
                base_name = '_'.join(parts[:2])  # e.g., "face_001"
                pattern = base_name + '*' + file_ext
            else:
                pattern = file_name + '*' + file_ext
        else:
            pattern = file_name + '*' + file_ext
        
        # Try multiple search directories
        search_dirs = [
            os.path.normpath(os.path.join(project_root, dir_part)),  # Relative to project root
            os.path.normpath(os.path.join(current_dir, dir_part)),   # Relative to current dir
            os.path.normpath(dir_part),                              # As-is if absolute
        ]
        
        for search_dir in search_dirs:
            if os.path.exists(search_dir):
                matches = glob.glob(os.path.join(search_dir, pattern))
                if matches:
                    print(f"  ℹ Found file with pattern matching: {os.path.basename(matches[0])}")
                    return matches[0]
        
        # Also try searching in detection/detected_faces directly
        detected_faces_dir = os.path.join(project_root, "detection", "detected_faces")
        if os.path.exists(detected_faces_dir):
            matches = glob.glob(os.path.join(detected_faces_dir, pattern))
            if matches:
                print(f"  ℹ Found file with pattern matching: {os.path.basename(matches[0])}")
                return matches[0]
    
    # File not found
    raise FileNotFoundError(
        f"Image not found: {original_path}\n"
        f"Tried:\n"
        f"  - {os.path.join(project_root, image_path)}\n"
        f"  - {os.path.normpath(os.path.join(current_dir, image_path))}\n"
        f"  - Pattern matching for: {file_part}"
    )


if __name__ == "__main__":
    """
    Simple command-line interface for inserting faces.
    
    Usage:
        python storage/insert_face.py <image_path> <name> [metadata_json]
    
    Examples:
        python storage/insert_face.py detection/detected_faces/face_001.jpg "John Doe"
        python storage/insert_face.py detection/detected_faces/face_001.jpg "Jane Smith" '{"age": 30, "department": "Engineering"}'
    """
    import json
    
    if len(sys.argv) < 3:
        print("Usage: python storage/insert_face.py <image_path> <name> [metadata_json]")
        print("\nExamples:")
        print('  python storage/insert_face.py detection/detected_faces/face_001.jpg "John Doe"')
        print('  python storage/insert_face.py detection/detected_faces/face_001.jpg "Jane Smith" \'{"age": 30}\'')
        sys.exit(1)
    
    image_path = sys.argv[1]
    name = sys.argv[2]
    metadata = None
    
    if len(sys.argv) > 3:
        try:
            metadata = json.loads(sys.argv[3])
        except json.JSONDecodeError:
            print(f"⚠ Warning: Invalid JSON for metadata: {sys.argv[3]}")
            print("  Continuing without metadata...")
    
    try:
        person_id = insert_face(image_path, name, metadata)
        print(f"\n{'='*70}")
        print(f"✓ Insertion completed successfully! Person ID: {person_id}")
        print(f"{'='*70}")
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

