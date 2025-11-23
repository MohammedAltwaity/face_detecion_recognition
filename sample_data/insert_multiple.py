"""
Insert Multiple Faces - Initialize database with sample faces.

Usage:
    python sample_data/insert_multiple.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from storage.insert_face import insert_face

# ============================================================================
# INSERT MULTIPLE FACES - Edit these values to add more people
# ============================================================================

# Person 1
IMAGE_1 = 'sample_data/faces/face_001_20251123_171052.jpg'
NAME_1 = 'Ali'
METADATA_1 = {"age": 30, "department": "Engineering"}

# Person 2
IMAGE_2 = 'sample_data/faces/face_002_20251123_171052.jpg'
NAME_2 = 'Sarah'
METADATA_2 = {"age": 28, "department": "Marketing"}

# Person 3
IMAGE_3 = 'sample_data/faces/face_003_20251123_171052.jpg'
NAME_3 = 'Mohammed'
METADATA_3 = {"age": 35, "department": "Engineering"}

# Person 4
IMAGE_4 = 'sample_data/faces/face_004_20251123_171052.jpg'
NAME_4 = 'Emma'
METADATA_4 = {"age": 26, "department": "Sales"}

# Person 5
IMAGE_5 = 'sample_data/faces/face_005_20251123_171052.jpg'
NAME_5 = 'Ahmed'
METADATA_5 = {"age": 32, "department": "Finance"}

# Person 6
IMAGE_6 = 'sample_data/faces/face_006_20251123_171052.jpg'
NAME_6 = 'Lisa'
METADATA_6 = {"age": 29, "department": "HR"}

# Person 7
IMAGE_7 = 'sample_data/faces/face_007_20251123_171052.jpg'
NAME_7 = 'Omar'
METADATA_7 = {"age": 31, "department": "Engineering"}

# Person 8
IMAGE_8 = 'sample_data/faces/face_008_20251123_171052.jpg'
NAME_8 = 'Sophia'
METADATA_8 = {"age": 27, "department": "Marketing"}

# ============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("Inserting Multiple Faces into Database")
    print("=" * 70)
    
    # Insert all faces
    insertions = [
        (IMAGE_1, NAME_1, METADATA_1),
        (IMAGE_2, NAME_2, METADATA_2),
        (IMAGE_3, NAME_3, METADATA_3),
        (IMAGE_4, NAME_4, METADATA_4),
        (IMAGE_5, NAME_5, METADATA_5),
        (IMAGE_6, NAME_6, METADATA_6),
        (IMAGE_7, NAME_7, METADATA_7),
        (IMAGE_8, NAME_8, METADATA_8),
    ]
    
    success_count = 0
    failed_count = 0
    
    for i, (image, name, metadata) in enumerate(insertions, 1):
        print(f"\n[{i}/8] Inserting: {name}")
        print(f"  Image: {image}")
        try:
            person_id = insert_face(image, name, metadata)
            print(f"  ✓ Success! Person ID: {person_id}")
            success_count += 1
        except Exception as e:
            print(f"  ✗ Failed: {e}")
            failed_count += 1
    
    print("\n" + "=" * 70)
    print("SUMMARY:")
    print(f"  ✓ Successfully inserted: {success_count}")
    print(f"  ✗ Failed: {failed_count}")
    print("=" * 70)
