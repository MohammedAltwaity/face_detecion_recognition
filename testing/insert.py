"""
DIRECT INSERT - No caching, no bullshit. Just provide values and insert.

Usage:
    python testing/insert.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from storage.insert_face import insert_face

# ============================================================================
# CHANGE THESE VALUES - That's it!
# ============================================================================
IMAGE = 'detection/detected_faces/face_001_20251123_114613.jpg'
NAME = 'Ali'
METADATA = {"age": 30, "department": "Engineering"}
# ============================================================================

# Insert directly - no caching, reads values above each time
insert_face(IMAGE, NAME, METADATA)

