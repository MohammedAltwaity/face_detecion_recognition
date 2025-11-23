"""
SIMPLE FACE INSERTION - Single file, no config needed!

Just edit the 3 values below and run. That's it.

Usage:
    python testing/test_isertion.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from storage.insert_face import insert_face

# ============================================================================
# EDIT THESE 3 VALUES - Save the file, then run!
# ============================================================================
IMAGE = 'image1.jpg'
NAME = 'Mohammed'
METADATA = {"age": 22, "department": "Engineering"}
# ============================================================================

# Insert into database
insert_face(IMAGE, NAME, METADATA)
