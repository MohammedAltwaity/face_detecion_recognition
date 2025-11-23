"""
Test script for embedding extraction.

This script tests the embedding extraction functionality.
It takes an image path, extracts the face embedding, and prints the embedding info.

Usage:
    python testing/test_embedding.py
"""

import sys
import os
import glob
import cv2
from datetime import datetime
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from embedding import EmbeddingExtractor
import numpy as np


# ============================================================================
# CONFIGURATION: Set your image path here
# ============================================================================
# You can set the image path here or pass it as an argument to test_embedding()
# Examples:
#   - '../detection/detected_faces/face_001_20251123_114613.jpg'
#   - 'path/to/your/single_face_image.jpg'
#   - 'group.jpg' (if it contains only one face)c
# ============================================================================
IMAGE_PATH = '../detection/detected_faces/face_012_20251123_114613.jpg'


def test_embedding(image_path=None):
    """
    Test embedding extraction from an image.
    
    Args:
        image_path: Path to the image file (must contain exactly one face)
                    If None, uses IMAGE_PATH from configuration
    """
    # Use path from configuration if not provided
    if image_path is None:
        image_path = IMAGE_PATH
    
    # Resolve image path relative to project root or current directory
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    testing_dir = os.path.dirname(os.path.abspath(__file__))
    original_path = image_path
    
    if not os.path.isabs(image_path):
        # First, try resolving relative to testing directory (handles ../ correctly)
        test_relative_path = os.path.normpath(os.path.join(testing_dir, image_path))
        if os.path.exists(test_relative_path):
            image_path = test_relative_path
        else:
            # Try relative to project root
            full_path = os.path.normpath(os.path.join(project_root, image_path))
            if os.path.exists(full_path):
                image_path = full_path
            elif not os.path.exists(image_path):
                # Try relative to current directory
                current_dir = os.getcwd()
                full_path = os.path.normpath(os.path.join(current_dir, image_path))
                if os.path.exists(full_path):
                    image_path = full_path
    
    # If file still doesn't exist, try pattern matching (e.g., face_001.jpg -> face_001*.jpg)
    if not os.path.exists(image_path):
        # Extract directory and filename pattern
        dir_part = os.path.dirname(original_path)
        file_part = os.path.basename(original_path)
        
        # Try to find matching files with pattern (e.g., face_001*.jpg)
        if '*' not in file_part and '.' in file_part:
            name, ext = os.path.splitext(file_part)
            # Create pattern: face_001*.jpg
            pattern = name + '*' + ext
            
            # Try searching relative to testing directory first (for ../ paths)
            search_dir = os.path.normpath(os.path.join(testing_dir, dir_part))
            if not os.path.exists(search_dir):
                # Fallback to project root
                search_dir = os.path.normpath(os.path.join(project_root, dir_part))
            
            if os.path.exists(search_dir):
                matches = glob.glob(os.path.join(search_dir, pattern))
                if matches:
                    image_path = matches[0]  # Use first match
                    print(f"  ℹ Found file with pattern: {os.path.basename(image_path)}")
    
    # Verify file exists
    if not os.path.exists(image_path):
        print(f"✗ Error: Image file not found: {original_path}")
        print(f"  Tried paths relative to project root and current directory.")
        print(f"  Also tried pattern matching for files like: {os.path.basename(original_path)}")
        return
    
    print("=" * 70)
    print("EMBEDDING EXTRACTION TEST")
    print("=" * 70)
    
    print(f"\nInitializing embedding extractor...")
    extractor = EmbeddingExtractor()  # Default: save_extracted_faces=False
    
    print(f"\nExtracting embedding from: {image_path}")
    print("-" * 70)
    
    # Create directory to save tested face images
    tested_images_embedding_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "tested_images_embedding")
    os.makedirs(tested_images_embedding_dir, exist_ok=True)
    
    try:
        # Load the image to save it later
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Could not load image: {image_path}")
        
        # Extract embedding (assume it's a pre-cropped face from detection)
        embedding = extractor.extract(image_path, is_cropped_face=True)
        
        # Save the tested face image to tested_images_embedding folder
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = os.path.basename(image_path)
        name, ext = os.path.splitext(filename)
        saved_filename = f"{name}_tested_{timestamp}{ext}"
        saved_path = os.path.join(tested_images_embedding_dir, saved_filename)
        success = cv2.imwrite(saved_path, img)
        if success and os.path.exists(saved_path):
            print(f"\n  ✓ Saved tested face image to: {saved_path}")
        else:
            print(f"\n  ✗ Failed to save tested face image to: {saved_path}")
        
        # Print embedding information
        print(f"\n✓ Embedding extracted successfully!")
        print(f"\nEmbedding Information:")
        print(f"  Shape: {embedding.shape}")
        print(f"  Data type: {embedding.dtype}")
        print(f"  Dimensions: {len(embedding)}")
        print(f"  Min value: {np.min(embedding):.6f}")
        print(f"  Max value: {np.max(embedding):.6f}")
        print(f"  Mean value: {np.mean(embedding):.6f}")
        print(f"  Norm (L2): {np.linalg.norm(embedding):.6f}")
        
        # Show first few values
        print(f"\n  First 10 values: {embedding[:10]}")
        print(f"  Last 10 values: {embedding[-10:]}")
        
        # Check if normalized (should be close to 1.0)
        norm = np.linalg.norm(embedding)
        if abs(norm - 1.0) < 0.01:
            print(f"\n  ✓ Embedding is normalized (L2 norm ≈ 1.0)")
        else:
            print(f"\n  ⚠ Embedding norm is {norm:.6f} (expected ~1.0)")
        
        print(f"\n{'='*70}")
        print("Test completed successfully!")
        print(f"{'='*70}")
        
    except ValueError as e:
        print(f"\n✗ Error: {e}")
        print("Make sure the image contains exactly ONE face.")
    except Exception as e:
        print(f"\n✗ Unexpected error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    # Test with path from configuration
    # You can also pass a custom path:
    # test_embedding('path/to/your/image.jpg')
    test_embedding()

