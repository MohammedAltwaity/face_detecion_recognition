"""

faces detected will be saved under detected_faces_dir in the root folder of the project






Test script for face detection - extract and save all faces.

This script demonstrates the detect_and_save_faces() method which
detects all faces in an image and saves them to a folder in one call.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from detection import FaceDetector


# ============================================================================
# CONFIGURATION: Set your image path here
# ============================================================================
# You can set the image path here or pass it as an argument to test_detect_and_save()
# Examples:
#   - 'group.jpg' (image with multiple faces)
#   - 'test.jpg' (any image with faces)
#   - 'path/to/your/image.jpg'
# ============================================================================
IMAGE_PATH = '../testing/group.jpg'


def test_detect_and_save(image_path=None):
    """
    Test the detect_and_save_faces method.
    This single function call will detect all faces and save them.
    
    Args:
        image_path: Path to the image file
                    If None, uses IMAGE_PATH from configuration
    """
    # Use path from configuration if not provided
    if image_path is None:
        image_path = IMAGE_PATH
    
    print(f"Initializing face detector...")
    # Get the root project directory
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    detected_faces_dir = os.path.join(project_root, "detected_faces_dir")
    detector = FaceDetector(detected_faces_dir=detected_faces_dir)
    
    print(f"\nDetecting and saving all faces from: {image_path}")
    print("=" * 60)
    
    # Single function call: detects all faces and saves them
    detections = detector.detect_and_save_faces(image_path)
    
    print(f"\n{'='*60}")
    print(f"RESULTS: Found {len(detections)} face(s)")
    print(f"{'='*60}")
    
    # Show details for each detected face
    for i, det in enumerate(detections, 1):
        print(f"\nFace {i}:")
        print(f"  Confidence: {det['det_score']:.2%}")
        print(f"  Bounding box: {det['bbox']}")
        if 'saved_path' in det and det['saved_path']:
            print(f"  ✓ Saved: {det['saved_path']}")


if __name__ == "__main__":
    # Test with path from configuration
    # You can also pass a custom path:
    # test_detect_and_save('path/to/your/image.jpg')
    test_detect_and_save()

