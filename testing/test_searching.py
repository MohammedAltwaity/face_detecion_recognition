"""
Test script for face searching in database.

This script tests searching for a face in the database.
It takes an image path, extracts the embedding, searches the database,
and displays the searched image and the found match side by side.

Usage:
    python testing/test_searching.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2
import numpy as np
import json
from embedding import EmbeddingExtractor
from storage import FaceSearch


# ============================================================================
# CONFIGURATION: Set your image path and threshold here
# ============================================================================
# You can set the query image path and threshold here or pass them as arguments
# Examples for query_image_path:
#   - '../detection/detected_faces/face_001_20251123_114613.jpg'
#   - 'path/to/your/query_image.jpg'
#   - 'group.jpg' (if it contains only one face)
# Threshold: Similarity threshold (0.0 to 1.0), higher = more strict matching
# ============================================================================
QUERY_IMAGE_PATH = '../image2.jpg'
THRESHOLD = 0.5


def test_searching(query_image_path=None, threshold=None):
    """
    Test searching for a face in the database.
    
    Args:
        query_image_path: Path to the query image (must contain exactly one face)
                          If None, uses QUERY_IMAGE_PATH from configuration
        threshold: Similarity threshold for matching (0.0 to 1.0)
                   If None, uses THRESHOLD from configuration
    """
    # Use values from configuration if not provided
    if query_image_path is None:
        query_image_path = QUERY_IMAGE_PATH
    if threshold is None:
        threshold = THRESHOLD
    
    print("=" * 70)
    print("FACE SEARCH TEST")
    print("=" * 70)
    
    # Database path: always use storage/faces.db
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    db_path = os.path.join(project_root, "storage", "faces.db")
    
    if not os.path.exists(db_path):
        print(f"\n✗ Error: Database '{db_path}' not found!")
        print("Please run test_isertion.py first to add faces to the database.")
        return
    
    print(f"\n✓ Database '{db_path}' found.")
    
    # Initialize extractor and searcher
    print(f"\nInitializing extractor and searcher...")
    extractor = EmbeddingExtractor()  # Default: save_extracted_faces=False
    searcher = FaceSearch(db_path=db_path)
    
    print(f"\nQuery image: {query_image_path}")
    print(f"Similarity threshold: {threshold}")
    print("-" * 70)
    
    try:
        # Extract embedding from query image
        print("Step 1: Extracting embedding from query image...")
        query_embedding = extractor.extract(query_image_path, is_cropped_face=True)
        print(f"  ✓ Query embedding extracted: {query_embedding.shape}")
        
        # Search database
        print(f"\nStep 2: Searching database...")
        all_matches = searcher.search(query_embedding, threshold=threshold)
        
        print(f"\nSearch Results:")
        print(f"  Found {len(all_matches)} match(es) above threshold {threshold}")
        
        if len(all_matches) == 0:
            print("\n✗ No matches found in database.")
            print("Make sure you have inserted faces using test_isertion.py first.")
            return
        
        # Take only the best match (highest similarity)
        matches = [all_matches[0]] if all_matches else []
        if len(all_matches) > 1:
            print(f"  Using best match (similarity: {all_matches[0][2]:.2%})")
        
        # Display results with all information
        print(f"\n{'='*70}")
        print("MATCHES FOUND:")
        print(f"{'='*70}")
        
        # Prepare JSON output
        matches_json = []
        
        for i, (person_id, name, similarity, metadata) in enumerate(matches, 1):
            print(f"\n{'─'*70}")
            print(f"Match {i}:")
            print(f"{'─'*70}")
            print(f"  Person ID: {person_id}")
            print(f"  Name: {name}")
            print(f"  Similarity: {similarity:.2%} ({similarity:.4f})")
            
            # Get full person information from database
            person_info = None
            created_at = None
            try:
                person_info = searcher.get_by_id(person_id)
                if person_info:
                    p_id, p_name, p_metadata, created_at = person_info
                    print(f"\n  Database Information:")
                    print(f"    Created At: {created_at}")
            except Exception as e:
                print(f"    (Could not retrieve additional database info: {e})")
            
            # Display all metadata information (excluding original_path)
            if metadata:
                print(f"\n  Metadata Information:")
                for key, value in metadata.items():
                    # Skip original_path - not needed
                    if key == 'original_path':
                        continue
                    if key == 'image_path':
                        print(f"    {key}: {value}")
                        # Try to verify if image exists
                        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
                        full_path = os.path.join(project_root, value)
                        if os.path.exists(full_path):
                            print(f"      ✓ Image file exists at: {full_path}")
                        else:
                            print(f"      ✗ Image file not found at: {full_path}")
                    else:
                        print(f"    {key}: {value}")
            else:
                print(f"  Metadata: None")
            
            # Build JSON object for this match (excluding original_path from metadata)
            metadata_for_json = {}
            if metadata:
                for key, value in metadata.items():
                    if key != 'original_path':  # Exclude original_path
                        metadata_for_json[key] = value
            
            match_data = {
                "match_number": i,
                "person_id": person_id,
                "name": name,
                "similarity": float(similarity),
                "similarity_percentage": f"{similarity:.2%}",
                "metadata": metadata_for_json,
                "created_at": created_at if created_at else None
            }
            matches_json.append(match_data)
        
        # Load query image
        query_image = cv2.imread(query_image_path)
        if query_image is None:
            print(f"✗ Could not load query image: {query_image_path}")
            return
        
        # Get the best match (first match) for image display
        best_match = matches[0]
        person_id, name, similarity, metadata = best_match
        
        # Try to load the matched person's image from database metadata
        matched_image = None
        matched_path = None
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        storage_images_dir = os.path.join(project_root, "storage", "images")
        
        # CRITICAL: Use image_path from database metadata, not from script
        if metadata and 'image_path' in metadata:
            db_image_path = metadata['image_path']
            print(f"\n  Searching for image from database:")
            print(f"    Database image_path: {db_image_path}")
            
            # Try different path resolutions
            possible_paths = [
                os.path.join(project_root, db_image_path),  # Relative to project root
                db_image_path,  # Try as-is
                os.path.join(storage_images_dir, os.path.basename(db_image_path)),  # Direct from storage/images/
            ]
            
            # Try exact path first
            for path in possible_paths:
                if os.path.exists(path):
                    matched_image = cv2.imread(path)
                    if matched_image is not None:
                        print(f"    ✓ Found image at: {path}")
                        matched_path = path
                        break
            
            # If not found, try searching by person_id and name pattern
            if matched_image is None:
                print(f"    ⚠ Exact path not found, searching by person_id and name...")
                import glob
                
                # Try pattern: {name}_{person_id}.jpg
                safe_name = "".join(c for c in name if c.isalnum() or c in (' ', '-', '_')).strip().replace(' ', '_')
                patterns = [
                    os.path.join(storage_images_dir, f"{safe_name}_{person_id}.jpg"),
                    os.path.join(storage_images_dir, f"{safe_name}_{person_id}.*"),
                    os.path.join(storage_images_dir, f"{safe_name}_*.jpg"),
                ]
                
                for pattern in patterns:
                    matches_files = glob.glob(pattern)
                    if matches_files:
                        matched_image = cv2.imread(matches_files[0])
                        if matched_image is not None:
                            print(f"    ✓ Found image by pattern '{pattern}': {matches_files[0]}")
                            matched_path = matches_files[0]
                            break
                
                # Last resort: search all files in storage/images/ and match by person_id
                if matched_image is None and os.path.exists(storage_images_dir):
                    all_files = glob.glob(os.path.join(storage_images_dir, "*.jpg"))
                    for file_path in all_files:
                        filename = os.path.basename(file_path)
                        # Check if filename contains person_id
                        if f"_{person_id}." in filename or filename.endswith(f"_{person_id}.jpg"):
                            matched_image = cv2.imread(file_path)
                            if matched_image is not None:
                                print(f"    ✓ Found image by person_id: {file_path}")
                                matched_path = file_path
                                break
        
        # Image loading status (for display only, not in JSON)
        if matched_path:
            print(f"\n  ✓ Image loaded successfully: {matched_path}")
        else:
            print(f"\n  ✗ Could not load image from database metadata")
        
        # Print all information in JSON format (after image loading)
        print(f"\n{'='*70}")
        print("ALL MATCH INFORMATION (JSON FORMAT):")
        print(f"{'='*70}")
        print(json.dumps(matches_json, indent=2, ensure_ascii=False))
        print(f"{'='*70}")
        
        # Display images section
        print(f"\n{'='*70}")
        print("Displaying images...")
        print("  Left: Searched image (query)")
        print("  Right: Found image from database")
        print("Press any key to close the window.")
        print(f"{'='*70}")
        
        # If we can't get the original image, create a placeholder
        if matched_image is None:
            print(f"  ⚠ Creating placeholder image (original image not found)")
            matched_image = np.zeros((400, 400, 3), dtype=np.uint8)
            cv2.putText(matched_image, f"Match: {name}", (50, 150),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            cv2.putText(matched_image, f"ID: {person_id}", (50, 200),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            cv2.putText(matched_image, f"Similarity: {similarity:.2%}", (50, 250),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            if matched_path:
                cv2.putText(matched_image, f"Path: {matched_path}", (50, 300),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Resize images for display
        def resize_for_display(img, max_size=600):
            h, w = img.shape[:2]
            if h > max_size or w > max_size:
                scale = max_size / max(h, w)
                new_w = int(w * scale)
                new_h = int(h * scale)
                return cv2.resize(img, (new_w, new_h))
            return img
        
        query_display = resize_for_display(query_image)
        matched_display = resize_for_display(matched_image)
        
        # Make both images same height
        h1, w1 = query_display.shape[:2]
        h2, w2 = matched_display.shape[:2]
        max_h = max(h1, h2)
        
        if h1 < max_h:
            pad = max_h - h1
            query_display = cv2.copyMakeBorder(query_display, 0, pad, 0, 0, 
                                              cv2.BORDER_CONSTANT, value=[0, 0, 0])
        if h2 < max_h:
            pad = max_h - h2
            matched_display = cv2.copyMakeBorder(matched_display, 0, pad, 0, 0,
                                                 cv2.BORDER_CONSTANT, value=[0, 0, 0])
        
        # Combine side by side
        comparison = np.hstack([query_display, matched_display])
        
        # Add labels
        cv2.putText(comparison, "Query Image", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(comparison, f"Match: {name} ({similarity:.2%})", 
                   (w1 + 10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # Show comparison
        cv2.imshow("Face Search: Query (Left) vs Match (Right)", comparison)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        
        print("\n✓ Search test completed successfully!")
        
    except ValueError as e:
        print(f"\n✗ Error: {e}")
        print("Make sure the query image contains exactly ONE face.")
    except Exception as e:
        print(f"\n✗ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        searcher.close()


if __name__ == "__main__":
    # Test with values from configuration
    # You can also pass custom values:
    # test_searching('path/to/query_image.jpg', threshold=0.7)
    test_searching()

