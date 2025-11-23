"""
Example: How to Search for Faces in Database

This script demonstrates step-by-step how to:
1. Extract embedding from a query image
2. Search the database for matching faces
3. Process and display results
"""

from detection import FaceDetector
from embedding import EmbeddingExtractor
from storage import FaceSearch
import cv2


def search_single_face_example():
    """Example: Search for a single face in the database."""
    
    print("=" * 50)
    print("EXAMPLE 1: Search for Single Face")
    print("=" * 50)
    
    # Step 1: Initialize components
    print("\nStep 1: Initializing components...")
    extractor = EmbeddingExtractor()
    searcher = FaceSearch("faces.db")
    
    # Step 2: Load query image
    print("\nStep 2: Loading query image...")
    query_image_path = "test.jpg"  # Change to your query image
    query_image = cv2.imread(query_image_path)
    
    if query_image is None:
        print(f"Error: Could not load image from {query_image_path}")
        return
    
    # Step 3: Extract embedding from query image
    print("\nStep 3: Extracting embedding from query image...")
    try:
        query_embedding = extractor.extract(query_image)
        print(f"✓ Embedding extracted: shape {query_embedding.shape}")
    except ValueError as e:
        print(f"Error: {e}")
        return
    
    # Step 4: Search in database
    print("\nStep 4: Searching database...")
    threshold = 0.6  # Similarity threshold
    matches = searcher.search(query_embedding, threshold=threshold)
    
    print(f"✓ Search completed with threshold {threshold}")
    print(f"  Found {len(matches)} match(es)")
    
    # Step 5: Process results
    print("\nStep 5: Processing results...")
    if matches:
        print("\n--- MATCHES FOUND ---")
        for i, (person_id, name, similarity, metadata) in enumerate(matches, 1):
            print(f"\nMatch {i}:")
            print(f"  Person ID: {person_id}")
            print(f"  Name: {name}")
            print(f"  Similarity: {similarity:.4f} ({similarity:.2%})")
            if metadata:
                print(f"  Metadata: {metadata}")
        
        # Best match
        best_match = matches[0]
        person_id, name, similarity, metadata = best_match
        print(f"\n--- BEST MATCH ---")
        print(f"Name: {name}")
        print(f"Confidence: {similarity:.2%}")
    else:
        print("\n--- NO MATCHES FOUND ---")
        print("Try:")
        print("  - Lowering the threshold (e.g., 0.5)")
        print("  - Checking if the person is in the database")
        print("  - Ensuring the query image has a clear face")
    
    # Step 6: Close connection
    print("\nStep 6: Closing database connection...")
    searcher.close()
    print("✓ Done!")


def search_multiple_faces_example():
    """Example: Search for multiple faces in a single image."""
    
    print("\n" + "=" * 50)
    print("EXAMPLE 2: Search for Multiple Faces")
    print("=" * 50)
    
    detector = FaceDetector()
    extractor = EmbeddingExtractor()
    searcher = FaceSearch("faces.db")
    
    # Load query image (may contain multiple faces)
    query_image_path = "test.jpg"
    query_image = cv2.imread(query_image_path)
    
    if query_image is None:
        print(f"Error: Could not load {query_image_path}")
        return
    
    # Detect all faces
    print("\nDetecting faces in query image...")
    detections = detector.detect(query_image)
    print(f"✓ Found {len(detections)} face(s)")
    
    if len(detections) == 0:
        print("No faces detected in query image")
        return
    
    # Extract embeddings for all faces
    print("\nExtracting embeddings...")
    embeddings = extractor.extract_from_image(query_image)
    print(f"✓ Extracted {len(embeddings)} embedding(s)")
    
    # Search for each face
    print("\nSearching database for each face...")
    threshold = 0.6
    
    for i, (detection, embedding) in enumerate(zip(detections, embeddings), 1):
        print(f"\n--- Face {i} ---")
        print(f"Location: {detection['bbox']}")
        print(f"Confidence: {detection['det_score']:.2%}")
        
        matches = searcher.search(embedding, threshold=threshold)
        
        if matches:
            person_id, name, similarity, metadata = matches[0]
            print(f"✓ Identified as: {name}")
            print(f"  Match confidence: {similarity:.2%}")
            if metadata:
                print(f"  Metadata: {metadata}")
        else:
            print("✗ Unknown person (no match found)")
    
    searcher.close()


def search_with_different_thresholds_example():
    """Example: Search with different similarity thresholds."""
    
    print("\n" + "=" * 50)
    print("EXAMPLE 3: Search with Different Thresholds")
    print("=" * 50)
    
    extractor = EmbeddingExtractor()
    searcher = FaceSearch("faces.db")
    
    query_image_path = "test.jpg"
    query_image = cv2.imread(query_image_path)
    
    if query_image is None:
        print(f"Error: Could not load {query_image_path}")
        return
    
    # Extract embedding
    query_embedding = extractor.extract(query_image)
    
    # Try different thresholds
    thresholds = [0.5, 0.6, 0.7, 0.8]
    
    print("\nTesting different similarity thresholds:")
    print("-" * 50)
    
    for threshold in thresholds:
        matches = searcher.search(query_embedding, threshold=threshold)
        print(f"\nThreshold: {threshold}")
        print(f"  Matches found: {len(matches)}")
        
        if matches:
            for person_id, name, similarity, _ in matches[:3]:  # Show top 3
                print(f"    - {name}: {similarity:.4f}")
    
    searcher.close()


def search_and_display_all_example():
    """Example: Search and display all persons in database."""
    
    print("\n" + "=" * 50)
    print("EXAMPLE 4: List All Persons in Database")
    print("=" * 50)
    
    searcher = FaceSearch("faces.db")
    
    # Get all persons
    all_persons = searcher.get_all()
    
    print(f"\nTotal persons in database: {len(all_persons)}")
    print("-" * 50)
    
    if all_persons:
        for person_id, name, metadata, created_at in all_persons:
            print(f"\nID: {person_id}")
            print(f"  Name: {name}")
            print(f"  Created: {created_at}")
            if metadata:
                print(f"  Metadata: {metadata}")
    else:
        print("Database is empty")
    
    searcher.close()


def search_with_context_manager_example():
    """Example: Using context manager for automatic cleanup."""
    
    print("\n" + "=" * 50)
    print("EXAMPLE 5: Search with Context Manager")
    print("=" * 50)
    
    extractor = EmbeddingExtractor()
    
    query_image_path = "test.jpg"
    query_image = cv2.imread(query_image_path)
    
    if query_image is None:
        print(f"Error: Could not load {query_image_path}")
        return
    
    # Extract embedding
    query_embedding = extractor.extract(query_image)
    
    # Use context manager (automatically closes connection)
    with FaceSearch("faces.db") as searcher:
        matches = searcher.search(query_embedding, threshold=0.6)
        
        if matches:
            person_id, name, similarity, metadata = matches[0]
            print(f"\n✓ Found: {name}")
            print(f"  Similarity: {similarity:.2%}")
        else:
            print("\n✗ No match found")
    
    # Connection is automatically closed here


if __name__ == "__main__":
    # Run examples
    search_single_face_example()
    # search_multiple_faces_example()  # Uncomment to run
    # search_with_different_thresholds_example()  # Uncomment to run
    # search_and_display_all_example()  # Uncomment to run
    # search_with_context_manager_example()  # Uncomment to run

