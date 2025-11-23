"""
Example: How to Insert Data into Database

This script demonstrates step-by-step how to:
1. Detect faces in an image
2. Extract embeddings from faces
3. Insert person data into the database
"""

from detection import FaceDetector
from embedding import EmbeddingExtractor
from storage import FaceInsert
import cv2


def insert_person_example():
    """Example: Insert a single person into the database."""
    
    print("=" * 50)
    print("EXAMPLE 1: Insert Single Person")
    print("=" * 50)
    
    # Step 1: Initialize all components
    print("\nStep 1: Initializing components...")
    detector = FaceDetector()
    extractor = EmbeddingExtractor()
    inserter = FaceInsert("faces.db")
    
    # Step 2: Load image
    print("\nStep 2: Loading image...")
    image_path = "test.jpg"  # Change to your image path
    image = cv2.imread(image_path)
    
    if image is None:
        print(f"Error: Could not load image from {image_path}")
        return
    
    # Step 3: Detect face (validate that exactly one face exists)
    print("\nStep 3: Detecting face...")
    detections = detector.detect(image)
    
    if len(detections) == 0:
        print("Error: No face detected in the image")
        return
    
    if len(detections) > 1:
        print(f"Warning: Multiple faces detected ({len(detections)}). Using first face.")
    
    print(f"✓ Found {len(detections)} face(s)")
    print(f"  Bounding box: {detections[0]['bbox']}")
    print(f"  Confidence: {detections[0]['det_score']:.2%}")
    
    # Step 4: Extract embedding
    print("\nStep 4: Extracting face embedding...")
    embedding = extractor.extract(image)
    print(f"✓ Embedding extracted: shape {embedding.shape}, dtype {embedding.dtype}")
    
    # Step 5: Insert into database
    print("\nStep 5: Inserting into database...")
    person_id = inserter.add(
        name="John Doe",
        embedding=embedding,
        metadata={"age": 30, "department": "Engineering", "role": "developer"}
    )
    print(f"✓ Person added successfully!")
    print(f"  Person ID: {person_id}")
    print(f"  Name: John Doe")
    print(f"  Metadata: {{'age': 30, 'department': 'Engineering', 'role': 'developer'}}")
    
    # Step 6: Close connection
    print("\nStep 6: Closing database connection...")
    inserter.close()
    print("✓ Done!")


def insert_multiple_people_example():
    """Example: Insert multiple people into the database."""
    
    print("\n" + "=" * 50)
    print("EXAMPLE 2: Insert Multiple People")
    print("=" * 50)
    
    # Initialize components
    extractor = EmbeddingExtractor()
    
    # List of people to add
    people = [
        ("test.jpg", "Alice", {"age": 25, "department": "Sales"}),
        ("test.jpg", "Bob", {"age": 30, "department": "Marketing"}),
        ("test.jpg", "Charlie", {"age": 28, "department": "HR"}),
    ]
    
    # Use context manager for automatic cleanup
    with FaceInsert("faces.db") as inserter:
        for image_path, name, metadata in people:
            try:
                print(f"\nProcessing {name}...")
                
                # Extract embedding
                embedding = extractor.extract(image_path)
                
                # Insert into database
                person_id = inserter.add(name, embedding, metadata)
                
                print(f"  ✓ Added {name} with ID {person_id}")
                
            except ValueError as e:
                print(f"  ✗ Failed to add {name}: {e}")
            except Exception as e:
                print(f"  ✗ Error adding {name}: {e}")


def insert_with_validation_example():
    """Example: Insert with proper validation."""
    
    print("\n" + "=" * 50)
    print("EXAMPLE 3: Insert with Validation")
    print("=" * 50)
    
    detector = FaceDetector()
    extractor = EmbeddingExtractor()
    
    image_path = "test.jpg"
    
    with FaceInsert("faces.db") as inserter:
        # Load image
        image = cv2.imread(image_path)
        if image is None:
            print(f"Error: Could not load {image_path}")
            return
        
        # Validate exactly one face
        detections = detector.detect(image)
        
        if len(detections) == 0:
            print("Error: No face detected. Cannot insert.")
            return
        
        if len(detections) > 1:
            print(f"Error: Multiple faces detected ({len(detections)}).")
            print("For registration, image must contain exactly one face.")
            return
        
        # Extract embedding
        embedding = extractor.extract(image)
        
        # Insert
        person_id = inserter.add(
            name="Validated Person",
            embedding=embedding,
            metadata={"validated": True}
        )
        
        print(f"✓ Successfully inserted person with ID: {person_id}")


if __name__ == "__main__":
    # Run examples
    insert_person_example()
    # insert_multiple_people_example()  # Uncomment to run
    # insert_with_validation_example()  # Uncomment to run

