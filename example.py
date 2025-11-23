"""
Example usage of the Face Recognition System.

This script demonstrates the main features using the new folder structure:
1. Adding persons to the database from images
2. Searching for persons in the database by face
3. Listing all persons in the database
4. Handling multiple faces in a single image
5. Processing search results

Run this script to see the system in action:
    python example.py

Make sure you have test.jpg in the same directory, or modify the image paths
in the examples below.
"""

from pipeline import FaceRecognitionPipeline


def main():
    # Initialize the face recognition pipeline
    print("Initializing Face Recognition Pipeline...")
    pipeline = FaceRecognitionPipeline(db_path="faces.db")
    
    try:
        # Example 1: Add a person to the database
        print("\n=== Example 1: Adding a person to the database ===")
        try:
            person_id = pipeline.add_person(
                image_path="test.jpg",
                name="John Doe",
                metadata={"age": 30, "department": "Engineering"}
            )
            print(f"Successfully added person with ID: {person_id}")
        except ValueError as e:
            print(f"Error adding person: {e}")
            print("Note: Make sure test.jpg contains exactly one face")
        
        # Example 2: Search for a person in the database
        print("\n=== Example 2: Searching for a person ===")
        try:
            results = pipeline.search(
                image_path="test.jpg",
                threshold=0.6
            )
            
            if results:
                for result in results:
                    print(f"\nFace {result['face_index']} detected:")
                    print(f"  Bounding box: {result['bbox']}")
                    print(f"  Detection score: {result['det_score']:.4f}")
                    
                    if result['matches']:
                        print(f"  Found {len(result['matches'])} match(es):")
                        for person_id, name, similarity, metadata in result['matches']:
                            print(f"    - Person ID: {person_id}, Name: {name}")
                            print(f"      Similarity: {similarity:.4f}")
                            if metadata:
                                print(f"      Metadata: {metadata}")
                    else:
                        print("  No matches found in database")
            else:
                print("No faces detected in the image")
        except ValueError as e:
            print(f"Error searching: {e}")
        
        # Example 3: Get all persons in the database
        print("\n=== Example 3: Listing all persons in database ===")
        all_persons = pipeline.get_all_persons()
        if all_persons:
            print(f"Total persons in database: {len(all_persons)}")
            for person_id, name, metadata, created_at in all_persons:
                print(f"  ID: {person_id}, Name: {name}, Created: {created_at}")
                if metadata:
                    print(f"    Metadata: {metadata}")
        else:
            print("Database is empty")
        
        # Example 4: Process image with multiple faces
        print("\n=== Example 4: Processing image (detect all faces) ===")
        try:
            results = pipeline.process_image(
                image_path="test.jpg",
                threshold=0.6
            )
            
            print(f"Detected {len(results)} face(s) in the image")
            for result in results:
                print(f"\nFace {result['face_index']}:")
                print(f"  Bounding box: {result['bbox']}")
                print(f"  Matches: {len(result['matches'])}")
        except ValueError as e:
            print(f"Error processing image: {e}")
    
    finally:
        # Close the database connection
        pipeline.close()
        print("\nDatabase connection closed.")


if __name__ == "__main__":
    main()
