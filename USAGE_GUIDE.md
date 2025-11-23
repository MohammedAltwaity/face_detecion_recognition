# Complete Usage Guide - Face Recognition System

This guide explains each module and how to use them step-by-step.

## Table of Contents

1. [Module Overview](#module-overview)
2. [Module 1: Detection](#module-1-detection)
3. [Module 2: Embedding](#module-2-embedding)
4. [Module 3: Insert Data](#module-3-insert-data)
5. [Module 4: Search Data](#module-4-search-data)
6. [Complete Examples](#complete-examples)

---

## Module Overview

The system is organized into 4 main modules:

1. **`detection/`** - Detects faces in images
2. **`embedding/`** - Extracts face embeddings (512-dim vectors)
3. **`storage/insert.py`** - Inserts face data into database
4. **`storage/search.py`** - Searches for faces in database

---

## Module 1: Detection

**Location**: `detection/detector.py`  
**Purpose**: Find faces in images and get their locations

### What it does:
- Detects faces in images using RetinaFace
- Returns bounding boxes (where faces are located)
- Returns confidence scores (how sure it is about detection)
- Returns facial landmarks (optional)

### How to use:

```python
from detection import FaceDetector
import cv2

# Initialize detector
detector = FaceDetector()

# Detect faces in an image
detections = detector.detect("image.jpg")

# Process results
for det in detections:
    bbox = det['bbox']          # [x1, y1, x2, y2] - face location
    confidence = det['det_score']  # 0.0 to 1.0 - how confident
    landmarks = det['landmark']    # Facial landmarks (optional)
    
    print(f"Face found at {bbox} with {confidence:.2%} confidence")
```

### Output:
```python
[
    {
        'bbox': [100, 150, 300, 350],  # [x1, y1, x2, y2]
        'det_score': 0.98,              # Confidence score
        'landmark': [...]               # Facial landmarks
    },
    # ... more faces if multiple detected
]
```

### Example:
```python
from detection import FaceDetector

detector = FaceDetector()
detections = detector.detect("group_photo.jpg")

print(f"Found {len(detections)} face(s)")
for i, det in enumerate(detections):
    print(f"Face {i+1}: {det['bbox']}, confidence: {det['det_score']:.2%}")
```

---

## Module 2: Embedding

**Location**: `embedding/extractor.py`  
**Purpose**: Extract 512-dimensional face embeddings from images

### What it does:
- Extracts face embeddings (mathematical representation of a face)
- Returns 512-dimensional vectors (normalized)
- These vectors are used to compare faces

### How to use:

```python
from embedding import EmbeddingExtractor
import numpy as np

# Initialize extractor
extractor = EmbeddingExtractor()

# Extract embedding from a single face image
embedding = extractor.extract("face.jpg")

# Check the embedding
print(f"Embedding shape: {embedding.shape}")  # (512,)
print(f"Embedding type: {embedding.dtype}")   # float32
```

### Methods:

#### 1. Extract from single image:
```python
embedding = extractor.extract("face.jpg")
# Returns: numpy array of shape (512,)
```

#### 2. Extract from multiple images:
```python
embeddings = extractor.extract_batch(["face1.jpg", "face2.jpg", "face3.jpg"])
# Returns: List of numpy arrays, each shape (512,)
```

#### 3. Extract from image with multiple faces:
```python
import cv2
image = cv2.imread("group_photo.jpg")
embeddings = extractor.extract_from_image(image)
# Returns: List of embeddings, one per face
```

### Example:
```python
from embedding import EmbeddingExtractor

extractor = EmbeddingExtractor()

# Single face
embedding = extractor.extract("alice.jpg")
print(f"Alice's embedding: {embedding.shape}")

# Multiple faces in one image
import cv2
image = cv2.imread("group.jpg")
embeddings = extractor.extract_from_image(image)
print(f"Found {len(embeddings)} face embeddings")
```

---

## Module 3: Insert Data

**Location**: `storage/insert.py`  
**Purpose**: Store person information and face embeddings in the database

### What it does:
- Stores person name, face embedding, and optional metadata
- Creates database table if it doesn't exist
- Returns the ID of the newly added person

### How to use:

#### Step-by-step: Insert a person

```python
# Step 1: Import required modules
from detection import FaceDetector
from embedding import EmbeddingExtractor
from storage import FaceInsert
import cv2

# Step 2: Initialize all components
detector = FaceDetector()
extractor = EmbeddingExtractor()
inserter = FaceInsert("faces.db")  # Database file

# Step 3: Load image
image = cv2.imread("person.jpg")

# Step 4: Detect face (optional, but good to validate)
detections = detector.detect(image)
if len(detections) != 1:
    print("Error: Image must contain exactly one face")
    exit()

# Step 5: Extract embedding
embedding = extractor.extract(image)

# Step 6: Insert into database
person_id = inserter.add(
    name="Alice",
    embedding=embedding,
    metadata={"age": 25, "department": "Engineering"}
)

print(f"Person added with ID: {person_id}")

# Step 7: Close connection
inserter.close()
```

### Complete Example: Insert Multiple People

```python
from detection import FaceDetector
from embedding import EmbeddingExtractor
from storage import FaceInsert
import cv2

# Initialize
detector = FaceDetector()
extractor = EmbeddingExtractor()
inserter = FaceInsert("faces.db")

# List of people to add
people = [
    ("alice.jpg", "Alice", {"age": 25, "role": "admin"}),
    ("bob.jpg", "Bob", {"age": 30, "role": "user"}),
    ("charlie.jpg", "Charlie", {"age": 28, "role": "guest"})
]

# Add each person
for image_path, name, metadata in people:
    try:
        # Load and process
        image = cv2.imread(image_path)
        embedding = extractor.extract(image)
        
        # Insert
        person_id = inserter.add(name, embedding, metadata)
        print(f"✓ Added {name} with ID {person_id}")
    except Exception as e:
        print(f"✗ Failed to add {name}: {e}")

inserter.close()
```

### Using Context Manager (Recommended):

```python
from storage import FaceInsert
from embedding import EmbeddingExtractor

extractor = EmbeddingExtractor()

# Automatically closes connection
with FaceInsert("faces.db") as inserter:
    embedding = extractor.extract("person.jpg")
    person_id = inserter.add("John Doe", embedding, {"age": 30})
    print(f"Added person with ID: {person_id}")
```

---

## Module 4: Search Data

**Location**: `storage/search.py`  
**Purpose**: Search for persons in the database based on face embeddings

### What it does:
- Searches database for matching faces
- Uses cosine similarity to compare embeddings
- Returns matching persons sorted by similarity

### How to use:

#### Step-by-step: Search for a face

```python
# Step 1: Import required modules
from embedding import EmbeddingExtractor
from storage import FaceSearch
import cv2

# Step 2: Initialize components
extractor = EmbeddingExtractor()
searcher = FaceSearch("faces.db")

# Step 3: Load query image
query_image = cv2.imread("unknown_face.jpg")

# Step 4: Extract embedding from query image
query_embedding = extractor.extract(query_image)

# Step 5: Search in database
matches = searcher.search(query_embedding, threshold=0.6)

# Step 6: Process results
if matches:
    # Get best match (first one, highest similarity)
    person_id, name, similarity, metadata = matches[0]
    print(f"Found: {name}")
    print(f"Similarity: {similarity:.2%}")
    print(f"Metadata: {metadata}")
else:
    print("No match found")

# Step 7: Close connection
searcher.close()
```

### Understanding the Results:

```python
matches = searcher.search(embedding, threshold=0.6)

# matches is a list of tuples:
# (person_id, name, similarity_score, metadata)

for person_id, name, similarity, metadata in matches:
    print(f"ID: {person_id}")
    print(f"Name: {name}")
    print(f"Similarity: {similarity:.2%}")  # 0.0 to 1.0
    print(f"Metadata: {metadata}")
    print("---")
```

### Similarity Threshold:

- **0.5-0.6**: More lenient (may have false positives)
- **0.6-0.7**: Balanced (recommended default)
- **0.7-0.8**: Strict (high accuracy)
- **0.8+**: Very strict (may miss valid matches)

### Complete Example: Search with Multiple Faces

```python
from detection import FaceDetector
from embedding import EmbeddingExtractor
from storage import FaceSearch
import cv2

# Initialize
detector = FaceDetector()
extractor = EmbeddingExtractor()
searcher = FaceSearch("faces.db")

# Load query image (may have multiple faces)
query_image = cv2.imread("group_photo.jpg")

# Detect all faces
detections = detector.detect(query_image)
print(f"Found {len(detections)} face(s) in query image")

# Extract embeddings for all faces
embeddings = extractor.extract_from_image(query_image)

# Search for each face
for i, (detection, embedding) in enumerate(zip(detections, embeddings)):
    print(f"\n--- Face {i+1} ---")
    matches = searcher.search(embedding, threshold=0.6)
    
    if matches:
        person_id, name, similarity, metadata = matches[0]
        print(f"Identified as: {name}")
        print(f"Confidence: {similarity:.2%}")
    else:
        print("Unknown person")

searcher.close()
```

### Using Context Manager:

```python
from storage import FaceSearch
from embedding import EmbeddingExtractor

extractor = EmbeddingExtractor()

with FaceSearch("faces.db") as searcher:
    query_embedding = extractor.extract("query.jpg")
    matches = searcher.search(query_embedding, threshold=0.65)
    
    if matches:
        person_id, name, similarity, _ = matches[0]
        print(f"Found: {name} ({similarity:.2%})")
```

---

## Complete Examples

### Example 1: Full Workflow - Add and Search

```python
from detection import FaceDetector
from embedding import EmbeddingExtractor
from storage import FaceInsert, FaceSearch
import cv2

# Initialize all components
detector = FaceDetector()
extractor = EmbeddingExtractor()
inserter = FaceInsert("faces.db")
searcher = FaceSearch("faces.db")

# ===== PART 1: ADD PERSON TO DATABASE =====
print("Adding person to database...")
image = cv2.imread("alice.jpg")
embedding = extractor.extract(image)
person_id = inserter.add("Alice", embedding, {"age": 25})
print(f"Added Alice with ID: {person_id}")

# ===== PART 2: SEARCH FOR PERSON =====
print("\nSearching for person...")
query_image = cv2.imread("query_alice.jpg")
query_embedding = extractor.extract(query_image)
matches = searcher.search(query_embedding, threshold=0.6)

if matches:
    person_id, name, similarity, metadata = matches[0]
    print(f"Found: {name}")
    print(f"Similarity: {similarity:.2%}")
else:
    print("No match found")

# Clean up
inserter.close()
searcher.close()
```

### Example 2: Using the Pipeline (Easier Way)

```python
from pipeline import FaceRecognitionPipeline

# Initialize pipeline (handles everything)
pipeline = FaceRecognitionPipeline("faces.db")

# Add person (automatic: detect → extract → insert)
person_id = pipeline.add_person("alice.jpg", "Alice", {"age": 25})
print(f"Added with ID: {person_id}")

# Search (automatic: detect → extract → search)
results = pipeline.search("query.jpg", threshold=0.6)

for result in results:
    if result['matches']:
        person_id, name, similarity, _ = result['matches'][0]
        print(f"Face {result['face_index']}: {name} ({similarity:.2%})")
    else:
        print(f"Face {result['face_index']}: Unknown")

pipeline.close()
```

### Example 3: Batch Operations

```python
from embedding import EmbeddingExtractor
from storage import FaceInsert, FaceSearch

extractor = EmbeddingExtractor()

# Batch insert
with FaceInsert("faces.db") as inserter:
    people_data = [
        ("alice.jpg", "Alice", {"age": 25}),
        ("bob.jpg", "Bob", {"age": 30}),
        ("charlie.jpg", "Charlie", {"age": 28})
    ]
    
    for image_path, name, metadata in people_data:
        embedding = extractor.extract(image_path)
        person_id = inserter.add(name, embedding, metadata)
        print(f"Added {name}")

# Batch search
with FaceSearch("faces.db") as searcher:
    query_images = ["query1.jpg", "query2.jpg", "query3.jpg"]
    
    for query_path in query_images:
        query_embedding = extractor.extract(query_path)
        matches = searcher.search(query_embedding, threshold=0.6)
        
        if matches:
            name = matches[0][1]
            similarity = matches[0][2]
            print(f"{query_path}: Found {name} ({similarity:.2%})")
        else:
            print(f"{query_path}: No match")
```

---

## Quick Reference

### Detection
```python
from detection import FaceDetector
detector = FaceDetector()
detections = detector.detect("image.jpg")
```

### Embedding
```python
from embedding import EmbeddingExtractor
extractor = EmbeddingExtractor()
embedding = extractor.extract("face.jpg")
```

### Insert
```python
from storage import FaceInsert
inserter = FaceInsert("faces.db")
person_id = inserter.add("Name", embedding, {"key": "value"})
```

### Search
```python
from storage import FaceSearch
searcher = FaceSearch("faces.db")
matches = searcher.search(embedding, threshold=0.6)
```

---

## Common Workflows

### Workflow 1: Register a New Person
1. Load image → 2. Detect face → 3. Extract embedding → 4. Insert to DB

### Workflow 2: Find a Person
1. Load query image → 2. Extract embedding → 3. Search in DB → 4. Get matches

### Workflow 3: Process Multiple Faces
1. Load image → 2. Detect all faces → 3. Extract all embeddings → 4. Search each

---

## Tips

1. **Always validate**: Check if exactly one face before inserting
2. **Use context managers**: Automatically close connections
3. **Adjust threshold**: Start with 0.6, adjust based on results
4. **Handle errors**: Images might not have faces
5. **Batch operations**: More efficient for multiple people

---

## Troubleshooting

**Problem**: "No face detected"
- **Solution**: Ensure image has a clear, visible face

**Problem**: "Multiple faces detected" (when inserting)
- **Solution**: Use image with exactly one face for registration

**Problem**: "No matches found"
- **Solution**: Lower threshold (try 0.5) or check if person is in database

**Problem**: "Too many false matches"
- **Solution**: Increase threshold (try 0.7 or 0.8)

