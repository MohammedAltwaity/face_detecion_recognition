# Module Structure Guide

This project is organized into modular components, each with a single responsibility.

## Module Overview

### 1. `detection.py` - Face Detection
**Purpose**: Detect faces in images using RetinaFace

**Key Class**: `FaceDetector`

**Usage**:
```python
from detection import FaceDetector

detector = FaceDetector()
detections = detector.detect("image.jpg")

for det in detections:
    print(f"Face at {det['bbox']} with confidence {det['det_score']}")
```

**Output**: List of detections with bounding boxes, landmarks, and confidence scores

---

### 2. `face_extraction.py` - Face Region Extraction
**Purpose**: Extract/crop face regions from images based on detections

**Key Class**: `FaceExtractor` (static methods)

**Usage**:
```python
from face_extraction import FaceExtractor
from detection import FaceDetector

# Detect faces first
detector = FaceDetector()
detections = detector.detect("image.jpg")

# Extract face regions
face_images = FaceExtractor.extract("image.jpg", detections)

for i, face_img in enumerate(face_images):
    cv2.imwrite(f"face_{i}.jpg", face_img)
```

**Output**: List of cropped face images (numpy arrays)

---

### 3. `embedding.py` - Embedding Extraction
**Purpose**: Extract 512-dimensional face embeddings using ArcFace

**Key Class**: `EmbeddingExtractor`

**Usage**:
```python
from embedding import EmbeddingExtractor

extractor = EmbeddingExtractor()

# Extract from single face image
embedding = extractor.extract("face.jpg")
print(f"Embedding shape: {embedding.shape}")  # (512,)

# Extract from multiple faces in one image
embeddings = extractor.extract_from_image(image_array)
```

**Output**: 512-dimensional numpy arrays (normalized, float32)

---

### 4. `storage.py` - Database Storage
**Purpose**: Store and retrieve face data from SQLite database

**Key Class**: `FaceStorage`

**Usage**:
```python
from storage import FaceStorage
from embedding import EmbeddingExtractor

storage = FaceStorage("faces.db")
extractor = EmbeddingExtractor()

# Add a person
embedding = extractor.extract("person.jpg")
person_id = storage.add("Alice", embedding, {"age": 25})

# Search for a person
query_embedding = extractor.extract("query.jpg")
matches = storage.search(query_embedding, threshold=0.6)

for person_id, name, similarity, metadata in matches:
    print(f"Found {name} with {similarity:.2%} similarity")
```

**Output**: Database operations (add, search, retrieve, delete)

---

### 5. `pipeline.py` - Main Pipeline
**Purpose**: Orchestrate all components for easy-to-use face recognition

**Key Class**: `FaceRecognitionPipeline`

**Usage**:
```python
from pipeline import FaceRecognitionPipeline

pipeline = FaceRecognitionPipeline()

# Add a person (handles detection + embedding + storage)
person_id = pipeline.add_person("alice.jpg", "Alice")

# Search (handles detection + embedding + search)
results = pipeline.search("query.jpg", threshold=0.6)

pipeline.close()
```

**Output**: High-level face recognition operations

---

## Using Modules Independently

Each module can be used independently, giving you full control over the pipeline:

### Example: Custom Pipeline

```python
from detection import FaceDetector
from face_extraction import FaceExtractor
from embedding import EmbeddingExtractor
from storage import FaceStorage

# Initialize components
detector = FaceDetector()
extractor = EmbeddingExtractor()
storage = FaceStorage("faces.db")

# Custom workflow
image = cv2.imread("image.jpg")

# Step 1: Detect faces
detections = detector.detect(image)

# Step 2: Extract face regions (optional)
face_images = FaceExtractor.extract(image, detections)

# Step 3: Extract embeddings
embeddings = extractor.extract_from_image(image)

# Step 4: Store or search
for i, embedding in enumerate(embeddings):
    matches = storage.search(embedding, threshold=0.6)
    print(f"Face {i}: {len(matches)} matches")
```

### Example: Only Detection

```python
from detection import FaceDetector

detector = FaceDetector()
detections = detector.detect("image.jpg")

for det in detections:
    bbox = det['bbox']
    confidence = det['det_score']
    print(f"Face at {bbox} with {confidence:.2%} confidence")
```

### Example: Only Embedding Extraction

```python
from embedding import EmbeddingExtractor

extractor = EmbeddingExtractor()
embedding = extractor.extract("face.jpg")

# Use embedding for your own purposes
# (e.g., save to file, send to API, etc.)
```

---

## Module Dependencies

```
pipeline.py
  ├── detection.py
  ├── embedding.py
  └── storage.py

face_extraction.py (standalone, uses detection results)

detection.py (standalone)
embedding.py (standalone)
storage.py (standalone)
```

---

## Migration from Old Structure

If you were using the old structure:

**Old**:
```python
from face_recognition import FaceRecognitionSystem
system = FaceRecognitionSystem()
```

**New**:
```python
from pipeline import FaceRecognitionPipeline
pipeline = FaceRecognitionPipeline()
```

The API is the same, just different module names!

