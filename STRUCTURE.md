# Project Folder Structure

The project is organized into folders by functionality:

```
face_detecion/
├── detection/              # Face detection module
│   ├── __init__.py
│   └── detector.py         # FaceDetector class
│
├── embedding/              # Embedding extraction module
│   ├── __init__.py
│   └── extractor.py       # EmbeddingExtractor class
│
├── storage/                # Database operations
│   ├── __init__.py
│   ├── insert.py          # FaceInsert class - Insert data into DB
│   └── search.py          # FaceSearch class - Search in DB
│
├── pipeline/               # Main orchestrator
│   ├── __init__.py
│   └── pipeline.py        # FaceRecognitionPipeline class
│
├── example.py              # Usage examples
├── requirements.txt        # Dependencies
├── README.md              # Main documentation
├── MODULES.md             # Module documentation
└── STRUCTURE.md           # This file
```

## Folder Responsibilities

### `detection/` - Face Detection
**Purpose**: Detect faces in images using RetinaFace

**Files**:
- `detector.py`: Contains `FaceDetector` class
  - Detects faces in images
  - Returns bounding boxes, landmarks, confidence scores

**Usage**:
```python
from detection import FaceDetector

detector = FaceDetector()
detections = detector.detect("image.jpg")
```

---

### `embedding/` - Embedding Extraction
**Purpose**: Extract 512-dimensional face embeddings using ArcFace

**Files**:
- `extractor.py`: Contains `EmbeddingExtractor` class
  - Extracts embeddings from face images
  - Returns normalized 512-dim vectors

**Usage**:
```python
from embedding import EmbeddingExtractor

extractor = EmbeddingExtractor()
embedding = extractor.extract("face.jpg")
```

---

### `storage/` - Database Operations
**Purpose**: Handle all database operations (insert and search)

**Files**:
- `insert.py`: Contains `FaceInsert` class
  - Insert person data with embeddings into database
  - Used after detection and embedding extraction
  
- `search.py`: Contains `FaceSearch` class
  - Search for persons in database using embeddings
  - Uses cosine similarity for matching

**Usage**:
```python
from storage import FaceInsert, FaceSearch

# Insert data
inserter = FaceInsert("faces.db")
person_id = inserter.add("Alice", embedding, {"age": 25})

# Search data
searcher = FaceSearch("faces.db")
matches = searcher.search(query_embedding, threshold=0.6)
```

---

### `pipeline/` - Main Orchestrator
**Purpose**: High-level API that combines all modules

**Files**:
- `pipeline.py`: Contains `FaceRecognitionPipeline` class
  - Orchestrates detection, embedding, and storage
  - Provides easy-to-use interface

**Usage**:
```python
from pipeline import FaceRecognitionPipeline

pipeline = FaceRecognitionPipeline()
person_id = pipeline.add_person("alice.jpg", "Alice")
results = pipeline.search("query.jpg")
```

---

## Workflow

### Adding a Person:
1. **Detection** (`detection/detector.py`): Detect face in image
2. **Embedding** (`embedding/extractor.py`): Extract embedding from face
3. **Insert** (`storage/insert.py`): Store in database

### Searching for a Person:
1. **Detection** (`detection/detector.py`): Detect faces in query image
2. **Embedding** (`embedding/extractor.py`): Extract embeddings from faces
3. **Search** (`storage/search.py`): Search database for matches

---

## Import Examples

### Using Individual Modules:
```python
from detection import FaceDetector
from embedding import EmbeddingExtractor
from storage import FaceInsert, FaceSearch

# Use each module independently
detector = FaceDetector()
extractor = EmbeddingExtractor()
inserter = FaceInsert()
searcher = FaceSearch()
```

### Using Pipeline:
```python
from pipeline import FaceRecognitionPipeline

# High-level API
pipeline = FaceRecognitionPipeline()
```

---

## Benefits of This Structure

1. **Clear Separation**: Each folder has a single responsibility
2. **Easy to Navigate**: Find code by functionality
3. **Modular**: Use only what you need
4. **Maintainable**: Easy to update individual components
5. **Scalable**: Easy to add new features to specific folders

