# Face Recognition System

A complete face recognition system using **RetinaFace** for face detection, **ArcFace** for embedding extraction, and **SQLite** for storage. The system can detect faces in images, extract 512-dimensional embeddings, store them in a database, and search for persons based on face similarity.

## Table of Contents

- [Overview](#overview)
- [Technologies Used](#technologies-used)
- [Installation](#installation)
- [Project Structure](#project-structure)
- [Database Initialization](#database-initialization)
- [Inserting Faces](#inserting-faces)
- [Searching for Faces](#searching-for-faces)
- [File Descriptions](#file-descriptions)
- [Quick Start Guide](#quick-start-guide)
- [API Reference](#api-reference)
- [Troubleshooting](#troubleshooting)

---

## Overview

This face recognition system provides:

- **Face Detection**: Detects single or multiple faces in images using RetinaFace
- **Embedding Extraction**: Generates 512-dimensional face embeddings using ArcFace
- **Database Storage**: Stores person information and face embeddings in SQLite
- **Face Search**: Finds matching faces using cosine similarity
- **Metadata Support**: Store additional information with each person
- **Simple API**: Easy-to-use functions for common operations

---

## Technologies Used

### InsightFace

**InsightFace** is a comprehensive face analysis toolkit that provides:

- **Face Detection**: Using RetinaFace model
- **Face Recognition**: Using ArcFace model
- **Facial Landmarks**: 106-point and 68-point landmark detection
- **Age/Gender Estimation**: Additional face attributes

**Models Used:**
- `buffalo_l` (default): Large model with best accuracy
- `buffalo_m`: Medium model (balanced)
- `buffalo_s`: Small model (faster, less accurate)

**Model Location**: Models are automatically downloaded to `~/.insightface/models/` on first use (~100-200MB).

### RetinaFace

**RetinaFace** is a state-of-the-art face detection model that:

- Detects faces with high accuracy
- Provides bounding boxes, confidence scores, and facial landmarks
- Works well with various face angles and lighting conditions
- Handles multiple faces in a single image

**Key Features:**
- Real-time face detection
- Robust to occlusions
- Works with profile and frontal faces

### ArcFace

**ArcFace** (Additive Angular Margin Loss) is a face recognition model that:

- Extracts 512-dimensional face embeddings
- Provides highly discriminative features for face matching
- Achieves state-of-the-art recognition accuracy
- Normalizes embeddings for cosine similarity comparison

**Embedding Properties:**
- **Dimensions**: 512 (float32)
- **Size**: 2048 bytes per embedding
- **Normalized**: Embeddings are L2-normalized
- **Usage**: Compare using cosine similarity

---

## Installation

### Prerequisites

- Python 3.7 or higher
- pip package manager

### Install Dependencies

```bash
pip install -r requirements.txt
```

**Dependencies:**
- `insightface` - Face detection and recognition models
- `opencv-python` - Image processing
- `numpy` - Numerical operations
- `onnxruntime` - Model inference engine

**Note**: On first run, InsightFace will automatically download model files (~100-200MB). This is a one-time download.

---

## Project Structure

```
face_detecion/
├── README.md                    # This file
├── requirements.txt             # Python dependencies
├── .gitignore                   # Git ignore rules
│
├── detection/                   # Face Detection Module
│   ├── __init__.py
│   ├── detector.py             # RetinaFace face detector
│   └── detected_faces/         # Saved detected face images (auto-created)
│
├── embedding/                   # Embedding Extraction Module
│   ├── __init__.py
│   ├── extractor.py            # ArcFace embedding extractor
│   └── extracted_faces/        # Saved extracted face images (optional)
│
├── storage/                     # Database and Storage Module
│   ├── __init__.py
│   ├── init_db.py              # Database initialization script
│   ├── insert.py               # Database insertion class
│   ├── insert_face.py          # Simple face insertion utility
│   ├── search.py               # Face search functionality
│   ├── faces.db                # SQLite database (auto-created)
│   └── images/                 # Saved person images (auto-created)
│
├── testing/                     # Test Scripts
│   ├── test_detecion.py        # Test face detection
│   ├── test_embedding.py       # Test embedding extraction
│   ├── test_isertion.py        # Simple face insertion test
│   └── test_searching.py       # Test face search
│
└── sample_data/                 # Sample Data and Scripts
    ├── faces/                  # Sample face images
    ├── insert_single.py        # Insert single face example
    └── insert_multiple.py      # Insert multiple faces example
```

---

## Database Initialization

### Automatic Initialization

The database is automatically created when you first insert a face. No manual initialization needed!

### Manual Initialization

If you want to initialize the database manually:

**Option 1: Using Python**

```python
from storage.init_db import init_database

# Initialize with default path (storage/faces.db)
init_database()

# Or specify custom path
init_database("path/to/faces.db")
```

**Option 2: Using Command Line**

```bash
python storage/init_db.py
```

### Database Schema

The SQLite database uses the following schema:

```sql
CREATE TABLE persons (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL,
    embedding BLOB NOT NULL,        -- 512-dim float32 array (2048 bytes)
    metadata TEXT,                  -- JSON string
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

**Fields:**
- `id`: Auto-incrementing unique identifier
- `name`: Person's name (required)
- `embedding`: Face embedding as binary data (2048 bytes)
- `metadata`: JSON string with additional information
- `created_at`: Timestamp of when record was created

**Metadata Structure** (stored as JSON):
```json
{
    "age": 30,
    "department": "Engineering",
    "image_path": "storage/images/John_1.jpg",
    "original_path": "detection/detected_faces/face_001.jpg"
}
```

---

## Inserting Faces

### Method 1: Simple Insert (Recommended)

Use `storage/insert_face.py` - handles everything automatically:

**Edit `testing/test_isertion.py`:**

```python
IMAGE = 'detection/detected_faces/face_001.jpg'
NAME = 'John Doe'
METADATA = {"age": 30, "department": "Engineering"}
```

**Run:**
```bash
python testing/test_isertion.py
```

**What it does:**
- Resolves image path (supports pattern matching)
- Extracts face embedding
- Saves person image to `storage/images/{name}_{id}.jpg`
- Inserts into database
- Updates metadata with image paths

### Method 2: Using insert_face Function

```python
from storage.insert_face import insert_face

person_id = insert_face(
    image_path='detection/detected_faces/face_001.jpg',
    name='John Doe',
    metadata={'age': 30, 'department': 'Engineering'}
)

print(f"Inserted person with ID: {person_id}")
```

### Method 3: Using Individual Modules

```python
from detection import FaceDetector
from embedding import EmbeddingExtractor
from storage import FaceInsert

# Step 1: Extract embedding
extractor = EmbeddingExtractor()
embedding = extractor.extract("face.jpg", is_cropped_face=True)

# Step 2: Insert into database
inserter = FaceInsert(db_path="storage/faces.db")
person_id = inserter.add(
    name="John Doe",
    embedding=embedding,
    metadata={"age": 30, "department": "Engineering"}
)
inserter.close()

print(f"Person ID: {person_id}")
```

### Method 4: Insert Multiple Faces

Use `sample_data/insert_multiple.py` to insert multiple people at once:

**Edit the file:**
```python
# Person 1
IMAGE_1 = 'sample_data/faces/face_001.jpg'
NAME_1 = 'Ali'
METADATA_1 = {"age": 30, "department": "Engineering"}

# Person 2
IMAGE_2 = 'sample_data/faces/face_002.jpg'
NAME_2 = 'Sarah'
METADATA_2 = {"age": 28, "department": "Marketing"}

# ... add more people
```

**Run:**
```bash
python sample_data/insert_multiple.py
```

---

## Searching for Faces

### Method 1: Using Test Script (Recommended)

**Edit `testing/test_searching.py`:**

```python
QUERY_IMAGE_PATH = 'query_face.jpg'
THRESHOLD = 0.6
```

**Run:**
```bash
python testing/test_searching.py
```

**Output:**
- Prints all match information in JSON format
- Displays query image and matched image side-by-side
- Shows similarity scores and metadata

### Method 2: Using Search Function

```python
from storage import FaceSearch
from embedding import EmbeddingExtractor

# Step 1: Extract embedding from query image
extractor = EmbeddingExtractor()
query_embedding = extractor.extract("query_face.jpg", is_cropped_face=True)

# Step 2: Search in database
searcher = FaceSearch(db_path="storage/faces.db")
matches = searcher.search(query_embedding, threshold=0.6)

# Step 3: Process results
for person_id, name, similarity, metadata in matches:
    print(f"Found: {name}")
    print(f"  Similarity: {similarity:.2%}")
    print(f"  Person ID: {person_id}")
    if metadata:
        print(f"  Metadata: {metadata}")

searcher.close()
```

### Similarity Threshold

The threshold determines how strict the matching is:

- **0.5-0.6**: More lenient, may match different people (lower accuracy)
- **0.6-0.7**: Balanced (recommended default)
- **0.7-0.8**: Strict, high accuracy
- **0.8+**: Very strict, may miss valid matches

**Recommendation**: Start with 0.6 and adjust based on your use case.

---

## File Descriptions

### Detection Module

#### `detection/detector.py`

**Class: `FaceDetector`**

Face detection using RetinaFace model.

**Initialization:**
```python
detector = FaceDetector(
    model_name='buffalo_l',           # Model variant
    save_detected_faces=True          # Save detected faces to folder
)
```

**Methods:**
- `detect(image_path)`: Detect all faces in an image
  - Returns: List of detection dictionaries
  - Each dict contains: `bbox`, `landmark`, `det_score`, `saved_path`
- `detect_and_save_faces(image_path)`: Detect and save all faces

**Example:**
```python
from detection import FaceDetector

detector = FaceDetector()
detections = detector.detect("image.jpg")

for det in detections:
    print(f"Face at {det['bbox']} with confidence {det['det_score']}")
```

### Embedding Module

#### `embedding/extractor.py`

**Class: `EmbeddingExtractor`**

Face embedding extraction using ArcFace model.

**Initialization:**
```python
extractor = EmbeddingExtractor(
    model_name='buffalo_l',              # Model variant
    save_extracted_faces=False           # Save extracted faces (default: False)
)
```

**Methods:**
- `extract(face_image, is_cropped_face=False)`: Extract embedding
  - `face_image`: Image path or numpy array
  - `is_cropped_face`: If True, assumes image contains only one face
  - Returns: 512-dimensional numpy array (float32)

**Example:**
```python
from embedding import EmbeddingExtractor

extractor = EmbeddingExtractor()
embedding = extractor.extract("face.jpg", is_cropped_face=True)
print(f"Embedding shape: {embedding.shape}")  # (512,)
```

### Storage Module

#### `storage/init_db.py`

**Function: `init_database(db_path=None)`**

Initializes the database with the correct schema.

**Usage:**
```python
from storage.init_db import init_database

init_database()  # Uses default: storage/faces.db
```

#### `storage/insert.py`

**Class: `FaceInsert`**

Handles database insertion operations.

**Methods:**
- `add(name, embedding, metadata=None)`: Insert a new person
  - Returns: Person ID (int)
- `update_metadata(person_id, metadata)`: Update person's metadata
  - Returns: True if updated, False if not found

**Example:**
```python
from storage import FaceInsert

inserter = FaceInsert(db_path="storage/faces.db")
person_id = inserter.add("John Doe", embedding, {"age": 30})
inserter.close()
```

#### `storage/insert_face.py`

**Function: `insert_face(image_path, name, metadata=None, db_path=None)`**

Simple utility function that handles the entire insertion process.

**What it does:**
1. Resolves image path (supports pattern matching)
2. Extracts face embedding
3. Saves person image to `storage/images/`
4. Inserts into database
5. Updates metadata with image paths

**Returns:** Person ID (int)

**Example:**
```python
from storage.insert_face import insert_face

person_id = insert_face(
    image_path='detection/detected_faces/face_001.jpg',
    name='John Doe',
    metadata={'age': 30, 'department': 'Engineering'}
)
```

#### `storage/search.py`

**Class: `FaceSearch`**

Handles face search operations.

**Methods:**
- `search(embedding, threshold=0.6)`: Search for matching faces
  - Returns: List of tuples `(person_id, name, similarity, metadata)`
  - Results sorted by similarity (highest first)
- `get_by_id(person_id)`: Get person by ID
  - Returns: Tuple `(id, name, metadata, created_at)` or None
- `get_all()`: Get all persons from database
  - Returns: List of tuples `(person_id, name, metadata, created_at)`

**Example:**
```python
from storage import FaceSearch

searcher = FaceSearch(db_path="storage/faces.db")
matches = searcher.search(query_embedding, threshold=0.6)

for person_id, name, similarity, metadata in matches:
    print(f"{name}: {similarity:.2%}")
```

### Test Scripts

#### `testing/test_detecion.py`

Tests face detection functionality.

**Configuration:**
```python
IMAGE_PATH = 'group.jpg'  # Image with faces
```

**Run:**
```bash
python testing/test_detecion.py
```

**Output:**
- Detects all faces in the image
- Saves detected faces to `detection/detected_faces/`
- Shows bounding boxes and confidence scores

#### `testing/test_embedding.py`

Tests embedding extraction.

**Configuration:**
```python
IMAGE_PATH = 'detection/detected_faces/face_001.jpg'  # Single face image
```

**Run:**
```bash
python testing/test_embedding.py
```

**Output:**
- Extracts embedding from face image
- Prints embedding information
- Saves tested face to `testing/tested_images_embedding/`

#### `testing/test_isertion.py`

Simple face insertion test.

**Configuration:**
```python
IMAGE = 'detection/detected_faces/face_001.jpg'
NAME = 'John Doe'
METADATA = {"age": 30, "department": "Engineering"}
```

**Run:**
```bash
python testing/test_isertion.py
```

**Output:**
- Inserts face into database
- Saves person image to `storage/images/`
- Prints person ID

#### `testing/test_searching.py`

Tests face search functionality.

**Configuration:**
```python
QUERY_IMAGE_PATH = 'query_face.jpg'
THRESHOLD = 0.6
```

**Run:**
```bash
python testing/test_searching.py
```

**Output:**
- Searches for matching faces
- Prints all match information in JSON format
- Displays query image and matched image side-by-side

---

## Quick Start Guide

### Step 1: Detect Faces

```bash
python testing/test_detecion.py
```

This will:
- Detect all faces in the image
- Save detected faces to `detection/detected_faces/`
- Files named: `face_001_timestamp.jpg`, `face_002_timestamp.jpg`, etc.

### Step 2: Insert Faces into Database

**Edit `testing/test_isertion.py`:**
```python
IMAGE = 'detection/detected_faces/face_001_20251123_114613.jpg'
NAME = 'John Doe'
METADATA = {"age": 30, "department": "Engineering"}
```

**Run:**
```bash
python testing/test_isertion.py
```

This will:
- Extract face embedding
- Save person image to `storage/images/John_1.jpg`
- Insert into database at `storage/faces.db`
- Return person ID

### Step 3: Search for a Face

**Edit `testing/test_searching.py`:**
```python
QUERY_IMAGE_PATH = 'detection/detected_faces/face_001_20251123_114613.jpg'
THRESHOLD = 0.6
```

**Run:**
```bash
python testing/test_searching.py
```

This will:
- Extract embedding from query image
- Search database for matches
- Display results with similarity scores
- Show images side-by-side

---

## API Reference

### Detection API

```python
from detection import FaceDetector

detector = FaceDetector(
    model_name='buffalo_l',        # 'buffalo_l', 'buffalo_m', 'buffalo_s'
    save_detected_faces=True       # Save detected faces to folder
)

# Detect faces
detections = detector.detect("image.jpg")
# Returns: List[Dict] with keys: 'bbox', 'landmark', 'det_score', 'saved_path'
```

### Embedding API

```python
from embedding import EmbeddingExtractor

extractor = EmbeddingExtractor(
    model_name='buffalo_l',
    save_extracted_faces=False
)

# Extract embedding
embedding = extractor.extract("face.jpg", is_cropped_face=True)
# Returns: numpy.ndarray, shape (512,), dtype float32
```

### Storage API

```python
from storage import FaceInsert, FaceSearch
from storage.insert_face import insert_face

# Simple insertion
person_id = insert_face(
    image_path='face.jpg',
    name='John Doe',
    metadata={'age': 30}
)

# Advanced insertion
inserter = FaceInsert(db_path="storage/faces.db")
person_id = inserter.add("John Doe", embedding, {"age": 30})
inserter.close()

# Search
searcher = FaceSearch(db_path="storage/faces.db")
matches = searcher.search(embedding, threshold=0.6)
# Returns: List[Tuple[int, str, float, Optional[Dict]]]
searcher.close()
```

---

## Troubleshooting

### Model Download Issues

**Problem**: Models fail to download automatically.

**Solution**:
1. Models are stored in `~/.insightface/models/`
2. Check internet connection
3. Manually download from: https://github.com/deepinsight/insightface
4. Place models in `.insightface/models/buffalo_l/` directory

### No Face Detected

**Problem**: No faces detected in image.

**Solutions**:
- Ensure image contains a clear, frontal face
- Check image format (JPG, PNG supported)
- Verify image is not corrupted
- Try different image sizes/resolutions
- Ensure good lighting in the image

### Low Similarity Scores

**Problem**: Similarity scores are too low.

**Solutions**:
- Use higher quality images
- Ensure faces are well-lit and clear
- Adjust threshold (try 0.5-0.6 for more lenient matching)
- Check that the same person's images are consistent
- Ensure faces are frontal (not heavily tilted)

### Performance Issues

**Problem**: Processing is too slow.

**Solutions**:
- Use `buffalo_s` model for faster processing (modify in `detector.py` and `extractor.py`)
- Process images in batches
- Consider GPU acceleration (modify `providers` in model initialization):
  ```python
  # Change from:
  providers=['CPUExecutionProvider']
  # To:
  providers=['CUDAExecutionProvider', 'CPUExecutionProvider']
  ```

### Database Location

**Default Locations**:
- Database: `storage/faces.db` (auto-created)
- Person images: `storage/images/` (auto-created)
- Detected faces: `detection/detected_faces/` (auto-created)
- Extracted faces: `embedding/extracted_faces/` (optional, disabled by default)

### Path Resolution Issues

**Problem**: Image file not found.

**Solution**: The system supports:
- Relative paths: `'detection/detected_faces/face_001.jpg'`
- Pattern matching: If file is `face_001_20251123_114613.jpg`, you can use `face_001*.jpg`
- Absolute paths: Full file paths are also supported

---

## Example Workflows

### Workflow 1: Single Face Insertion

```bash
# 1. Detect faces
python testing/test_detecion.py

# 2. Edit test_isertion.py with image path and name
# 3. Insert face
python testing/test_isertion.py

# 4. Search for face
python testing/test_searching.py
```

### Workflow 2: Batch Insertion

```bash
# 1. Edit sample_data/insert_multiple.py with multiple people
# 2. Run batch insertion
python sample_data/insert_multiple.py
```

### Workflow 3: Programmatic Usage

```python
from storage.insert_face import insert_face
from storage import FaceSearch
from embedding import EmbeddingExtractor

# Insert multiple people
people = [
    ('face1.jpg', 'Alice', {'age': 25}),
    ('face2.jpg', 'Bob', {'age': 30}),
    ('face3.jpg', 'Charlie', {'age': 28}),
]

for image, name, metadata in people:
    person_id = insert_face(image, name, metadata)
    print(f"Inserted {name} with ID: {person_id}")

# Search
extractor = EmbeddingExtractor()
query_embedding = extractor.extract("query.jpg", is_cropped_face=True)

searcher = FaceSearch()
matches = searcher.search(query_embedding, threshold=0.6)

for person_id, name, similarity, metadata in matches:
    print(f"Match: {name} ({similarity:.2%})")
```

---

## Technical Details

### Face Embeddings

- **Dimensions**: 512 (float32)
- **Size**: 2048 bytes per embedding
- **Normalization**: L2-normalized vectors
- **Comparison**: Cosine similarity (dot product of normalized vectors)
- **Range**: Similarity scores range from 0.0 to 1.0

### Face Detection

- **Model**: RetinaFace (from InsightFace)
- **Input Size**: 640x640 pixels
- **Output**: Bounding boxes, landmarks (106 points), confidence scores
- **Performance**: ~100-200ms per image on CPU

### Face Recognition

- **Model**: ArcFace (from InsightFace)
- **Input Size**: 112x112 pixels (cropped and aligned face)
- **Output**: 512-dimensional embedding vector
- **Performance**: ~50-100ms per face on CPU

---

## License

This project is open source and available for use.

## Acknowledgments

- [InsightFace](https://github.com/deepinsight/insightface) - Face analysis toolkit
- **RetinaFace** - Face detection model
- **ArcFace** - Face recognition model

---

## Support

For issues or questions:
1. Check the [Troubleshooting](#troubleshooting) section
2. Review the test scripts in `testing/` folder
3. Check the code comments in each module
