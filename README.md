# Face Recognition System

A complete face recognition system using RetinaFace for face detection, ArcFace for embedding extraction, and SQLite for storage. The system can detect faces in images, extract 512-dimensional embeddings, store them in a database, and search for persons based on face similarity.

## Features

- **Face Detection**: Uses RetinaFace to detect single or multiple faces in images
- **Embedding Extraction**: Uses ArcFace to generate 512-dimensional face embeddings
- **Database Storage**: SQLite database for storing person information and face embeddings
- **Face Search**: Cosine similarity-based search to find matching faces in the database
- **Multiple Faces**: Handles multiple faces in a single image
- **Metadata Support**: Store additional information with each person

## Installation

### Prerequisites

- Python 3.7 or higher
- pip package manager

### Install Dependencies

```bash
pip install -r requirements.txt
```

This will install:
- `insightface` - Face detection and recognition models (RetinaFace + ArcFace)
- `opencv-python` - Image processing
- `numpy` - Numerical operations
- `Pillow` - Image handling
- `onnxruntime` - Model inference

**Note**: On first run, InsightFace will automatically download model files (~100-200MB). This is a one-time download.

## Project Structure

The project is organized into modular components:

- **`detection.py`** - Face detection using RetinaFace
- **`face_extraction.py`** - Extract/crop face regions from images
- **`embedding.py`** - Extract face embeddings using ArcFace
- **`storage.py`** - Database operations for storing and retrieving face data
- **`pipeline.py`** - Main orchestrator that combines all components

See [MODULES.md](MODULES.md) for detailed module documentation.

## Quick Start

### Basic Usage (Using Pipeline)

```python
from pipeline import FaceRecognitionPipeline

# Initialize the pipeline
pipeline = FaceRecognitionPipeline(db_path="faces.db")

# Add a person to the database (image must contain exactly one face)
person_id = pipeline.add_person(
    image_path="person.jpg",
    name="John Doe",
    metadata={"age": 30, "department": "Engineering"}
)

# Search for a person (can handle multiple faces)
results = pipeline.search(
    image_path="query.jpg",
    threshold=0.6
)

# Process results
for result in results:
    if result['matches']:
        person_id, name, similarity, metadata = result['matches'][0]
        print(f"Found: {name} (Similarity: {similarity:.2%})")
    else:
        print("No match found")

# Close the connection
pipeline.close()
```

### Run Example Script

```bash
python example.py
```

### Using Individual Modules

You can also use modules independently for more control:

```python
from detection import FaceDetector
from embedding import EmbeddingExtractor
from storage import FaceStorage

# Use only detection
detector = FaceDetector()
detections = detector.detect("image.jpg")

# Use only embedding extraction
extractor = EmbeddingExtractor()
embedding = extractor.extract("face.jpg")

# Use only storage
storage = FaceStorage("faces.db")
storage.add("Alice", embedding)
```

See [MODULES.md](MODULES.md) for detailed module usage.

## API Reference

### FaceRecognitionPipeline

Main class that orchestrates the entire face recognition pipeline.

#### Methods

##### `__init__(db_path="faces.db", model_name='buffalo_l')`

Initialize the face recognition pipeline.

**Parameters:**
- `db_path` (str): Path to SQLite database file (default: "faces.db")
- `model_name` (str): InsightFace model name - 'buffalo_l' (large, most accurate), 'buffalo_m' (medium), or 'buffalo_s' (small, faster)

##### `add_person(image_path, name, metadata=None)`

Add a new person to the database from an image.

**Parameters:**
- `image_path` (str): Path to image file containing exactly one face
- `name` (str): Name of the person
- `metadata` (dict, optional): Additional metadata (e.g., {"age": 30, "role": "admin"})

**Returns:**
- `int`: ID of the newly added person

**Raises:**
- `ValueError`: If no face or multiple faces are detected

##### `search(image_path, threshold=0.6)`

Search for a person in the database based on their face.

**Parameters:**
- `image_path` (str): Path to image file (can contain multiple faces)
- `threshold` (float): Minimum similarity threshold (0.0 to 1.0, default: 0.6)

**Returns:**
- `List[Dict]`: List of results, one per detected face. Each result contains:
  - `face_index`: Index of the face
  - `bbox`: Bounding box coordinates [x1, y1, x2, y2]
  - `det_score`: Detection confidence score
  - `matches`: List of matching persons (person_id, name, similarity, metadata)

##### `process_image(image_path, threshold=0.6)`

Full pipeline: detect faces → extract embeddings → search in database.

**Parameters:**
- `image_path` (str): Path to image file
- `threshold` (float): Similarity threshold for matching

**Returns:**
- `List[Dict]`: Same format as `search_person_in_db()`

##### `get_all_persons()`

Retrieve all persons from the database.

**Returns:**
- `List[Tuple]`: List of (person_id, name, metadata, created_at) tuples

##### `delete_person(person_id)`

Delete a person from the database.

**Parameters:**
- `person_id` (int): ID of the person to delete

**Returns:**
- `bool`: True if deleted, False if not found

## Usage Examples

### Example 1: Adding Multiple Persons

```python
from pipeline import FaceRecognitionPipeline

pipeline = FaceRecognitionPipeline()

# Add multiple persons
persons = [
    ("alice.jpg", "Alice", {"role": "admin"}),
    ("bob.jpg", "Bob", {"role": "user"}),
    ("charlie.jpg", "Charlie", {"role": "guest"})
]

for image_path, name, metadata in persons:
    try:
        person_id = pipeline.add_person(image_path, name, metadata)
        print(f"Added {name} with ID {person_id}")
    except ValueError as e:
        print(f"Error adding {name}: {e}")

pipeline.close()
```

### Example 2: Using Context Manager

```python
from pipeline import FaceRecognitionPipeline

# Automatically closes database connection
with FaceRecognitionPipeline() as pipeline:
    # Add person
    person_id = pipeline.add_person("person.jpg", "Jane Smith")
    
    # Search
    results = pipeline.search("query.jpg", threshold=0.7)
    
    # List all
    all_persons = pipeline.get_all_persons()
    for person_id, name, metadata, created_at in all_persons:
        print(f"{name} - {created_at}")
```

### Example 3: Handling Multiple Faces

```python
from pipeline import FaceRecognitionPipeline

pipeline = FaceRecognitionPipeline()

# Process image with multiple faces
results = pipeline.process_image("group_photo.jpg", threshold=0.6)

for result in results:
    print(f"\nFace {result['face_index']} at {result['bbox']}")
    
    if result['matches']:
        # Get best match
        person_id, name, similarity, metadata = result['matches'][0]
        print(f"  Identified as: {name} (confidence: {similarity:.2%})")
    else:
        print("  Unknown person")

pipeline.close()
```

### Example 4: Using Individual Modules

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

# Step 1: Detect
detections = detector.detect(image)

# Step 2: Extract face regions (optional)
face_images = FaceExtractor.extract(image, detections)

# Step 3: Extract embeddings
embeddings = extractor.extract_from_image(image)

# Step 4: Store
for i, embedding in enumerate(embeddings):
    storage.add(f"Person_{i}", embedding)
```

## Similarity Threshold

The similarity threshold determines how strict the matching is:

- **0.5-0.6**: More lenient, may match different people (lower accuracy)
- **0.6-0.7**: Balanced (recommended default)
- **0.7-0.8**: Strict, high accuracy
- **0.8+**: Very strict, may miss valid matches

**Recommendation**: Start with 0.6 and adjust based on your use case.

## Database Schema

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

## Integration with Other SQL Databases

The current implementation uses SQLite, but you can easily adapt it to work with other SQL databases like PostgreSQL, MySQL, or SQL Server. Here are the key considerations:

### Option 1: Modify the Database Class

Create a new database adapter class that uses your preferred database:

#### PostgreSQL Example

```python
import psycopg2
import psycopg2.extras
import numpy as np
import json
from typing import List, Tuple, Optional, Dict

class PostgreSQLFaceDatabase:
    """PostgreSQL adapter for face database."""
    
    def __init__(self, connection_string: str):
        """
        Initialize PostgreSQL connection.
        
        Args:
            connection_string: PostgreSQL connection string
                Example: "host=localhost dbname=faces user=postgres password=pass"
        """
        self.conn = psycopg2.connect(connection_string)
        self._create_table()
    
    def _create_table(self):
        """Create the persons table if it doesn't exist."""
        cursor = self.conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS persons (
                id SERIAL PRIMARY KEY,
                name VARCHAR(255) NOT NULL,
                embedding BYTEA NOT NULL,
                metadata JSONB,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        self.conn.commit()
    
    def add_person(self, name: str, embedding: np.ndarray, metadata: Optional[Dict] = None) -> int:
        """Add a new person with their face embedding."""
        cursor = self.conn.cursor()
        
        # Serialize embedding to bytes
        embedding_bytes = embedding.tobytes()
        
        # PostgreSQL supports JSON natively
        cursor.execute("""
            INSERT INTO persons (name, embedding, metadata)
            VALUES (%s, %s, %s)
            RETURNING id
        """, (name, embedding_bytes, json.dumps(metadata) if metadata else None))
        
        person_id = cursor.fetchone()[0]
        self.conn.commit()
        return person_id
    
    def search_person(self, embedding: np.ndarray, threshold: float = 0.6) -> List[Tuple]:
        """Search for a person using cosine similarity."""
        cursor = self.conn.cursor()
        cursor.execute("SELECT id, name, embedding, metadata FROM persons")
        rows = cursor.fetchall()
        
        results = []
        for person_id, name, embedding_bytes, metadata in rows:
            stored_embedding = np.frombuffer(embedding_bytes, dtype=np.float32)
            similarity = self._cosine_similarity(embedding, stored_embedding)
            
            if similarity >= threshold:
                metadata_dict = json.loads(metadata) if metadata else None
                results.append((person_id, name, similarity, metadata_dict))
        
        results.sort(key=lambda x: x[2], reverse=True)
        return results
    
    def _cosine_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """Calculate cosine similarity between two vectors."""
        vec1_norm = vec1 / (np.linalg.norm(vec1) + 1e-8)
        vec2_norm = vec2 / (np.linalg.norm(vec2) + 1e-8)
        similarity = np.dot(vec1_norm, vec2_norm)
        return float(np.clip(similarity, 0.0, 1.0))
    
    # ... implement other methods similarly
```

#### MySQL Example

```python
import mysql.connector
import numpy as np
import json
from typing import List, Tuple, Optional, Dict

class MySQLFaceDatabase:
    """MySQL adapter for face database."""
    
    def __init__(self, host: str, database: str, user: str, password: str):
        """Initialize MySQL connection."""
        self.conn = mysql.connector.connect(
            host=host,
            database=database,
            user=user,
            password=password
        )
        self._create_table()
    
    def _create_table(self):
        """Create the persons table."""
        cursor = self.conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS persons (
                id INT AUTO_INCREMENT PRIMARY KEY,
                name VARCHAR(255) NOT NULL,
                embedding BLOB NOT NULL,
                metadata JSON,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        self.conn.commit()
    
    def add_person(self, name: str, embedding: np.ndarray, metadata: Optional[Dict] = None) -> int:
        """Add a new person."""
        cursor = self.conn.cursor()
        embedding_bytes = embedding.tobytes()
        
        cursor.execute("""
            INSERT INTO persons (name, embedding, metadata)
            VALUES (%s, %s, %s)
        """, (name, embedding_bytes, json.dumps(metadata) if metadata else None))
        
        self.conn.commit()
        return cursor.lastrowid
    
    # ... implement other methods
```

### Option 2: Use SQLAlchemy ORM

For a more database-agnostic approach, use SQLAlchemy:

```python
from sqlalchemy import create_engine, Column, Integer, String, LargeBinary, JSON, DateTime
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
import numpy as np
from datetime import datetime

Base = declarative_base()

class Person(Base):
    __tablename__ = 'persons'
    
    id = Column(Integer, primary_key=True)
    name = Column(String(255), nullable=False)
    embedding = Column(LargeBinary, nullable=False)
    metadata = Column(JSON)
    created_at = Column(DateTime, default=datetime.utcnow)

class SQLAlchemyFaceDatabase:
    """SQLAlchemy-based face database (works with any SQL database)."""
    
    def __init__(self, database_url: str):
        """
        Initialize SQLAlchemy connection.
        
        Args:
            database_url: Database URL
                Examples:
                - SQLite: "sqlite:///faces.db"
                - PostgreSQL: "postgresql://user:pass@localhost/faces"
                - MySQL: "mysql://user:pass@localhost/faces"
        """
        self.engine = create_engine(database_url)
        Base.metadata.create_all(self.engine)
        Session = sessionmaker(bind=self.engine)
        self.session = Session()
    
    def add_person(self, name: str, embedding: np.ndarray, metadata: dict = None):
        """Add a new person."""
        person = Person(
            name=name,
            embedding=embedding.tobytes(),
            metadata=metadata
        )
        self.session.add(person)
        self.session.commit()
        return person.id
    
    # ... implement other methods
```

### Option 3: Vector Database Integration

For large-scale applications, consider using specialized vector databases:

- **PostgreSQL with pgvector**: Add vector similarity search extension
- **Milvus**: Open-source vector database
- **Pinecone**: Managed vector database service
- **Weaviate**: Vector search engine

#### PostgreSQL with pgvector Example

```sql
-- Install pgvector extension
CREATE EXTENSION IF NOT EXISTS vector;

-- Create table with vector column
CREATE TABLE persons (
    id SERIAL PRIMARY KEY,
    name VARCHAR(255) NOT NULL,
    embedding vector(512),  -- 512-dimensional vector
    metadata JSONB,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Create index for fast similarity search
CREATE INDEX ON persons USING ivfflat (embedding vector_cosine_ops);
```

Then use vector similarity operators:

```python
# Search using cosine similarity
cursor.execute("""
    SELECT id, name, metadata, 
           1 - (embedding <=> %s) as similarity
    FROM persons
    WHERE 1 - (embedding <=> %s) >= %s
    ORDER BY embedding <=> %s
    LIMIT 10
""", (embedding.tobytes(), embedding.tobytes(), threshold, embedding.tobytes()))
```

### Key Considerations for Database Migration

1. **Data Type Mapping**:
   - SQLite: `BLOB` for embeddings
   - PostgreSQL: `BYTEA` or `vector(512)` (with pgvector)
   - MySQL: `BLOB` or `VARBINARY`
   - SQL Server: `VARBINARY(MAX)`

2. **JSON Support**:
   - SQLite: Store as `TEXT` (JSON string)
   - PostgreSQL: Use `JSONB` (native JSON with indexing)
   - MySQL 5.7+: Use `JSON` type
   - SQL Server: Use `NVARCHAR(MAX)` with JSON functions

3. **Performance**:
   - For large datasets (>10K faces), consider vector databases
   - Add indexes on frequently queried columns
   - Use connection pooling for production

4. **Modify FaceRecognitionSystem**:
   ```python
   # In face_recognition.py, change:
   self.database = FaceDatabase(db_path=db_path)
   
   # To:
   self.database = PostgreSQLFaceDatabase(connection_string)
   # or
   self.database = SQLAlchemyFaceDatabase(database_url)
   ```

## Project Structure

```
face_detecion/
├── README.md              # This file
├── MODULES.md             # Detailed module documentation
├── requirements.txt       # Python dependencies
├── detection.py           # Face detection module
├── face_extraction.py    # Face region extraction module
├── embedding.py           # Embedding extraction module
├── storage.py             # Database storage module
├── pipeline.py            # Main pipeline orchestrator
├── example.py             # Usage examples
├── faces.db               # SQLite database (created at runtime)
└── test.jpg               # Test image
```

### Module Responsibilities

- **`detection.py`**: Detects faces in images (RetinaFace)
- **`face_extraction.py`**: Extracts/crops face regions from images
- **`embedding.py`**: Extracts 512-dim embeddings from faces (ArcFace)
- **`storage.py`**: Database operations (store, search, retrieve)
- **`pipeline.py`**: High-level API that orchestrates all modules

## Troubleshooting

### Model Download Issues

If models fail to download automatically:
1. Models are stored in `~/.insightface/models/`
2. You can manually download from: https://github.com/deepinsight/insightface
3. Place models in the `.insightface/models/` directory

### No Face Detected

- Ensure the image contains a clear, frontal face
- Check image format (JPG, PNG supported)
- Verify image is not corrupted
- Try different image sizes/resolutions

### Low Similarity Scores

- Use higher quality images
- Ensure faces are well-lit and clear
- Adjust threshold (try 0.5-0.6 for more lenient matching)
- Check that the same person's images are consistent

### Performance Issues

- Use `buffalo_s` model for faster processing
- Process images in batches
- Consider GPU acceleration (modify `providers` in model initialization)

## License

This project is open source and available for use.

## Contributing

Contributions are welcome! Please feel free to submit issues or pull requests.

## Acknowledgments

- [InsightFace](https://github.com/deepinsight/insightface) - Face analysis toolkit
- RetinaFace - Face detection model
- ArcFace - Face recognition model

