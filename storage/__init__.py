"""
Storage module for database operations.

This package contains:
- insert.py: Functions for inserting data into the database
- search.py: Functions for searching in the database
"""

from .insert import FaceInsert
from .search import FaceSearch

__all__ = ['FaceInsert', 'FaceSearch']

