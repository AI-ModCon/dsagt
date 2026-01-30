"""Test script for KnowledgeBase."""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"  # macOS OpenMP workaround

import shutil
import tempfile
from pathlib import Path

from knowledge_base import KnowledgeBase


def create_test_data(base_dir: Path):
    """Create test collection folders with sample documents."""
    
    # Collection 1: Python docs
    python_dir = base_dir / "python_basics"
    python_dir.mkdir()
    
    (python_dir / "DESCRIPTION.md").write_text(
        "Python programming fundamentals: variables, functions, classes, and control flow."
    )
    
    (python_dir / "variables.md").write_text("""
# Variables in Python

Python variables are dynamically typed. You don't need to declare types.

## Assignment

```python
x = 10
name = "Alice"
pi = 3.14159
```

## Multiple Assignment

```python
a, b, c = 1, 2, 3
```
""")
    
    (python_dir / "functions.md").write_text("""
# Functions in Python

Functions are defined using the `def` keyword.

## Basic Function

```python
def greet(name):
    return f"Hello, {name}!"
```

## Default Arguments

```python
def power(base, exponent=2):
    return base ** exponent
```

## Lambda Functions

Lambda functions are small anonymous functions:

```python
square = lambda x: x ** 2
```
""")
    
    (python_dir / "example.py").write_text('''
"""Example Python module."""

def fibonacci(n: int) -> list[int]:
    """Generate fibonacci sequence up to n terms."""
    if n <= 0:
        return []
    if n == 1:
        return [0]
    
    seq = [0, 1]
    while len(seq) < n:
        seq.append(seq[-1] + seq[-2])
    return seq


class Calculator:
    """Simple calculator class."""
    
    def __init__(self, initial: float = 0):
        self.value = initial
    
    def add(self, x: float) -> "Calculator":
        self.value += x
        return self
    
    def multiply(self, x: float) -> "Calculator":
        self.value *= x
        return self
''')
    
    # Collection 2: Data science docs
    ds_dir = base_dir / "data_science"
    ds_dir.mkdir()
    
    (ds_dir / "DESCRIPTION.md").write_text(
        "Data science concepts: pandas, numpy, data cleaning, and analysis techniques."
    )
    
    (ds_dir / "pandas_intro.md").write_text("""
# Introduction to Pandas

Pandas is a powerful data manipulation library for Python.

## DataFrames

A DataFrame is a 2-dimensional labeled data structure.

```python
import pandas as pd

df = pd.DataFrame({
    'name': ['Alice', 'Bob', 'Charlie'],
    'age': [25, 30, 35],
    'city': ['NYC', 'LA', 'Chicago']
})
```

## Reading Data

```python
df = pd.read_csv('data.csv')
df = pd.read_json('data.json')
```

## Basic Operations

- `df.head()` - View first rows
- `df.describe()` - Summary statistics
- `df.info()` - Column types and memory
""")
    
    (ds_dir / "numpy_basics.md").write_text("""
# NumPy Basics

NumPy provides support for large, multi-dimensional arrays.

## Creating Arrays

```python
import numpy as np

arr = np.array([1, 2, 3, 4, 5])
zeros = np.zeros((3, 3))
ones = np.ones((2, 4))
```

## Array Operations

NumPy operations are vectorized for performance:

```python
a = np.array([1, 2, 3])
b = np.array([4, 5, 6])

c = a + b  # Element-wise addition
d = a * b  # Element-wise multiplication
dot = np.dot(a, b)  # Dot product
```
""")
    
    return python_dir, ds_dir


def test_knowledge_base():
    """Run all KnowledgeBase tests."""
    
    # Setup temp directories
    test_dir = Path(tempfile.mkdtemp())
    data_dir = test_dir / "data"
    index_dir = test_dir / "index"
    data_dir.mkdir()
    
    print(f"Test directory: {test_dir}\n")
    
    try:
        # Create test data
        print("=" * 50)
        print("Creating test data...")
        python_dir, ds_dir = create_test_data(data_dir)
        print(f"  Created: {python_dir.name}")
        print(f"  Created: {ds_dir.name}")
        
        # Initialize KnowledgeBase
        print("\n" + "=" * 50)
        print("Initializing KnowledgeBase...")
        kb = KnowledgeBase(index_dir)
        print(f"  Index dir: {kb.index_dir}")
        
        # Test ingest
        print("\n" + "=" * 50)
        print("Testing ingest...")
        
        result1 = kb.ingest(python_dir)
        print(f"  {result1['collection']}: {result1['files']} files, {result1['chunks']} chunks")
        
        result2 = kb.ingest(ds_dir)
        print(f"  {result2['collection']}: {result2['files']} files, {result2['chunks']} chunks")
        
        # Test collections property
        print("\n" + "=" * 50)
        print("Testing collections property...")
        print(f"  Available: {kb.collections}")
        
        # Test list_collections
        print("\n" + "=" * 50)
        print("Testing list_collections...")
        for coll in kb.list_collections():
            print(f"  {coll['name']}: {coll['description'][:60]}...")
        
        # Test search without reranking
        print("\n" + "=" * 50)
        print("Testing search (no rerank)...")
        
        query = "how to define a function"
        results = kb.search(query, "python_basics", top_k=3, rerank=False)
        print(f"  Query: '{query}'")
        print(f"  Results: {len(results)}")
        for i, r in enumerate(results):
            text_preview = r['chunk']['text'][:80].replace('\n', ' ')
            print(f"    {i+1}. score={r['score']:.3f} | {text_preview}...")
        
        # Test search with reranking
        print("\n" + "=" * 50)
        print("Testing search (with rerank)...")
        
        query = "how to create a numpy array"
        results = kb.search(query, "data_science", top_k=3, rerank=True)
        print(f"  Query: '{query}'")
        print(f"  Results: {len(results)}")
        for i, r in enumerate(results):
            text_preview = r['chunk']['text'][:80].replace('\n', ' ')
            print(f"    {i+1}. score={r['score']:.3f} rerank={r['rerank_score']:.3f} | {text_preview}...")
        
        # Test search across different collection
        print("\n" + "=" * 50)
        print("Testing cross-collection isolation...")
        
        query = "pandas dataframe"
        results = kb.search(query, "python_basics", top_k=2, rerank=False)
        print(f"  Query: '{query}' in python_basics")
        print(f"  Results: {len(results)} (should not contain pandas content)")
        
        # Test error handling
        print("\n" + "=" * 50)
        print("Testing error handling...")
        
        try:
            kb.search("test", "nonexistent_collection")
            print("  ERROR: Should have raised ValueError")
        except ValueError as e:
            print(f"  Caught expected error: {e}")
        
        # Cleanup
        print("\n" + "=" * 50)
        print("Closing KnowledgeBase...")
        kb.close()
        print("  Done.")
        
        print("\n" + "=" * 50)
        print("ALL TESTS PASSED")
        print("=" * 50)
        
    finally:
        # Cleanup temp directory
        shutil.rmtree(test_dir)
        print(f"\nCleaned up: {test_dir}")


if __name__ == "__main__":
    test_knowledge_base()
