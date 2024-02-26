# Building Persistent Vector Store from Scratch

## Description

This project implements a persistent vector store from scratch using the Hierarchical Navigable Small World (HNSW) algorithm. The vector store allows users to efficiently store, update, and query vectors representing sentences or text data.

## Features

- Implementation of a vector store using the HNSW algorithm.
- Automatic serialization and deserialization of the vector store during updates and similarity queries.
- Ability to add new data to the vector store without losing existing data.
- Querying capability to find similar sentences based on cosine similarity.
- Integration with sentence transformer for embedding text data.

## Files

- `vector_store.py`: Contains the class/package for vector store functionality, including adding vectors, updating the index, and querying similar vectors.
- `demo.py`: A demo script demonstrating how to use the vector store, including examples of adding vectors, querying similar sentences, and updating the vector store.

## Installation

To install the necessary dependencies, run:

```bash
pip install -r requirements.txt
```

## Usage

### Creating a Vector Store

Instantiate a `VectorStore` object and add vectors to it using the `add_vector` method.

```python
from vector_store import VectorStore

# Create VectorStore object
vector_store = VectorStore(vector_dim)

# Add vectors to the vector store
new_id_vectors = {'sentence1': [vector1], 'sentence2': [vector2]}
vector_store.add_vector(new_id_vectors)
```

### Querying Similar Sentences

Use the `get_similar_vectors` method to find similar sentences to a given query vector.

```python
query_vector = [query_vector]
similar_sentences = vector_store.get_similar_vectors(query_vector, top_n=5)
print(similar_sentences)
```

### Updating the Vector Store

To update the vector store with new data, simply use the `update_vector_store` method.

```python
new_id_vectors = {'sentence3': [vector3], 'sentence4': [vector4]}
vector_store.update_vector_store(new_id_vectors)
```
