from sentence_transformers import SentenceTransformer
from vector_store import VectorStore
import pandas as pd


# Load the Embedding Model
model = SentenceTransformer('BAAI/bge-large-zh-v1.5')

# Get data from csv
data = pd.read_csv('data.csv', delimiter=';')

# Take 80% of the data to create the vector store, remaining 20% to update the vector store
data_1 = data.sample(frac=0.8, random_state=200)
data_2 = data.drop(data_1.index)

# Convert data to list
data_1 = data_1['text'].tolist()

# Encode the data, have data and vector in dictionary, data keys, vector values
vectors_1 = model.encode(data_1)

# Get the vector dimension
vector_dimension = len(vectors_1[0])

# Create a dictionary with id and vectors
new_sentence_vectors_1 = {data_1[i]: vectors_1[i] for i in range(len(data_1))}

# Create a vector store
vector_store = VectorStore(vector_dimension)

# Create the vector store, set persist to True to save the vector store on disk
vector_store.create_vector_store(new_sentence_vectors_1, persist=True)

# Query the vector store
query = "I want to buy a car"
query_vector = model.encode(query)
similar_vectors = vector_store.get_similar_vectors(query_vector, top_n=5)
print("Similarity Vectors:")
for sentence, similarity_score in similar_vectors:
    print(f"- Sentence: {sentence}")
    print(f"  Similarity Score: {similarity_score}")
    print()

# Update the vector store
data_2 = data_2['text'].tolist()
vectors_2 = model.encode(data_2)
new_sentence_vectors_2 = {data_2[i]: vectors_2[i] for i in range(len(data_2))}
vector_store.update_vector_store(new_sentence_vectors_2)

# Query the vector store
query = "I want to buy a cycle"
query_vector = model.encode(query)
similar_vectors = vector_store.get_similar_vectors(query_vector, top_n=5)
print("Similarity Vectors:")
for sentence, similarity_score in similar_vectors:
    print(f"- Sentence: {sentence}")
    print(f"  Similarity Score: {similarity_score}")
    print()

