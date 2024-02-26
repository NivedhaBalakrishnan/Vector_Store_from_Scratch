import os
import hnswlib
import numpy as np
import pickle
from configparser import ConfigParser

class VectorStore:
    def __init__(self, vector_dim, M=128, efC=1500, efS=1500, metric='cosine'):
        print(vector_dim)
        self.vector_dim = vector_dim
        self.index = hnswlib.Index(space=metric, dim=vector_dim)
        self.index.init_index(max_elements=10000, ef_construction=efC, M=M)
        self.index.set_ef(efS)
        self.sentences = {}  # Dictionary to store sentences corresponding to vectors
        self.id_counter = 0



    def create_vector_store(self, new_sentence_vectors, persist=True, persist_path="vector_store"):
        """
        Add a vector to the store

        id: the unique id for the vector
        vector: the vector to be added
        """
        try: 
            vectors = []
            ids = []
            for sentence, vector in new_sentence_vectors.items():
                vectors.append(vector)
                ids.append(self.id_counter)  # Assign unique integer id
                self.sentences[self.id_counter] = sentence  # Store the sentence
                self.id_counter += 1  # Increment the id counter
            self.index.add_items(vectors, ids)

            if persist:
                # Create the directory if it doesn't exist
                os.makedirs(persist_path, exist_ok=True)

                # Serialize and save the index
                with open(os.path.join(persist_path, 'index.pkl'), 'wb') as f:
                    pickle.dump(self.index, f)

                # Serialize and save the sentences
                with open(os.path.join(persist_path, 'sentences.pkl'), 'wb') as f:
                    pickle.dump(self.sentences, f)
            
            print("Vector store created successfully", end="\n\n")
        except Exception as e:
            raise e
        
    


    def _load_vector_store(self, persist_path="vector_store"):
        index_file = os.path.join(persist_path, 'index.pkl')
        sentences_file = os.path.join(persist_path, 'sentences.pkl')
        if not (os.path.exists(index_file) and os.path.exists(sentences_file)):
            raise FileNotFoundError("Index and sentences files not found in the specified directory.")
        
        with open(index_file, 'rb') as f:
            self.index = pickle.load(f)
        with open(sentences_file, 'rb') as f:
            self.sentences = pickle.load(f)
        
        return self.index, self.sentences
    


    
    def update_vector_store(self, new_sentence_vectors, persist_path="vector_store"):
        """
        Update the existing vector store with new vectors

        new_id_vectors: Dictionary containing new vectors to be added
        persist_path: Path to the directory where the existing vector store is saved
        """
        try: 
            # load existing index and sentences
            self.index, self.sentences = self._load_vector_store(persist_path)
            

            # Update the id counter
            self.id_counter = max(self.sentences.keys()) + 1

            # Add new vectors to the index and sentences
            vectors = []
            ids = []
            for sentence, vector in new_sentence_vectors.items():
                vectors.append(vector)
                ids.append(self.id_counter)
                self.sentences[self.id_counter] = sentence
                self.id_counter += 1
            self.index.add_items(vectors, ids)
            print("Vector store updated successfully", end="\n\n")
        except Exception as e:
            raise e

       

    def get_similar_vectors(self, query_vector, top_n=5, persist_path="vector_store"):
        """
        Get the most similar vectors to the given vector

        query_vector: the vector to compare with the vectors in the store
        top_n: the number of similar vectors to return
        persist_path: Path to the directory where the existing vector store is saved

        return: the most similar vectors
        """
        # load existing index and sentences
        self.index, self.sentences = self._load_vector_store(persist_path)

        # Search for similar vectors and return the sentences and similarity scores
        labels, distances = self.index.knn_query(np.array([query_vector]), k=top_n)
        similar_vectors = [(self.sentences[label], distance) for label, distance in zip(labels[0], distances[0])]
        return similar_vectors