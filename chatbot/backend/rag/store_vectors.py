import os
import faiss
import numpy as np
import json

def save_index(embeddings, texts, index_path="vectorstore/index.faiss", texts_path="vectorstore/texts.json"):
    """
    Save FAISS index and associated texts
    """
    # Create vectorstore directory if it doesn't exist
    os.makedirs("vectorstore", exist_ok=True)
    
    # Ensure embeddings are in the correct format
    if not isinstance(embeddings, np.ndarray):
        embedding_matrix = np.array(embeddings, dtype="float32")
    else:
        embedding_matrix = embeddings.astype("float32")
    
    # Ensure 2D array
    if embedding_matrix.ndim == 1:
        embedding_matrix = embedding_matrix.reshape(1, -1)
    
    print(f"Embedding matrix shape: {embedding_matrix.shape}")
    
    # Create FAISS index
    dimension = embedding_matrix.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embedding_matrix)
    
    # Save FAISS index
    faiss.write_index(index, index_path)
    
    # Save texts as JSON
    with open(texts_path, "w", encoding="utf-8") as f:
        json.dump(texts, f, ensure_ascii=False, indent=2)
    
    print(f"Index saved to {index_path}")
    print(f"Texts saved to {texts_path}")

def load_index(path="vectorstore/index.faiss"):
    """
    Load FAISS index
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Index file not found: {path}")
    return faiss.read_index(path)

def load_texts(path="vectorstore/texts.json"):
    """
    Load texts from JSON file
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Texts file not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)