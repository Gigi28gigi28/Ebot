from sentence_transformers import SentenceTransformer

# Initialize the model once
model = SentenceTransformer("all-MiniLM-L6-v2")

def embed_texts(texts):
    """
    Embed a list of texts and return embeddings as numpy array
    """
    embeddings = model.encode(texts)
    return embeddings

def embed_text(text):
    """
    Embed a single text and return as 1D vector
    """
    return model.encode([text])[0]