from sentence_transformers import SentenceTransformer

model = SentenceTransformer("all-MiniLM-L6-v2", device='cpu')

def encode(text: str):
    return model.encode(text).tolist()

def encode_vector(text: str):
    return model.encode(text)
