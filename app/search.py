import faiss
import numpy as np
import pandas as pd
from app.utils import get_image_embedding

# Load FAISS index and product data once
embedding_path = "C:/Users/chira/Desktop/fashion-visual-search/app/model/fashion_image_embeddings.npy"
index_path = "C:/Users/chira/Desktop/fashion-visual-search/app/model/faiss_index.idx"
products_path = "C:/Users/chira/Desktop/fashion-visual-search/app/data/valid_products.csv"

# Load FAISS index
index = faiss.read_index(index_path)

# Load product metadata
products_df = pd.read_csv(products_path)

def search_similar(image_url, top_k=5, exact_match_threshold=0.05):
    """
    Given an image URL, return the top_k most similar product items.
    """
    embedding = get_image_embedding(image_url)
    if embedding is None:
        return None

    # Convert to 2D array
    embedding = np.expand_dims(embedding, axis=0)

    # Search FAISS index
    distances, indices = index.search(embedding, top_k)

    # Get matching products
    results = products_df.iloc[indices[0]].copy()
    results["image_url"] = results["image_url"].astype(str)
    results["distance"] = distances[0]
    results["exact_match"] = results["distance"] < exact_match_threshold

    return results
