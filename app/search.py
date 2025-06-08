import faiss
import numpy as np
import pandas as pd
from app.utils import get_image_embedding

# Load FAISS index and product data once
embedding_path = "C:/Users/chira/Desktop/fashion-visual-search/app/model/fashion_image_embeddings.npy"
index_path = "C:/Users/chira/Desktop/fashion-visual-search/app/model/faiss_index.idx"
products_path = "C:/Users/chira/Desktop/fashion-visual-search/app/data/valid_products.csv"

embeddings = np.load(embedding_path)
index = faiss.read_index(index_path)
products_df = pd.read_csv(products_path)

def search_similar(image_url, top_k=5, exact_match_threshold=0.05):
    embedding = get_image_embedding(image_url)
    if embedding is None:
        return None

    embedding = np.expand_dims(embedding, axis=0)
    distances, indices = index.search(embedding, top_k)

    results = products_df.iloc[indices[0]].copy()
    results["distance"] = distances[0]
    results["exact_match"] = results["distance"] < exact_match_threshold

    # Ensure pdp_images_s3 is deserialized if stored as string
    if isinstance(results.iloc[0]['pdp_images_s3'], str):
        results["pdp_images_s3"] = results["pdp_images_s3"].apply(eval)

    return results[["pdp_images_s3", "category_type", "image_url", "distance"]]
