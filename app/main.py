from fastapi import FastAPI, Query
from pydantic import BaseModel
from typing import List
from app.search import search_similar

app = FastAPI()

@app.get("/")
async def root():
    return {"message": "Welcome to Fashion Visual Search API"}

class SearchResponseItem(BaseModel):
    pdp_images_s3: List[str]
    category_type: str
    image_url: str
    distance: float

@app.get("/search", response_model=List[SearchResponseItem])
async def search(image_url: str = Query(...), top_k: int = 5):
    results = search_similar(image_url, top_k=top_k)
    if results is None or len(results) == 0:
        return []
    return results.to_dict(orient="records")