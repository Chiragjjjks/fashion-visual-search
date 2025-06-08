from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from app.search import search_similar
from pydantic import BaseModel

app = FastAPI()

# Setup templates folder
templates = Jinja2Templates(directory="frontend")

# Define input model
class SearchRequest(BaseModel):
    image_url: str
    top_k: int = 5

@app.get("/", response_class=HTMLResponse)
async def homepage(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/search")
async def search_items(request: SearchRequest):
    results = search_similar(request.image_url, top_k=request.top_k)
    return results.to_dict(orient="records")
