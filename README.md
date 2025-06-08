# 👗 Fashion Visual Search & Intelligent Styling Assistant

An AI-powered fashion recommendation and visual search engine that helps users discover and style outfits using image and text inputs. Built with OpenAI’s CLIP model and FAISS for fast and intelligent retrieval.

---

## 🚀 Features

- 🔍 **Visual Search**: Find visually similar fashion products using an input image.
- 🧠 **Semantic Embeddings**: Uses OpenAI CLIP to understand both image and text descriptions.
- 🧾 **Intelligent Styling**: Suggests outfit combinations based on category, occasion, and visual similarity.
- 🖼️ **Multi-image Support**: Displays all related images for each fashion item.
- 🧵 **Category-Aware Filtering**: Filters search results based on fashion categories like `jeans`, `dress`, etc.
- ⚡ **Fast Retrieval**: Leverages FAISS for fast nearest-neighbor search across thousands of embeddings.

---

## 🛠️ Tech Stack

- 🧠 **CLIP (Contrastive Language-Image Pretraining)** – for multimodal embeddings
- 🧲 **FAISS** – for efficient similarity search
- 🐍 **Python** – core logic and preprocessing
- 🔧 **FastAPI** – for backend APIs
- 💻 **HTML/CSS/JS** – simple frontend
- 📊 **Pandas / NumPy** – data processing
- 📝 **Jupyter Notebooks** – for prototyping and development

---

## 📂 Project Structure

```bash
fashion-visual-search/
│
├── app/                            # FastAPI application
│   ├── __init__.py
│   ├── main.py                     # Entry point for FastAPI
│   ├── model/
│   │   ├── index.faiss            # FAISS index file (precomputed)
│   │   ├── image_embeddings.npy   # NumPy image embeddings (CLIP features)
│   │   
│   ├── search.py                  # Core logic for visual similarity
│   └── utils.py                   # Utilities: image loader, preprocessing, etc.
│
├── data/                           # (Optional) Raw or cleaned CSV/data files
│   ├── valid_products.csv         # Cleaned metadata (product, URLs, labels)
│   
│
├── frontend/                       # Simple HTML/CSS/JS frontend
│   └── index.html                 # User UI to search via image URL
│
├── notebooks/                      # Jupyter notebooks (for development)
│   ├── 01_preprocessing.ipynb     # Cleaning, feature extraction
│   ├── 02_generate_embeddings.ipynb # CLIP or ResNet50 embeddings
│   └── 03_visual_search_demo.ipynb # Demo of visual similarity using FAISS
│
├── .gitignore
├── requirements.txt               # All dependencies (FastAPI, faiss, torch, etc.)
└── README.md                      # Project overview and instructions
