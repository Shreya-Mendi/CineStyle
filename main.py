# AI tools used: Claude (Anthropic) assisted with FastAPI endpoint
# structure, CORS configuration, and static file serving setup.
"""
CineStyle — FastAPI backend entry point.

Endpoints:
  POST /identify   — receives an image crop, returns garment attributes + embedding
  POST /recommend  — receives an embedding, returns ranked product recommendations
  GET  /health     — liveness check
"""

import io
import json
import os
from pathlib import Path

import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from PIL import Image
from pydantic import BaseModel

from scripts.build_features import embed_image
from scripts.model import recommend

# ---------------------------------------------------------------------------
# Personas — loaded once at startup, optional
# ---------------------------------------------------------------------------
_PERSONAS_PATH = Path("data/personas.json")
_PERSONAS: list[dict] = []
if _PERSONAS_PATH.exists():
    with open(_PERSONAS_PATH) as _f:
        _PERSONAS = json.load(_f)
_PERSONA_BY_ID: dict[int, dict] = {p["id"]: p for p in _PERSONAS}

# Restrict CORS to known origins. Override via ALLOWED_ORIGINS env var
# (comma-separated list). Defaults to Vercel deployment + local dev.
_default_origins = [
    "https://cine-style.vercel.app",
    "http://localhost:3000",
    "http://localhost:8000",
]
_env_origins = os.environ.get("ALLOWED_ORIGINS", "")
allowed_origins: list[str] = (
    [o.strip() for o in _env_origins.split(",") if o.strip()] or _default_origins
)

app = FastAPI(title="CineStyle", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_methods=["*"],
    allow_headers=["*"],
)

Path("data/raw/crops").mkdir(parents=True, exist_ok=True)
app.mount("/static", StaticFiles(directory="data/raw/crops"), name="crops")

class GarmentResponse(BaseModel):
    garment_type: str
    color: str
    aesthetic: str
    embedding: list[float]


class RecommendRequest(BaseModel):
    embedding: list[float]
    top_k: int = 12
    price_min: float | None = None
    price_max: float | None = None
    # Optional — supply either user_id (int, maps to persona) or persona_id (same thing)
    # If omitted, recommendations are purely embedding-based (no NCF/SASRec persona bias).
    persona_id: int | None = None


class ProductCard(BaseModel):
    id: str
    title: str
    brand: str
    price: float
    image_url: str
    product_url: str
    similarity: float


class PersonaCard(BaseModel):
    id: int
    name: str
    description: str
    avatar_emoji: str
    favorite_categories: list[str]
    price_range: list[float]
    aesthetic: str


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/personas", response_model=list[PersonaCard])
def list_personas():
    """
    Return all named demo personas.
    Each persona has distinct category preferences that make FAISS vs NCF vs SASRec
    rankings visibly different — useful for demo and evaluation slides.
    Returns an empty list if data/personas.json is not present.
    """
    return _PERSONAS


@app.post("/identify", response_model=GarmentResponse)
async def identify(file: UploadFile = File(...)):
    """
    Accept a cropped garment image.
    Returns FashionCLIP attributes and the 512-dim embedding.
    """
    contents = await file.read()
    try:
        image = Image.open(io.BytesIO(contents)).convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid image file.")

    result = embed_image(image)
    return GarmentResponse(**result)


@app.post("/recommend", response_model=list[ProductCard])
def get_recommendations(req: RecommendRequest):
    """
    Given a garment embedding, run FAISS retrieval + optional NCF/SASRec re-ranking.

    persona_id (optional): integer ID from /personas.
      - Activates NCF user-preference re-ranking using that persona's interaction history.
      - If persona_id is not provided, returns purely embedding-based KNN results.
      - price_min/price_max from the persona are applied automatically unless overridden
        in the request.
    """
    embedding = np.array(req.embedding, dtype=np.float32)

    # Resolve persona overrides
    price_min = req.price_min
    price_max = req.price_max
    user_id = 0

    if req.persona_id is not None:
        persona = _PERSONA_BY_ID.get(req.persona_id)
        if persona is None:
            raise HTTPException(status_code=404, detail=f"Persona {req.persona_id} not found.")
        user_id = persona["id"]
        # Apply persona price range only if caller didn't supply their own
        if price_min is None and price_max is None:
            lo, hi = persona.get("price_range", [None, None])
            price_min = lo
            price_max = hi

    products = recommend(
        embedding,
        top_k=req.top_k,
        price_min=price_min,
        price_max=price_max,
        user_id=user_id,
    )
    return products
