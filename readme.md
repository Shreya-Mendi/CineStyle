# CineStyle

> *You're watching a show. Someone walks in wearing the perfect outfit. You want it. Now you can find it.*

**AIPI 540 · Module Project · Duke University**

---

## What It Does

CineStyle is a real-time fashion identification and recommendation system for film and TV viewers. While watching a scene, a user can capture a frame, select a character, and ask: *"What is she wearing?"* — and get both an identification of the garment and style-matched recommendations to buy or recreate it.

```
User captures a frame from a show
              ↓
  Character + garment region selected
              ↓
  Computer vision pipeline
  → Garment type / color / style attributes extracted (FashionCLIP)
              ↓
  Four-stage recommendation engine
  → FAISS KNN (visual similarity)
  → NeuMF re-ranking (collaborative filtering)
  → SASRec re-ranking (sequential Transformer)
  → Diversity filter (price spread + deduplication)
              ↓
  Product cards with purchase links
```

The key contribution is the **end-to-end pipeline from passive viewing to active discovery** — combining vision-based garment parsing with a hybrid recommendation system that uses visual embeddings, collaborative filtering, and sequential Transformer-based personalization.

---

## Rubric Alignment

### Problem & Motivation

Viewers regularly notice clothing in shows and films but have no frictionless way to identify or shop it. Existing tools (Google Lens, ShopLook) require manual image search with no scene context. CineStyle closes this gap by making fashion discovery a native part of the viewing experience.

### Three Required Modeling Approaches

| Approach | Model | Role |
|---|---|---|
| Naive baseline | Popularity-based recommender (most-interacted items globally) | Baseline |
| Classical ML | FAISS KNN with FashionCLIP visual embeddings + cosine similarity | Candidate retrieval |
| Deep learning (feedforward) | NeuMF re-ranker — GMF + MLP branches with BPR loss | Personalized re-ranking |
| Deep learning (Transformer) | SASRec — causal self-attention over user sequences with BCE loss | Sequential re-ranking |

All four are implemented, documented, and benchmarked. The deployed app uses the full four-stage pipeline (FAISS → NeuMF → SASRec → diversity filter).

### Evaluation Strategy

**Offline:**
- Precision@K, Recall@K, NDCG@K (K = 5, 10) — FAISS KNN vs SASRec comparison
- Mean Average Precision (MAP@K)
- Visual similarity score (cosine distance in embedding space)

**Online (in-app):**
- Click-through rate on recommendations
- "Save to wishlist" rate
- Time-to-first-click

### Experiments

**Experiment 1: Frame quality vs. recommendation accuracy**

We vary input frame quality (full HD, JPEG-compressed at quality=15, Gaussian-blurred radius=4) and measure how retrieval precision degrades. Tests robustness of the vision pipeline to real-world screenshot quality variation.

**Experiment 2: NCF hyperparameter tuning (embed_dim)**

We sweep `embed_dim` across [16, 32, 64, 128, 256] for the NeuMF re-ranker, training each variant for 10 epochs and evaluating NDCG@10. Results in `data/outputs/hyperparam_tuning.json`.

**Experiment 3: Persona comparison — FAISS vs NeuMF vs SASRec**

Six named user personas (see [Personas](#personas)) with distinct style profiles are used to show how each stage of the pipeline produces measurably different rankings. The same probe image yields different top-3 categories at the FAISS, NCF, and SASRec stages — demonstrating that deeper personalization shifts results toward each persona's taste.

**Error Analysis: Category mispredictions**

Five representative cases where FAISS retrieval returns a top-5 item from a different garment category than the query are identified and visualized. Results in `data/outputs/error_analysis.json`.

---

## Technical Architecture

### Vision Pipeline (Garment Identification)

```
Input: video frame (image)
  ↓
Garment region crop (user-drawn bounding box)
  ↓
FashionCLIP (patrickjohncyh/fashion-clip)
  → garment type, color, aesthetic label
  → 512-dim embedding vector
```

**Model:** `patrickjohncyh/fashion-clip` — CLIP fine-tuned on the Farfetch dataset (700K fashion items). Classification is done via cosine similarity against text label embeddings — no separate classification head needed.

### Recommendation Engine

Four-stage pipeline:

**Stage 1 — Candidate Retrieval (Classical ML)**
- FAISS `IndexFlatIP` over L2-normalized FashionCLIP product embeddings
- Inner product on normalized vectors = cosine similarity
- Returns top-50 visually similar items
- Baseline: global popularity ranking (naive model)

**Stage 2 — NCF Re-ranking (Deep Learning, Feedforward)**
- **NeuMF** (He et al. 2017): GMF branch (elementwise product of user/item embeddings) + MLP branch (concatenate → [256→128→64] → ReLU)
- Trained with **BPR loss** (Bayesian Personalized Ranking) on implicit feedback
- Input: `(user_id, item_embedding_512d)` → scalar relevance score

**Stage 2b — SASRec Re-ranking (Deep Learning, Transformer)**
- **SASRec** (Kang & McAuley, ICDM 2018): Self-Attentive Sequential Recommendation
- Architecture: FashionCLIP embeddings (512d) → linear projection → d_model=128, learned positional embeddings, N=2 causal Transformer blocks (multi-head self-attention + GELU FFN + pre-LayerNorm), final dot product with target item projection
- Trained with **BCE loss** on (sequence → next item) prediction
- Activated when `user_history` (sequence of prior interaction embeddings) is provided

**Stage 3 — Diversity Filter**
- Deduplicates by price quartile — at most 3 items per low/mid/high price bucket
- Ensures spread across price tiers in the final recommendation grid

### Dataset

| Source | Purpose |
|---|---|
| [detection-datasets/fashionpedia](https://huggingface.co/datasets/detection-datasets/fashionpedia) | Product catalog — bounding-box crops of 46 garment/accessory categories |
| Synthetic interactions | NCF + SASRec training — taste-cohesive implicit feedback (500 users, 30 interactions each) |
| Persona-biased interactions *(optional)* | Stronger training signal for named persona demo |
| Duke LiteLLM API (Claude) | Agentic price enrichment via tool-calling loop |

The catalog is built from Fashionpedia's editorial images: each annotated bounding box becomes one product record with a cropped JPEG, category label, and LLM-enriched price.

### API Endpoints

| Method | Path | Description |
|---|---|---|
| `GET` | `/health` | Liveness check |
| `POST` | `/identify` | Upload a garment image crop → returns type, color, aesthetic, 512-dim embedding |
| `POST` | `/recommend` | Send embedding + optional filters → returns ranked product cards |
| `GET` | `/personas` | List all demo personas (empty list if `data/personas.json` absent) |

**`/recommend` request fields:**

```json
{
  "embedding": [/* 512 floats */],
  "top_k": 12,
  "price_min": null,
  "price_max": null,
  "persona_id": null
}
```

`persona_id` is optional. When set, the persona's price range and user ID are applied to NCF/SASRec re-ranking — making recommendations visibly persona-specific. Omitting it gives purely embedding-based results (no change to the deployed behaviour).

---

## Personas

Six named demo personas with distinct style profiles, used to demonstrate how the three model stages produce different rankings:

| # | Name | Aesthetic | Favourite categories | Price range |
|---|------|-----------|----------------------|-------------|
| 0 | Aria Chen 🖤 | Minimalist streetwear | jacket, pants, shoe, tee | $40–300 |
| 1 | Sofia Rossi 🤍 | Old money / quiet luxury | coat, bag, sweater, shoe | $150–850 |
| 2 | Juno Park 💜 | Y2K / dark academia | top, skirt, dress, shoe | $20–180 |
| 3 | Paloma Vega 🌸 | Cottagecore | dress, blouse, bag, shoe | $30–250 |
| 4 | Marcus Webb 🕴️ | Sharp formal | pants, blazer, watch, shoe | $80–700 |
| 5 | Zara Okonkwo ✨ | Maximalist accessories | bag, scarf, hat, shoe | $25–500 |

Personas are defined in `data/personas.json` and served by `GET /personas`. The notebook Step 9 runs a side-by-side comparison showing how each persona's probe image produces different top-3 categories at the FAISS, NCF, and SASRec stages.

---

## Project Structure

```
cinestyle/
├── readme.md
├── requirements.txt
├── main.py                          # FastAPI backend — /identify, /recommend, /personas
├── Dockerfile
├── scripts/
│   ├── make_dataset.py              # Fashionpedia download → catalog.jsonl + interactions.jsonl
│   │                                #   --personas flag: use persona-biased interaction data
│   ├── build_features.py            # FashionCLIP embeddings → FAISS index
│   ├── model.py                     # NeuMF + SASRec training + four-stage recommend()
│   ├── evaluate.py                  # Offline eval, hyperparameter tuning, error analysis
│   ├── fetch_prices.py              # Agentic price enrichment via Duke LiteLLM
│   ├── generate_charts.py           # Output chart generation
│   └── download_assets.py           # Asset download helpers
├── models/
│   ├── faiss_index/                 # products.index + meta.json
│   ├── ncf_reranker/                # ncf.pt + config.pt
│   └── sasrec/                      # sasrec.pt + config.pt
├── data/
│   ├── personas.json                # Named persona definitions (6 personas)
│   ├── raw/crops/                   # Fashionpedia garment crops
│   ├── processed/                   # catalog.jsonl, interactions.jsonl
│   └── outputs/                     # Eval results, experiment charts
├── notebooks/
│   └── cinestyle_pipeline.ipynb     # End-to-end Colab A100 notebook (Steps 0–9)
├── frontend/
│   ├── app/
│   │   ├── page.tsx                 # Main viewer interface
│   │   ├── globals.css              # Cinema dark theme, amber accents
│   │   └── components/
│   │       ├── FrameCapture.tsx     # Image/video upload + frame selection slider
│   │       ├── GarmentHighlight.tsx # Drag-to-select garment overlay
│   │       ├── ResultPanel.tsx      # Slide-in identification + recommendation panel
│   │       └── ProductCard.tsx      # Shoppable product card
│   └── lib/api.ts                   # Typed API client
└── .gitignore
```

---

## Notebook (Colab A100)

`notebooks/cinestyle_pipeline.ipynb` runs the complete pipeline end-to-end. Steps:

| Step | What it does | Time (A100) |
|------|--------------|-------------|
| 0 | Install dependencies | ~3 min |
| 1 | Clone repo & configure paths | <1 min |
| 2 | Download Fashionpedia + build catalog (20k items) | ~15 min |
| 2b *(opt)* | Build persona-biased interactions | ~2 min |
| 3 | Enrich prices with Duke LiteLLM agent | ~5 min |
| 4 | FashionCLIP embeddings + FAISS index | ~15 min |
| 5 | Train NeuMF (10 epochs, BPR) | ~5 min |
| 5b | Train SASRec (20 epochs, BCE) | ~8 min |
| 6 | Offline eval — FAISS vs SASRec Precision/Recall/NDCG/MAP | ~3 min |
| 6b | NCF hyperparameter sweep (embed_dim) | ~10 min |
| 6c | Error analysis — 5 category mispredictions | ~2 min |
| 7 | Frame quality degradation experiment | ~3 min |
| 8 | Quick inference demo | ~1 min |
| 9 *(opt)* | Persona comparison — visual grids per persona | ~3 min |

---

## Running Locally

**Backend:**
```bash
pip install -r requirements.txt
uvicorn main:app --reload --port 8000
```

**Frontend:**
```bash
cd frontend
npm install
npm run dev   # http://localhost:3000
```

**Build product index:**
```bash
# Standard (anonymous synthetic users)
python scripts/make_dataset.py --max_items 20000

# With persona-biased interactions (recommended for persona demo)
python scripts/make_dataset.py --max_items 20000 --personas

python scripts/build_features.py
```

**Enrich prices (requires Duke LiteLLM API key):**
```bash
export DUKE_LLM_API_KEY="<your key>"
python scripts/fetch_prices.py
```

**Train re-rankers:**
```bash
python scripts/model.py --train --epochs 10 --batch_size 256
python scripts/model.py --train_sasrec --epochs 20 --batch_size 256
```

**Evaluate:**
```bash
python scripts/evaluate.py --k 5 --k 10
python scripts/evaluate.py --tune              # NCF hyperparameter sweep
python scripts/evaluate.py --error_analysis    # category misprediction analysis
python scripts/evaluate.py --experiment        # frame quality degradation
```

---

## Application Design (UX)

**Flow:**
1. User uploads a show still or pastes a frame
2. Garment region is selected by dragging a bounding box over any item
3. A panel slides in with:
   - Identified item (garment type, color, aesthetic label)
   - 6–12 shoppable recommendations in a scroll rail
   - Optional persona selector — switches the NCF/SASRec persona to show how recommendations change
4. User can save items to a wishlist or filter by price

**Design language:** Dark cinema aesthetic (`#0c0a09`), warm amber accents (`#d97706`), editorial typography — slideInRight + fadeUp + pulse-amber animations.

**Tech stack:**

| Layer | Tool |
|---|---|
| Frontend | Next.js + Tailwind CSS |
| Backend | FastAPI + uvicorn |
| Vision | FashionCLIP (`patrickjohncyh/fashion-clip`) via HuggingFace Transformers |
| Vector search | FAISS GPU (`IndexFlatIP`) |
| Re-rankers | NeuMF (BPR) + SASRec (BCE) — PyTorch |
| Price enrichment | Duke LiteLLM (OpenAI-compatible) — agentic tool-calling loop |
| Deployment | Railway (backend) + Vercel (frontend) |
| Training | Google Colab A100 |

---

## Ethics Statement

- Fashion data encodes narrow body and aesthetic standards — outputs may skew toward slim, Western, Eurocentric styles
- Affiliate/purchase links create a commercial incentive that can conflict with genuine user interest
- Character identification (linking outfits to actors) risks misidentification across demographic groups
- Visual similarity search does not account for accessibility, sizing, or sustainability — future versions should surface these filters prominently

---

## Commercial Viability

The "shop the look" market is validated (LTK, ShopLook, Amazon's "Find on Amazon"). CineStyle's differentiation:

1. **Context awareness** — recommendations are anchored to a specific scene, not a generic style board
2. **Passive discovery** — no manual search; the viewer's natural behavior triggers the pipeline
3. **Persona-driven personalization** — sequential Transformer captures evolving taste over a session
4. **Extensibility** — the same architecture applies to sports gear, home décor, or any visual domain

Monetization path: affiliate revenue on purchases, white-label API licensing to streaming platforms.

---

## Related Work

- He et al. (2017) — Neural Collaborative Filtering (NCF) / NeuMF
- Kang & McAuley (2018) — Self-Attentive Sequential Recommendation (SASRec)
- Chia et al. (2022) — FashionCLIP: CLIP fine-tuned on the Farfetch fashion catalog
- Guo et al. (2019) — FashionBERT: cross-modal fashion retrieval
- Wu et al. (2022) — Graph Neural Networks in Recommender Systems: A Survey
- Han et al. (2017) — Learning Fashion Compatibility with Bidirectional LSTMs (Polyvore)
