import os
import re
import sys
import logging
from pathlib import Path
from contextlib import asynccontextmanager
import traceback
import numpy as np
import torch
import torch.nn as nn
import joblib
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from sentence_transformers import SentenceTransformer

# ── Ścieżki ──────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))
MODELS_DIR = PROJECT_ROOT / "models"

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ── Definicje modeli (muszą zgadzać się z notebookiem) ───
class SimpleNLPClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_classes, pad_idx=0):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)
        self.classifier = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, input_ids, attention_mask):
        embedded = self.embedding(input_ids)
        mask = attention_mask.unsqueeze(-1)
        pooled = (embedded * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
        return self.classifier(pooled)


class E5Classifier(nn.Module):
    def __init__(self, in_dim, hidden_dim, num_classes, dropout=0.3):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, features):
        return self.classifier(features)


# ── Helpers ───────────────────────────────────────────────
TEXT_COLS = ["Name_EN", "Description_EN", "Composition_EN"]
CAT_COLS  = ["Gender", "Tree"]
NUM_COLS  = ["Grammage", "WeightNet"]
DEVICE    = torch.device("cpu")


def tokenize(text: str) -> list[str]:
    text = text.lower()
    text = re.sub(r"[^a-z0-9 ]", " ", text)
    return text.strip().split()


def encode_text(text: str, vocab: dict, max_len: int):
    tokens = tokenize(text)

    unk_idx = vocab.get("UNK", vocab.get("<UNK>", 1))
    pad_idx = vocab.get("PAD", vocab.get("<PAD>", 0))
    ids = [vocab.get(t, unk_idx) for t in tokens]
    if len(ids) > max_len:
        ids = ids[:max_len]

    mask = [1] * len(ids)
    while len(ids) < max_len:
        ids.append(pad_idx)
        mask.append(0)

    return (
        torch.tensor(ids, dtype=torch.long).unsqueeze(0),
        torch.tensor(mask, dtype=torch.float).unsqueeze(0),
    )


# ── Lifespan — wczytanie modeli raz przy starcie ─────────
loaded = {}


@asynccontextmanager
async def lifespan(app):
    # v1 — TF-IDF (sklearn Pipeline: ColumnTransformer + LinearSVC)
    loaded["tfidf"] = joblib.load(MODELS_DIR / "tfidf_pipeline.joblib")
    loaded["tfidf_label_encoder"] = joblib.load(MODELS_DIR / "tfidf_label_encoder.joblib")
    logger.info("TF-IDF wczytany")
    # DEBUG
    print(loaded["tfidf"].classes_[:10])

    # v2 — SimpleNLP (PyTorch)
    ckpt_mlp = torch.load(
        MODELS_DIR / "simple_nlp_classifier.pt", map_location=DEVICE
    )
    mlp = SimpleNLPClassifier(
        vocab_size=ckpt_mlp["vocab_size"],
        embed_dim=ckpt_mlp["embed_dim"],
        hidden_dim=ckpt_mlp["hidden_dim"],
        num_classes=ckpt_mlp["num_classes"],
    ).to(DEVICE)
    mlp.load_state_dict(ckpt_mlp["model_state_dict"])
    mlp.eval()
    loaded["simple_nlp"]        = mlp
    loaded["simple_nlp_vocab"]  = ckpt_mlp["vocab"]
    loaded["simple_nlp_labels"] = ckpt_mlp["label_classes"]
    loaded["simple_nlp_maxlen"] = ckpt_mlp["max_len"]
    logger.info("SimpleNLP wczytany")

    # v3 — E5 (SentenceTransformer + E5Classifier + meta preprocessor)
    ckpt_e5 = torch.load(MODELS_DIR / "e5_embeddings_classifier.pt", map_location=DEVICE)

    state = ckpt_e5["model_state_dict"]
    actual_hidden_dim = state["classifier.0.weight"].shape[0]  # ← 256
    actual_in_dim = state["classifier.0.weight"].shape[1]  # ← 776
    actual_num_classes = state["classifier.3.weight"].shape[0]  # ← 197

    embed_model = SentenceTransformer("intfloat/multilingual-e5-base", device=str(DEVICE))
    meta_preprocessor = joblib.load(MODELS_DIR / "e5_meta_preprocessor.joblib")

    in_dim = ckpt_e5["input_dim"]
    e5_cls = E5Classifier(
        in_dim=actual_in_dim,
        hidden_dim=actual_hidden_dim,
        num_classes=actual_num_classes,
        dropout=ckpt_e5.get("dropout", 0.3),
    ).to(DEVICE)
    e5_cls.load_state_dict(state)
    e5_cls.eval()
    loaded["e5"]               = e5_cls
    loaded["e5_embed_model"]   = embed_model
    loaded["e5_meta_prep"]     = meta_preprocessor
    loaded["e5_labels"]        = ckpt_e5["label_classes"]
    logger.info("E5 wczytany")

    yield
    loaded.clear()


app = FastAPI(
    title="E-commerce Category Classification API",
    description="3 modele klasyfikacji kategorii: TF-IDF (v1), SimpleNLP (v2), E5 (v3)",
    version="1.0.0",
    lifespan=lifespan,
)

# ── Schematy Pydantic ────────────────────────────────────
class ProductInput(BaseModel):
    Name_EN: str         = Field(..., description="Nazwa produktu po angielsku")
    Description_EN: str  = Field(..., description="Opis produktu")
    Composition_EN: str  = Field(default="", description="Skład materiałowy")
    Gender: str         = Field(default="Unisex")
    Tree: str           = Field(default="")
    Grammage: float | None = Field(default=None)
    Weight_net: float | None = Field(default=None)

    model_config = {"json_schema_extra": {"example": {
        "Name_EN": "Men softshell winter jacket with hood",
        "Description_EN": "Water-resistant jacket with zipper pockets and warm fleece lining",
        "Composition_EN": "94% polyester 6% elastane",
        "Gender": "Male",
        "Tree": "Textile",
        "Grammage": 280,
        "Weight_net": 0.82,
    }}}


class PredictionResult(BaseModel):
    model:      str
    category:   str   = Field(..., description="Kod kategorii, np. TOZB")
    top5: list[dict]  = Field(default=[], description="Top-5 kategorii z prawdopodobieństwami")


class CompareResponse(BaseModel):
    v1_tfidf:      PredictionResult
    v2_simple_nlp: PredictionResult
    v3_e5:         PredictionResult


# ── Funkcje predykcji ────────────────────────────────────
def predict_tfidf(product: ProductInput) -> PredictionResult:
    import pandas as pd
    text = " ".join([product.Name_EN, product.Description_EN, product.Composition_EN]).strip()
    row  = pd.DataFrame([{
        "text":     text,
        "Gender":   product.Gender,
        "Tree":     product.Tree,
        "Grammage": product.Grammage if product.Grammage is not None else 0.0,
        "Weight_net": product.Weight_net if product.Weight_net is not None else 0.0,
    }])

    pred_idx = int(loaded["tfidf"].predict(row)[0])
    le = loaded["tfidf_label_encoder"]
    pred_label = le.inverse_transform([pred_idx])[0]

    # LinearSVC nie ma predict_proba — zwróć decision_function jako score
    scores = loaded["tfidf"].decision_function(row)[0]
    top5_idx = np.argsort(scores)[::-1][:5]
    classes = loaded["tfidf"].classes_
    top5 = [
        {
            "category": str(le.inverse_transform([classes[i]])[0]),
            "score": round(float(scores[i]), 4)
        }
        for i in top5_idx
    ]

    return PredictionResult(model="TF-IDF", category=str(pred_label), top5=top5)


def predict_simple_nlp(product: ProductInput) -> PredictionResult:
    text   = " ".join([product.Name_EN, product.Description_EN, product.Composition_EN]).strip()
    vocab  = loaded["simple_nlp_vocab"]
    maxlen = loaded["simple_nlp_maxlen"]
    labels = loaded["simple_nlp_labels"]

    input_ids, attention_mask = encode_text(text, vocab, maxlen)

    # DEBUG
    print("=== DEBUG SIMPLE NLP ===")
    print("labels type:", type(labels))
    print("labels sample:", labels[:5])

    print("UNK in vocab:", "UNK" in vocab)
    print("PAD in vocab:", "PAD" in vocab)

    print("input_ids shape:", input_ids.shape)
    print("attention_mask shape:", attention_mask.shape)

    print("max input id:", input_ids.max().item())
    print("vocab size:", len(vocab))
    print(list(vocab.keys())[:20])

    with torch.no_grad():
        logits = loaded["simple_nlp"](input_ids.to(DEVICE), attention_mask.to(DEVICE))
        probs  = torch.softmax(logits, dim=-1).cpu().numpy()[0]

    top5_idx = np.argsort(probs)[::-1][:5]
    top5     = [{"category": labels[i], "score": round(float(probs[i]), 4)} for i in top5_idx]

    return PredictionResult(
        model="SimpleNLP",
        category=labels[int(np.argmax(probs))],
        top5=top5
    )


def predict_e5(product: ProductInput) -> PredictionResult:
    import pandas as pd
    text = " ".join([f"passage: {product.Name_EN}", product.Description_EN, product.Composition_EN]).strip()
    emb  = loaded["e5_embed_model"].encode([text], normalize_embeddings=True, convert_to_numpy=True)

    meta_row = pd.DataFrame([{
        "Gender":    product.Gender,
        "Tree":      product.Tree,
        "Grammage":  product.Grammage if product.Grammage is not None else 0.0,
        "Weight_net": product.Weight_net if product.Weight_net is not None else 0.0,
    }])
    meta = loaded["e5_meta_prep"].transform(meta_row)
    if hasattr(meta, "toarray"):
        meta = meta.toarray()

    X = np.concatenate([emb, meta.astype(np.float32)], axis=1)
    tensor = torch.from_numpy(X).float().to(DEVICE)

    labels = loaded["e5_labels"]
    with torch.no_grad():
        logits = loaded["e5"](tensor)
        probs  = torch.softmax(logits, dim=-1).cpu().numpy()[0]

    top5_idx = np.argsort(probs)[::-1][:5]
    top5     = [{"category": labels[i], "score": round(float(probs[i]), 4)} for i in top5_idx]
    return PredictionResult(model="E5", category=labels[int(np.argmax(probs))], top5=top5)


# ── Endpointy ────────────────────────────────────────────
@app.get("/")
def root():
    return {"message": "E-commerce Category Classification API", "docs": "/docs"}


@app.get("/health")
def health():
    return {
        "status": "OK" if len(loaded) >= 3 else "ERROR",
        "loaded": [k for k in loaded if not k.endswith(("_vocab", "_labels", "_maxlen", "_prep", "_model"))],
    }


@app.post("/predict/TF-IDF", response_model=PredictionResult, summary="TF-IDF + LinearSVC")
def predict_v1(product: ProductInput):
    try:
        return predict_tfidf(product)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict/SimpleNLP")
def predict_v2(product: ProductInput):
    try:
        return predict_simple_nlp(product)
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict/e5NLP", response_model=PredictionResult, summary="E5 Embeddings + MLP")
def predict_v3(product: ProductInput):
    try:
        return predict_e5(product)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict/compare", response_model=CompareResponse, summary="Porównanie wszystkich 3 modeli")
def predict_compare(product: ProductInput):
    try:
        return CompareResponse(
            v1_tfidf=predict_tfidf(product),
            v2_simple_nlp=predict_simple_nlp(product),
            v3_e5=predict_e5(product),
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8886)
