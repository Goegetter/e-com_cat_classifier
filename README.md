# e-com_cat_classifier

An e-commerce product category classifier for English product texts, based on several benchmark approaches:
- a classic TF-IDF + tabular features + LinearSVC pipeline,
- a simple NLP model with a custom vocabulary and a PyTorch neural network,
- an embedding-based classifier using `intfloat/multilingual-e5-base` with additional meta-features.

The project follows a typical data science project structure: raw and processed data, experimental notebooks, source code in `src/`, and trained model artifacts in `models/`.

---

## Project Structure

```text
e-com_cat_classifier/
├── data/
│   ├── embeddings/
│   │   └── e5_all_cpu.npy              # precomputed E5 embeddings for product descriptions
│   ├── external/
│   │   └── Category_Tree.ots           # external category tree / supporting files
│   ├── processed/
│   │   ├── Product Data - Clean.csv    # cleaned dataset
│   │   └── Product Data - CleanEn.csv  # cleaned English-only dataset
│   └── raw/
│       ├── Product Data.csv            # original full dataset
│       └── Product Data - Sample.csv   # smaller sample for quick experiments
├── models/
│   ├── e5_embeddings_classifier.pt     # classifier trained on E5 embeddings (+ meta-features)
│   ├── e5_meta_preprocessor.joblib     # OneHotEncoder / Scaler for meta-features
│   ├── simple_nlp_classifier.pt        # simple PyTorch NLP model with nn.Embedding + MLP
│   ├── tfidf_label_encoder.joblib      # LabelEncoder for Item_Quality_Code
│   └── tfidf_pipeline.joblib           # full TF-IDF + meta-features + LinearSVC pipeline
├── notebooks/
│   ├── 01_eda.ipynb                    # exploratory data analysis
│   ├── 02_preprocessing.ipynb          # data cleaning and preprocessing
│   └── 03_modeling.ipynb               # training and comparison of baseline/final models
├── src/
│   └── api.py                          # inference / API logic
├── .gitignore
├── requirements.txt
└── TODO.txt
```

> Note: this README intentionally reflects the structure shown in the project screenshot, excluding `04_predictions` and `tests`.

---

## Overview

This repository contains a machine learning workflow for assigning product category labels to e-commerce items based on product descriptions and selected structured attributes.

The target variable is:

- `Item_Quality_Code`

The main text inputs are:

- `Name_EN`
- `Description_EN`
- `Composition_EN`

Additional structured features may include:

- `Gender`
- `Tree`
- `Grammage`
- `Weight_net`

---

## Data

### Input files

The project uses the following data layout:

- `data/raw/Product Data.csv` – the original source dataset
- `data/raw/Product Data - Sample.csv` – a smaller subset for testing and fast iteration
- `data/processed/Product Data - Clean.csv` – cleaned dataset after preprocessing
- `data/processed/Product Data - CleanEn.csv` – cleaned English-focused dataset used for modeling
- `data/embeddings/e5_all_cpu.npy` – precomputed E5 embeddings generated from the English text fields

### Typical columns

Example columns used in the project:

- `Name_EN`, `Description_EN`, `Composition_EN` – text features
- `Gender`, `Tree`, `Grammage`, `Weight_net` – structured/meta features
- `Item_Quality_Code` – classification target
- `Item_Quality_Code_desc` – human-readable target description

---

## Models

### 1. TF-IDF + Meta-features + LinearSVC

This is the main classical baseline model.

**Input:**
- concatenated product text (`Name_EN + Description_EN + Composition_EN`)
- categorical meta-features: `Gender`, `Tree`
- numeric meta-features: `Grammage`, `Weight_net`

**Pipeline:**
- `TfidfVectorizer` for text
- `OneHotEncoder` for categorical features
- `StandardScaler` for numeric features
- `LinearSVC` classifier

Saved artifacts:
- `models/tfidf_pipeline.joblib`
- `models/tfidf_label_encoder.joblib`

---

### 2. Simple NLP Classifier

A lightweight PyTorch baseline built from scratch.

**Architecture:**
- tokenization + vocabulary building
- `nn.Embedding`
- mean pooling over token embeddings
- MLP head: `Linear -> ReLU -> Dropout -> Linear`

This model serves as a neural baseline without external pretrained embeddings.

Saved artifact:
- `models/simple_nlp_classifier.pt`

---

### 3. E5 Embeddings Classifier

This model uses pretrained multilingual E5 sentence embeddings.

**Embedding model:**
- `intfloat/multilingual-e5-base`

**Workflow:**
- build a single product text from English columns
- prefix each sample with `passage: `
- compute embeddings offline
- save them to `data/embeddings/e5_all_cpu.npy`

**Classifier input:**
- E5 embeddings
- optional concatenated meta-features (`Gender`, `Tree`, `Grammage`, `Weight_net`)

**Classifier head:**
- MLP on top of embeddings (or embeddings + meta-features)

Saved artifacts:
- `models/e5_embeddings_classifier.pt`
- `models/e5_meta_preprocessor.joblib`

---

## Notebooks

### `01_eda.ipynb`
Exploratory data analysis:
- class distribution
- missing values
- text length inspection
- basic visual analysis of product data

### `02_preprocessing.ipynb`
Data cleaning and preparation:
- handling missing values
- joining English text columns
- filtering empty texts
- preparing clean output files for modeling

### `03_modeling.ipynb`
Model training and evaluation:
- shared train/validation/test split
- TF-IDF baseline training
- Simple NLP model training
- E5 embedding-based classifier training
- model comparison

---

## Inference

`src/api.py` contains the logic for loading trained artifacts and running predictions on new product records.

Depending on the selected model, inference may use:
- the saved TF-IDF pipeline,
- the saved PyTorch simple NLP model,
- the saved E5 classifier together with its preprocessing artifacts.

---

## Installation

### Requirements

- Python 3.10+
- `pip`
- virtual environment support (`venv`, Conda, etc.)

Main dependencies are listed in `requirements.txt`, including:
- `pandas`
- `numpy`
- `scikit-learn`
- `torch`
- `sentence-transformers`
- `joblib`

### Setup

```bash
git clone https://github.com/Goegetter/e-com_cat_classifier.git
cd e-com_cat_classifier

python -m venv .venv
source .venv/bin/activate
# On Windows:
# .venv\Scripts\activate

pip install -r requirements.txt
```

---

## How to Run

1. Place the source files in the `data/raw/` directory.
2. Run `notebooks/02_preprocessing.ipynb` to generate cleaned datasets.
3. Run `notebooks/03_modeling.ipynb` to train and compare models.
4. Saved models will be stored in the `models/` directory.
5. Use `src/api.py` for inference or API integration.

---

## Notes

- The repository structure is designed for experimentation and reproducibility.
- Precomputed embeddings are stored separately to avoid recomputing them every time.
- The E5-based model can be extended with meta-features for better performance.
- The TF-IDF baseline may still outperform embedding models on fine-grained product taxonomy tasks.

---

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.