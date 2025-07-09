# src/store_vectors_joblib.py
import os
import joblib
import pickle
import numpy as np
from scipy import sparse

base = os.path.join(os.path.dirname(__file__), '..', 'data')

for ds in ['trec', 'msmarco']:
    print(f"🔁 Saving compressed vectors for {ds}...")

    # Load vectorizer and matrix
    tfidf_vec_path = os.path.join(base, f"{ds}_tfidf_vectorizer.pkl")
    tfidf_mat_path = os.path.join(base, f"{ds}_docs_tfidf.npz")

    with open(tfidf_vec_path, 'rb') as f:
        vec = pickle.load(f)
    mat = sparse.load_npz(tfidf_mat_path)

    # Save compressed
    joblib.dump(vec, os.path.join(base, f"{ds}_tfidf_vectorizer.joblib"))
    joblib.dump(mat, os.path.join(base, f"{ds}_docs_tfidf.joblib"))

    # Word2Vec
    with open(os.path.join(base, f"{ds}_word2vec.pkl"), 'rb') as f:
        w2v = pickle.load(f)
    joblib.dump(w2v, os.path.join(base, f"{ds}_word2vec.joblib"))

    # BERT model & tokenizer
    with open(os.path.join(base, f"{ds}_bert.pkl"), 'rb') as f:
        tokenizer, model = pickle.load(f)
    joblib.dump((tokenizer, model), os.path.join(base, f"{ds}_bert.joblib"))

    print(f"✅ Done for {ds}")
