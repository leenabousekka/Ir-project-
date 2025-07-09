# src/build_tfidf_trec.py

import os
import json
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy import sparse


base_dir = os.path.join(os.path.dirname(__file__), '..', 'data')


def load_texts(path):
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return [' '.join(entry['tokens']) for entry in data]


print("Loading TREC-COVID data...")
docs = load_texts(os.path.join(base_dir, 'trec_docs.json'))
queries = load_texts(os.path.join(base_dir, 'trec_queries.json'))


print("Building TF-IDF model for TREC-COVID...")
vectorizer = TfidfVectorizer()
doc_matrix = vectorizer.fit_transform(docs)
query_matrix = vectorizer.transform(queries)


with open(os.path.join(base_dir, 'trec_tfidf_vectorizer.pkl'), 'wb') as f:
    pickle.dump(vectorizer, f)


sparse.save_npz(os.path.join(base_dir, 'trec_docs_tfidf.npz'), doc_matrix)
sparse.save_npz(os.path.join(base_dir, 'trec_queries_tfidf.npz'), query_matrix)

print("✅ TREC-COVID TF-IDF saved.")
