# src/generate_run.py

import os
import json
import pickle
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from scipy import sparse

base = os.path.join(os.path.dirname(__file__), '..', 'data')

def load_data(ds):
    with open(os.path.join(base, f'{ds}_docs.json'), encoding='utf-8') as f:
        docs = json.load(f)
    with open(os.path.join(base, f'{ds}_queries.json'), encoding='utf-8') as f:
        queries = json.load(f)
    with open(os.path.join(base, f'{ds}_tfidf_vectorizer.pkl'), 'rb') as f:
        vec = pickle.load(f)
    doc_mat = sparse.load_npz(os.path.join(base, f'{ds}_docs_tfidf.npz'))
    return docs, queries, vec, doc_mat

def main(ds, top_k=100):
    docs, queries, vec, doc_mat = load_data(ds)
    output_path = os.path.join(base, f'{ds}_results.txt')

    with open(output_path, 'w', encoding='utf-8') as out:
        for q in queries:
            q_id = q['query_id']
            q_text = ' '.join(q['tokens'])
            q_vec = vec.transform([q_text])
            scores = cosine_similarity(q_vec, doc_mat)[0]
            top_idxs = np.argsort(scores)[::-1][:top_k]

            for rank, idx in enumerate(top_idxs):
                doc_id = docs[idx].get('doc_id', f'DOC{idx}')
                score = scores[idx]
                out.write(f"{q_id} Q0 {doc_id} {rank+1} {score:.4f} STANDARD\n")
    
    print(f"✅ Saved: {output_path}")

if __name__ == '__main__':
    main('trec')
