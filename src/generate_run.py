
import os
import json
import joblib
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from scipy import sparse
from rank_bm25 import BM25Okapi
from preprocess import preprocess_text
import torch

base = os.path.join(os.path.dirname(__file__), '..', 'data')


def load_data(ds):
    with open(os.path.join(base, f'{ds}_queries.json'), encoding='utf-8') as f:
        queries = json.load(f)

    tfidf_vec = joblib.load(os.path.join(base, f"{ds}_tfidf_vectorizer.joblib"))
    tfidf_mat = joblib.load(os.path.join(base, f"{ds}_docs_tfidf.joblib"))
    w2v = joblib.load(os.path.join(base, f"{ds}_word2vec.joblib"))
    tokenizer, bert_model = joblib.load(os.path.join(base, f"{ds}_bert.joblib"))

    with open(os.path.join(base, f"{ds}_docs.json"), encoding='utf-8') as f:
        docs = json.load(f)

    bm25 = BM25Okapi([doc['tokens'] for doc in docs])

    return queries, docs, tfidf_vec, tfidf_mat, w2v, tokenizer, bert_model, bm25


def compute_scores(method, query, docs, tfidf_vec, tfidf_mat, w2v, tokenizer, bert_model, bm25):
    tokens = preprocess_text(query['text'])
    q_text = ' '.join(tokens)
    scores = np.zeros(len(docs))

    if method == 'tfidf':
        qv = tfidf_vec.transform([q_text])
        scores = cosine_similarity(qv, tfidf_mat)[0]

    elif method == 'bm25':
        scores = bm25.get_scores(tokens)

    elif method == 'word2vec':
        vecs = [w2v.wv[t] for t in tokens if t in w2v.wv]
        if vecs:
            qv = np.mean(vecs, axis=0).reshape(1, -1)
            doc_vecs = []
            for doc in docs:
                dvecs = [w2v.wv[t] for t in doc['tokens'] if t in w2v.wv]
                doc_vecs.append(np.mean(dvecs, axis=0) if dvecs else np.zeros(w2v.vector_size))
            scores = cosine_similarity(qv, np.vstack(doc_vecs))[0]

    elif method == 'bert':
        with torch.no_grad():
            inputs = tokenizer(q_text, return_tensors='pt', truncation=True, padding=True)
            qv = bert_model(**inputs).last_hidden_state.mean(dim=1).detach().numpy()
            doc_vecs = []
            for doc in docs[:1000]:  # تحسين الأداء
                inputs = tokenizer(doc['text'], return_tensors='pt', truncation=True, padding=True)
                dv = bert_model(**inputs).last_hidden_state.mean(dim=1).detach().numpy()
                doc_vecs.append(dv[0])
            scores = cosine_similarity(qv, np.vstack(doc_vecs))[0]

    elif method == 'hybrid':
        tfidf_scores = cosine_similarity(tfidf_vec.transform([q_text]), tfidf_mat)[0]
        vecs = [w2v.wv[t] for t in tokens if t in w2v.wv]
        if vecs:
            qv = np.mean(vecs, axis=0).reshape(1, -1)
            doc_vecs = []
            for doc in docs:
                dvecs = [w2v.wv[t] for t in doc['tokens'] if t in w2v.wv]
                doc_vecs.append(np.mean(dvecs, axis=0) if dvecs else np.zeros(w2v.vector_size))
            w2v_scores = cosine_similarity(qv, np.vstack(doc_vecs))[0]
        else:
            w2v_scores = np.zeros(len(docs))
        scores = 0.5 * tfidf_scores + 0.5 * w2v_scores

    return scores


def main(ds, method='tfidf', top_k=100):
    queries, docs, tfidf_vec, tfidf_mat, w2v, tokenizer, bert_model, bm25 = load_data(ds)
    run_dict = {}

    for q in queries:
        scores = compute_scores(method, q, docs, tfidf_vec, tfidf_mat, w2v, tokenizer, bert_model, bm25)
        top_idxs = np.argsort(scores)[::-1][:top_k]
        run_dict[q['query_id']] = {
            docs[i].get('id') or docs[i].get('doc_id') or f"DOC{i}": float(scores[i])
            for i in top_idxs
        }

    # حفظ بصيغة run.json
    output_path = os.path.join(base, f'{ds}_{method}_run.json')
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(run_dict, f, indent=2)

    print(f"✅ Saved: {output_path}")


if __name__ == '__main__':
    import sys
    if len(sys.argv) < 3:
        print("⚠️ usage: python generate_run.py <dataset> <method>")
    else:
        main(sys.argv[1], sys.argv[2])