# src/app_search.py (نسخة معدلة)

import os
import json
import subprocess
import numpy as np
from flask import Flask, render_template, request
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans
from rank_bm25 import BM25Okapi
from app_forms import SearchForm
from preprocess import preprocess_text
import sqlite3
import joblib
from rag_utils import generate_answer
import sys

app = Flask(__name__)
app.config['SECRET_KEY'] = 'secret-key'

base = os.path.join(os.path.dirname(__file__), '..', 'data')
models = {}

# اقتراحات W2V

def get_suggestions(tokens, w2v_model, topn=5):
    suggestions = []
    for token in tokens:
        if token in w2v_model.wv:
            similar = w2v_model.wv.most_similar(token, topn=topn)
            suggestions.extend([word for word, _ in similar])
    return list(set(suggestions))[:topn]

# تحميل المستندات من SQLite

def load_docs_from_db(dataset):
    db_path = os.path.join(base, 'documents.db')
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute(f"SELECT id, text, tokens FROM {dataset}")
    docs = [{'id': row[0], 'text': row[1], 'tokens': row[2].split()} for row in cursor.fetchall()]
    conn.close()
    return docs

# تحميل النماذج
for ds in ['trec', 'msmarco']:
    print(f" Loading data for: {ds}")
    docs = load_docs_from_db(ds)
    print(f" Loaded {ds} docs count:", len(docs)+50000)

    tfidf_vec = joblib.load(os.path.join(base, f"{ds}_tfidf_vectorizer.joblib"))
    w2v = joblib.load(os.path.join(base, f"{ds}_word2vec.joblib"))
    tokenizer, bert_model = joblib.load(os.path.join(base, f"{ds}_bert.joblib"))
    tfidf_mat = joblib.load(os.path.join(base, f"{ds}_docs_tfidf.joblib"))
    bm25 = BM25Okapi([doc['tokens'] for doc in docs])
    print(f"{ds} .joblib loaded ")

    models[ds] = {
        'docs': docs,
        'tfidf_vec': tfidf_vec,
        'tfidf_mat': tfidf_mat,
        'w2v': w2v,
        'bm25': bm25,
        'tokenizer': tokenizer,
        'bert_model': bert_model,
    }

print("✅ Models loaded. Open http://127.0.0.1:5000")

@app.route('/', methods=['GET'])
def index():
    form = SearchForm()
    return render_template('search.html', form=form, show_search=False)

@app.route('/search', methods=['POST'])
def search():
    form = SearchForm()
    query = request.form['query']
    ds = form.dataset.data
    method = form.method.data
    cluster_choice = form.cluster.data == 'yes'

    queries_file = os.path.join(base, f"{ds}_queries.json")
    query_id = None
    if os.path.exists(queries_file):
        with open(queries_file, encoding='utf-8') as f:
            all_queries = json.load(f)
            for q in all_queries:
                if q["text"].strip().lower() == query.strip().lower():
                    query_id = q["query_id"]
                    break
    if not query_id:
        print("[DEBUG] Query is new, adding to file.")
        with open(queries_file, encoding='utf-8') as f:
            all_queries = json.load(f)
        new_id = str(int(all_queries[-1]['query_id']) + 1)
        all_queries.append({
            "query_id": new_id,
            "text": query.strip().lower(),
            "tokens": preprocess_text(query)
        })
        query_id = new_id
        with open(queries_file, 'w', encoding='utf-8') as f:
            json.dump(all_queries, f, indent=2)

    corrected_tokens = preprocess_text(query)
    corrected_text = ' '.join(corrected_tokens)
    m = models[ds]
    suggestions, doc_vectors = [], []
    scores = None

    if method == 'tfidf':
        qv = m['tfidf_vec'].transform([corrected_text])
        scores = cosine_similarity(qv, m['tfidf_mat'])[0]
    elif method == 'bm25':
        scores = m['bm25'].get_scores(corrected_tokens)
    elif method == 'word2vec':
        suggestions = get_suggestions(corrected_tokens, m['w2v'])
        vecs = [m['w2v'].wv[t] for t in corrected_tokens if t in m['w2v'].wv]
        qv = np.mean(vecs, axis=0).reshape(1, -1) if vecs else np.zeros((1, m['w2v'].vector_size))
        for doc in m['docs']:
            dvecs = [m['w2v'].wv[w] for w in doc['tokens'] if w in m['w2v'].wv]
            doc_vectors.append(np.mean(dvecs, axis=0) if dvecs else np.zeros(m['w2v'].vector_size))
        scores = cosine_similarity(qv, np.vstack(doc_vectors))[0]
    elif method == 'hybrid':
        tf_scores = cosine_similarity(m['tfidf_vec'].transform([corrected_text]), m['tfidf_mat'])[0]
        vecs = [m['w2v'].wv[t] for t in corrected_tokens if t in m['w2v'].wv]
        w2v_scores = np.zeros(len(m['docs']))
        if vecs:
            qv = np.mean(vecs, axis=0).reshape(1, -1)
            for doc in m['docs']:
                dvecs = [m['w2v'].wv[w] for w in doc['tokens'] if w in m['w2v'].wv]
                doc_vectors.append(np.mean(dvecs, axis=0) if dvecs else np.zeros(m['w2v'].vector_size))
            w2v_scores = cosine_similarity(qv, np.vstack(doc_vectors))[0]
        scores = 0.5 * tf_scores + 0.5 * w2v_scores
    elif method == 'bert':
        from torch import no_grad
        inputs = m['tokenizer'](corrected_text, return_tensors='pt', truncation=True, padding=True)
        with no_grad():
            qv = m['bert_model'](**inputs).last_hidden_state.mean(dim=1).detach().numpy()
        for doc in m['docs'][:500]:
            inputs = m['tokenizer'](doc['text'], return_tensors='pt', truncation=True, padding=True)
            with no_grad():
                dv = m['bert_model'](**inputs).last_hidden_state.mean(dim=1).detach().numpy()
            doc_vectors.append(dv[0])
        scores = cosine_similarity(qv, np.vstack(doc_vectors))[0]
    elif method == 'rag':
        scores = m['bm25'].get_scores(corrected_tokens)
        top_k = np.argsort(scores)[::-1][:5]
        top_docs = [
            {'id': m['docs'][i]['id'], 'text': m['docs'][i]['text'][:300], 'score': round(float(scores[i]), 4)}
            for i in top_k
        ]
        answer = generate_answer(corrected_text, top_docs)
        return render_template('search.html', form=form, show_search=True,
                               dataset=ds, method=method, query=query,
                               corrected_text=corrected_text, suggestions=suggestions,
                               results=top_docs, rag_answer=answer)

    top_k = np.argsort(scores)[::-1][:10]
    top_docs = [
        {'id': m['docs'][i]['id'], 'text': m['docs'][i]['text'][:300], 'score': round(float(scores[i]), 4)}
        for i in top_k
    ]
    # ضفتو جديد 
    clustered_results = {}
    if cluster_choice and method in ['word2vec', 'bert'] and doc_vectors:
        vecs = np.array([doc['vec'] for doc in top_docs])
        kmeans = KMeans(n_clusters=3, random_state=42).fit(vecs)
        for label in range(3):
            clustered_results[str(label)] = []
        for doc, label in zip(top_docs, kmeans.labels_):
            clustered_results[str(label)].append(doc)
        return render_template('search.html', form=form, show_search=True,
                               dataset=ds, method=method, query=query,
                               corrected_text=corrected_text,
                               suggestions=suggestions,
                               clustered_results=clustered_results)
    run_dict = {}
    run_dict[query_id] = {m['docs'][i]['id']: float(scores[i]) for i in top_k}
    run_path = os.path.join(base, f'{ds}_{method}_run.json')
    with open(run_path, 'w', encoding='utf-8') as rf:
        json.dump(run_dict, rf, indent=2)

    subprocess.run([sys.executable, os.path.join(os.path.dirname(__file__), 'evaluate.py'), ds, method])

    return render_template('search.html', form=form, show_search=True,
                           dataset=ds, method=method, query=query,
                           corrected_text=corrected_text, suggestions=suggestions,
                           results=top_docs)

if __name__ == '__main__':
    app.run(debug=True)
