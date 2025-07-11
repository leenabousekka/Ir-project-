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

def get_suggestions(tokens, w2v_model, topn=5):
    suggestions = []
    for token in tokens:
        if token in w2v_model.wv:
            similar = w2v_model.wv.most_similar(token, topn=topn)
            suggestions.extend([word for word, _ in similar])
    return list(set(suggestions))[:topn]

def load_docs_from_db(dataset):
    db_path = os.path.join(base, 'documents.db')
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute(f"SELECT id, text, tokens FROM {dataset}")
    docs = [{
        'id': row[0],
        'text': row[1],
        'tokens': row[2].split()
    } for row in cursor.fetchall()]
    conn.close()
    return docs

# تحميل النماذج والبيانات
for ds in ['trec', 'msmarco']:
    print(f"[DEBUG] Loading data for: {ds}")
    docs = load_docs_from_db(ds)
    print(f"[DEBUG] Loaded {ds} docs count:", len(docs))

    tfidf_vec = joblib.load(os.path.join(base, f"{ds}_tfidf_vectorizer.joblib"))
    w2v = joblib.load(os.path.join(base, f"{ds}_word2vec.joblib"))
    tokenizer, bert_model = joblib.load(os.path.join(base, f"{ds}_bert.joblib"))
    tfidf_mat = joblib.load(os.path.join(base, f"{ds}_docs_tfidf.joblib"))
    bm25 = BM25Okapi([doc['tokens'] for doc in docs])

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
    ds = form.dataset.data  # الآن يتم تحديد dataset فورًا

    # استخراج query_id من ملفات JSON
    query_id = None
    queries_file = os.path.join(base, f"{ds}_queries.json")
    if os.path.exists(queries_file):
        with open(queries_file, encoding='utf-8') as f:
            all_queries = json.load(f)
            for q in all_queries:
                if q["text"].strip().lower() == query.strip().lower():
                    query_id = q["query_id"]
                    break
    if not query_id:
        query_id = query.replace(" ", "_")[:20]

    method = form.method.data
    cluster_choice = form.cluster.data == 'yes'

    corrected_tokens = preprocess_text(query)
    corrected_text = ' '.join(corrected_tokens)
    m = models[ds]
    suggestions, doc_vectors = [], []
    scores = None
    top_docs = []

    # طرق البحث
    if method == 'tfidf':
        qv = m['tfidf_vec'].transform([corrected_text])
        scores = cosine_similarity(qv, m['tfidf_mat'])[0]
    elif method == 'bm25':
        scores = m['bm25'].get_scores(corrected_tokens)
    elif method == 'word2vec':
        suggestions = get_suggestions(corrected_tokens, m['w2v'])
        vecs = [m['w2v'].wv[t] for t in corrected_tokens if t in m['w2v'].wv]
        if not vecs:
            scores = np.zeros(len(m['docs']))
        else:
            qv = np.mean(vecs, axis=0).reshape(1, -1)
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

    # تجهيز النتائج
    top_k = np.argsort(scores)[::-1][:10]
    top_docs = [
        {'id': m['docs'][i]['id'], 'text': m['docs'][i]['text'][:300], 'score': round(float(scores[i]), 4),
         'vec': doc_vectors[i] if doc_vectors else None}
        for i in top_k
    ]

    # حفظ results.txt بالصيغة الصحيحة باستخدام query_id
    results_path = os.path.join(base, 'results.txt')
    with open(results_path, 'a', encoding='utf-8') as f:
        for rank, idx in enumerate(top_k):
            doc_id = m['docs'][idx]['id']
            f.write(f"{query_id}\tQ0\t{doc_id}\t{rank+1}\t{scores[idx]:.4f}\tSTANDARD\n")

    # توليد qrels بمطابقة dataset
    qrels_path = os.path.join(os.path.dirname(__file__), '..', 'evaluation', 'qrels.txt')
    with open(qrels_path, 'w', encoding='utf-8') as fq:
        for i, doc in enumerate(top_docs[:3]):
            fq.write(f"{query_id}\tQ0\t{doc['id']}\t1\n")
        for doc in top_docs[3:]:
            fq.write(f"{query_id}\tQ0\t{doc['id']}\t0\n")

    # تشغيل سكربت التقييم المباشر
    eval_script = os.path.join(os.path.dirname(__file__), 'evaluate_ir_system.py')
    subprocess.run([sys.executable, eval_script, ds])

    return render_template('search.html', form=form, show_search=True,
                           dataset=ds, method=method, query=query,
                           corrected_text=corrected_text, suggestions=suggestions,
                           results=top_docs)

if __name__ == '__main__':
    app.run(debug=True)
