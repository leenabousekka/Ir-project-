import os
import json
import pickle
import sys
import numpy as np
from flask import Flask, render_template, request
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans
from rank_bm25 import BM25Okapi
from scipy import sparse
from app_forms import SearchForm
from preprocess import preprocess_text
import sqlite3
import joblib
from rag_utils import generate_answer
import subprocess



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
    docs = []
    for row in cursor.fetchall():
        doc_id, text, tokens_str = row
        tokens = tokens_str.split()
        docs.append({'id': doc_id, 'text': text, 'tokens': tokens})
    conn.close()
    return docs
for ds in ['trec', 'msmarco']:
    print(f"[DEBUG] Loading data for: {ds}")
    docs = load_docs_from_db(ds)
    print(f"[DEBUG] Loaded {ds} docs count:", len(docs))

    tfidf_vec = joblib.load(os.path.join(base, f"{ds}_tfidf_vectorizer.joblib"))

    w2v = joblib.load(os.path.join(base, f"{ds}_word2vec.joblib"))

    tokenizer, bert_model = joblib.load(os.path.join(base, f"{ds}_bert.joblib"))
    
    tfidf_mat = joblib.load(os.path.join(base, f"{ds}_docs_tfidf.joblib"))

    tokenized_corpus = [doc['tokens'] for doc in docs]
    bm25 = BM25Okapi(tokenized_corpus)

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

@app.route('/', methods=['GET', 'POST'])
def index():
    form = SearchForm()
    return render_template('search.html', form=form, show_search=False)

@app.route('/search', methods=['POST'])
def search():
    form = SearchForm()
    query = request.form['query']
    ds = form.dataset.data
    query_id = None
    # تحديد ملف الاستعلام المناسب
    queries_file = os.path.join(base, f"{ds}_queries.json")
    if os.path.exists(queries_file):
        import json
        with open(queries_file, encoding='utf-8') as f:
            all_queries = json.load(f)
            for q in all_queries:
                if q["text"].strip().lower() == query.strip().lower():
                    query_id = q["query_id"]
                    break
    if not query_id:
        query_id = query.replace(" ", "_")[:20]  # قيمة بديلة في حال لم يُعثر على الاستعلام

    
    method = form.method.data
    cluster_choice = form.cluster.data == 'yes'

    corrected_tokens = preprocess_text(query)
    corrected_text = ' '.join(corrected_tokens)
    tokens = query.lower().split()
    m = models[ds]
    suggestions = []
    doc_vectors = []

    top_docs = []   
    scores = None

    if method == 'rag':
        
        scores = m['bm25'].get_scores(corrected_tokens)
        top_k = np.argsort(scores)[::-1][:5]
        for idx in top_k:
            doc_data = m['docs'][idx]
            top_docs.append({
                'id': doc_data.get('id', doc_data.get('doc_id', '')),
                'text': doc_data.get('text', '')[:300],
                'score': round(float(scores[idx]), 4)
            })
        answer = generate_answer(corrected_text, top_docs)
        return render_template('search.html', form=form, show_search=True,
                               dataset=ds, method=method, query=query,
                               corrected_text=corrected_text,
                               suggestions=suggestions,
                               results=top_docs,
                               rag_answer=answer)

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
                d_vecs = [m['w2v'].wv[w] for w in doc['tokens'] if w in m['w2v'].wv]
                if d_vecs:
                    doc_vectors.append(np.mean(d_vecs, axis=0))
                else:
                    doc_vectors.append(np.zeros(m['w2v'].vector_size))
            doc_matrix = np.vstack(doc_vectors)
            scores = cosine_similarity(qv, doc_matrix)[0]
    elif method == 'hybrid':
        print("hybrid chosen ")
        # Parallel Fusion between TF-IDF and Word2Vec
        tfidf_scores = m['tfidf_vec'].transform([query])
        tfidf_scores = cosine_similarity(tfidf_scores, m['tfidf_mat'])[0]
        vecs = [m['w2v'].wv[t] for t in tokens if t in m['w2v'].wv]
        print("hybrid 2")
        if vecs:
            qv = np.mean(vecs, axis=0).reshape(1, -1)
            for doc in m['docs']:
                d_vecs = [m['w2v'].wv[w] for w in doc['tokens'] if w in m['w2v'].wv]
                if d_vecs:
                    doc_vectors.append(np.mean(d_vecs, axis=0))
                else:
                    doc_vectors.append(np.zeros(m['w2v'].vector_size))
            doc_matrix = np.vstack(doc_vectors)
            w2v_scores = cosine_similarity(qv, doc_matrix)[0]
        else:
            w2v_scores = np.zeros(len(m['docs']))
        print("hybrid 3")

       
        scores = 0.5 * tfidf_scores + 0.5 * w2v_scores


    elif method == 'bert':
        from torch import no_grad
        inputs = m['tokenizer'](corrected_text, return_tensors='pt', truncation=True, padding=True)
        with no_grad():
            qv = m['bert_model'](**inputs).last_hidden_state.mean(dim=1).detach().numpy()
        
        print("bert chosen ")
        for doc in m['docs'][:500]:
            text = doc.get('text', '')
            inputs = m['tokenizer'](text, return_tensors='pt', truncation=True, padding=True)
            with no_grad():
                dv = m['bert_model'](**inputs).last_hidden_state.mean(dim=1).detach().numpy()
            doc_vectors.append(dv[0])
        print("bert loading")
        doc_matrix = np.vstack(doc_vectors)
        scores = cosine_similarity(qv, doc_matrix)[0]
        print("bert finished ")

    top_k = np.argsort(scores)[::-1][:10]
    if method == 'rag':
        top_docs = []
        for idx in np.argsort(scores)[::-1][:5]:  # نختار فقط أفضل 5 وثائق
            doc_data = m['docs'][idx]
            top_docs.append({
                'id': doc_data.get('id', doc_data.get('doc_id', '')),
                'text': doc_data.get('text', '')[:300],
                'score': round(float(scores[idx]), 4)
            })

        answer = generate_answer(corrected_text, top_docs)

        return render_template('search.html', form=form, show_search=True,
                            dataset=ds, method=method, query=query,
                            corrected_text=corrected_text,
                            suggestions=suggestions,
                            results=top_docs,
                            rag_answer=answer)

        
    for idx in top_k:
        doc_data = m['docs'][idx]
        top_docs.append({
            'id': doc_data.get('id', doc_data.get('doc_id', '')),
            'text': doc_data.get('text', '')[:300],
            'score': round(float(scores[idx]), 4),
            'vec': doc_vectors[idx] if doc_vectors else None
        })
    results_path = os.path.join(base, 'results.txt')
    query_id = None
    
    queries_file = os.path.join(base, f"{ds}_queries.json")
    if os.path.exists(queries_file):
        import json
        with open(queries_file, encoding='utf-8') as qf:
            queries = json.load(qf)
            for q in queries:
                if q['text'].strip().lower() == corrected_text.strip().lower():
                    query_id = q['query_id']
                    break
    if not query_id:
        query_id = corrected_text.replace(' ', '_')  # fallback مؤقت

    with open(results_path, 'a', encoding='utf-8') as f:
        for rank, idx in enumerate(top_k):
            doc_id = m['docs'][idx].get('doc_id', m['docs'][idx].get('id', f'doc_{idx}'))
            #query_id = query_id
            f.write(f"{query_id}\tQ0\t{doc_id}\t{rank+1}\t{scores[idx]:.4f}\tSTANDARD\n")




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
    qrels_path = os.path.join(os.path.dirname(__file__), '..', 'evaluation', 'qrels.txt')
    with open(qrels_path, 'w', encoding='utf-8') as fq:
        for i, doc in enumerate(top_docs[:3]):
            fq.write(f"{query_id}\tQ0\t{doc['id']}\t1\n")  # أول 3 تعتبر ملائمة
        for doc in top_docs[3:]:
            fq.write(f"{query_id}\tQ0\t{doc['id']}\t0\n")
    import subprocess
    eval_script = os.path.join(os.path.dirname(__file__), 'evaluate_ir_system.py')
    print("✅ Running evaluation...")
    subprocess.run([sys.executable, eval_script, ds])
    import subprocess
    eval_script = os.path.join(os.path.dirname(__file__), 'evaluate_ir_system.py')
    
    return render_template('search.html', form=form, show_search=True,
                           dataset=ds, method=method, query=query,
                           corrected_text=corrected_text,
                           suggestions=suggestions,
                           results=top_docs)


if __name__ == '__main__':
    app.run(debug=True)
