import os
import json
from preprocess import preprocess_text
import ir_datasets
from tqdm import tqdm 

# saving data to json files 
def save_json(data, path):
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

# make sure that data file exists
output_dir = os.path.join(os.path.dirname(__file__), '..', 'data')
os.makedirs(output_dir, exist_ok=True)

#  proccessing MSMARCO 
print("📥 Loading MSMARCO...")
msmarco = ir_datasets.load("msmarco-passage/train")
msmarco_docs = []
for i, doc in enumerate(tqdm(msmarco.docs_iter(), desc="📝 Processing 200K MSMARCO docs")):
    if i >= 200000:
        break
    processed = preprocess_text(doc.text)
    msmarco_docs.append({
        'doc_id': doc.doc_id,
        'text': doc.text,
        'tokens': processed
    })

msmarco_queries = []
for i, query in enumerate(tqdm(msmarco.queries_iter(), desc="🔍 Processing 2K MSMARCO queries")):
    if i >= 2000:
        break
    processed = preprocess_text(query.text)
    msmarco_queries.append({
        'query_id': query.query_id,
        'text': query.text,
        'tokens': processed
    })

save_json(msmarco_docs, os.path.join(output_dir, "msmarco_docs.json"))
save_json(msmarco_queries, os.path.join(output_dir, "msmarco_queries.json"))
print("✅ MSMARCO saved.")

# process  TREC-COVID 
print("📥 Loading TREC-COVID...")
trec = ir_datasets.load("beir/trec-covid")
trec_docs = []
for doc in tqdm(trec.docs_iter(), desc="📄 Processing TREC-COVID docs"):
    processed = preprocess_text(doc.text)
    trec_docs.append({
        'doc_id': doc.doc_id,
        'text': doc.text,
        'tokens': processed
    })

trec_queries = []
for query in tqdm(trec.queries_iter(), desc="🔎 Processing TREC-COVID queries"):
    processed = preprocess_text(query.text)
    trec_queries.append({
        'query_id': query.query_id,
        'text': query.text,
        'tokens': processed
    })

save_json(trec_docs, os.path.join(output_dir, "trec_docs.json"))
save_json(trec_queries, os.path.join(output_dir, "trec_queries.json"))
print("✅ TREC-COVID saved.")
