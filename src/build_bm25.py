import os, json, pickle
from rank_bm25 import BM25Okapi

def build_bm25(dataset_prefix):
    base = os.path.join(os.path.dirname(__file__), '..', 'data')
    docs = json.load(open(os.path.join(base, f"{dataset_prefix}_docs.json"), encoding='utf-8'))
    tokenized = [entry['tokens'] for entry in docs]
    bm25 = BM25Okapi(tokenized)
    with open(os.path.join(base, f"{dataset_prefix}_bm25.pkl"), 'wb') as f:
        pickle.dump(bm25, f)
    print(f"✅ BM25 model saved: {dataset_prefix}_bm25.pkl")

if __name__ == "__main__":
    build_bm25('trec')
    build_bm25('msmarco')
