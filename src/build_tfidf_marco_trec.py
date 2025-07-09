import os
import json
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from preprocess import preprocess_text
from tqdm import tqdm

def load_docs(dataset_name):
    json_path = os.path.join(os.path.dirname(__file__), '..', 'data', f"{dataset_name}_docs.json")
    with open(json_path, encoding='utf-8') as f:
        return json.load(f)

def build_tfidf(dataset_name):
    print(f"📦 Building TF-IDF for {dataset_name.upper()}...")
    docs = load_docs(dataset_name)
    corpus = [' '.join(doc['tokens']) for doc in docs]

    vectorizer = TfidfVectorizer(
        tokenizer=preprocess_text,
        lowercase=False,
        preprocessor=None,
        token_pattern=None
    )

    tfidf_matrix = vectorizer.fit_transform(tqdm(corpus, desc="🔢 Encoding TF-IDF"))

    output_dir = os.path.join(os.path.dirname(__file__), '..', 'data')
    joblib.dump(vectorizer, os.path.join(output_dir, f"{dataset_name}_tfidf_vectorizer.joblib"))
    joblib.dump(tfidf_matrix, os.path.join(output_dir, f"{dataset_name}_docs_tfidf.joblib"))

    print(f"✅ Done for {dataset_name.upper()}.")

if __name__ == "__main__":
    build_tfidf("trec")
    build_tfidf("msmarco")

