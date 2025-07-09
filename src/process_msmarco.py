import ir_datasets
import os
import json
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import string

STOPWORDS = set(stopwords.words('english'))
PUNCT = set(string.punctuation)
STEMMER = PorterStemmer()

def normalize(text):
    tokens = word_tokenize(text.lower())
    tokens = [t for t in tokens if t.isalpha() and t not in STOPWORDS and t not in PUNCT]
    return tokens

def stem(tokens):
    return [STEMMER.stem(t) for t in tokens]

def process():
    dataset = ir_datasets.load("msmarco-passage/train")
    out_dir = "data/msmarco_processed"
    os.makedirs(out_dir, exist_ok=True)

    for i, doc in enumerate(dataset.docs_iter()):
        if i >= 1000:
            break  # نأخذ فقط 1000 وثيقة لتقليل الوقت
        tokens = stem(normalize(doc.text))
        with open(f"{out_dir}/{doc.doc_id}.json", "w", encoding="utf-8") as f:
            json.dump({
                "doc_id": doc.doc_id,
                "tokens": tokens
            }, f)

    print("✅ تم معالجة 1000 وثيقة من msmarco وحفظها في data/msmarco_processed")

if __name__ == "__main__":
    process()
