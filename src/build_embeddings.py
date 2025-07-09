import os
import json
import pickle
from gensim.models import Word2Vec
from transformers import BertTokenizer, BertModel

def build_word2vec_model(token_lists, save_path):
    print("🔄 تدريب نموذج Word2Vec...")
    w2v = Word2Vec(sentences=token_lists, vector_size=100, window=5, min_count=2, workers=4)
    with open(save_path, 'wb') as f:
        pickle.dump(w2v, f)
    print(f" تم حفظ نموذج Word2Vec في: {save_path}")


def save_bert_model(save_path):
    print(" تحميل نموذج BERT...")
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertModel.from_pretrained('bert-base-uncased')
    with open(save_path, 'wb') as f:
        pickle.dump((tokenizer, model), f)
    print(f" تم حفظ نموذج BERT في: {save_path}")


def main():
    base = os.path.join(os.path.dirname(__file__), '..', 'data')

    for ds in ['trec', 'msmarco']:
        print(f"\n📁 معالجة المجموعة: {ds.upper()}")
        json_path = os.path.join(base, f"{ds}_docs.json")

        if not os.path.exists(json_path):
            print(f" الملف غير موجود: {json_path}")
            continue

        with open(json_path, encoding='utf-8') as f:
            docs = json.load(f)
        token_lists = [doc['tokens'] for doc in docs]

        # Word2Vec
        w2v_path = os.path.join(base, f"{ds}_word2vec.pkl")
        build_word2vec_model(token_lists, w2v_path)

        # BERT
        bert_path = os.path.join(base, f"{ds}_bert.pkl")
        save_bert_model(bert_path)


if __name__ == '__main__':
    main()
