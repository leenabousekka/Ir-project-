# src/evaluate.py

import os
import json
import ir_datasets
import pytrec_eval
import sys

# ✅ تحديد مجموعة البيانات وتمثيل البحث
if len(sys.argv) < 3:
    print("⚠️ يرجى تمرير اسم مجموعة البيانات وطريقة التمثيل: trec/msmarco tfidf/bm25/w2v/bert/hybrid")
    sys.exit(1)

dataset_name = sys.argv[1].lower()
method_name = sys.argv[2].lower()

if dataset_name not in ['trec', 'msmarco']:
    print("⚠️ اسم مجموعة البيانات غير صحيح. استخدم 'trec' أو 'msmarco'")
    sys.exit(1)

if method_name not in ['tfidf', 'bm25', 'word2vec', 'bert', 'hybrid', 'rag']:
    print("⚠️ اسم طريقة التمثيل غير صحيح. استخدم tfidf/bm25/word2vec/bert/hybrid/rag")
    sys.exit(1)

# تحميل qrels الرسمية
print(f"📥 Loading qrels for {dataset_name.upper()}...")
if dataset_name == 'trec':
    dataset = ir_datasets.load("beir/trec-covid")
else:
    dataset = ir_datasets.load("msmarco-passage/train")
qrels = dataset.qrels_dict()

# تحميل ملف run.json المناسب
base_dir = os.path.join(os.path.dirname(__file__), '..', 'data')
run_file = os.path.join(base_dir, f'{dataset_name}_{method_name}_run.json')

if not os.path.exists(run_file):
    print(f"❌ لم يتم العثور على {run_file}. الرجاء توليده أولاً باستخدام generate_run.py")
    sys.exit(1)

with open(run_file, 'r', encoding='utf-8') as f:
    run = json.load(f)

# تحويل القيم إلى float
for q_id in run:
    for doc_id in run[q_id]:
        run[q_id][doc_id] = float(run[q_id][doc_id])

# إعداد المقيم
evaluator = pytrec_eval.RelevanceEvaluator(
    qrels, {"map", "recip_rank", "P_10"}
)

results = evaluator.evaluate(run)

# حساب المتوسطات
metrics = {'map': 0, 'recip_rank': 0, 'P_10': 0}
for query_id in results:
    for metric in metrics:
        metrics[metric] += results[query_id][metric]

num_queries = len(results)
print(f"\n✅ Evaluated {num_queries} queries from {dataset_name.upper()} using {method_name.upper()}")
for metric in metrics:
    value = metrics[metric] / num_queries if num_queries > 0 else 0
    print(f"📊 {metric}: {value:.4f}")
