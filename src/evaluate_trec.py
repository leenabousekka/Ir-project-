# src/evaluate_trec.py

import os
import json
import ir_datasets
import pytrec_eval

# تحميل qrels من مجموعة TREC-COVID
print("📥 Loading qrels for TREC-COVID...")
dataset = ir_datasets.load("beir/trec-covid")
qrels = dataset.qrels_dict()

# تحميل نتائج البحث المحفوظة (run.json)
base_dir = os.path.join(os.path.dirname(__file__), '..', 'data')
with open(os.path.join(base_dir, 'trec_run.json'), 'r', encoding='utf-8') as f:
    run = json.load(f)

# تحويل القيم إلى float
for q_id in run:
    for doc_id in run[q_id]:
        run[q_id][doc_id] = float(run[q_id][doc_id])

# إعداد المقيم (بدون recall لتفادي الخطأ)
evaluator = pytrec_eval.RelevanceEvaluator(
    qrels,
    {'map', 'recip_rank', 'P_10'}
)

# التقييم
results = evaluator.evaluate(run)

# حساب المتوسطات
metrics = {'map': 0, 'recip_rank': 0, 'P_10': 0}
for query_id in results:
    for metric in metrics:
        metrics[metric] += results[query_id][metric]

num_queries = len(results)
print("\n📊 Evaluation Metrics for TREC-COVID:")
for metric in metrics:
    value = metrics[metric] / num_queries if num_queries > 0 else 0
    print(f"{metric}: {value:.4f}")
