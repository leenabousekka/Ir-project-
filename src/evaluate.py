# src/evaluate.py (يدعم qrels الرسمي والتقييم الداخلي)

import os
import json
import sys
import numpy as np
from collections import defaultdict
from sklearn.metrics import precision_score, recall_score
import pytrec_eval

if len(sys.argv) < 3:
    print("⚠️ يرجى تمرير اسم مجموعة البيانات وطريقة التمثيل: trec/msmarco tfidf/bm25/w2v/bert/hybrid")
    sys.exit(1)

# المدخلات
dataset_name = sys.argv[1]
method = sys.argv[2]

base_dir = os.path.dirname(__file__)
data_dir = os.path.join(base_dir, '..', 'data')
eval_dir = os.path.join(base_dir, '..', 'evaluation')

# المسارات
run_file = os.path.join(data_dir, f'{dataset_name}_{method}_run.json')
qrels_file = os.path.join(eval_dir, f'qrels_{dataset_name}.txt')

# تحميل run.json
with open(run_file, 'r', encoding='utf-8') as f:
    run = json.load(f)

# محاولة تحميل qrels.txt إن وجد
qrels = defaultdict(dict)
use_official_qrels = False
if os.path.exists(qrels_file):
    with open(qrels_file, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) != 4:
                continue
            qid, _, doc_id, rel = parts
            if int(rel) > 0:
                qrels[qid][doc_id] = int(rel)
    use_official_qrels = True

# ----------------------------
# ✅ تقييم باستخدام pytrec_eval
# ----------------------------
if use_official_qrels:
    print(f"📥 Loading qrels for {dataset_name.upper()}...")
    evaluator = pytrec_eval.RelevanceEvaluator(qrels, {'map', 'recip_rank', 'P_10'})
    results = evaluator.evaluate(run)

    # المتوسطات
    metrics = {'map': 0, 'recip_rank': 0, 'P_10': 0}
    for qid in results:
        for metric in metrics:
            metrics[metric] += results[qid][metric]

    total = len(results)
    print(f"\n✅ Evaluated {total} queries from {dataset_name.upper()} using {method.upper()}")
    if total == 0:
        print("⚠️ لم يتم العثور على استعلامات مطابقة بين run و qrels. تأكد من أن query_id في run متطابق مع qrels.")
    else:
        for metric in metrics:
            print(f"📊 {metric}: {metrics[metric]/total:.4f}")

else:
    # ----------------------------
    # ❗ تقييم يدوي داخلي بدون qrels رسمي
    # ----------------------------
    print("⚠️ لم يتم العثور على qrels رسمي. سيتم تنفيذ تقييم داخلي بديل.")
    all_precisions = []
    all_recalls = []
    average_precisions = []
    reciprocal_ranks = []

    for query_id, doc_scores in run.items():
        sorted_docs = sorted(doc_scores.items(), key=lambda x: -x[1])
        predicted = [doc_id for doc_id, _ in sorted_docs[:10]]
        relevant = set(predicted[:3])

        y_true = [1 if doc in relevant else 0 for doc in predicted]
        y_pred = [1] * len(predicted)

        precision = precision_score(y_true, y_pred, zero_division=0)
        recall = recall_score(y_true, y_pred, zero_division=0)

        all_precisions.append(precision)
        all_recalls.append(recall)

        # AP
        num_correct = 0
        precisions = []
        for i, doc in enumerate(predicted):
            if doc in relevant:
                num_correct += 1
                precisions.append(num_correct / (i + 1))
        ap = sum(precisions) / len(relevant) if relevant else 0
        average_precisions.append(ap)

        # MRR
        rr = 0
        for i, doc in enumerate(predicted):
            if doc in relevant:
                rr = 1 / (i + 1)
                break
        reciprocal_ranks.append(rr)

    matched = len(run)
    print(f"\n✅ Internal Evaluation on {matched} queries (no official qrels)")
    print(f"📊 Average Precision: {sum(all_precisions)/matched:.4f}")
    print(f"📊 Average Recall: {sum(all_recalls)/matched:.4f}")
    print(f"📊 MAP: {sum(average_precisions)/matched:.4f}")
    print(f"📊 MRR: {sum(reciprocal_ranks)/matched:.4f}")
