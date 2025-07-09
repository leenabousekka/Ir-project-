import os
from collections import defaultdict
from sklearn.metrics import precision_score, recall_score

# المسارات
base_dir = os.path.dirname(__file__)
data_dir = os.path.join(base_dir, '..', 'data')
eval_dir = os.path.join(base_dir, '..', 'evaluation')

results_file = os.path.join(data_dir, 'results.txt')

# 📌 تحديد مجموعة البيانات من أول استعلام في results.txt
dataset_name = "trec"  # القيمة الافتراضية

with open(results_file, encoding='utf-8') as f:
    for line in f:
        first_query = line.strip().split()[0]
        if first_query.isdigit():
            dataset_name = "msmarco"
        else:
            dataset_name = "trec"
        break

# اختيار ملف qrels المناسب
if dataset_name == "msmarco":
    qrels_file = os.path.join(eval_dir, 'qrels_marco.txt')
else:
    qrels_file = os.path.join(eval_dir, 'qrels_trec.txt')

# قراءة qrels
qrels = defaultdict(set)
with open(qrels_file, 'r', encoding='utf-8') as f:
    for line in f:
        parts = line.strip().split()
        if len(parts) != 4:
            continue
        qid, _, doc_id, rel = parts
        if int(rel) > 0:
            qrels[qid].add(doc_id)

# قراءة النتائج
results = defaultdict(list)
with open(results_file, 'r', encoding='utf-8') as f:
    for line in f:
        parts = line.strip().split()
        if len(parts) < 6:
            continue
        query_id = parts[0]
        doc_id = parts[2]
        score = float(parts[4])
        results[query_id].append((doc_id, score))

# التقييم
all_precisions = []
all_recalls = []
average_precisions = []
reciprocal_ranks = []
matched_queries = 0

for query_id, retrieved_docs in results.items():
    predicted = [doc_id for doc_id, _ in sorted(retrieved_docs, key=lambda x: -x[1])]
    relevant = qrels.get(query_id, set())

    if not relevant:
        continue

    matched_queries += 1
    y_true = [1 if doc in relevant else 0 for doc in predicted]
    y_pred = [1] * len(predicted)

    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)

    all_precisions.append(precision)
    all_recalls.append(recall)

    # Average Precision (AP)
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

# الطباعة
if matched_queries > 0:
    print(f"✅ Evaluated {matched_queries} matched queries from {dataset_name.upper()}")
    print(f"📊 Average Precision: {sum(all_precisions)/matched_queries:.4f}")
    print(f"📊 Average Recall: {sum(all_recalls)/matched_queries:.4f}")
    print(f"📊 MAP (Mean Average Precision): {sum(average_precisions)/matched_queries:.4f}")
    print(f"📊 MRR (Mean Reciprocal Rank): {sum(reciprocal_ranks)/matched_queries:.4f}")
else:
    print("⚠️ لم يتم العثور على تطابق بين الاستعلامات و qrels. تأكد من التنسيق الصحيح.")
