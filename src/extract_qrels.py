import ir_datasets
import os

output_dir = os.path.join(os.path.dirname(__file__), '..', 'evaluation')
os.makedirs(output_dir, exist_ok=True)

def write_qrels(dataset_name, dataset, filename):
    print(f"✅ Processing qrels for: {dataset_name}")
    qrels = dataset.qrels_iter()
    qrels_path = os.path.join(output_dir, filename)

    with open(qrels_path, 'w', encoding='utf-8') as f:
        for qrel in qrels:
            # صيغة qrels المطلوبة هي: query-id  Q0  doc-id  relevance
            f.write(f"{qrel.query_id} Q0 {qrel.doc_id} {qrel.relevance}\n")

    print(f"✅ Saved qrels to {qrels_path}")


# MSMARCO
msmarco = ir_datasets.load("msmarco-passage/train")
write_qrels("MSMARCO", msmarco, "qrels_msmarco.txt")

# TREC-COVID
trec = ir_datasets.load("beir/trec-covid")
write_qrels("TREC-COVID", trec, "qrels_trec.txt")
