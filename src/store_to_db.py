# src/store_to_db.py

import sqlite3
import os
import json

data_dir = os.path.join(os.path.dirname(__file__), '..', 'data')
db_path = os.path.join(data_dir, 'documents.db')

def create_table(cursor, name):
    cursor.execute(f'''
        CREATE TABLE IF NOT EXISTS {name} (
            id TEXT PRIMARY KEY,
            text TEXT,
            tokens TEXT
        )
    ''')

def insert_docs(cursor, name, docs):
    for doc in docs:
        doc_id = doc.get('id') or doc.get('doc_id')
        text = doc.get('text', '')
        tokens = ' '.join(doc.get('tokens', []))
        cursor.execute(f'''
            INSERT OR REPLACE INTO {name} (id, text, tokens)
            VALUES (?, ?, ?)
        ''', (doc_id, text, tokens))

def load_and_store(dataset_name, json_file):
    with open(os.path.join(data_dir, json_file), encoding='utf-8') as f:
        docs = json.load(f)

    with sqlite3.connect(db_path) as conn:
        cursor = conn.cursor()
        create_table(cursor, dataset_name)
        insert_docs(cursor, dataset_name, docs)
        conn.commit()
        print(f"✅ Stored {len(docs)} docs in table '{dataset_name}'.")

# datasets to load
datasets = {
    'trec': 'trec_docs.json',
    'msmarco': 'msmarco_docs.json'
}

for name, file in datasets.items():
    load_and_store(name, file)
