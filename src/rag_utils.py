# src/rag_utils.py

from transformers import pipeline

# يمكنكِ استبداله بنموذج أقوى لاحقًا
qa_pipeline = pipeline("question-answering", model="distilbert-base-cased-distilled-squad")

def generate_answer(query, docs):
    # نجمع النصوص من أفضل المستندات المسترجعة
    context = " ".join([doc['text'] for doc in docs if doc.get('text')])
    result = qa_pipeline(question=query, context=context)
    return result['answer']
