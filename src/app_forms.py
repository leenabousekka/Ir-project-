from flask_wtf import FlaskForm
from wtforms import StringField, SubmitField, SelectField
from wtforms.validators import DataRequired

class SearchForm(FlaskForm):
    query = StringField('أدخل استعلامك', validators=[DataRequired()])
    dataset = SelectField('اختر مجموعة البيانات', choices=[('trec', 'TREC-COVID'), ('msmarco', 'MSMARCO')])
    method = SelectField('اختر طريقة التمثيل', choices=[
        ('tfidf', 'TF‑IDF'),
        ('bm25', 'BM25'),
        ('word2vec', 'Word2Vec'),
        ('bert', 'BERT'),
        ('hybrid', 'Hybrid (Parallel)'),
        ('rag', 'RAG (Retrieval + Generation)')
    ])
    cluster = SelectField('هل ترغب بتجميع النتائج؟', choices=[('no', 'لا'), ('yes', 'نعم')])
    submit = SubmitField('بحث')
