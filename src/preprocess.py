import re
import nltk
#import os, json
from textblob import Word
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.stem import PorterStemmer
#import string
#nltk.download('wordnet')


stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()
stemmer = PorterStemmer()

def preprocess_text(text):  
    # lowercase
    text = text.lower()

    # remove num 
    text = re.sub(r'\d+', '', text)

    # remove علامات الترقيم 
    text = re.sub(r'[^\w\s]', '', text)

    
    text = text.strip()

    
    tokens = word_tokenize(text)

    
    tokens = [t for t in tokens if t not in stop_words]

    # 7. Lemmatization
    tokens = [lemmatizer.lemmatize(t) for t in tokens]
    cleaned = []
    for word in tokens:
        if word not in stop_words:
            corrected = Word(word).correct()      
            lemmatized = Word(corrected).lemmatize()
            stemmed = stemmer.stem(lemmatized)
            cleaned.append(stemmed)
    return cleaned
