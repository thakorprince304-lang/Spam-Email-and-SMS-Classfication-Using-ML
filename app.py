import pandas as pd
import numpy as np
import pickle, re, string
import nltk

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import ComplementNB
from sklearn.pipeline import Pipeline
from sklearn.metrics import (accuracy_score, classification_report,
                               confusion_matrix, precision_score,
                               recall_score, f1_score)
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

nltk.download('stopwords', quiet=True)
nltk.download('punkt', quiet=True)

# ── 1. Load Dataset ──────────────────────────────────────────
df = pd.read_csv('spam.csv', encoding='latin-1')
df = df[['v1', 'v2']].rename(columns={'v1': 'label', 'v2': 'text'})
df['label_num'] = df['label'].map({'ham': 0, 'spam': 1})
print(df['label'].value_counts())

# ── 2. Feature Engineering ───────────────────────────────────
def extract_features(df):
    df = df.copy()
    df['num_chars']     = df['text'].apply(len)
    df['num_words']     = df['text'].apply(lambda x: len(x.split()))
    df['num_sentences'] = df['text'].apply(lambda x: len(nltk.sent_tokenize(x)))
    df['num_upper']     = df['text'].apply(lambda x: sum(1 for c in x if c.isupper()))
    df['upper_ratio']   = df['num_upper'] / (df['num_chars'] + 1)
    df['num_special']   = df['text'].apply(lambda x: len([c for c in x if c in '$!?%#@']))
    df['has_url']       = df['text'].apply(lambda x: 1 if 'http' in x.lower() else 0)
    return df

df = extract_features(df)
print(df.groupby('label')[['num_chars','num_words','upper_ratio']].mean().round(2))

# ── 3. Preprocess Text ───────────────────────────────────────
ps         = PorterStemmer()
stop_words = set(stopwords.words('english'))

def preprocess(text):
    text = text.lower()
    text = re.sub(r'http\S+', 'URL', text)
    text = re.sub(r'\d+', 'NUM', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    tokens = [ps.stem(w) for w in text.split() if w not in stop_words]
    return ' '.join(tokens)

df['processed'] = df['text'].apply(preprocess)

# ── 4. Train / Test Split ────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    df['processed'], df['label_num'],
    test_size=0.2, random_state=42, stratify=df['label_num']
)

# ── 5. Build Pipeline ────────────────────────────────────────
pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(
        max_features=5000,
        ngram_range=(1, 2),
        sublinear_tf=True
    )),
    ('model', ComplementNB(alpha=0.1))
])
pipeline.fit(X_train, y_train)

# ── 6. Evaluate ──────────────────────────────────────────────
y_pred = pipeline.predict(X_test)
print(f"Accuracy : {accuracy_score(y_test, y_pred)*100:.2f}%")
print(f"Precision: {precision_score(y_test, y_pred)*100:.2f}%")
print(f"Recall   : {recall_score(y_test, y_pred)*100:.2f}%")
print(f"F1 Score : {f1_score(y_test, y_pred)*100:.2f}%")
print(classification_report(y_test, y_pred, target_names=['Ham', 'Spam']))

# ── 7. Save Model ────────────────────────────────────────────
with open('spam_model.pkl', 'wb') as f:
    pickle.dump(pipeline, f)
print("Model saved → spam_model.pkl")

# ── 8. Predict Function ──────────────────────────────────────
def predict(message):
    processed = preprocess(message)
    proba     = pipeline.predict_proba([processed])[0]
    label     = 'SPAM' if proba[1] > 0.5 else 'HAM'
    return {'label': label, 'confidence': round(max(proba)*100,2),
            'spam_prob': round(proba[1]*100,2)}

print(predict("Congratulations! You won a FREE iPhone. Click here NOW!"))
print(predict("Hey, are you coming to class tomorrow?"))
