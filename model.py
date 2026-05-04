from flask import Flask, request, jsonify
import pickle, re, string, nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

app        = Flask(__name__)
ps         = PorterStemmer()
stop_words = set(stopwords.words('english'))

with open('spam_model.pkl', 'rb') as f:
    model = pickle.load(f)

def preprocess(text):
    text = text.lower()
    text = re.sub(r'http\S+', 'URL', text)
    text = re.sub(r'\d+', 'NUM', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    tokens = [ps.stem(w) for w in text.split() if w not in stop_words]
    return ' '.join(tokens)

@app.route('/predict', methods=['POST'])
def predict():
    data    = request.get_json()
    message = data.get('message', '')
    if not message.strip():
        return jsonify({'error': 'Empty message'}), 400

    processed = preprocess(message)
    proba     = model.predict_proba([processed])[0]
    label     = 'spam' if proba[1] > 0.5 else 'ham'

    features = {
        'num_words'    : len(message.split()),
        'num_chars'    : len(message),
        'upper_ratio'  : round(sum(1 for c in message if c.isupper()) / len(message), 3),
        'num_special'  : len([c for c in message if c in '$!?%#@']),
        'has_url'      : 'http' in message.lower()
    }

    return jsonify({
        'label'     : label,
        'spam_prob' : round(proba[1] * 100, 2),
        'ham_prob'  : round(proba[0] * 100, 2),
        'confidence': round(max(proba) * 100, 2),
        'features'  : features
    })

# Example usage:
# curl -X POST http://localhost:5000/predict \
#      -H "Content-Type: application/json" \
#      -d '{"message": "FREE prize winner click now!!"}'

if __name__ == '__main__':
    app.run(debug=True, port=5000)
