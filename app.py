from flask import Flask, render_template, request, jsonify
import numpy as np
import re
from gensim.models import Word2Vec

app = Flask(__name__)
print("Loading Word2Vec model...")
word2vec_model = Word2Vec.load("model/word2vec_review.model")

print("Loading model parameters...")
params = np.load("model/model_params.npz")
wei1 = params["wei1"]
wei2 = params["wei2"]
bai1 = params["bai1"]
bai2 = params["bai2"]
def clean_text(text):
    text = text.lower()
    text = re.sub(r"[^\w\s]", "", text)
    return text
def get_review_vector(text, model):
    words = clean_text(text).split()
    vectors = [model.wv[word] for word in words if word in model.wv]
    return np.mean(vectors, axis=0) if vectors else np.zeros(model.vector_size)
def sigmoid(x):
    return 1 / (1 + np.exp(-x))
def predict_sentiment(review_text):
    vec = get_review_vector(review_text, word2vec_model).reshape(1, -1)
    hidden = np.dot(vec, wei1.T) + bai1
    hidden_act = np.tanh(hidden)
    out = np.dot(hidden_act, wei2.T) + bai2
    out = np.clip(out, -100, 100)
    pred = sigmoid(out)
    
    return pred[0][0]

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    review = request.form['review']
    try:
        prediction_score = predict_sentiment(review)
        sentiment = "Positive" if prediction_score >= 0.5 else "Negative"
        confidence = f"{prediction_score:.2%}"
        
        return render_template('index.html', 
                            review=review,
                            sentiment=sentiment,
                            confidence=confidence,
                            score=prediction_score)
    except Exception as e:
        return render_template('index.html', 
                            error=str(e),
                            review=review)

if __name__ == '__main__':
    app.run(debug=True)