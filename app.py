from flask import Flask, request, jsonify, render_template
import pickle
import pandas as pd
import re
from nltk.corpus import stopwords

app = Flask(__name__)

# Load the model and vectorizer
with open('random_forest_model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

with open('tfidf_vectorizer.pkl', 'rb') as vectorizer_file:
    vectorizer = pickle.load(vectorizer_file)

stop_words = set(stopwords.words('english'))

def clean_text(text):
    text = re.sub(r'@\w+|#\w+|https?://(?:www\.)?[^\s/$.?#].[^\s]*', '', text)  # Remove mentions, hashtags, and URLs
    text = re.sub(r"[^a-zA-Z0-9\s]", '', text)  # Remove non-alphanumeric characters
    text = ' '.join([word for word in text.split() if word.lower() not in stop_words])
    return text.strip().lower()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    tweet = data['tweet']
    clean_tweet = clean_text(tweet)
    vectorized_text = vectorizer.transform([clean_tweet]).toarray()
    prediction = model.predict(vectorized_text)[0]
    
    sentiment_mapping = {0: 'Negative', 1: 'Neutral', 2: 'Positive'}
    sentiment_label = sentiment_mapping.get(prediction, 'Unknown')
    
    return jsonify({'sentiment': sentiment_label})


