import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
import nltk
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE
import pickle
from nltk.corpus import stopwords
nltk.download('stopwords')


"""Loading and Analyzing"""


data = pd.read_csv("AirlineTwitterData.csv", encoding = "ISO-8859-1")

df = data[['text', 'airline_sentiment']].copy()

df.rename(columns = {"text" : "tweet", "airline_sentiment" : "sentiment"}, inplace = True)


"""Data Preprocessing"""


le = LabelEncoder()
df.sentiment = le.fit_transform(df.sentiment)

label_mapping = dict(zip(le.classes_, range(len(le.classes_))))

stop_words = set(stopwords.words('english'))

df['clean_tweet'] = df['tweet'].apply(lambda x : ' '.join([word for word in x.split() if word.lower() not in stop_words]))

def clean_text(text):
    text = re.sub(r'@\w+|#\w+|https?://(?:www\.)?[^\s/$.?#].[^\s]*', '', text)  # Remove mentions, hashtags, and URLs
    text = re.sub(r"[^a-zA-Z0-9\s]", '', text)  # Remove non-alphanumeric characters
    return text.strip().lower()  # Strip whitespace and convert to lowercase

# Apply the cleaning function to the DataFrame
df['clean_tweet'] = df['clean_tweet'].apply(clean_text)

df.drop('tweet', axis = 1, inplace = True)

df.head()

Tfidf = TfidfVectorizer(max_features=10000, ngram_range=(1, 3))
X = Tfidf.fit_transform(df.clean_tweet).toarray()
y = df['sentiment']

# Handling class imbalance using SMOTE
smote = SMOTE(random_state=42)
X_res, y_res = smote.fit_resample(X, y)


"""Model Building"""


X_test, X_train, y_test, y_train = train_test_split(X_res,y_res, test_size = 0.3, random_state = 42)


"""Random Forest Classifier"""


rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
y_pred = rf.predict(X_test)

# Save the trained model
with open('random_forest_model.pkl', 'wb') as model_file:
    pickle.dump(rf, model_file)

# Save the TfidfVectorizer
with open('tfidf_vectorizer.pkl', 'wb') as vectorizer_file:
    pickle.dump(Tfidf, vectorizer_file)


print(classification_report(y_pred, y_test))
