# Airline_TwitterSentimentAnalysis

This project focuses on analyzing sentiment within Twitter data related to various airlines. The growing importance of social media as a platform for customer feedback makes sentiment analysis a crucial tool for businesses to understand public perception and respond effectively. By leveraging machine learning techniques, this project aims to automatically classify the sentiment expressed in tweets about airlines as positive, negative, or neutral.

## Objective

The primary objective of this project is to develop a robust machine learning model that can accurately predict the sentiment of tweets related to airlines. This capability can be instrumental for airlines in monitoring their brand reputation, identifying areas of customer dissatisfaction, and making data-driven decisions to improve customer experience.

## Project Structure


- `nltk_data/` - Contains necessary NLTK stopwords for preprocessing.
- `static/` - Directory for static files (CSS).
- `templates/` - Directory for HTML templates.
- `AirlineTwitterData.csv` - Dataset containing Twitter data related to airlines.
- `Airline_TSA.ipynb` - Jupyter notebook with exploratory data analysis and model comparison.
- `airline_tsa.py` - Script to preprocess data and train the model.
- `app.py` - Main Flask application file.
- `random_forest_model.pkl` - Pre-trained Random Forest model.
- `tfidf_vectorizer.pkl` - TF-IDF vectorizer used for transforming the text data into matrices.
- `requirements.txt` - List of required Python packages.

## Getting Started

### Prerequisites

- Python 3.x
- pip

### Installation

1. **Clone the repository:**

   ```bash
   git clone https://github.com/your-username/airline-sentiment-analysis.git
   cd airline-sentiment-analysis
   ```

2. Create a virtual environment:

   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   ```

3. Install dependencies:
 
    ```bash
    pip install -r requirements.txt
    ```

### Running the Flask App

1. Run the Flask app:
   
    ```bash
    python app.py
    ```

2. Open the app in your browser:

  Go to `http://127.0.0.1:5000/`

### Contribution

If you would like to contribute to this project, feel free to fork the repository and submit a pull request.

