# Smart Book Recommendation System

## Project Description
A basic Python application that recommends books based on genre similarity.

## Objective
To help readers discover books similar to a selected book using machine-learning techniques.

## Technologies
- Python
- Pandas
- Scikit-learn
- Streamlit
- TF-IDF
- Cosine similarity

## How It Works
The app converts book genres into numerical features using TF-IDF, compares them with cosine similarity, and displays up to five similar books.

## How to Run
Install the required libraries:

pip install -r requirements.txt

Start the app:

streamlit run app.py

## Limitation
Recommendations are based on genre information and a small book collection.
