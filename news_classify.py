import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import feedparser
import urllib.parse
import joblib
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import nltk
from datetime import datetime, timedelta
from collections import defaultdict
import streamlit as st

# Download stopwords if not already downloaded
nltk.download('stopwords')

# Preprocessing function to clean and normalize text data
def preprocess_text(text):
    text = text.lower()  # Lowercase
    text = re.sub(r'\W+', ' ', text)  # Remove non-word characters
    text = re.sub(r'\d+', '', text)  # Remove numbers
    text = re.sub(r'\s+', ' ', text).strip()  # Remove extra spaces
    stemmer = PorterStemmer()
    stop_words = set(stopwords.words('english'))
    text = ' '.join([stemmer.stem(word) for word in text.split() if word not in stop_words])  # Stemming and stopword removal
    return text

# Function to load dataset from an Excel file
def load_kaggle_dataset(file_path):
    df = pd.read_excel(file_path)
    return df

# Prepare data for training
def prepare_data(df):
    news_articles = []
    for index, row in df.iterrows():
        title = str(row['title']) if pd.notna(row['title']) else ""
        content = str(row['content']) if pd.notna(row['content']) else ""
        category = row['category'] if pd.notna(row['category']) else "Unknown"
        news_articles.append((title, content, category))
    X = [preprocess_text(article[0] + " " + article[1]) for article in news_articles]
    y = [article[2] for article in news_articles]
    return X, y

# Train and evaluate the model
def train_model(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    pipelines = {
        'SVM': Pipeline([
            ('tfidf', TfidfVectorizer(stop_words='english')),
            ('clf', LinearSVC())
        ]),
        'RandomForest': Pipeline([
            ('tfidf', TfidfVectorizer(stop_words='english')),
            ('clf', RandomForestClassifier())
        ])
    }

    param_grids = {
        'SVM': {
            'clf__C': [0.1, 1, 10, 100],
            'clf__loss': ['hinge', 'squared_hinge'],
            'clf__max_iter': [1000, 5000, 10000]
        },
        'RandomForest': {
            'clf__n_estimators': [100, 200, 500],
            'clf__max_depth': [10, 20, 50]
        }
    }

    best_model = None
    best_score = 0
    for model_name, pipeline in pipelines.items():
        grid_search = GridSearchCV(pipeline, param_grids[model_name], cv=5, n_jobs=-1, scoring='accuracy')
        grid_search.fit(X_train, y_train)
        score = grid_search.best_score_
        print(f"{model_name} Best Score: {score * 100:.2f}%")

        if score > best_score:
            best_score = score
            best_model = grid_search.best_estimator_

    joblib.dump(best_model, 'best_news_classifier_model.pkl')
    loaded_model = joblib.load('best_news_classifier_model.pkl')

    y_pred = loaded_model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Model Accuracy: {accuracy * 100:.2f}%")

    return best_model
def fetch_and_classify_news(topics, days=1):
    model = joblib.load('best_news_classifier_model.pkl')
    target_date = datetime.now() - timedelta(days=int(days))
    categorized_news = defaultdict(list)

    for topic in topics:
        encoded_topic = urllib.parse.quote(topic.strip())
        rss_url = f"https://news.google.com/rss/search?q={encoded_topic}"
        print(f"Fetching news from URL: {rss_url}")
        feed = feedparser.parse(rss_url)

        if not feed.entries:
            print(f"No entries found in RSS feed for topic '{topic}'.")
            continue

        for entry in feed.entries:
            published_date = datetime.strptime(entry.published, "%a, %d %b %Y %H:%M:%S %Z")
            if published_date >= target_date:
                text_for_classification = (entry.title or "") + " " + (entry.summary or "")
                text_for_classification = preprocess_text(text_for_classification)
                predicted_category = model.predict([text_for_classification])[0]
                
                # Only add the article if the predicted category is in the selected topics
                if predicted_category in topics:
                    categorized_news[predicted_category].append({
                        'title': entry.title,
                        'summary': entry.summary,
                        'published': published_date.strftime("%Y-%m-%d %H:%M:%S"),
                        'link': entry.link
                    })

    return categorized_news

# Streamlit app
st.title("News Classification Dashboard")

# User input for topics
topics = st.multiselect(
    "Select Topics:",
    options=["business", "health", "entertainment", "tech", "politics", "sports"],
    default=["business"]
)

days = st.number_input("Days Back:", min_value=1, value=1)

if st.button("Fetch News"):
    if not topics:
        st.warning("Please select at least one topic.")
    else:
        news = fetch_and_classify_news(topics, days)
        if not news:
            st.info("No news available for the selected topics.")
        else:
            # Only display the news for selected categories
            for category in topics:
                if category in news:
                    st.subheader(f"Category: {category.upper()}")
                    for article in news[category]:
                        st.write(f"**Title:** {article['title']}")
                        st.write(f"**Published:** {article['published']}")
                        st.write(f"[Read more]({article['link']})")
                    st.markdown("---")
                else:
                    st.info(f"No news available for the {category} category.")
