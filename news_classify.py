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

        print(f"Total articles fetched for {topic}: {len(feed.entries)}")
        for entry in feed.entries:
            published_date = datetime.strptime(entry.published, "%a, %d %b %Y %H:%M:%S %Z")
            print(f"Article published date: {published_date}")  # Debugging line

            # Check if the news is within the range of 'days' back from today
            if published_date >= target_date:
                text_for_classification = (entry.title or "") + " " + (entry.summary or "")
                text_for_classification = preprocess_text(text_for_classification)
                predicted_category = model.predict([text_for_classification])[0]
                
                if predicted_category in topics:
                    categorized_news[predicted_category].append({
                        'title': entry.title,
                        'summary': entry.summary,
                        'published': published_date.strftime("%Y-%m-%d %H:%M:%S"),
                        'link': entry.link
                    })
            else:
                print(f"Skipping article. Published on {published_date}, older than target date: {target_date}")

    return categorized_news
# Streamlit app
st.markdown("<h1 style='text-align: center; color: #4CAF50;'>News Classification Dashboard</h1>", unsafe_allow_html=True)

# User input for topics
topics = st.multiselect(
    "Select Topics:",
    options=["business", "health", "entertainment", "tech", "politics", "sports"],
    default=["business"],
    help="Choose the topics you'd like to fetch news for"
)

# Input for how many days back to fetch news
days = st.number_input("Days Back:", min_value=1, value=1)

# Add Fetch News button with improved style
fetch_button = st.button("Fetch News", key='fetch_button')

# Button styling
if fetch_button:
    if not topics:
        st.warning("Please select at least one topic.")
    else:
        news = fetch_and_classify_news(topics, days)
        if not news:
            st.info("No news available for the selected topics.")
        else:
            # Display the news for each category
            for category in topics:
                if category in news:
                    st.markdown(f"<h2 style='color: #1E90FF;'>{category.upper()} News</h2>", unsafe_allow_html=True)
                    for article in news[category]:
                        st.markdown(f"**Title:** {article['title']} [Read more]({article['link']})", unsafe_allow_html=True)
                        st.markdown(f"**Published:** {article['published']}")
                    st.markdown("<hr>", unsafe_allow_html=True)
                else:
                    st.info(f"No news available for the {category} category.")

# Styling the sidebar
st.sidebar.markdown("<h2 style='color: #FFD700;'>Options</h2>", unsafe_allow_html=True)
st.sidebar.markdown("This app classifies and displays news articles fetched from Google News based on selected topics.")

# Adding a footer with styling
st.markdown("<footer style='text-align: center; color: grey;'>Powered by Streamlit</footer>", unsafe_allow_html=True)
