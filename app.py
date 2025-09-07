import streamlit as st
from transformers import pipeline
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import pandas as pd
from datetime import datetime, timedelta
import plotly.express as px
import plotly.graph_objects as go
from wordcloud import WordCloud
import base64
from io import BytesIO
import nltk
from textblob import TextBlob
import praw
from googleapiclient.discovery import build
import os
import time
from functools import lru_cache
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline

# --------------------------
# Initial Setup
# --------------------------
st.set_page_config(
    page_title="🚀 SentimentSync Pro",
    page_icon="📈",
    layout="wide")

# --------------------------
# Performance Optimizations
# --------------------------
@st.cache_resourcedef load_models():
    """Load models with progress indicators"""
    progress = st.progress(0, text="Loading sentiment models...")

    try:
        with st.spinner("Loading BERT model..."):
            bert_sentiment = pipeline(
                "sentiment-analysis",
                model="nlptown/bert-base-multilingual-uncased-sentiment"
            )
        progress.progress(50)

        with st.spinner("Loading VADER analyzer..."):
            vader_analyzer = SentimentIntensityAnalyzer()
        progress.progress(100)

        return bert_sentiment, vader_analyzer
    except Exception as e:
        st.error(f"Model loading failed: {str(e)}")
        return None, None

@st.cache_resourcedef setup_api_clients():
    """Initialize API clients with error handling"""
    try:
        with st.spinner("Initializing Reddit API..."):
            reddit = praw.Reddit(
                client_id=st.secrets["reddit_client_id"],
                client_secret=st.secrets["reddit_client_secret"],
                user_agent="SentimentSync/1.0"
            )

        with st.spinner("Initializing YouTube API..."):
            youtube = build('youtube', 'v3', developerKey=st.secrets["youtube_api_key"])

        return reddit, youtube
    except Exception as e:
        st.error(f"API initialization failed: {str(e)}")
        return None, None

# --------------------------
# Core Functions
# --------------------------
def analyze_text(text, models):
    """Optimized text analysis with batch processing"""
    bert_sentiment, vader_analyzer = models

    # Truncate very long texts to improve performance
    truncated_text = text[:2000] if text else ""

    try:
        if not truncated_text.strip():
            return {
                'vader': 0,
                'bert': 0,
                'textblob': 0,
                'bert_label': 'Neutral',
                'bert_confidence': 0
            }

        vader_score = vader_analyzer.polarity_scores(truncated_text)['compound']
        textblob_score = TextBlob(truncated_text).sentiment.polarity

        bert_result = bert_sentiment(truncated_text[:512])[0]  # BERT 512 token limit

        label_map = {
            '1 star': -1,
            '2 stars': -0.5,
            '3 stars': 0,
            '4 stars': 0.5,
            '5 stars': 1
        }
        bert_num = label_map.get(bert_result['label'], 0)

        return {
            'vader': vader_score,
            'bert': bert_num,
            'textblob': textblob_score,
            'bert_label': bert_result['label'],
            'bert_confidence': bert_result['score']
        }
    except Exception as e:
        st.error(f"Analysis error: {str(e)}")
        return {
            'vader': 0,
            'bert': 0,
            'textblob': 0,
            'bert_label': 'Error',
            'bert_confidence': 0
        }

@st.cache_data(ttl=3600, show_spinner="Fetching data...")
def fetch_reddit_data(keyword, limit=30):
    """Optimized Reddit data fetching"""
    try:
        reddit, _ = setup_api_clients()
        if not reddit:
            return pd.DataFrame()

        posts = list(reddit.subreddit("all").search(keyword, limit=limit))

        return pd.DataFrame([{
            'date': datetime.fromtimestamp(post.created_utc),
            'text': f"{post.title}\n{post.selftext}",
            'source': 'Reddit',
            'url': f"https://reddit.com{post.permalink}"
        } for post in posts])

    except Exception as e:
        st.error(f"Reddit fetch error: {str(e)}")
        return pd.DataFrame()

@st.cache_data(ttl=3600, show_spinner="Fetching data...")
def fetch_youtube_data(keyword, limit=30):
    """Optimized YouTube data fetching"""
    try:
        _, youtube = setup_api_clients()
        if not youtube:
            return pd.DataFrame()

        response = youtube.search().list(
            q=keyword,
            part="snippet",
            maxResults=limit,
            type="video",
            order="relevance"
        ).execute()

        return pd.DataFrame([{
            'date': datetime.strptime(item['snippet']['publishedAt'], '%Y-%m-%dT%H:%M:%SZ'),
            'text': f"{item['snippet']['title']}\n{item['snippet']['description']}",
            'source': 'YouTube',
            'url': f"https://youtube.com/watch?v={item['id']['videoId']}"
        } for item in response['items']])

    except Exception as e:
        st.error(f"YouTube fetch error: {str(e)}")
        return pd.DataFrame()

# --------------------------
# Visualization Functions
# --------------------------
def generate_wordcloud(text):
    """Fast word cloud generation"""
    try:
        if not text.strip():
            return ""

        wordcloud = WordCloud(
            width=800,
            height=400,
            background_color='white',
            collocations=False,  # Faster processing
            stopwords=nltk.corpus.stopwords.words('english')
        ).generate(text)

        img = BytesIO()
        wordcloud.to_image().save(img, format='PNG')
        return base64.b64encode(img.getvalue()).decode()
    except Exception as e:
        st.error(f"Word cloud generation error: {str(e)}")
        return ""

# --------------------------
# Prediction Functions
# --------------------------
def prepare_data_for_prediction(data):
    """Prepare time series data for prediction, handling NaN values"""
    try:
        if data.empty:
            st.warning("No data available for prediction")
            return None

        # Ensure data is sorted by date
        data = data.sort_values('date')

        # Filter out rows with invalid sentiment scores
        data = data.dropna(subset=['average'])

        # Create daily aggregates
        daily_data = data.groupby(pd.Grouper(key='date', freq='D'))['average'].mean().reset_index()

        # Remove any remaining NaN values from aggregation
        daily_data = daily_data.dropna(subset=['average'])

        # Check if enough data points remain
        if len(daily_data) < 5:
            st.warning("Insufficient valid data points for prediction (minimum 5 required)")
            return None

        # Create numerical features (days since first date)
        daily_data['days'] = (daily_data['date'] - daily_data['date'].min()).dt.days

        return daily_data
    except Exception as e:
        st.error(f"Data preparation error: {str(e)}")
        return None

def train_sentiment_model(data):
    """Train Ridge regression model, ensuring valid input"""
    try:
        if data is None:
            st.warning("No valid data for model training")
            return None, None

        # Verify sufficient data points
        if len(data) < 5:
            st.warning("Not enough data points for reliable prediction (minimum 5 required)")
            return None, None

        # Extract features and target
        X = data['days'].values.reshape(-1, 1)
        y = data['average'].values

        # Check for NaN values
        if np.any(np.isnan(X)) or np.any(np.isnan(y)):
            st.warning("Invalid values detected in data. Skipping prediction.")
            return None, None

        # Train polynomial Ridge regression
        model = make_pipeline(
            PolynomialFeatures(degree=2),
            Ridge(alpha=1.0)
        )

        model.fit(X, y)

        return model, data
    except Exception as e:
        st.error(f"Model training error: {str(e)}")
        return None, None

def predict_future_sentiment(model, training_data, days_to_predict=15):
    """Predict future sentiment using trained model"""
    try:
        if model is None or training_data is None:
            st.warning("No valid model or data for prediction")
            return None

        # Create future dates
        last_date = training_data['date'].max()
        future_dates = [last_date + timedelta(days=i) for i in range(1, days_to_predict+1)]

        # Create feature matrix for future dates
        min_date = training_data['date'].min()
        future_days = [(date - min_date).days for date in future_dates]
        X_future = np.array(future_days).reshape(-1, 1)

        # Make predictions
        predictions = model.predict(X_future)

        # Create prediction dataframe
        pred_df = pd.DataFrame({
            'date': future_dates,
            'average': predictions,
            'type': 'prediction'
        })

        # Add training data for plotting
        training_df = training_data.copy()
        training_df['type'] = 'actual'

        return pd.concat([training_df, pred_df], ignore_index=True)
    except Exception as e:
        st.error(f"Prediction error: {str(e)}")
        return None

def plot_sentiment(data, keyword):
    """Plot sentiment trends, handling missing data"""
    try:
        if data is None or data.empty:
            st.warning("No data available for plotting sentiment trends")
            return None

        # Separate actual and predicted data
        actual_data = data[data['type'] == 'actual']
        pred_data = data[data['type'] == 'prediction']

        fig = go.Figure()

        # Add actual data
        if not actual_data.empty:
            fig.add_trace(go.Scatter(
                x=actual_data['date'],
                y=actual_data['average'],
                name='Actual Sentiment',
                mode='lines+markers',
                line=dict(color='#636EFA')
            ))

        # Add predicted data if available
        if not pred_data.empty:
            fig.add_trace(go.Scatter(
                x=pred_data['date'],
                y=pred_data['average'],
                name='Predicted Sentiment',
                mode='lines+markers',
                line=dict(color='#EF553B', dash='dot')
            ))

            # Add confidence interval
            fig.add_trace(go.Scatter(
                x=pred_data['date'],
                y=pred_data['average'] + 0.1,
                mode='lines',
                line=dict(width=0),
                showlegend=False,
                hoverinfo='skip'
            ))

            fig.add_trace(go.Scatter(
                x=pred_data['date'],
                y=pred_data['average'] - 0.1,
                mode='lines',
                fill='tonexty',
                line=dict(width=0),
                fillcolor='rgba(239, 85, 59, 0.2)',
                name='Prediction Range'
            ))

        fig.update_layout(
            title=f'Sentiment Analysis and Prediction for "{keyword}"',
            xaxis_title="Date",
            yaxis_title="Sentiment Score",
            hovermode="x unified",
            legend_title="Data Type"
        )

        return fig
    except Exception as e:
        st.error(f"Plotting error: {str(e)}")
        return None

# --------------------------
# Main Application
# --------------------------
def main():
    st.title("🚀 SentimentSync Pro - Real-time Analysis Dashboard")

    # Sidebar controls
    with st.sidebar:
        st.header("🔧 Analysis Controls")
        analysis_mode = st.radio(
            "Mode",
            ["Text Analysis", "Live Data Analysis"],
            index=0
        )

        if analysis_mode == "Text Analysis":
            user_input = st.text_area(
                "Enter text to analyze",
                height=200,
                placeholder="Paste your content here..."
            )
            analyze_btn = st.button("Analyze Now")
        else:
            keyword = st.text_input(
                "Search keyword",
                placeholder="e.g., Apple, Tesla, etc."
            )
            analyze_btn = st.button("Fetch & Analyze")

        st.markdown("---")
        st.markdown("### Options")
        show_details = st.checkbox("Show detailed results", value=False)
        enable_prediction = st.checkbox("Enable sentiment prediction", value=True)
        st.markdown("---")

    # Main content
    if analyze_btn:
        models = load_models()
        if not all(models):
            st.error("Required models failed to load")
            return

        if analysis_mode == "Text Analysis":
            if not user_input.strip():
                st.warning("Please enter some text to analyze")
                return

            with st.spinner("Analyzing content..."):
                start_time = time.time()
                result = analyze_text(user_input, models)
                processing_time = time.time() - start_time

                st.success(f"Analysis completed in {processing_time:.2f} seconds")

                cols = st.columns(3)
                cols[0].metric("VADER Score", f"{result['vader']:.2f}",
                              "Positive" if result['vader'] > 0 else "Negative" if result['vader'] < 0 else "Neutral")
                cols[1].metric("BERT Sentiment", result['bert_label'], f"Confidence: {result['bert_confidence']:.2f}")
                cols[2].metric("TextBlob Score", f"{result['textblob']:.2f}",
                              "Positive" if result['textblob'] > 0 else "Negative" if result['textblob'] < 0 else "Neutral")

                st.subheader("📊 Text Visualization")
                wordcloud_img = f'data:image/png;base64,{generate_wordcloud(user_input)}'
                if wordcloud_img:
                    st.image(wordcloud_img, use_column_width=True)
                else:
                    st.info("No word cloud generated due to insufficient text")

        else:  # Live Data Analysis
            if not keyword.strip():
                st.warning("Please enter a search keyword")
                return

            with st.spinner(f"Gathering data for '{keyword}'..."):
                start_time = time.time()

                reddit_data = fetch_reddit_data(keyword)
                youtube_data = fetch_youtube_data(keyword)

                if reddit_data.empty and youtube_data.empty:
                    st.error("No data found. Try a different keyword.")
                    return

                combined_data = pd.concat([reddit_data, youtube_data], ignore_index=True)

                # Filter out empty or invalid texts
                combined_data = combined_data[combined_data['text'].str.strip() != '']

                # Analyze in batches
                analysis_results = []
                for _, row in combined_data.iterrows():
                    analysis_results.append(analyze_text(row['text'], models))

                # Add results to dataframe
                combined_data['vader'] = [r['vader'] for r in analysis_results]
                combined_data['bert'] = [r['bert'] for r in analysis_results]
                combined_data['textblob'] = [r['textblob'] for r in analysis_results]

                # Ensure no NaN values in sentiment scores
                combined_data = combined_data.dropna(subset=['vader', 'bert', 'textblob'])
                combined_data['average'] = combined_data[['vader', 'bert', 'textblob']].mean(axis=1)

                processing_time = time.time() - start_time
                st.success(f"Analyzed {len(combined_data)} sources in {processing_time:.2f} seconds")

                st.subheader(f"📈 Overall Sentiment for '{keyword}'")

                cols = st.columns(3)
                avg_sentiment = combined_data['average'].mean()
                pos_pct = (combined_data['average'] > 0.1).mean() * 100
                neg_pct = (combined_data['average'] < -0.1).mean() * 100

                cols[0].metric("Avg Sentiment", f"{avg_sentiment:.2f}",
                              "Positive" if avg_sentiment > 0 else "Negative" if avg_sentiment < 0 else "Neutral")
                cols[1].metric("Positive Content", f"{pos_pct:.1f}%")
                cols[2].metric("Negative Content", f"{neg_pct:.1f}%")

                st.subheader("📊 Content Visualization")
                all_text = " ".join(combined_data['text'])
                wordcloud_img = f'data:image/png;base64,{generate_wordcloud(all_text)}'
                if wordcloud_img:
                    st.image(wordcloud_img, use_column_width=True)
                else:
                    st.info("No word cloud generated due to insufficient text")

                # Filter recent data
                combined_data['date'] = pd.to_datetime(combined_data['date'])
                recent_data = combined_data[combined_data['date'] >= (datetime.now() - timedelta(days=60))]

                if not recent_data.empty:
                    st.subheader("📅 Sentiment Over Time")

                    if enable_prediction:
                        with st.spinner("Training prediction model..."):
                            daily_data = prepare_data_for_prediction(recent_data)
                            model, training_data = train_sentiment_model(daily_data)

                            if model is not None and training_data is not None:
                                full_data = predict_future_sentiment(model, training_data)
                                fig = plot_sentiment(full_data, keyword)
                            else:
                                daily_data = daily_data if daily_data is not None else recent_data[['date', 'average']].assign(type='actual')
                                fig = plot_sentiment(daily_data, keyword)
                    else:
                        daily_data = prepare_data_for_prediction(recent_data)
                        fig = plot_sentiment(daily_data.assign(type='actual') if daily_data is not None else recent_data[['date', 'average']].assign(type='actual'), keyword)

                    if fig:
                        st.plotly_chart(fig, use_container_width=True)

                    if enable_prediction and 'full_data' in locals() and full_data is not None:
                        last_actual = full_data[full_data['type'] == 'actual']['average'].iloc[-1]
                        last_pred = full_data[full_data['type'] == 'prediction']['average'].iloc[-1]

                        if last_pred > last_actual + 0.1:
                            st.success("📈 Prediction: Sentiment is expected to improve in the next 15 days")
                        elif last_pred < last_actual - 0.1:
                            st.warning("📉 Prediction: Sentiment is expected to decline in the next 15 days")
                        else:
                            st.info("📊 Prediction: Sentiment is expected to remain stable in the next 15 days")

                    if show_details:
                        st.subheader("🔍 Detailed Results")
                        st.dataframe(recent_data[['date', 'source', 'text', 'average']], use_container_width=True)
                else:
                    st.info("No recent data found (within last 60 days).")

if __name__ == "__main__":
    try:
        nltk.data.path.append(os.path.join(os.path.expanduser("~"), "nltk_data"))
        nltk.download('punkt', quiet=True)
        nltk.download('stopwords', quiet=True)
    except:
        pass

    main()
