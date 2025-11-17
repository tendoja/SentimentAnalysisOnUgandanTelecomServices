import streamlit as st
import pandas as pd # type: ignore
import numpy as np
import re
import json
from datetime import datetime, timedelta
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score # type: ignore
from sklearn.model_selection import cross_val_score
import joblib

# Page configuration
st.set_page_config(
    page_title="Ugandan Telecom Sentiment Analysis",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #1f77b4;
    }
    .positive { color: #2ecc71; }
    .negative { color: #e74c3c; }
    .neutral { color: #3498db; }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'data_loaded' not in st.session_state:
    st.session_state.data_loaded = False
if 'df' not in st.session_state:
    st.session_state.df = None
if 'ml_trained' not in st.session_state:
    st.session_state.ml_trained = False

# Define stopwords manually
manual_stopwords = {
    'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', "you're", 
    "you've", "you'll", "you'd", 'your', 'yours', 'yourself', 'yourselves', 'he', 
    'him', 'his', 'himself', 'she', "she's", 'her', 'hers', 'herself', 'it', 
    "it's", 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves', 
    'what', 'which', 'who', 'whom', 'this', 'that', "that'll", 'these', 'those', 
    'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 
    'having', 'do', 'does', 'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 
    'or', 'because', 'as', 'until', 'while', 'of', 'at', 'by', 'for', 'with', 
    'about', 'against', 'between', 'into', 'through', 'during', 'before', 'after', 
    'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 
    'under', 'again', 'further', 'then', 'once', 'here', 'there', 'when', 'where', 
    'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other', 
    'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 
    'very', 's', 't', 'can', 'will', 'just', 'don', "don't", 'should', "should've", 
    'now', 'd', 'll', 'm', 'o', 're', 've', 'y', 'ain', 'aren', "aren't", 'couldn', 
    "couldn't", 'didn', "didn't", 'doesn', "doesn't", 'hadn', "hadn't", 'hasn', 
    "hasn't", 'haven', "haven't", 'isn', "isn't", 'ma', 'mightn', "mightn't", 
    'mustn', "mustn't", 'needn', "needn't", 'shan', "shan't", 'shouldn', "shouldn't", 
    'wasn', "wasn't", 'weren', "weren't", 'won', "won't", 'wouldn', "wouldn't"
}

telecom_stopwords = {'mtn', 'airtel', 'uganda', 'ug', 'telco', 'network', 'service', 'mobile', 'please', 'thank', 'thanks'}
stop_words = manual_stopwords.union(telecom_stopwords)

# Simple lemmatization mapping
lemmatization_map = {
    'networks': 'network', 'services': 'service', 'payments': 'payment',
    'charges': 'charge', 'fees': 'fee', 'problems': 'problem', 'issues': 'issue',
    'improvements': 'improvement', 'customers': 'customer', 'providers': 'provider',
    'connections': 'connection', 'speeds': 'speed', 'packages': 'package',
    'promotions': 'promotion', 'bundles': 'bundle', 'transactions': 'transaction',
    'works': 'work', 'working': 'work', 'worked': 'work', 'helps': 'help',
    'helping': 'help', 'helped': 'help', 'uses': 'use', 'using': 'use', 'used': 'use'
}

def simple_lemmatize(word):
    """Simple lemmatizer using mapping"""
    return lemmatization_map.get(word, word)

# Initialize VADER sentiment analyzer
vader_analyzer = SentimentIntensityAnalyzer()

def collect_telco_tweets():
    """Collect telecom-related tweets for Ugandan context"""
    base_date = datetime(2024, 10, 20)
    sample_data = [
        # Negative sentiments
        {'content': 'MTN network in Kampala is terrible today! No internet for hours #MTNUganda', 'date': base_date, 'username': 'user1', 'sentiment': 'negative'},
        {'content': 'Airtel mobile money charges are too high! When will they reduce fees?', 'date': base_date - timedelta(days=1), 'username': 'user2', 'sentiment': 'negative'},
        {'content': 'Very frustrated with MTN customer service. Been on hold for 30 minutes!', 'date': base_date - timedelta(days=2), 'username': 'user3', 'sentiment': 'negative'},
        {'content': 'Airtel data bundles finish too quickly. Something is wrong with their billing', 'date': base_date - timedelta(days=3), 'username': 'user4', 'sentiment': 'negative'},
        {'content': 'MTN network down in Entebbe area. This is becoming too frequent @MTNUganda', 'date': base_date - timedelta(days=1), 'username': 'user5', 'sentiment': 'negative'},
        
        # Positive sentiments
        {'content': 'Great service from Airtel today! Their 4G network is super fast in Kampala', 'date': base_date - timedelta(days=2), 'username': 'user6', 'sentiment': 'positive'},
        {'content': 'MTN mobile money saved me today! Easy and quick transaction. #MoMo', 'date': base_date - timedelta(days=3), 'username': 'user7', 'sentiment': 'positive'},
        {'content': 'Airtel customer care was very helpful with my SIM swap. Thank you!', 'date': base_date - timedelta(days=1), 'username': 'user8', 'sentiment': 'positive'},
        {'content': 'MTN double data promo is amazing! Great value for money', 'date': base_date - timedelta(days=4), 'username': 'user9', 'sentiment': 'positive'},
        {'content': 'Airtel network improvement in Gulu is noticeable. Good work!', 'date': base_date - timedelta(days=2), 'username': 'user10', 'sentiment': 'positive'},
        
        # Neutral sentiments
        {'content': 'Just bought a new Airtel SIM card in Kampala', 'date': base_date - timedelta(days=1), 'username': 'user11', 'sentiment': 'neutral'},
        {'content': 'MTN is having a promotion on data bundles this week', 'date': base_date - timedelta(days=3), 'username': 'user12', 'sentiment': 'neutral'},
        {'content': 'Airtel Uganda announced new CEO today', 'date': base_date - timedelta(days=2), 'username': 'user13', 'sentiment': 'neutral'},
        {'content': 'Mobile money transactions increased by 20% according to UCC report', 'date': base_date - timedelta(days=4), 'username': 'user14', 'sentiment': 'neutral'},
    ]
    
    # Expand dataset
    expanded_data = sample_data.copy()
    telecom_issues = ['network', 'data', 'mobile money', 'customer service', 'pricing', 'coverage', 'signal', 'internet']
    locations = ['Kampala', 'Entebbe', 'Gulu', 'Mbarara', 'Jinja', 'Lira', 'Mbale', 'Fort Portal']
    
    for i in range(86):
        issue = np.random.choice(telecom_issues)
        location = np.random.choice(locations)
        telco = np.random.choice(['MTN', 'Airtel'])
        sentiment = np.random.choice(['positive', 'negative', 'neutral'], p=[0.3, 0.4, 0.3])
        
        if sentiment == 'negative':
            templates = [
                f"{telco} {issue} in {location} is terrible! Very disappointed!",
                f"Frustrated with {telco} {issue} in {location}. Needs improvement!",
                f"{telco} {issue} in {location} is not working properly. So annoying!",
                f"Bad experience with {telco} {issue} in {location}. Very poor service!"
            ]
            content = np.random.choice(templates)
        elif sentiment == 'positive':
            templates = [
                f"Great {issue} from {telco} in {location}! Excellent service!",
                f"Very happy with {telco} {issue} in {location}. Good work!",
                f"{telco} {issue} in {location} is amazing! Keep it up!",
                f"Excellent {issue} experience with {telco} in {location}. Thank you!"
            ]
            content = np.random.choice(templates)
        else:
            templates = [
                f"Using {telco} {issue} in {location}. Regular service.",
                f"{telco} {issue} in {location} is okay. Normal performance.",
                f"Checking {telco} {issue} in {location}. Nothing special.",
                f"{telco} {issue} service in {location}. Average experience."
            ]
            content = np.random.choice(templates)
            
        expanded_data.append({
            'content': content,
            'date': base_date - timedelta(days=np.random.randint(1, 30)),
            'username': f'user_{i+15}',
            'sentiment': sentiment
        })
    
    df = pd.DataFrame(expanded_data)
    return df

def clean_text(text):
    """Comprehensive text cleaning function"""
    if not isinstance(text, str):
        return ''
    
    # Convert to lowercase
    text = text.lower()
    
    # Remove URLs, mentions, and hashtags
    text = re.sub(r'http\S+|www\.\S+', '', text)
    text = re.sub(r'@\w+', '', text)
    text = re.sub(r'#(\w+)', r'\1', text)
    
    # Remove special characters and digits but keep words
    text = re.sub(r'[^a-zA-Z\s]', ' ', text)
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

def preprocess_text(text):
    """Text preprocessing with manual tokenization"""
    # Clean text first
    text = clean_text(text)
    
    # Simple word splitting (manual tokenization)
    words = text.split()
    
    # Remove stopwords, short words, and apply simple lemmatization
    processed_words = []
    for word in words:
        if word not in stop_words and len(word) > 2:
            processed_words.append(simple_lemmatize(word))
    
    return ' '.join(processed_words)

def get_vader_sentiment(text):
    """Get sentiment using VADER (optimized for social media)"""
    if not text.strip():
        return 'neutral'
    
    scores = vader_analyzer.polarity_scores(text)
    compound = scores['compound']
    
    if compound >= 0.05:
        return 'positive'
    elif compound <= -0.05:
        return 'negative'
    else:
        return 'neutral'

def get_top_words_by_sentiment(df, sentiment_category, n_words=8):
    """Get most frequent words for a given sentiment"""
    sentiment_texts = df[df['final_sentiment'] == sentiment_category]['processed_text']
    all_words = ' '.join(sentiment_texts).split()
    word_freq = Counter(all_words).most_common(n_words)
    return word_freq

# Main application
def main():
    st.markdown('<h1 class="main-header">📊 Ugandan Telecom Services Sentiment Analysis</h1>', unsafe_allow_html=True)
    
    # Sidebar
    st.sidebar.title("Navigation")
    app_mode = st.sidebar.selectbox("Choose Analysis Section", 
                                   ["Data Overview", "Sentiment Analysis", "Visualizations", "Business Insights"])
    
    # Load data
    if not st.session_state.data_loaded:
        with st.spinner("Loading and processing data..."):
            df = collect_telco_tweets()
            
            # Apply cleaning and preprocessing
            df['cleaned_text'] = df['content'].apply(clean_text)
            df['processed_text'] = df['content'].apply(preprocess_text)
            
            # Text length analysis
            df['text_length'] = df['cleaned_text'].apply(len)
            df['word_count'] = df['cleaned_text'].apply(lambda x: len(x.split()))
            
            # Company mentions
            df['mentions_mtn'] = df['cleaned_text'].str.contains('mtn')
            df['mentions_airtel'] = df['cleaned_text'].str.contains('airtel')
            
            # Apply sentiment analysis
            df['vader_sentiment'] = df['cleaned_text'].apply(get_vader_sentiment)
            df['final_sentiment'] = df['vader_sentiment']
            
            st.session_state.df = df
            st.session_state.data_loaded = True
    
    df = st.session_state.df
    
    if app_mode == "Data Overview":
        show_data_overview(df)
    elif app_mode == "Sentiment Analysis":
        show_sentiment_analysis(df)
    elif app_mode == "Visualizations":
        show_visualizations(df)
    elif app_mode == "Business Insights":
        show_business_insights(df)

def show_data_overview(df):
    st.header("📋 Data Overview")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Tweets", f"{len(df):,}")
    
    with col2:
        st.metric("Date Range", f"{df['date'].min().date()} to {df['date'].max().date()}")
    
    with col3:
        avg_chars = df['text_length'].mean()
        st.metric("Avg. Characters per Tweet", f"{avg_chars:.1f}")
    
    with col4:
        avg_words = df['word_count'].mean()
        st.metric("Avg. Words per Tweet", f"{avg_words:.1f}")
    
    # Company mentions
    st.subheader("🏢 Company Mentions")
    col1, col2 = st.columns(2)
    
    with col1:
        mtn_mentions = df['mentions_mtn'].sum()
        st.metric("MTN Mentions", f"{mtn_mentions:,}", f"{mtn_mentions/len(df)*100:.1f}%")
    
    with col2:
        airtel_mentions = df['mentions_airtel'].sum()
        st.metric("Airtel Mentions", f"{airtel_mentions:,}", f"{airtel_mentions/len(df)*100:.1f}%")
    
    # Show sample data
    st.subheader("Sample Data")
    st.dataframe(df[['content', 'cleaned_text', 'processed_text', 'sentiment']].head(10), use_container_width=True)
    
    # Text cleaning examples
    st.subheader("Text Cleaning Examples")
    for i in range(3):
        with st.expander(f"Example {i+1}"):
            st.write(f"**Original:** {df['content'].iloc[i]}")
            st.write(f"**Cleaned:** {df['cleaned_text'].iloc[i]}")
            st.write(f"**Processed:** {df['processed_text'].iloc[i]}")

def show_sentiment_analysis(df):
    st.header("🎭 Sentiment Analysis")
    
    # Run ML analysis if requested
    if st.button("Run Machine Learning Analysis") and not st.session_state.ml_trained:
        with st.spinner("Training machine learning model..."):
            run_ml_analysis(df)
    
    # Sentiment distribution
    st.subheader("Sentiment Distribution")
    
    sentiment_counts = df['final_sentiment'].value_counts()
    total = len(df)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        positive_pct = (df['final_sentiment'] == 'positive').mean() * 100
        st.metric("Positive", f"{sentiment_counts.get('positive', 0):,}", f"{positive_pct:.1f}%", delta_color="off")
    
    with col2:
        neutral_pct = (df['final_sentiment'] == 'neutral').mean() * 100
        st.metric("Neutral", f"{sentiment_counts.get('neutral', 0):,}", f"{neutral_pct:.1f}%", delta_color="off")
    
    with col3:
        negative_pct = (df['final_sentiment'] == 'negative').mean() * 100
        st.metric("Negative", f"{sentiment_counts.get('negative', 0):,}", f"{negative_pct:.1f}%", delta_color="off")
    
    # Sentiment by company
    st.subheader("Sentiment by Telecom Company")
    
    if df['mentions_mtn'].sum() > 0 and df['mentions_airtel'].sum() > 0:
        col1, col2 = st.columns(2)
        
        with col1:
            mtn_positive = (df[df['mentions_mtn']]['final_sentiment'] == 'positive').mean() * 100
            mtn_negative = (df[df['mentions_mtn']]['final_sentiment'] == 'negative').mean() * 100
            mtn_neutral = (df[df['mentions_mtn']]['final_sentiment'] == 'neutral').mean() * 100
            
            st.write("**MTN Uganda**")
            st.write(f"✅ Positive: {mtn_positive:.1f}%")
            st.write(f"⚪ Neutral: {mtn_neutral:.1f}%")
            st.write(f"❌ Negative: {mtn_negative:.1f}%")
        
        with col2:
            airtel_positive = (df[df['mentions_airtel']]['final_sentiment'] == 'positive').mean() * 100
            airtel_negative = (df[df['mentions_airtel']]['final_sentiment'] == 'negative').mean() * 100
            airtel_neutral = (df[df['mentions_airtel']]['final_sentiment'] == 'neutral').mean() * 100
            
            st.write("**Airtel Uganda**")
            st.write(f"✅ Positive: {airtel_positive:.1f}%")
            st.write(f"⚪ Neutral: {airtel_neutral:.1f}%")
            st.write(f"❌ Negative: {airtel_negative:.1f}%")
    
    # Sample tweets with sentiment
    st.subheader("Sample Tweets with Sentiment Analysis")
    sample_data = df[['content', 'vader_sentiment', 'final_sentiment']].head(10)
    st.dataframe(sample_data, use_container_width=True)

def run_ml_analysis(df):
    """Run machine learning sentiment classification"""
    ml_df = df[df['processed_text'].str.strip() != ''].copy()
    
    if len(ml_df) > 10:
        X = ml_df['processed_text']
        y = ml_df['final_sentiment']

        # Convert labels to numerical values
        label_mapping = {'negative': 0, 'neutral': 1, 'positive': 2}
        y_encoded = y.map(label_mapping)

        # TF-IDF Vectorization
        vectorizer = TfidfVectorizer(
            max_features=500,
            ngram_range=(1, 2),
            min_df=1,
            max_df=0.9,
            stop_words=list(stop_words)
        )

        X_tfidf = vectorizer.fit_transform(X)

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X_tfidf, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
        )

        # Train Logistic Regression model
        lr_model = LogisticRegression(
            random_state=42,
            max_iter=1000,
            class_weight='balanced'
        )

        lr_model.fit(X_train, y_train)

        # Predictions
        y_pred = lr_model.predict(X_test)

        # Evaluation
        accuracy = accuracy_score(y_test, y_pred)
        
        st.success(f"Machine Learning Model Trained Successfully!")
        st.metric("Model Accuracy", f"{accuracy:.3f}")
        
        # Display classification report
        st.subheader("Classification Report")
        report = classification_report(y_test, y_pred, target_names=label_mapping.keys(), output_dict=True)
        report_df = pd.DataFrame(report).transpose()
        st.dataframe(report_df.round(3), use_container_width=True)
        
        # Update final sentiment with ML results
        X_full = vectorizer.transform(ml_df['processed_text'])
        ml_predictions = lr_model.predict(X_full)
        
        reverse_mapping = {v: k for k, v in label_mapping.items()}
        ml_indices = ml_df.index
        df.loc[ml_indices, 'ml_sentiment'] = [reverse_mapping[pred] for pred in ml_predictions]
        df['final_sentiment'] = df['ml_sentiment']
        
        st.session_state.df = df
        st.session_state.ml_trained = True

def show_visualizations(df):
    st.header("📈 Visualizations")
    
    # Set style
    plt.style.use('default')
    sns.set_palette("husl")
    
    # Create tabs for different visualizations
    tab1, tab2, tab3, tab4 = st.tabs(["Sentiment Overview", "Word Analysis", "Trends Over Time", "Company Comparison"])
    
    with tab1:
        col1, col2 = st.columns(2)
        
        with col1:
            # Overall sentiment distribution
            fig, ax = plt.subplots(figsize=(8, 6))
            sentiment_order = ['positive', 'neutral', 'negative']
            sentiment_colors = ['#2ecc71', '#3498db', '#e74c3c']
            sentiment_counts = df['final_sentiment'].value_counts().reindex(sentiment_order)
            
            bars = ax.bar(sentiment_order, sentiment_counts, color=sentiment_colors, alpha=0.8, edgecolor='black')
            ax.set_title('Overall Sentiment Distribution', fontsize=14, fontweight='bold')
            ax.set_xlabel('Sentiment Category', fontweight='bold')
            ax.set_ylabel('Number of Tweets', fontweight='bold')
            ax.grid(axis='y', alpha=0.3)
            
            # Add value labels on bars
            for bar, count in zip(bars, sentiment_counts):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, 
                       f'{count}\n({count/len(df)*100:.1f}%)', 
                       ha='center', va='bottom', fontweight='bold', fontsize=10)
            
            st.pyplot(fig)
            plt.close()
        
        with col2:
            # Text length distribution by sentiment
            fig, ax = plt.subplots(figsize=(8, 6))
            sns.boxplot(data=df, x='final_sentiment', y='text_length', order=sentiment_order, palette=sentiment_colors, ax=ax)
            ax.set_title('Tweet Length Distribution by Sentiment', fontsize=14, fontweight='bold')
            ax.set_xlabel('Sentiment Category', fontweight='bold')
            ax.set_ylabel('Text Length (characters)', fontweight='bold')
            ax.grid(True, alpha=0.3)
            st.pyplot(fig)
            plt.close()
    
    with tab2:
        col1, col2 = st.columns(2)
        
        with col1:
            # Positive word cloud
            st.subheader("Positive Sentiment Word Cloud")
            positive_text = ' '.join(df[df['final_sentiment'] == 'positive']['processed_text'])
            if positive_text.strip():
                fig, ax = plt.subplots(figsize=(10, 6))
                wordcloud = WordCloud(width=400, height=300, background_color='white', 
                                    max_words=30, colormap='Greens').generate(positive_text)
                ax.imshow(wordcloud, interpolation='bilinear')
                ax.axis('off')
                st.pyplot(fig)
                plt.close()
            else:
                st.info("No positive text data available for word cloud")
            
            # Top positive words
            st.subheader("Top Positive Words")
            top_positive = get_top_words_by_sentiment(df, 'positive', 8)
            if top_positive:
                pos_df = pd.DataFrame(top_positive, columns=['Word', 'Frequency'])
                st.dataframe(pos_df, use_container_width=True)
        
        with col2:
            # Negative word cloud
            st.subheader("Negative Sentiment Word Cloud")
            negative_text = ' '.join(df[df['final_sentiment'] == 'negative']['processed_text'])
            if negative_text.strip():
                fig, ax = plt.subplots(figsize=(10, 6))
                wordcloud = WordCloud(width=400, height=300, background_color='white', 
                                    max_words=30, colormap='Reds').generate(negative_text)
                ax.imshow(wordcloud, interpolation='bilinear')
                ax.axis('off')
                st.pyplot(fig)
                plt.close()
            else:
                st.info("No negative text data available for word cloud")
            
            # Top negative words
            st.subheader("Top Negative Words")
            top_negative = get_top_words_by_sentiment(df, 'negative', 8)
            if top_negative:
                neg_df = pd.DataFrame(top_negative, columns=['Word', 'Frequency'])
                st.dataframe(neg_df, use_container_width=True)
    
    with tab3:
        # Sentiment over time
        try:
            df['date'] = pd.to_datetime(df['date'])
            df['date_only'] = df['date'].dt.date
            daily_sentiment = df.groupby(['date_only', 'final_sentiment']).size().unstack(fill_value=0)
            daily_sentiment = daily_sentiment.reindex(columns=sentiment_order, fill_value=0)
            
            fig, ax = plt.subplots(figsize=(12, 6))
            daily_sentiment.plot(kind='line', color=sentiment_colors, ax=ax, linewidth=2.5)
            ax.set_title('Sentiment Trends Over Time', fontsize=14, fontweight='bold')
            ax.set_xlabel('Date', fontweight='bold')
            ax.set_ylabel('Number of Tweets', fontweight='bold')
            ax.legend(title='Sentiment', title_fontsize=10)
            ax.grid(True, alpha=0.3)
            plt.xticks(rotation=45)
            st.pyplot(fig)
            plt.close()
        except Exception as e:
            st.error(f"Could not plot time trends: {e}")
    
    with tab4:
        if df['mentions_mtn'].sum() > 0 and df['mentions_airtel'].sum() > 0:
            col1, col2 = st.columns(2)
            
            with col1:
                # Sentiment by company
                company_data = []
                labels = []
                
                mtn_sentiment = df[df['mentions_mtn']]['final_sentiment'].value_counts().reindex(sentiment_order, fill_value=0)
                company_data.append(mtn_sentiment.values)
                labels.append('MTN')
                
                airtel_sentiment = df[df['mentions_airtel']]['final_sentiment'].value_counts().reindex(sentiment_order, fill_value=0)
                company_data.append(airtel_sentiment.values)
                labels.append('Airtel')
                
                fig, ax = plt.subplots(figsize=(10, 6))
                x = np.arange(len(sentiment_order))
                width = 0.35
                
                for i, data in enumerate(company_data):
                    ax.bar(x + i*width, data, width, label=labels[i], alpha=0.8, edgecolor='black')
                
                ax.set_xlabel('Sentiment Category', fontweight='bold')
                ax.set_ylabel('Number of Tweets', fontweight='bold')
                ax.set_title('Sentiment Distribution by Telecom Company', fontsize=14, fontweight='bold')
                ax.set_xticks(x + width/2)
                ax.set_xticklabels(sentiment_order)
                ax.legend()
                ax.grid(axis='y', alpha=0.3)
                st.pyplot(fig)
                plt.close()
            
            with col2:
                # Positive sentiment rate by company
                mtn_positive = (df[df['mentions_mtn']]['final_sentiment'] == 'positive').mean() * 100
                airtel_positive = (df[df['mentions_airtel']]['final_sentiment'] == 'positive').mean() * 100
                
                fig, ax = plt.subplots(figsize=(8, 6))
                companies = ['MTN', 'Airtel']
                positive_rates = [mtn_positive, airtel_positive]
                colors = ['#ff6b6b', '#4ecdc4']
                
                ax.pie(positive_rates, labels=companies, colors=colors, autopct='%1.1f%%', 
                      startangle=90, textprops={'fontweight': 'bold'})
                ax.set_title('Positive Sentiment Rate by Company', fontsize=14, fontweight='bold')
                st.pyplot(fig)
                plt.close()

def show_business_insights(df):
    st.header("💡 Business Insights and Recommendations")
    
    # Calculate key metrics
    positive_pct = (df['final_sentiment'] == 'positive').mean() * 100
    negative_pct = (df['final_sentiment'] == 'negative').mean() * 100
    neutral_pct = (df['final_sentiment'] == 'neutral').mean() * 100
    
    # Executive summary
    st.subheader("🎯 Executive Summary")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Total Tweets Analyzed", f"{len(df):,}")
    
    with col2:
        st.metric("Overall Positive Sentiment", f"{positive_pct:.1f}%")
    
    with col3:
        st.metric("Overall Negative Sentiment", f"{negative_pct:.1f}%")
    
    # Company performance
    if df['mentions_mtn'].sum() > 0 and df['mentions_airtel'].sum() > 0:
        st.subheader("🏢 Company Performance Analysis")
        
        mtn_positive = (df[df['mentions_mtn']]['final_sentiment'] == 'positive').mean() * 100
        airtel_positive = (df[df['mentions_airtel']]['final_sentiment'] == 'positive').mean() * 100
        mtn_negative = (df[df['mentions_mtn']]['final_sentiment'] == 'negative').mean() * 100
        airtel_negative = (df[df['mentions_airtel']]['final_sentiment'] == 'negative').mean() * 100
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**📱 MTN Uganda**")
            st.write(f"✅ Positive: {mtn_positive:.1f}%")
            st.write(f"❌ Negative: {mtn_negative:.1f}%")
            st.write(f"📊 Net Sentiment: {mtn_positive - mtn_negative:+.1f}%")
        
        with col2:
            st.write("**📶 Airtel Uganda**")
            st.write(f"✅ Positive: {airtel_positive:.1f}%")
            st.write(f"❌ Negative: {airtel_negative:.1f}%")
            st.write(f"📊 Net Sentiment: {airtel_positive - airtel_negative:+.1f}%")
    
    # Key themes
    st.subheader("🔍 Key Themes Identified")
    
    negative_words = get_top_words_by_sentiment(df, 'negative', 8)
    positive_words = get_top_words_by_sentiment(df, 'positive', 8)
    
    col1, col2 = st.columns(2)
    
    with col1:
        if negative_words:
            negative_topics = ', '.join([word[0] for word in negative_words[:5]])
            st.error(f"**Customer Pain Points:** {negative_topics}")
        else:
            st.info("No negative themes identified")
    
    with col2:
        if positive_words:
            positive_topics = ', '.join([word[0] for word in positive_words[:5]])
            st.success(f"**Customer Satisfaction Drivers:** {positive_topics}")
        else:
            st.info("No positive themes identified")
    
    # Strategic recommendations
    st.subheader("💡 Strategic Recommendations")
    
    st.write("**🎯 IMMEDIATE ACTIONS:**")
    if negative_pct > 40:
        st.warning(f"• Address critical issues causing {negative_pct:.1f}% negative sentiment")
        st.warning("• Focus on network reliability and customer service responsiveness")
        st.warning("• Implement proactive communication about service improvements")
    
    if positive_pct < 30:
        st.info(f"• Boost positive sentiment from current {positive_pct:.1f}%")
        st.info("• Highlight success stories and positive customer experiences")
        st.info("• Increase engagement with satisfied customers")
    
    st.write("**📈 COMPETITIVE POSITIONING:**")
    if 'mtn_positive' in locals() and 'airtel_positive' in locals():
        if mtn_positive > airtel_positive:
            st.success("• MTN leads in customer satisfaction - maintain competitive advantage")
            st.warning("• Airtel should conduct gap analysis to identify improvement areas")
        else:
            st.success("• Airtel shows stronger positive sentiment - leverage this in marketing")
            st.warning("• MTN should benchmark Airtel's customer experience strategies")
    
    st.write("**🔄 CONTINUOUS IMPROVEMENT:**")
    st.info("• Establish real-time sentiment monitoring system")
    st.info("• Create response protocol for negative feedback")
    st.info("• Regularly analyze emerging themes and trends")
    
    # Download results
    st.subheader("📥 Download Results")
    
    if st.button("Export Analysis Results"):
        # Create downloadable CSV
        csv = df[['content', 'cleaned_text', 'processed_text', 'final_sentiment', 'date']].to_csv(index=False)
        st.download_button(
            label="Download CSV",
            data=csv,
            file_name="telecom_sentiment_analysis.csv",
            mime="text/csv"
        )

if __name__ == "__main__":
    main()