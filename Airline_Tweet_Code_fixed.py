import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from wordcloud import WordCloud, STOPWORDS
import matplotlib.pyplot as plt

# --- Page Configuration ---
st.set_page_config(page_title="Airline Sentiment Analysis", layout="wide")
st.title("Sentiment Analysis of Tweets about U.S. Airlines")
st.sidebar.title("Dashboard Controls")

filename = 'tweets.csv'

# --- Data Loading & Cleaning ---
@st.cache_data(persist=True)
def load_data():
    try:
        data = pd.read_csv(filename)
        # Rename common coordinate variations if they exist
        data = data.rename(columns={'lat': 'latitude', 'lon': 'longitude'})
        
        # Ensure date format
        data["tweet_created"] = pd.to_datetime(data['tweet_created'])
        
        # st.map requires numeric latitude/longitude; drop rows missing these
        data['latitude'] = pd.to_numeric(data['latitude'], errors='coerce')
        data['longitude'] = pd.to_numeric(data['longitude'], errors='coerce')
        data = data.dropna(subset=['latitude', 'longitude'])
        return data
    except FileNotFoundError:
        st.error(f"Error: '{filename}' not found. Please ensure the CSV is in the same directory.")
        return pd.DataFrame()

data = load_data()

if not data.empty:
    # --- Data Processing: Color Mapping ---
    sentiment_colors = {
        'positive': '#3cb371',
        'neutral': '#ffa500',
        'negative': '#ff0000'
    }
    data['palette'] = data['airline_sentiment'].map(sentiment_colors)

    # --- Sidebar: Random Tweet ---
    st.sidebar.subheader("Show random tweet")
    random_tweet_sentiment = st.sidebar.radio('Sentiment Category', ('positive', 'neutral', 'negative'))
    
    filtered_tweets = data.query('airline_sentiment == @random_tweet_sentiment')
    if not filtered_tweets.empty:
        random_text = filtered_tweets["text"].sample(n=1).iat[0]
        st.sidebar.info(f'"{random_text}"')
    else:
        st.sidebar.write("No tweets found for this category.")

    # --- Visualization: Histogram/Pie ---
    st.sidebar.markdown('### Number of tweets by sentiment')
    select = st.sidebar.selectbox('Visualization type', ['Histogram', 'Pie chart'], key='1')
    
    sentiment_count = data['airline_sentiment'].value_counts().reset_index()
    sentiment_count.columns = ['Sentiment', 'Tweets']

    if not st.sidebar.checkbox('Hide Charts', False):
        st.markdown('### Total Tweet Distribution')
        if select == "Histogram":
            fig = px.bar(sentiment_count, x='Sentiment', y='Tweets', color='Sentiment', 
                         color_discrete_map=sentiment_colors, height=500)
            # FIX: Updated use_container_width to width="stretch"
            st.plotly_chart(fig, width="stretch")
        else:
            fig = px.pie(sentiment_count, values='Tweets', names='Sentiment',
                         color='Sentiment', color_discrete_map=sentiment_colors)
            # FIX: Updated use_container_width to width="stretch"
            st.plotly_chart(fig, width="stretch")

    # --- Map Section: Time of Day ---
    st.sidebar.subheader('When and where are users tweeting from?')
    hour = st.sidebar.slider("Hour of day", 0, 23)
    modified_data = data[data['tweet_created'].dt.hour == hour]
    
    if not st.sidebar.checkbox("Hide Map", False, key='2'):
        st.markdown('### Tweet locations based on time of day')
        st.markdown(f'**{len(modified_data)}** tweets recorded between {hour}:00 and {(hour+1)%24}:00')
        
        # Explicitly passing column names for better compatibility
        st.map(modified_data, latitude='latitude', longitude='longitude', color='palette')
        
        if st.sidebar.checkbox("Show raw data for this hour", False):
            st.write(modified_data)

    # --- Multiselect: Airline Breakdown ---
    st.sidebar.subheader("Breakdown by Airline")
    airlines = ('US Airways', 'United', 'American', 'Southwest', 'Delta', 'Virgin America')
    choice = st.sidebar.multiselect('Pick airlines', airlines, key='0')

    if len(choice) > 0:
        choice_data = data[data.airline.isin(choice)]
        fig_choice = px.histogram(
            choice_data, x='airline', color='airline_sentiment', 
            barmode='group', color_discrete_map=sentiment_colors,
            labels={'airline_sentiment': 'Sentiment'}, height=600
        )
        # FIX: Updated use_container_width to width="stretch"
        st.plotly_chart(fig_choice, width="stretch")

    # --- Word Cloud Section ---
    st.sidebar.header("Word Cloud")
    word_sentiment = st.sidebar.radio('Word cloud sentiment', ('positive', 'neutral', 'negative'))

    if not st.sidebar.checkbox('Hide Word Cloud', True, key='3'):
        st.header(f'Word Cloud for "{word_sentiment}" Sentiment')

        df_word = data[data['airline_sentiment'] == word_sentiment]
        words = ' '.join(df_word['text'].dropna().astype(str))
        
        processed_words = ' '.join([word for word in words.split()
                                    if 'http' not in word and
                                    not word.startswith('@') and
                                    word.upper() != 'RT'])

        if processed_words.strip():
            wordcloud = WordCloud(stopwords=STOPWORDS, background_color='white',
                                  height=640, width=800).generate(processed_words)

            fig, ax = plt.subplots()
            ax.imshow(wordcloud, interpolation='bilinear')
            ax.axis('off')
            st.pyplot(fig)
        else:
            st.warning("Not enough text data to generate a word cloud for this category.")

else:
    st.warning("The dataset is empty or could not be loaded. Check your CSV file.")