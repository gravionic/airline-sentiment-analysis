import streamlit as st
import joblib as jl
import pandas as pd
import string
import nltk
import plotly.express as px
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# Load model and vectorizer
model = jl.load('model/airlines_sentiment_model.pkl')
tfidf = jl.load('model/tfidf_vectorizer.pkl')

# Page configuration
st.set_page_config(page_title="Airline Sentiment Analysis", page_icon="✈️")

st.title("✈️ Airline Sentiment Analysis")
st.write("Analyze customer reviews and predict their sentiment using a trained Logistic Regression machine learning model.")

st.markdown("---")
st.subheader("Try an example")

# Example reviews
col1, col2, col3 = st.columns(3)

if col1.button("Negative"):
    st.session_state.review = "worse than the budget airlines Ljubljana to ZÃ¼rich. Firstly, Ljubljana airport is terrible. Badly trained staff, unfriendly. Toilets are very dirty. Flight 2 hours delayed without any information."
    
if col2.button("Neutral"):
    st.session_state.review = "service onboard was poor Frankfurt to Ljubljana with Adria Airways. Boarding was on-time, but aircraft was parked on remote gate about 15 minutes from main terminal. So we departed about 25 minutes late. Service onboard was poor, with no complimentary drinks or snacks - they are available only for sale. Arrival was 15 minutes later than scheduled. Crew was friendly."
    
if col3.button("Positive"):
    st.session_state.review = "pretty decent airline Moroni to Moheli.Turned out to be a pretty decent airline. Online booking worked well, checkin and boarding was fine and the plane looked well maintained"

# Review input
review = st.text_area("Enter an airline review:", height=150, key="review")

# Preprocessing
nltk.download('stopwords')
nltk.download('punkt_tab')
stop_words = set(stopwords.words('english'))

negations = {"no", "not", "nor", "never", "don't", "doesn't", "didn't", "isn't", "wasn't", "can't", "won't"}
stop_words = stop_words - negations

def preprocess(txt):

    # Convert to lowercase
    txt = txt.lower()

    # Remove punctuation
    txt = txt.translate(str.maketrans('', '', string.punctuation))

    # Remove numbers
    new = ""
    for i in txt:
        if not i.isdigit():
            new = new + i
    txt = new

    # Remove emojis and non-ASCII characters
    new = ""
    for i in txt:
        if i.isascii():
            new = new + i
    txt = new

    # Tokenize and remove stopwords
    words = word_tokenize(txt)
    cleaned = []

    for i in words:
        if i not in stop_words:
            cleaned.append(i)

    return ' '.join(cleaned)

# Prediction
if st.button("Predict Sentiment", type="primary"):
    st.markdown("---")

    if review.strip():

        # Convert review into TF-IDF features
        cleaned_review = preprocess(review)
        review_tfidf = tfidf.transform([cleaned_review])

        # Predict sentiment and probabilities
        prediction = model.predict(review_tfidf)[0]
        probabilities = model.predict_proba(review_tfidf)[0]

        # Display prediction
        st.subheader("Predicted Sentiment")

        if prediction == "Negative":
            st.error("Negative")
        elif prediction == "Neutral":
            st.warning("Neutral")
        else:
            st.success("Positive")

        st.markdown("---")
        # Probability chart
        st.subheader("Sentiment Probabilities")

        prob_df = pd.DataFrame({
            'Sentiment': model.classes_, 
            'Probability': probabilities})

       
        #prob_df = prob_df.set_index('Sentiment')

        fig=px.bar(prob_df, x='Sentiment', y='Probability', color='Sentiment', color_discrete_map={'Negative':'red',
                                                                                                   'Neutral':'orange',
                                                                                                   'Positive':'green'})

        st.plotly_chart(fig, width='stretch')

        # Neutral warning
        st.caption(
            "Note: Neutral predictions are less reliable due to the smaller number of Neutral training samples in our dataset")

    else:
        st.warning("Please enter a review first.")
