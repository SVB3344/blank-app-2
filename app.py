import streamlit as st
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer

nltk.download('vader_lexicon')

st.title("Анализатор настроений")
text = st.text_input("Введите текст:")
if text:
    sia = SentimentIntensityAnalyzer()
    score = sia.polarity_scores(text)
    st.write(f"Результат анализа: {score}")
