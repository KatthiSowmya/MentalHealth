import streamlit as st
import pickle
import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Load files
model = pickle.load(open("model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

# NLTK
nltk.download('stopwords')
nltk.download('wordnet')

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

# Text cleaning
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'\d+', '', text)
    text = re.sub(r'http\S+|www\S+', '', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = re.sub(r'\s+', ' ', text).strip()
    tokens = text.split()
    tokens = [lemmatizer.lemmatize(w) for w in tokens if w not in stop_words]
    return " ".join(tokens)

# UI
st.title("Mental Health Text Classification")
st.write("Enter a sentence to predict mental health condition")

user_input = st.text_area("Enter text here")

if st.button("Predict"):
    if user_input.strip() == "":
        st.warning("Please enter some text")
    else:
        clean_text = preprocess_text(user_input)
        vec = vectorizer.transform([clean_text])
        prediction = model.predict(vec)[0]

        if prediction == 1:
            st.error("⚠️ Mental Health Risk Detected")
        else:
            st.success("✅ No Mental Health Risk Detected")
