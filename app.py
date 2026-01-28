import streamlit as st
import pickle
import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Load model and vectorizer
model = pickle.load(open("model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

# NLTK downloads (safe)
nltk.download("stopwords", quiet=True)
nltk.download("wordnet", quiet=True)

stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()

# Text preprocessing
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"\d+", "", text)
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"\s+", " ", text).strip()
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
            st.info(
                "If you’re feeling overwhelmed or unsafe, please consider reaching out to "
                "a trusted person or a mental health professional. "
                "If you are in immediate danger, contact local emergency services or a suicide prevention helpline."
            )
        else:
            st.success("✅ No Mental Health Risk Detected")
