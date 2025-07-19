import streamlit as st
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer

# Load model and vectorizer
model = pickle.load(open("model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

# App title
st.title("📰 Fake News Detector")
st.markdown("Check if a news article is **Fake** or **Real** using Machine Learning 🧠")

# User input
news_input = st.text_area("Paste your news content below:", height=200)

# Prediction
if st.button("Check News"):
    if news_input.strip() == "":
        st.warning("⚠️ Please enter some news text.")
    else:
        # Vectorize input
        transformed = vectorizer.transform([news_input])
        prediction = model.predict(transformed)[0]
        proba = model.predict_proba(transformed).max()

        # Result display
        if prediction == 1:
            st.success("✅ This news is **REAL**.")
        else:
            st.error("❌ This news is **FAKE**.")
        
        # Show confidence score
        st.info(f"🧠 Model confidence: **{proba * 100:.2f}%**")

