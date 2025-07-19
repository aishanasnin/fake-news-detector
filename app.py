import streamlit as st
import joblib

# Load model and vectorizer
model = joblib.load("news_model.pkl")
vectorizer = joblib.load("news_vectorizer.pkl")

# App title
st.title("📰 Fake News Detector")
st.markdown("Check if a news article is **Fake** or **Real** using Machine Learning 🧠")

# Input box
news_input = st.text_area("Paste your news content below:", height=200)

# Predict button
if st.button("Check News"):
    if news_input.strip() == "":
        st.warning("⚠️ Please enter some news text.")
    else:
        transformed = vectorizer.transform([news_input])
        prediction = model.predict(transformed)[0]
        proba = model.predict_proba(transformed).max()

        if prediction == 1:
            st.success("✅ This news is **REAL**.")
        else:
            st.error("❌ This news is **FAKE**.")
        
        st.info(f"🧠 Model confidence: **{proba * 100:.2f}%**")

st.markdown("<hr style='border:1px solid #999;'>", unsafe_allow_html=True)
st.markdown(
    "<p style='text-align: center; font-size: 12px; color: gray;'>Created by Aisha Nasnin</p>", 
    unsafe_allow_html=True
)