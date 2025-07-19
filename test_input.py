import joblib

# Load the saved model and vectorizer
model = joblib.load("news_model.pkl")
vectorizer = joblib.load("news_vectorizer.pkl")

# User input
news = input("📰 Enter a news article or headline to check: ")

# Vectorize the input
news_vec = vectorizer.transform([news])

# Predict
prediction = model.predict(news_vec)

# Output result
if prediction[0] == 1:
    print("✅ This news is likely REAL.")
else:
    print("⚠️ This news is likely FAKE.")
