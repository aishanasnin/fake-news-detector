# Step 1: Import libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib

# Step 2: Load and label datasets
fake = pd.read_csv("Fake.csv")
real = pd.read_csv("True.csv")
fake['label'] = 0
real['label'] = 1

# Step 3: Combine, shuffle, and clean data
data = pd.concat([fake, real])
data = data.sample(frac=1, random_state=42).reset_index(drop=True)
data.dropna(inplace=True)

# Step 4: Split data into training and test sets
X = data['text']
y = data['label']
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)

# Step 5: TF-IDF Vectorization
vectorizer = TfidfVectorizer(stop_words='english', max_df=0.7)
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# Step 6: Train the model
model = PassiveAggressiveClassifier()
model.fit(X_train_tfidf, y_train)

# Step 7: Evaluate
y_pred = model.predict(X_test_tfidf)
print("✅ Accuracy:", accuracy_score(y_test, y_pred))
print("\n✅ Classification Report:\n", classification_report(y_test, y_pred))

# Step 8: Save model and vectorizer
joblib.dump(model, "model.pkl")
joblib.dump(vectorizer, "vectorizer.pkl")

print("✅ Model and vectorizer saved successfully.")

import pickle

# Save the trained model
with open("model.pkl", "wb") as f:
    pickle.dump(model, f)

# Save the TF-IDF vectorizer
with open("vectorizer.pkl", "wb") as f:
    pickle.dump(TfidfVectorizer, f)

