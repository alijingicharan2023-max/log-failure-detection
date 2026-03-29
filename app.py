import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# Load data
data = pd.read_csv("data.csv")
data.columns = data.columns.str.strip().str.lower()

# Train model
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(data['log'])
y = data['label']

model = LogisticRegression()
model.fit(X, y)

# UI
st.title("Predictive Configuration Failure Detection")

log_input = st.text_input("Enter system log:")

if st.button("Predict"):
    new_X = vectorizer.transform([log_input])
    prediction = model.predict(new_X)

    if prediction[0] == 1:
        st.error("⚠️ Failure Likely!")
    else:
        st.success("✅ System Normal")