import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# Page config
st.set_page_config(page_title="Failure Detection", layout="centered")

# 🎨 FINAL UI CSS (Fixed Background + Premium Look)
st.markdown("""
<style>

/* Full background (IMPORTANT FIX) */
.stApp {
    background: linear-gradient(135deg, #e3f2fd, #f9fbfd);
}

/* Center container */
.container {
    max-width: 700px;
    margin: auto;
    padding-top: 40px;
}

/* Title */
.title {
    text-align: center;
    font-size: 42px;
    font-weight: 700;
    color: #1f2d3d;
}

/* Subtitle */
.subtitle {
    text-align: center;
    font-size: 16px;
    color: #6c757d;
    margin-bottom: 30px;
}

/* Card */
.card {
    background: white;
    padding: 30px;
    border-radius: 15px;
    box-shadow: 0 10px 30px rgba(0,0,0,0.08);
}

/* Input */
.stTextInput input {
    border-radius: 10px;
    padding: 10px;
    border: 1px solid #ddd;
}

/* Button */
.stButton button {
    border-radius: 10px;
    padding: 10px 24px;
    background-color: #1f2d3d;
    color: white;
    border: none;
    font-weight: 500;
}

/* Button hover */
.stButton button:hover {
    background-color: #34495e;
}

/* Result box */
.result {
    margin-top: 20px;
    padding: 14px;
    border-radius: 10px;
    font-weight: 500;
    text-align: center;
    font-size: 16px;
}

</style>
""", unsafe_allow_html=True)

# Container
st.markdown('<div class="container">', unsafe_allow_html=True)

# Title
st.markdown('<div class="title">Predictive Configuration Failure Detection</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Smart log analysis using Machine Learning</div>', unsafe_allow_html=True)

# Card start
st.markdown('<div class="card">', unsafe_allow_html=True)

# Load data
data = pd.read_csv("data.csv")
data.columns = data.columns.str.strip().str.lower()

# Train model
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(data['log'])
y = data['label']

model = LogisticRegression()
model.fit(X, y)

# Input
log_input = st.text_input("Enter system log")

# Predict
if st.button("Predict"):
    if log_input.strip() == "":
        st.warning("Please enter a log message")
    else:
        new_X = vectorizer.transform([log_input])
        prediction = model.predict(new_X)

        if prediction[0] == 1:
            st.markdown(
                '<div class="result" style="background:#fdecea;color:#c0392b;">⚠️ Failure Likely</div>',
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                '<div class="result" style="background:#eafaf1;color:#27ae60;">✅ System Normal</div>',
                unsafe_allow_html=True
            )

# Card end
st.markdown('</div>', unsafe_allow_html=True)

# Footer
st.markdown("<br>", unsafe_allow_html=True)
st.markdown(
    '<div style="text-align:center;color:gray;font-size:13px;">Developed using Python, Machine Learning & Streamlit</div>',
    unsafe_allow_html=True
)

# Container end
st.markdown('</div>', unsafe_allow_html=True)