import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

# Load dataset
data = pd.read_csv("data.csv")

# Clean column names
data.columns = data.columns.str.strip().str.lower()

# Feature extraction
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(data['log'])
y = data['label']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Train model
model = LogisticRegression()
model.fit(X_train, y_train)

# Evaluate model
accuracy = model.score(X_test, y_test)
print("Model Accuracy:", accuracy)

# Prediction
new_log = ["Disk error occurred"]
new_X = vectorizer.transform(new_log)

prediction = model.predict(new_X)

if prediction[0] == 1:
    print("⚠️ ALERT: Configuration Failure Likely!")
else:
    print("✅ System is stable")