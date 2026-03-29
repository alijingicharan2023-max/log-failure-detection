import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

# Step 1: Load dataset
data = pd.read_csv("data.csv")

# 🔍 DEBUG LINE (ADD HERE)
print(data.columns)

# Step 2: Feature extraction
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(data['log'])
y = data['label']

print("Feature extraction done")