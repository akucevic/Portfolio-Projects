# NLP Classifier for IMDB Reviews using Scikit-learn

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, classification_report

# Load data
df = pd.read_csv('imdb_reviews.csv')  # Columns: review, sentiment

# Preprocessing
df['review'] = df['review'].str.lower().str.replace(r"<.*?>", "", regex=True)

# Vectorization
vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
X = vectorizer.fit_transform(df['review'])
y = df['sentiment'].map({'positive': 1, 'negative': 0})

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model training
clf = LogisticRegression(max_iter=1000)
clf.fit(X_train, y_train)

# Prediction
y_pred = clf.predict(X_test)

# Evaluation
print("F1 Score:", f1_score(y_test, y_pred))
print(classification_report(y_test, y_pred))
