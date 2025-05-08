import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score

df = pd.read_csv('imdb_reviews.csv')
X = df['review']
y = df['sentiment']

vectorizer = TfidfVectorizer(max_features=10000)
X_vec = vectorizer.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_vec, y, test_size=0.2)
model = LogisticRegression()
model.fit(X_train, y_train)
preds = model.predict(X_test)
print('F1 Score:', f1_score(y_test, preds, pos_label='positive'))
