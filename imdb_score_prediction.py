import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Lasso
from sklearn.metrics import r2_score
import torch
import tensorflow as tf

# Load data
df = pd.read_csv('movies.csv')
X = df.drop(columns=['imdb_score'])
y = df['imdb_score']

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Random Forest
rf = RandomForestRegressor()
rf.fit(X_train, y_train)
print('RF R2:', r2_score(y_test, rf.predict(X_test)))

# LASSO
lasso = Lasso()
lasso.fit(X_train, y_train)
print('LASSO R2:', r2_score(y_test, lasso.predict(X_test)))
