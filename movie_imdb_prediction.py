# Movie IMDB Score Prediction using LASSO and Random Forest

import pandas as pd
import numpy as np
from sklearn.linear_model import LassoCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

# Load dataset
df = pd.read_csv('movie_data.csv')

# Feature Engineering
df['main_actor_known'] = df['main_actor'].apply(lambda x: 1 if x in ['Brad Pitt', 'Tom Hanks', 'Meryl Streep'] else 0)
df['genre_encoded'] = pd.factorize(df['genre'])[0]

# Select features and target
X = df[['budget', 'duration', 'main_actor_known', 'genre_encoded']]
y = df['imdb_score']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# LASSO Regression
lasso = LassoCV(cv=5)
lasso.fit(X_train, y_train)
y_pred_lasso = lasso.predict(X_test)

# Random Forest Regression
rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)

# Metrics
print("LASSO R^2:", r2_score(y_test, y_pred_lasso))
print("Random Forest R^2:", r2_score(y_test, y_pred_rf))

# Plot feature importances
importances = rf.feature_importances_
plt.bar(X.columns, importances)
plt.title('Feature Importances - Random Forest')
plt.show()
