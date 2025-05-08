# Predicting Disease Progression using Random Forest and Time-Series Features

import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# Load dataset
df = pd.read_csv('disease_progression.csv')  # Includes vitals, lifestyle, genetic markers

# Feature selection
features = [col for col in df.columns if col not in ['date', 'patient_id', 'progression_score']]
X = df[features]
y = df['progression_score']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model
rf = RandomForestRegressor(n_estimators=100)
rf.fit(X_train, y_train)
y_pred = rf.predict(X_test)

# Evaluation
print("RMSE:", mean_squared_error(y_test, y_pred, squared=False))
print("R² Score:", r2_score(y_test, y_pred))
