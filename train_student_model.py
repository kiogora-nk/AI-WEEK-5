# scripts/train_student_model.py

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, f1_score
import pickle

# Load dataset
data = pd.read_csv("data/student_data.csv")

# Encode categorical columns
le_gender = LabelEncoder()
data['gender'] = le_gender.fit_transform(data['gender'])

le_region = LabelEncoder()
data['region'] = le_region.fit_transform(data['region'])

# Fill missing values
data['grades'].fillna(data['grades'].median(), inplace=True)
data['attendance'].fillna(data['attendance'].median(), inplace=True)

# Normalize numeric features
scaler = StandardScaler()
data[['age', 'attendance', 'grades']] = scaler.fit_transform(data[['age', 'attendance', 'grades']])

# Split dataset
X = data.drop('dropout', axis=1)
y = data['dropout']
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# Train model
model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
model.fit(X_train, y_train)

# Evaluate model
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("F1 Score:", f1_score(y_test, y_pred))

# Save model
with open("models/student_dropout_model.pkl", "wb") as f:
    pickle.dump(model, f)
