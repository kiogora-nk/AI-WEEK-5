# scripts/train_hospital_model.py

import pandas as pd
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix, precision_score, recall_score
import pickle

# Load dataset
data = pd.read_csv("data/hospital_data.csv")

# Encode categorical columns
le_gender = LabelEncoder()
data['gender'] = le_gender.fit_transform(data['gender'])

# Fill missing values
data['prior_admissions'].fillna(0, inplace=True)
data['chronic_conditions'].fillna(0, inplace=True)

# Split dataset
X = data.drop('readmission', axis=1)
y = data['readmission']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = XGBClassifier(use_label_encoder=False, eval_metric='logloss')
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
cm = confusion_matrix(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)

print("Confusion Matrix:\n", cm)
print("Precision:", precision)
print("Recall:", recall)

# Save model
with open("models/readmission_model.pkl", "wb") as f:
    pickle.dump(model, f)
