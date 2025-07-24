# train_model.py

import pandas as pd
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score

# Load dataset
df = pd.read_csv('dataset/dataset.csv')
print("✅ Loaded columns:", df.columns.tolist())

# Use 'Disease' as target column
target_col = 'Disease'
symptom_cols = [col for col in df.columns if col.startswith('Symptom_')]

# Fill missing symptoms with 'none'
df[symptom_cols] = df[symptom_cols].fillna('none')

# Convert symptoms to one-hot encoded features
symptom_set = set()
for col in symptom_cols:
    symptom_set.update(df[col].unique())

symptom_set.discard('none')
all_symptoms = sorted(list(symptom_set))

# Create binary symptom matrix
def encode_symptoms(row):
    present = set(row[symptom_cols])
    return [1 if symptom in present else 0 for symptom in all_symptoms]

X = df.apply(encode_symptoms, axis=1, result_type='expand')
X.columns = all_symptoms

# Encode diseases
disease_encoder = LabelEncoder()
y = disease_encoder.fit_transform(df[target_col])

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model = RandomForestClassifier(n_estimators=200, random_state=42)
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f"✅ Model Accuracy: {acc * 100:.2f}%")

# Save model
with open('model/disease_model.pkl', 'wb') as f:
    pickle.dump((model, disease_encoder, all_symptoms), f)

print("✅ Model saved successfully")
