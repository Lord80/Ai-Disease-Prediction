import pandas as pd
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Load dataset
df = pd.read_csv("dataset/Disease_symptom_and_patient_profile_dataset.csv")
df.columns = df.columns.str.strip()
df = df.map(lambda x: x.strip() if isinstance(x, str) else x)  # safer

# Map Yes/No symptoms
binary_map = {"Yes": 1, "No": 0}
df["Fever"] = df["Fever"].map(binary_map)
df["Cough"] = df["Cough"].map(binary_map)
df["Fatigue"] = df["Fatigue"].map(binary_map)
df["Difficulty Breathing"] = df["Difficulty Breathing"].map(binary_map)

# Encode categorical values
df["Gender"] = df["Gender"].map({"Male": 1, "Female": 0})
df["Blood Pressure"] = df["Blood Pressure"].map({"High": 1, "Low": -1, "Normal": 0})
df["Cholesterol Level"] = df["Cholesterol Level"].map({"High": 1, "Normal": 0})

# Encode outcome
df["Outcome Variable"] = df["Outcome Variable"].map({"Positive": 1, "Negative": 0})

# Encode disease names
disease_encoder = LabelEncoder()
df["Disease"] = disease_encoder.fit_transform(df["Disease"])

# Prepare features and labels
X_disease = df[[
    "Fever", "Cough", "Fatigue", "Difficulty Breathing",
    "Age", "Gender", "Blood Pressure", "Cholesterol Level"
]]
y_disease = df["Disease"]

# Train-test split for disease prediction
X_train_d, X_test_d, y_train_d, y_test_d = train_test_split(X_disease, y_disease, test_size=0.2, random_state=42)

# Train disease model
disease_model = RandomForestClassifier(random_state=42)
disease_model.fit(X_train_d, y_train_d)

# Outcome prediction
y_outcome = df["Outcome Variable"]
X_train_o, X_test_o, y_train_o, y_test_o = train_test_split(X_disease, y_outcome, test_size=0.2, random_state=42)

outcome_model = RandomForestClassifier(random_state=42)
outcome_model.fit(X_train_o, y_train_o)

# Save models with encoder
with open("model/disease_model_v2.pkl", "wb") as f:
    pickle.dump((disease_model, disease_encoder), f)  # save encoder, not just mapping

with open("model/outcome_model.pkl", "wb") as f:
    pickle.dump(outcome_model, f)

print("✅ Disease and outcome models saved successfully.")
