# app.py

from flask import Flask, render_template, request
import pickle
import numpy as np
import pandas as pd

app = Flask(__name__)

# Load model and metadata
with open('model/disease_model.pkl', 'rb') as f:
    model, disease_encoder, symptom_list = pickle.load(f)

@app.route('/')
def index():
    return render_template('index.html', symptoms=symptom_list)

@app.route('/predict', methods=['POST'])
def predict():
    selected_symptoms = request.form.getlist('symptoms')

    # Convert selected symptoms to binary input vector
    input_vector = [1 if symptom in selected_symptoms else 0 for symptom in symptom_list]

    # Convert to DataFrame to match training format
    input_df = pd.DataFrame([input_vector], columns=symptom_list)

    # Predict disease
    prediction = model.predict(input_df)[0]
    predicted_disease = disease_encoder.inverse_transform([prediction])[0]

    return render_template('result.html', disease=predicted_disease)


if __name__ == '__main__':
    app.run(debug=True)
