from flask import Flask, render_template, request, redirect, session, url_for
import pickle
import numpy as np
import pandas as pd
from werkzeug.security import generate_password_hash, check_password_hash
import os
import datetime

app = Flask(__name__)

app.secret_key = 'your_secret_key_here'  # Change this to something secure

USER_CSV = 'dataset/users.csv'
HISTORY_CSV = 'dataset/history.csv'

# Ensure user CSV exists
if not os.path.exists(USER_CSV):
    pd.DataFrame(columns=['username', 'password']).to_csv(USER_CSV, index=False)

# Load model and metadata
with open('model/disease_model.pkl', 'rb') as f:
    model, disease_encoder, symptom_list = pickle.load(f)

@app.route('/')
def index():
    username = session.get('username')
    return render_template('index.html', symptoms=symptom_list, username=username)

@app.route('/predict', methods=['POST'])
def predict():
    selected_symptoms = request.form.getlist('symptoms')

    # 🔷 Create input vector and predict
    input_vector = [1 if symptom in selected_symptoms else 0 for symptom in symptom_list]
    input_df = pd.DataFrame([input_vector], columns=symptom_list)

    prediction = model.predict(input_df)[0]
    predicted_disease = disease_encoder.inverse_transform([prediction])[0]

    # 🔷 Load and extract symptoms from dataset.csv for this disease
    df = pd.read_csv('dataset/dataset.csv')
    symptom_cols = [col for col in df.columns if col.startswith('Symptom_')]

    related_rows = df[df['Disease'] == predicted_disease]
    disease_symptoms = set()

    for _, row in related_rows.iterrows():
        for col in symptom_cols:
            symptom = row[col]
            if pd.notna(symptom) and symptom.strip().lower() != 'none' and symptom.strip() != '':
                disease_symptoms.add(symptom.strip())

    disease_symptoms = sorted(disease_symptoms)

    # 🔷 Save prediction history if user is logged in
    if 'username' in session:
        history_path = 'dataset/history.csv'

        # Ensure the history.csv file exists with correct columns
        if not os.path.exists(history_path):
            pd.DataFrame(columns=['username', 'symptoms', 'prediction', 'timestamp']).to_csv(history_path, index=False)

        history_df = pd.read_csv(history_path)
        history_df.loc[len(history_df)] = [
            session['username'],
            ', '.join(selected_symptoms),
            predicted_disease,
            datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        ]
        history_df.to_csv(history_path, index=False)

    return render_template('result.html',
                           disease=predicted_disease,
                           symptoms=disease_symptoms)

@app.route('/history')
def view_history():
    if 'username' not in session:
        return redirect('/login')
    
    history_df = pd.read_csv(HISTORY_CSV)
    user_history = history_df[history_df['username'] == session['username']]
    return render_template('history.html', history=user_history.to_dict(orient='records'))

@app.route('/admin/history')
def admin_history():
    if 'username' not in session or session.get('role') != 'admin':
        return "Access denied. Admins only.", 403

    if not os.path.exists(HISTORY_CSV):
        return "No history found."

    history_df = pd.read_csv(HISTORY_CSV)
    history_df = history_df.sort_values(by='timestamp', ascending=False)

    return render_template('admin_history.html', history=history_df.to_dict(orient='records'))




def load_users():
    return pd.read_csv(USER_CSV)

def save_user(username, hashed_password):
    df = load_users()
    df.loc[len(df)] = [username, hashed_password]
    df.to_csv(USER_CSV, index=False)

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form['username']
        password = generate_password_hash(request.form['password'])

        users = pd.read_csv(USER_CSV)
        if username in users['username'].values:
            return "Username already exists."

        # Default role = 'user'
        new_user = pd.DataFrame([[username, password, 'user']], columns=['username', 'password', 'role'])
        users = pd.concat([users, new_user], ignore_index=True)
        users.to_csv(USER_CSV, index=False)

        return redirect('/login')
    return render_template('register.html')


@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']

        users = pd.read_csv(USER_CSV)
        user = users[users['username'] == username]

        if not user.empty and check_password_hash(user.iloc[0]['password'], password):
            session['username'] = username
            session['role'] = user.iloc[0]['role']  # ✅ Save role in session
            return redirect('/')
        else:
            return "Invalid credentials"
    return render_template('login.html')


@app.route('/logout')
def logout():
    session.pop('username', None)
    return redirect('/')



if __name__ == '__main__':
    app.run(debug=True)