from flask import Flask, render_template, request, redirect, session, url_for
import pickle
import numpy as np
import pandas as pd
from werkzeug.security import generate_password_hash, check_password_hash
from datetime import datetime
import mysql.connector
from mysql.connector import Error

app = Flask(__name__)

app.secret_key = 'your_secret_key_here'  # Change this to something secure

def get_db_connection():
    return mysql.connector.connect(
        host='localhost',
        user='lord',
        password='2005',
        database='disease_prediction'
    )

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

    # ⬇️ Insert history into MySQL
    if 'username' in session:
        username = session['username']
        symptoms_str = ", ".join(selected_symptoms)
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

        conn = get_db_connection()
        cursor = conn.cursor(dictionary=True)
        insert_query = """
            INSERT INTO history (username, symptoms, prediction, timestamp)
            VALUES (%s, %s, %s, %s)
        """
        cursor.execute(insert_query, (username, symptoms_str, predicted_disease, timestamp))
        conn.commit()
        cursor.close()
        conn.close()

    return render_template('result.html',
                           disease=predicted_disease,
                           symptoms=disease_symptoms)

@app.route('/history')
def view_history():
    if 'username' not in session:
        return redirect('/login')
    
    username = session['username']

    conn = get_db_connection()
    cursor = conn.cursor(dictionary=True)
    cursor.execute("SELECT symptoms, prediction, timestamp FROM history WHERE username = %s ORDER BY timestamp DESC", (username,))
    history = cursor.fetchall()
    cursor.close()
    conn.close()

    return render_template('history.html', history=history)

@app.route('/admin/history')
def admin_history():
    if 'username' not in session or session.get('role') != 'admin':
        return "Access denied", 403

    conn = get_db_connection()
    cursor = conn.cursor(dictionary=True)
    cursor.execute("SELECT * FROM history ORDER BY timestamp DESC")
    records = cursor.fetchall()
    cursor.close()
    conn.close()

    return render_template('admin_history.html', history=records)

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form['username']
        password = generate_password_hash(request.form['password'])

        conn = get_db_connection()
        cursor = conn.cursor()
        try:
            cursor.execute("INSERT INTO users (username, password) VALUES (%s, %s)", (username, password))
            conn.commit()
        except mysql.connector.IntegrityError:
            return "Username already exists."
        finally:
            cursor.close()
            conn.close()

        return redirect('/login')
    return render_template('register.html')



@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']

        conn = get_db_connection()
        cursor = conn.cursor(dictionary=True)
        cursor.execute("SELECT * FROM users WHERE username = %s", (username,))
        user = cursor.fetchone()
        cursor.close()
        conn.close()

        if user and check_password_hash(user['password'], password):
            session['username'] = user['username']
            session['role'] = user['role']
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