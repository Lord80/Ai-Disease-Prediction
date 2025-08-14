from flask import Flask, render_template, request, redirect, session, url_for
import pickle
import numpy as np
import pandas as pd
from werkzeug.security import generate_password_hash, check_password_hash
from datetime import datetime
import mysql.connector
from mysql.connector import Error
from remedies_v2 import remedy_dict_v2

app = Flask(__name__)
app.secret_key = 'your_secret_key_here'  # Change to secure secret in production

def get_db_connection():
    try:
        conn = mysql.connector.connect(
            host='localhost',
            user='lord',
            password='2005',
            database='disease_prediction'
        )
        return conn
    except Error as e:
        print(f"[DB] Connection error: {e}")
        return None

try:
    with open('model/disease_model.pkl', 'rb') as f:
        temp = pickle.load(f)
        if isinstance(temp, tuple) and len(temp) >= 3:
            model, disease_encoder, symptom_list = temp[0], temp[1], temp[2]
        elif isinstance(temp, tuple) and len(temp) == 2:
            model, disease_encoder = temp
            symptom_list = None
        else:
            model = temp
            disease_encoder = None
            symptom_list = None
    print("[MODEL] Loaded disease_model (old).")
except Exception as e:
    print(f"[MODEL] Failed to load model/disease_model.pkl: {e}")
    model = None
    disease_encoder = None
    symptom_list = None

with open("model/disease_model_v2.pkl", "rb") as f:
    disease_model, disease_encoder = pickle.load(f)

with open("model/outcome_model.pkl", "rb") as f:
    outcome_model = pickle.load(f)


@app.route('/')
def index():
    username = session.get('username')
    return render_template('index.html', username=username)

# ----- Old model predict -----
@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        selected_symptoms = request.form.getlist('symptoms')

        # Create input vector and predict
        if symptom_list is None:
            return "Symptom list not loaded on server.", 500

        input_vector = [1 if symptom in selected_symptoms else 0 for symptom in symptom_list]
        input_df = pd.DataFrame([input_vector], columns=symptom_list)

        prediction = model.predict(input_df)[0]
        predicted_disease = disease_encoder.inverse_transform([prediction])[0] if disease_encoder is not None else str(prediction)

        # Load dataset to extract related symptoms
        df = pd.read_csv('dataset/dataset.csv')
        symptom_cols = [col for col in df.columns if col.startswith('Symptom_')]

        related_rows = df[df['Disease'] == predicted_disease]
        disease_symptoms = set()
        for _, row in related_rows.iterrows():
            for col in symptom_cols:
                symptom = row[col]
                if pd.notna(symptom) and str(symptom).strip().lower() != 'none' and str(symptom).strip() != '':
                    disease_symptoms.add(str(symptom).strip())

        disease_symptoms = sorted(disease_symptoms)

        # Save history for old model (if logged in)
        if 'username' in session:
            username = session['username']
            symptoms_str = ", ".join(selected_symptoms)
            timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

            conn = get_db_connection()
            if conn:
                try:
                    cursor = conn.cursor()
                    insert_query = """
                        INSERT INTO history (username, symptoms, prediction, timestamp)
                        VALUES (%s, %s, %s, %s)
                    """
                    cursor.execute(insert_query, (username, symptoms_str, predicted_disease, timestamp))
                    conn.commit()
                except Error as e:
                    print(f"[DB] Failed to insert old-model history: {e}")
                finally:
                    cursor.close()
                    conn.close()

        return render_template('result.html', disease=predicted_disease, symptoms=disease_symptoms)

    return render_template('predict.html', symptoms=symptom_list)


@app.route('/predict_v2', methods=['GET', 'POST']) 
def predict_v2():
    if request.method == 'POST':
        # Get numeric values from form
        fever = int(request.form.get('fever', 0))
        cough = int(request.form.get('cough', 0))
        fatigue = int(request.form.get('fatigue', 0))
        breathing = int(request.form.get('breathing', 0))
        age = int(request.form.get('age', 0))
        gender = int(request.form.get('gender', 0))
        bp = int(request.form.get('bp', 0))
        cholesterol = int(request.form.get('cholesterol', 0))

        # Create DataFrame with correct column names
        input_df = pd.DataFrame(
            [[fever, cough, fatigue, breathing, age, gender, bp, cholesterol]],
            columns=[
                "Fever", "Cough", "Fatigue", "Difficulty Breathing",
                "Age", "Gender", "Blood Pressure", "Cholesterol Level"
            ]
        )

        # Disease predictions
        probs = disease_model.predict_proba(input_df)[0]
        top_indices = np.argsort(probs)[::-1][:3]  # top 3

        top_diseases = []
        for idx in top_indices:
            disease_name = disease_encoder.inverse_transform([idx])[0]
            probability = round(float(probs[idx]) * 100, 2)
            remedy_text = remedy_dict_v2.get(disease_name, "No remedy available.")
            top_diseases.append({
                "name": disease_name,
                "probability": probability,
                "remedy": remedy_text
            })


        # Outcome prediction
        outcome_pred = int(outcome_model.predict(input_df)[0])
        outcome_text = "Positive" if outcome_pred == 1 else "Negative"

        # Store in MySQL
        conn = get_db_connection()
        cursor = conn.cursor()
        
        sql = """
            INSERT INTO prediction_history 
            (username, fever, cough, fatigue, breathing, age, gender, bp, cholesterol,
            top_disease_1, probability_1, remedy_1,
            top_disease_2, probability_2, remedy_2,
            top_disease_3, probability_3, remedy_3,
            outcome)
            VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,
                    %s,%s,%s,
                    %s,%s,%s,
                    %s,%s,%s,
                    %s)
        """
        cursor.execute(sql, (
            session['username'],  # 👈 store logged-in user's ID
            fever, cough, fatigue, breathing, age, gender, bp, cholesterol,
            top_diseases[0]['name'], top_diseases[0]['probability'], top_diseases[0]['remedy'],
            top_diseases[1]['name'], top_diseases[1]['probability'], top_diseases[1]['remedy'],
            top_diseases[2]['name'], top_diseases[2]['probability'], top_diseases[2]['remedy'],
            outcome_text
        ))
        conn.commit()
        conn.close()

        return render_template('result_v2.html',
                               top_diseases=top_diseases,
                               outcome=outcome_text)

    return render_template('predict_v2.html')




# ----- History for old model -----
@app.route('/history')
def view_history():
    if 'username' not in session:
        return redirect('/login')

    username = session['username']
    conn = get_db_connection()
    history = []
    if conn:
        try:
            cursor = conn.cursor(dictionary=True)
            cursor.execute("SELECT symptoms, prediction, timestamp FROM history WHERE username = %s ORDER BY timestamp DESC", (username,))
            history = cursor.fetchall()
        except Error as e:
            print(f"[DB] Failed to fetch history: {e}")
        finally:
            cursor.close()
            conn.close()

    return render_template('history.html', history=history)

@app.route('/admin_history')
def admin_history():
    if 'username' not in session or session.get('role') != 'admin':
        return "Access denied", 403

    conn = get_db_connection()
    records = []
    if conn:
        try:
            cursor = conn.cursor(dictionary=True)
            cursor.execute("SELECT * FROM history ORDER BY timestamp DESC")
            records = cursor.fetchall()
        except Error as e:
            print(f"[DB] Failed to fetch admin history: {e}")
        finally:
            cursor.close()
            conn.close()

    return render_template('admin_history.html', history=records)

# ----- History for v2 model (user) -----
@app.route('/history_v2')
def history_v2():
    if 'username' not in session:
        return redirect(url_for('login'))

    username = session['username']

    conn = get_db_connection()
    cursor = conn.cursor(dictionary=True)

    cursor.execute("""
        SELECT * FROM prediction_history
        WHERE username = %s
        ORDER BY predicted_at DESC
    """, (username,))
    
    history = cursor.fetchall()
    conn.close()

    return render_template('history_v2.html', history=history)

# ----- Admin history for v2 -----


@app.route('/admin_history_v2')
def admin_history_v2():
    if 'username' not in session or session.get('role') != 'admin':
        return redirect(url_for('login'))

    conn = get_db_connection()
    cursor = conn.cursor(dictionary=True)

    cursor.execute("""
        SELECT *
        FROM prediction_history
        ORDER BY predicted_at DESC
    """)
    
    history = cursor.fetchall()
    conn.close()

    return render_template('admin_history_v2.html', history=history)



# ----- Auth routes -----
@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form['username']
        password = generate_password_hash(request.form['password'])

        conn = get_db_connection()
        if not conn:
            return "DB connection error", 500
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
        if not conn:
            return "DB connection error", 500
        cursor = conn.cursor(dictionary=True)
        cursor.execute("SELECT * FROM users WHERE username = %s", (username,))
        user = cursor.fetchone()
        cursor.close()
        conn.close()

        if user and check_password_hash(user['password'], password):
            session['username'] = user['username']
            session['role'] = user.get('role', None)
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
