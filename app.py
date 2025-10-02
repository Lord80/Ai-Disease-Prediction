from flask import Flask, render_template, request, redirect, session, url_for, send_file
import pickle
import numpy as np
import pandas as pd
from werkzeug.security import generate_password_hash, check_password_hash
from datetime import datetime
import mysql.connector
from mysql.connector import Error
from remedies_v2 import remedy_dict_v2
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from reportlab.lib import colors
from reportlab.lib.units import inch
import io
import os


app = Flask(__name__)

app.secret_key = 'your_secret_key_here'

try:
    from disease_info import disease_info as disease_info_map
except Exception as e:
    print(f"[WARN] Could not load disease_info: {e}")
    disease_info_map = {}


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


# load model 1
try:
    with open('model/disease_model.pkl', 'rb') as f:
        temp = pickle.load(f)
        if isinstance(temp, tuple) and len(temp) >= 3:
            model_old, disease_encoder_old, symptom_list = temp[0], temp[1], temp[2]
        elif isinstance(temp, tuple) and len(temp) == 2:
            model_old, disease_encoder_old = temp
            symptom_list = None
        else:
            model_old = temp
            disease_encoder_old = None
            symptom_list = None
    print("[MODEL] Loaded disease_model.pkl (old symptom-based).")
except Exception as e:
    print(f"[MODEL] Failed to load model/disease_model.pkl: {e}")
    model_old = None
    disease_encoder_old = None
    symptom_list = None

# load model 2
try:
    with open("model/disease_model_v2.pkl", "rb") as f:
        disease_model_v2, disease_encoder_v2 = pickle.load(f)
    print("[MODEL] Loaded disease_model_v2.pkl (v2).")
except Exception as e:
    print(f"[MODEL] Failed to load model/disease_model_v2.pkl: {e}")
    disease_model_v2 = None
    disease_encoder_v2 = None

try:
    with open("model/outcome_model.pkl", "rb") as f:
        outcome_model = pickle.load(f)
    print("[MODEL] Loaded outcome_model.pkl.")
except Exception as e:
    print(f"[MODEL] Failed to load model/outcome_model.pkl: {e}")
    outcome_model = None


@app.route('/')
def index():
    username = session.get('username')
    return render_template('index.html', username=username)


@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        selected_symptoms = request.form.getlist('symptoms')

        # Create input vector and predict
        if symptom_list is None:
            return "Symptom list not loaded on server.", 500

        input_vector = [
            1 if symptom in selected_symptoms else 0 for symptom in symptom_list]
        input_df = pd.DataFrame([input_vector], columns=symptom_list)

        # ---- Prediction + Confidence ----
        try:
            probs = model_old.predict_proba(input_df)[0]
            top_idx = int(np.argmax(probs))
            encoded_label = model_old.classes_[top_idx]
            predicted_disease = (
                disease_encoder_old.inverse_transform([encoded_label])[0]
                if disease_encoder_old is not None else str(encoded_label)
            )
            confidence = round(float(probs[top_idx]) * 100, 2)
        except Exception:
            pred_enc = model_old.predict(input_df)[0]
            predicted_disease = (
                disease_encoder_old.inverse_transform([pred_enc])[0]
                if disease_encoder_old is not None else str(pred_enc)
            )
            confidence = None

        # ---- Load dataset to extract related symptoms ----
        df = pd.read_csv('dataset/dataset.csv')
        symptom_cols = [
            col for col in df.columns if col.startswith('Symptom_')]

        related_rows = df[df['Disease'] == predicted_disease]
        disease_symptoms = set()
        for _, row in related_rows.iterrows():
            for col in symptom_cols:
                symptom = row[col]
                if pd.notna(symptom) and str(symptom).strip().lower() != 'none' and str(symptom).strip() != '':
                    disease_symptoms.add(str(symptom).strip())
        disease_symptoms = sorted(disease_symptoms)

        info = disease_info_map.get(predicted_disease, {})

        # ---- Build medical report fields ----
        description = info.get(
            "description",
            f"No curated description available for {predicted_disease} yet."
        )

        remedies_list = info.get("remedies")
        if not remedies_list:
            rem_text = remedy_dict_v2.get(predicted_disease)
            remedies_list = [rem_text] if rem_text else [
                "General care: rest, stay hydrated, and consider OTC symptom relief.",
                "Consult a clinician if symptoms persist or worsen."
            ]

        red_flags = info.get("red_flags", [
            "Severe or worsening shortness of breath.",
            "Persistent high fever (>39°C / 102.2°F) for more than 48 hours.",
            "Chest pain, confusion, fainting, or severe dehydration.",
            "Symptoms persist beyond 10–14 days or suddenly get worse."
        ])

        # ---- Save history ----
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
                    cursor.execute(
                        insert_query, (username, symptoms_str, predicted_disease, timestamp))
                    conn.commit()
                except Error as e:
                    print(f"[DB] Failed to insert history: {e}")
                finally:
                    cursor.close()
                    conn.close()

        return render_template(
            'result.html',
            disease=predicted_disease,
            symptoms=disease_symptoms,
            confidence=confidence,
            description=description,
            remedies=remedies_list,
            red_flags=red_flags
        )

    return render_template('predict.html', symptoms=symptom_list)


@app.route('/predict_v2', methods=['GET', 'POST'])
def predict_v2():
    if request.method == 'POST':
        fever = int(request.form.get('fever', 0))
        cough = int(request.form.get('cough', 0))
        fatigue = int(request.form.get('fatigue', 0))
        breathing = int(request.form.get('breathing', 0))
        age = int(request.form.get('age', 0))
        gender = int(request.form.get('gender', 0))
        bp = int(request.form.get('bp', 0))
        cholesterol = int(request.form.get('cholesterol', 0))

        input_df = pd.DataFrame(
            [[fever, cough, fatigue, breathing, age, gender, bp, cholesterol]],
            columns=[
                "Fever", "Cough", "Fatigue", "Difficulty Breathing",
                "Age", "Gender", "Blood Pressure", "Cholesterol Level"
            ]
        )

        probs = disease_model_v2.predict_proba(input_df)[0]
        top_indices = np.argsort(probs)[::-1][:3]

        top_diseases = []
        for idx in top_indices:
            disease_name = disease_encoder_v2.inverse_transform([idx])[0]
            probability = round(float(probs[idx]) * 100, 2)
            remedy_text = remedy_dict_v2.get(
                disease_name, "No remedy available.")
            top_diseases.append({
                "name": disease_name,
                "probability": probability,
                "remedy": remedy_text
            })

        # Outcome prediction
        outcome_pred = int(outcome_model.predict(input_df)[0])
        outcome_text = "Positive" if outcome_pred == 1 else "Negative"

        username = session.get('username')
        if not username:
            username = "Guest"  # Allow predictions as Guest

        # Store in MySQL
        conn = get_db_connection()
        cursor = conn.cursor()

        sql = """
            INSERT INTO prediction_history
            (username, fever, cough, fatigue, breathing, age, gender, bp, cholesterol,
            top_disease_1, top_disease_2, top_disease_3, outcome)
            VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,
                    %s,%s,%s,%s)
        """


        cursor.execute(sql, (
            username,
            fever, cough, fatigue, breathing, age, gender, bp, cholesterol,
            top_diseases[0]['name'],
            top_diseases[1]['name'],
            top_diseases[2]['name'],
            outcome_text
        ))
        conn.commit()
        conn.close()

        return render_template('result_v2.html',
                               top_diseases=top_diseases,
                               outcome=outcome_text)

    return render_template('predict_v2.html')


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

@app.route('/download_report')
def download_report():

    disease = request.args.get('disease', 'Unknown')
    confidence = request.args.get('confidence', 'N/A')
    description = request.args.get('description', 'No description available')
    symptoms = request.args.get('symptoms', '').split(",") if request.args.get('symptoms') else []
    remedies = request.args.get('remedies', '').split(",") if request.args.get('remedies') else []
    red_flags = request.args.get('red_flags', '').split(",") if request.args.get('red_flags') else []
    username = session.get('username', 'Guest')

    buffer = io.BytesIO()
    p = canvas.Canvas(buffer, pagesize=A4)
    width, height = A4

    # === HEADER BAR WITH LOGO + TITLE ===
    p.setFillColor(colors.HexColor("#1a5276"))
    p.rect(0, height - 90, width, 90, fill=1, stroke=0)

    logo_path = os.path.join("static", "images", "logo.png")
    if os.path.exists(logo_path):
        p.drawImage(logo_path, 40, height - 80, width=60, height=60, mask='auto')

    p.setFillColor(colors.white)
    p.setFont("Helvetica-Bold", 22)
    p.drawString(120, height - 50, "AI Disease Prediction System")

    # === PATIENT INFO BOX ===
    y = height - 120
    p.setFillColor(colors.HexColor("#f8f9f9"))
    p.roundRect(40, y - 50, width - 80, 60, 10, fill=1, stroke=0)

    p.setFillColor(colors.HexColor("#2c3e50"))
    p.setFont("Helvetica", 12)
    p.drawString(60, y - 20, f"👤 Patient: {username}")
    p.drawString(320, y - 20, f"📅 Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # === DISEASE PREDICTION ===
    y -= 90
    p.setFillColor(colors.HexColor("#f4f6f7"))
    p.roundRect(40, y - 70, width - 80, 80, 10, fill=1, stroke=0)

    p.setFillColor(colors.HexColor("#1a5276"))
    p.setFont("Helvetica-Bold", 14)
    p.drawString(60, y - 20, "Predicted Disease:")
    p.setFillColor(colors.black)
    p.setFont("Helvetica", 12)
    p.drawString(220, y - 20, disease)

    p.setFillColor(colors.HexColor("#1a5276"))
    p.setFont("Helvetica-Bold", 14)
    p.drawString(60, y - 45, "Confidence:")
    p.setFillColor(colors.black)
    p.setFont("Helvetica", 12)
    p.drawString(220, y - 45, f"{confidence}%")

    # === SECTION HELPER ===
    def section(title, color, y):
        p.setFillColor(color)
        p.rect(40, y - 28, width - 80, 28, fill=1, stroke=0)
        p.setFillColor(colors.white)
        p.setFont("Helvetica-Bold", 13)
        p.drawString(50, y - 17, title)
        return y - 45

    # === SUMMARY ===
    y = section("📖 Summary", colors.HexColor("#2980b9"), y - 100)
    p.setFillColor(colors.black)
    p.setFont("Helvetica", 11)
    text = p.beginText(50, y)
    text.setLeading(15)
    for line in description.split(". "):
        text.textLine(f"● {line.strip()}")
    p.drawText(text)
    y = text.getY() - 20

    # === SYMPTOMS ===
    if symptoms:
        y = section("📝 Related Symptoms", colors.HexColor("#8e44ad"), y)
        p.setFont("Helvetica", 11)
        p.setFillColor(colors.HexColor("#2c3e50"))
        for s in symptoms:
            p.drawString(60, y, f"● {s.strip()}")
            y -= 15
        y -= 10

    # === REMEDIES ===
    if remedies:
        y = section("✅ Suggested Remedies", colors.HexColor("#27ae60"), y)
        p.setFont("Helvetica", 11)
        p.setFillColor(colors.HexColor("#2c3e50"))
        for r in remedies:
            p.drawString(60, y, f"● {r.strip()}")
            y -= 15
        y -= 10

    # === RED FLAGS ===
    if red_flags:
        y = section("⚠️ When to Seek Medical Care", colors.HexColor("#c0392b"), y)
        p.setFont("Helvetica", 11)
        p.setFillColor(colors.HexColor("#2c3e50"))
        for f in red_flags:
            p.drawString(60, y, f"● {f.strip()}")
            y -= 15

    # === FOOTER ===
    p.setFont("Helvetica-Oblique", 9)
    p.setFillColor(colors.HexColor("#7f8c8d"))
    p.drawCentredString(width / 2, 30, "© 2025 AI Disease Prediction System")

    p.showPage()
    p.save()
    buffer.seek(0)

    return send_file(buffer, as_attachment=True,
                     download_name=f"prediction_report_{disease}.pdf",
                     mimetype='application/pdf')




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
    session.clear()
    return redirect('/')


if __name__ == '__main__':
    app.run(debug=True)
