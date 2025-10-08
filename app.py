from flask import Flask, render_template, request, redirect, session, url_for, send_file, flash, Response, jsonify
import pickle
import numpy as np
import pandas as pd
from werkzeug.security import generate_password_hash, check_password_hash
from datetime import datetime, date, timedelta
import json
import mysql.connector
from mysql.connector import Error
from remedies_v2 import remedy_dict_v2
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from reportlab.lib import colors
from reportlab.lib.units import inch
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import Paragraph
import io
import os
import csv
from functools import wraps


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

def admin_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        if 'username' not in session:
            return redirect(url_for('login'))
        # role may be lowercase/uppercase depending on DB; normalize
        role = session.get('role') or ''
        if str(role).lower() != 'admin':
            return "Access denied", 403
        return f(*args, **kwargs)
    return decorated


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

@app.route('/download_report_v2')
def download_report_v2():

    username = session.get('username', 'Guest')

    # Collect params
    top1 = request.args.get('top1', 'Unknown')
    prob1 = request.args.get('prob1', '0')
    top2 = request.args.get('top2', 'Unknown')
    prob2 = request.args.get('prob2', '0')
    top3 = request.args.get('top3', 'Unknown')
    prob3 = request.args.get('prob3', '0')
    outcome = request.args.get('outcome', 'N/A')

    # Remedies
    from remedies_v2 import remedy_dict_v2
    remedies = []
    for disease in [top1, top2, top3]:
        if disease in remedy_dict_v2:
            remedies.append(f"<b>{disease}:</b> {remedy_dict_v2[disease]}")

    buffer = io.BytesIO()
    p = canvas.Canvas(buffer, pagesize=A4)
    width, height = A4

    # === HEADER BAR ===
    p.setFillColor(colors.HexColor("#154360"))
    p.rect(0, height - 100, width, 100, fill=1, stroke=0)

    logo_path = os.path.join("static", "images", "logo.png")
    if os.path.exists(logo_path):
        p.drawImage(logo_path, 40, height - 90, width=70, height=70, mask='auto')

    p.setFillColor(colors.white)
    p.setFont("Helvetica-Bold", 24)
    p.drawString(130, height - 55, "AI Disease Prediction System")

    # === PATIENT INFO ===
    y = height - 140
    p.setFillColor(colors.HexColor("#f4f6f7"))
    p.roundRect(40, y - 70, width - 80, 60, 12, fill=1, stroke=0)

    p.setFillColor(colors.HexColor("#2c3e50"))
    p.setFont("Helvetica", 11)
    p.drawString(60, y - 30, f"👤 Patient: {username}")
    p.drawString(60, y - 50, f"📅 Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    p.drawString(320, y - 30, "🧬 Model: Prediction V2")

    # === PREDICTIONS ===
    y -= 110
    p.setFillColor(colors.HexColor("#2874A6"))
    p.roundRect(40, y - 30, width - 80, 30, 8, fill=1, stroke=0)
    p.setFillColor(colors.white)
    p.setFont("Helvetica-Bold", 13)
    p.drawString(50, y - 18, "Predicted Diseases & Confidence")

    y -= 50
    diseases = [(top1, prob1), (top2, prob2), (top3, prob3)]
    p.setFont("Helvetica", 11)

    for disease, prob in diseases:
        # disease label
        p.setFillColor(colors.HexColor("#2c3e50"))
        p.drawString(60, y, f"● {disease} ({prob}%)")

        # bar aligned
        try:
            prob_val = float(prob)
        except:
            prob_val = 0
        bar_len = (prob_val / 100) * 200
        p.setFillColor(colors.HexColor("#5DADE2"))
        p.roundRect(300, y - 6, bar_len, 8, 3, fill=1, stroke=0)

        y -= 25

    # add confidence scale
    p.setFont("Helvetica-Oblique", 9)
    p.setFillColor(colors.HexColor("#7f8c8d"))
    p.drawString(300, y, "0%")
    p.drawRightString(500, y, "100%")

    # === OUTCOME ===
    y -= 60
    outcome_color = colors.HexColor("#27ae60") if outcome == "Negative" else colors.HexColor("#c0392b")
    icon = "✅" if outcome == "Negative" else "❌"

    p.setFillColor(outcome_color)
    p.roundRect(70, y - 90, width - 140, 90, 18, fill=1, stroke=0)

    p.setFillColor(colors.white)
    p.setFont("Helvetica-Bold", 22)
    p.drawCentredString(width / 2, y - 50, f"{icon} Final Outcome: {outcome}")

    # === REMEDIES ===
    if remedies:
        y -= 140
        p.setFillColor(colors.HexColor("#138D75"))
        p.roundRect(40, y - 30, width - 80, 30, 8, fill=1, stroke=0)
        p.setFillColor(colors.white)
        p.setFont("Helvetica-Bold", 13)
        p.drawString(50, y - 18, "Suggested Remedies")

        y -= 40
        styles = getSampleStyleSheet()
        remedy_style = ParagraphStyle(
            "remedyStyle",
            parent=styles["Normal"],
            fontName="Helvetica",
            fontSize=11,
            textColor=colors.HexColor("#2c3e50"),
            leading=14,
        )

        for r in remedies:
            para = Paragraph(f"✓ {r}", remedy_style)
            w, h = para.wrap(width - 120, 100)
            para.drawOn(p, 60, y - h)
            y -= h + 12

    # === FOOTER ===
    p.setFont("Helvetica-Oblique", 9)
    p.setFillColor(colors.HexColor("#7f8c8d"))
    p.drawCentredString(width / 2, 30, "© 2025 AI Disease Prediction System | Report generated by AI")

    p.showPage()
    p.save()
    buffer.seek(0)

    return send_file(buffer, as_attachment=True,
                     download_name=f"prediction_report_v2_{username}.pdf",
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

@app.route('/admin/reset_password/<int:user_id>')
def reset_password(user_id):
    if 'username' not in session or session.get('role') != 'admin':
        return redirect(url_for('login'))

    conn = get_db_connection()
    cursor = conn.cursor()
    new_password = generate_password_hash("12345")  # default reset
    cursor.execute("UPDATE users SET password=%s WHERE id=%s", (new_password, user_id))
    conn.commit()
    conn.close()

    return redirect(url_for('admin_users'))


@app.route('/admin/change_role/<int:user_id>')
def change_role(user_id):
    if 'username' not in session or session.get('role') != 'admin':
        return redirect(url_for('login'))

    conn = get_db_connection()
    cursor = conn.cursor(dictionary=True)
    cursor.execute("SELECT role FROM users WHERE id=%s", (user_id,))
    user = cursor.fetchone()
    new_role = "admin" if user['role'] == "user" else "user"
    cursor.execute("UPDATE users SET role=%s WHERE id=%s", (new_role, user_id))
    conn.commit()
    conn.close()

    return redirect(url_for('admin_users'))


@app.route('/admin/delete_user/<int:user_id>')
def delete_user(user_id):
    if 'username' not in session or session.get('role') != 'admin':
        return redirect(url_for('login'))

    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("DELETE FROM users WHERE id=%s", (user_id,))
    conn.commit()
    conn.close()

    return redirect(url_for('admin_users'))


@app.route('/logout')
def logout():
    session.clear()
    return redirect('/')

@app.route('/profile')
def profile():
    if 'username' not in session:
        return redirect(url_for('login'))

    username = session['username']
    conn = get_db_connection()
    user = {}
    total_predictions = 0
    last_prediction = None
    join_date = '—'
    account_type = session.get('role', 'User')

    if conn:
        try:
            cursor = conn.cursor(dictionary=True)

            # 1️⃣ Fetch user info
            cursor.execute("""
                SELECT username, role, created_at, email, phone, location
                FROM users WHERE username = %s
            """, (username,))
            user = cursor.fetchone() or {}

            # 2️⃣ Total predictions (combined)
            cursor.execute("""
                SELECT 
                    (SELECT COUNT(*) FROM history WHERE username=%s)
                    + 
                    (SELECT COUNT(*) FROM prediction_history WHERE username=%s)
                    AS total_predictions
            """, (username, username))
            total_predictions = cursor.fetchone()['total_predictions']

            # 3️⃣ Last prediction date (newest of both tables)
            cursor.execute("""
                SELECT MAX(ts) AS last_pred FROM (
                    SELECT timestamp AS ts FROM history WHERE username=%s
                    UNION ALL
                    SELECT predicted_at AS ts FROM prediction_history WHERE username=%s
                ) AS combined
            """, (username, username))
            result = cursor.fetchone()
            last_dt = result['last_pred'] if result else None
            last_prediction = last_dt.strftime("%b %d, %Y %I:%M %p") if last_dt else "—"

            # 4️⃣ Join date
            join_date = user.get('created_at')
            if join_date:
                try:
                    join_date = join_date.strftime("%b %d, %Y")
                except:
                    join_date = str(join_date)
            else:
                join_date = "—"

        except Exception as e:
            print("PROFILE ERROR:", e)
        finally:
            cursor.close()
            conn.close()

    return render_template(
        "profile.html",
        username=username,
        role=user.get("role", "User").capitalize(),
        join_date=join_date,
        total_predictions=total_predictions,
        last_prediction=last_prediction,
        email=user.get("email", ""),
        phone=user.get("phone", ""),
        location=user.get("location", "")
    )



@app.route('/about')
def about():
    return render_template("about.html")


@app.route('/contact', methods=["GET", "POST"])
def contact():
    if request.method == "POST":
        name = request.form.get("name")
        email = request.form.get("email")
        message = request.form.get("message")
        username = session.get("username", "Guest")

        if not all([name, email, message]):
            flash("⚠️ Please fill all fields.", "danger")
            return redirect(url_for("contact"))

        conn = get_db_connection()
        if conn:
            try:
                cursor = conn.cursor()
                cursor.execute("""
                    INSERT INTO contact_messages (username, name, email, message, created_at)
                    VALUES (%s, %s, %s, %s, %s)
                """, (username, name, email, message, datetime.now()))
                conn.commit()
                flash("✅ Message sent successfully!", "success")
            except Exception as e:
                print("CONTACT ERROR:", e)
                flash("❌ Error saving message. Try again later.", "danger")
            finally:
                cursor.close()
                conn.close()
        else:
            flash("❌ Database connection error.", "danger")

        return redirect(url_for("contact"))

    return render_template("contact.html")

@app.route('/update_profile', methods=['POST'])
def update_profile():
    if 'username' not in session:
        return redirect(url_for('login'))

    username = session['username']
    email = request.form.get('email', '')
    phone = request.form.get('phone', '')
    location = request.form.get('location', '')

    conn = get_db_connection()
    if conn:
        try:
            cursor = conn.cursor()
            cursor.execute("""
                UPDATE users
                SET email=%s, phone=%s, location=%s
                WHERE username=%s
            """, (email, phone, location, username))
            conn.commit()
            flash("✅ Profile updated successfully!", "success")
        except Exception as e:
            print("UPDATE PROFILE ERROR:", e)
            flash("❌ Failed to update profile.", "danger")
        finally:
            cursor.close()
            conn.close()

    return redirect(url_for('profile'))

@app.route('/admin/users')
@admin_required
def admin_users_enhanced():
    """
    Show enhanced user management UI: filtering, sorting, pagination.
    Query params:
      - q: search query (username/email)
      - role: user/admin
      - active: 1/0
      - page: int
    """
    q = request.args.get('q', '').strip()
    role_filter = request.args.get('role', '').strip()
    active_filter = request.args.get('active', '').strip()
    page = int(request.args.get('page', 1))
    per_page = 20
    offset = (page - 1) * per_page

    conn = get_db_connection()
    users = []
    total = 0
    if conn:
        try:
            cursor = conn.cursor(dictionary=True)
            base = "SELECT SQL_CALC_FOUND_ROWS id, username, role, created_at, email, phone, location, active, last_login FROM users WHERE 1=1"
            params = []
            if q:
                base += " AND (username LIKE %s OR email LIKE %s)"
                params.extend([f"%{q}%", f"%{q}%"])
            if role_filter:
                base += " AND role = %s"
                params.append(role_filter)
            if active_filter in ('0', '1'):
                base += " AND active = %s"
                params.append(active_filter)

            base += " ORDER BY id ASC LIMIT %s OFFSET %s"
            params.extend([per_page, offset])

            cursor.execute(base, tuple(params))
            users = cursor.fetchall()

            cursor.execute("SELECT FOUND_ROWS() AS total")
            total = cursor.fetchone().get('total', 0)
        except Exception as e:
            print("[ADMIN USERS] DB error:", e)
        finally:
            cursor.close()
            conn.close()

    total_pages = (total + per_page - 1) // per_page if total else 1
    return render_template('admin_users_enhanced.html',
                           users=users,
                           page=page,
                           total_pages=total_pages,
                           q=q,
                           role_filter=role_filter,
                           active_filter=active_filter,
                           total=total)

@app.route('/admin/user/<int:user_id>/edit', methods=['POST'])
@admin_required
def admin_edit_user(user_id):
    # allow admin to update email/phone/location/role/active
    email = request.form.get('email') or ''
    phone = request.form.get('phone') or ''
    location = request.form.get('location') or ''
    role = request.form.get('role') or 'user'
    active = 1 if request.form.get('active') == '1' else 0

    conn = get_db_connection()
    if conn:
        try:
            cursor = conn.cursor()
            cursor.execute("""
                UPDATE users
                SET email=%s, phone=%s, location=%s, role=%s, active=%s
                WHERE id=%s
            """, (email, phone, location, role, active, user_id))
            conn.commit()
            flash("✅ User updated.", "success")
        except Exception as e:
            print("[ADMIN EDIT USER] Error:", e)
            flash("❌ Failed to update user.", "danger")
        finally:
            cursor.close()
            conn.close()
    return redirect(url_for('admin_users_enhanced'))

@app.route('/admin/user/<int:user_id>/toggle_active')
@admin_required
def admin_toggle_active(user_id):
    conn = get_db_connection()
    if conn:
        try:
            cursor = conn.cursor(dictionary=True)
            cursor.execute("SELECT active FROM users WHERE id=%s", (user_id,))
            row = cursor.fetchone()
            if row is None:
                flash("User not found.", "danger")
                return redirect(url_for('admin_users_enhanced'))
            new_active = 0 if row['active'] == 1 else 1
            cursor.execute("UPDATE users SET active=%s WHERE id=%s", (new_active, user_id))
            conn.commit()
            flash("✅ User status updated.", "success")
        except Exception as e:
            print("[ADMIN TOGGLE ACTIVE] Error:", e)
            flash("❌ Failed to update status.", "danger")
        finally:
            cursor.close()
            conn.close()
    return redirect(url_for('admin_users_enhanced'))

@app.route('/admin/user/<int:user_id>/reset_password')
@admin_required
def admin_reset_password(user_id):
    # set default password to '12345' (hashed)
    try:
        new_hash = generate_password_hash("12345")
        conn = get_db_connection()
        if conn:
            cursor = conn.cursor()
            cursor.execute("UPDATE users SET password=%s WHERE id=%s", (new_hash, user_id))
            conn.commit()
            cursor.close()
            conn.close()
            flash("✅ Password reset to default (12345).", "success")
    except Exception as e:
        print("[ADMIN RESET PW] Error:", e)
        flash("❌ Failed to reset password.", "danger")
    return redirect(url_for('admin_users_enhanced'))

@app.route('/admin/users/export')
@admin_required
def admin_export_users():
    q = request.args.get('q', '').strip()
    role_filter = request.args.get('role', '').strip()
    active_filter = request.args.get('active', '').strip()

    conn = get_db_connection()
    rows = []
    if conn:
        try:
            cursor = conn.cursor()
            base = "SELECT id, username, role, email, phone, location, active, created_at, last_login FROM users WHERE 1=1"
            params = []
            if q:
                base += " AND (username LIKE %s OR email LIKE %s)"
                params.extend([f"%{q}%", f"%{q}%"])
            if role_filter:
                base += " AND role = %s"
                params.append(role_filter)
            if active_filter in ('0', '1'):
                base += " AND active = %s"
                params.append(active_filter)
            base += " ORDER BY id ASC"
            cursor.execute(base, tuple(params))
            rows = cursor.fetchall()
        except Exception as e:
            print("[ADMIN EXPORT] Error:", e)
        finally:
            cursor.close()
            conn.close()

    # Generate CSV in memory
    output = io.StringIO()
    writer = csv.writer(output)
    header = ['id', 'username', 'role', 'email', 'phone', 'location', 'active', 'created_at', 'last_login']
    writer.writerow(header)
    for row in rows:
        writer.writerow(row)

    response = Response(output.getvalue(), mimetype='text/csv')
    response.headers['Content-Disposition'] = 'attachment; filename=users_export.csv'
    return response

@app.route('/admin/dashboard')
@admin_required
def admin_dashboard():
    """
    Admin analytics dashboard: summary cards, daily predictions (last 14 days),
    top predicted diseases, user stats.
    """
    conn = get_db_connection()
    # default safe values
    total_users = 0
    active_users = 0
    total_predictions = 0
    predictions_last_14 = []  # list of (date_str, count)
    top_diseases = []  # list of (disease, count)

    if conn:
        try:
            cursor = conn.cursor(dictionary=True)

            # 1) users counts
            cursor.execute("SELECT COUNT(*) AS cnt FROM users")
            row = cursor.fetchone()
            total_users = row['cnt'] if row else 0

            cursor.execute("SELECT COUNT(*) AS cnt FROM users WHERE active = 1")
            row = cursor.fetchone()
            active_users = row['cnt'] if row else 0

            # 2) total predictions (both tables)
            # history uses `timestamp`, prediction_history uses `predicted_at`
            cursor.execute("""
                SELECT
                  (SELECT COUNT(*) FROM history) + (SELECT COUNT(*) FROM prediction_history) AS total
            """)
            row = cursor.fetchone()
            total_predictions = row['total'] if row else 0

            # 3) daily predictions for last 14 days (merged from both tables)
            # build date range
            today = date.today()
            start_dt = today - timedelta(days=13)  # 14 days inclusive
            # Query: group by date over union
            cursor.execute("""
                SELECT d AS day, SUM(cnt) AS total
                FROM (
                  SELECT DATE(predicted_at) AS d, COUNT(*) AS cnt
                  FROM prediction_history
                  WHERE predicted_at >= %s
                  GROUP BY DATE(predicted_at)
                  UNION ALL
                  SELECT DATE(timestamp) AS d, COUNT(*) AS cnt
                  FROM history
                  WHERE timestamp >= %s
                  GROUP BY DATE(timestamp)
                ) t
                GROUP BY d
                ORDER BY d ASC
            """, (start_dt, start_dt))
            rows = cursor.fetchall()
            # Map row day->count
            counts_map = { (r['day'].strftime('%Y-%m-%d') if isinstance(r['day'], (date,)) else str(r['day'])): r['total'] for r in rows }


            # build full 14-day list (fill zeros)
            predictions_last_14 = []
            for i in range(14):
                d = start_dt + timedelta(days=i)
                ds = d.strftime('%Y-%m-%d')
                predictions_last_14.append({'date': ds, 'count': int(counts_map.get(ds, 0))})

            # 4) top predicted diseases (aggregate top_disease_1/2/3)
            cursor.execute("""
                SELECT disease, SUM(cnt) AS total
                FROM (
                  SELECT top_disease_1 AS disease, COUNT(*) AS cnt FROM prediction_history WHERE top_disease_1 IS NOT NULL GROUP BY top_disease_1
                  UNION ALL
                  SELECT top_disease_2 AS disease, COUNT(*) AS cnt FROM prediction_history WHERE top_disease_2 IS NOT NULL GROUP BY top_disease_2
                  UNION ALL
                  SELECT top_disease_3 AS disease, COUNT(*) AS cnt FROM prediction_history WHERE top_disease_3 IS NOT NULL GROUP BY top_disease_3
                ) AS u
                GROUP BY disease
                ORDER BY total DESC
                LIMIT 10
            """)
            rows = cursor.fetchall()
            top_diseases = [{'disease': r['disease'], 'count': int(r['total'])} for r in rows if r['disease']]

        except Exception as e:
            print("[ADMIN DASHBOARD] DB error:", e)
        finally:
            cursor.close()
            conn.close()

    # Prepare data for charts (JSON serializable)
    chart_dates = [p['date'] for p in predictions_last_14]
    chart_counts = [p['count'] for p in predictions_last_14]
    top_labels = [td['disease'] for td in top_diseases]
    top_counts = [td['count'] for td in top_diseases]

    print("===DEBUG=== dates:", chart_dates)
    print("===DEBUG=== counts:", chart_counts)

    return render_template('admin_dashboard.html',
                           total_users=total_users,
                           active_users=active_users,
                           total_predictions=total_predictions,
                           chart_dates=json.dumps(chart_dates),
                           chart_counts=json.dumps(chart_counts),
                           top_labels=json.dumps(top_labels),
                           top_counts=json.dumps(top_counts),
                           top_diseases=top_diseases)


if __name__ == '__main__':
    app.run(debug=True)
