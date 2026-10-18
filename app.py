import os
import sqlite3
import joblib

from flask import Flask, render_template, request, redirect, session
from pathlib import Path
from dotenv import load_dotenv

ENV_FILE = Path(__file__).resolve().parent / ".env"
load_dotenv(ENV_FILE, override=True)


app = Flask(__name__)

app.secret_key = os.getenv("FLASK_SECRET_KEY")

if not app.secret_key:
    raise RuntimeError(
        "FLASK_SECRET_KEY is missing. Set it in your .env file."
    )

# ---------------- ADMIN LOGIN ----------------

@app.route("/admin/login", methods=["GET", "POST"])
def admin_login():

    if request.method == "POST":

        username = request.form["username"]
        password = request.form["password"]

        
        if (
            os.getenv("ADMIN_USERNAME")
            and os.getenv("ADMIN_PASSWORD")
            and username == os.getenv("ADMIN_USERNAME")
            and password == os.getenv("ADMIN_PASSWORD")
        ):
            session["admin_logged_in"] = True
            return redirect("/admin")


        return render_template(
            "login.html",
            error="Invalid username or password"
        )

    return render_template("login.html")

# ---------------- ADMIN LOGOUT ----------------

@app.route("/admin/logout")
def admin_logout():

    session.pop("admin_logged_in", None)

    return redirect("/admin/login")

# Load trained ML model
model = joblib.load("complaint_model.pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")


# ---------------- AI COMPLAINT ANALYZER ----------------

def analyze_complaint(complaint):

    # Convert complaint text into TF-IDF features
    complaint_vector = vectorizer.transform([complaint])

    
    # Predict category using trained ML model
    category = model.predict(complaint_vector)[0]


    # Get AI confidence
    probabilities = model.predict_proba(complaint_vector)[0]
    confidence = max(probabilities) * 100

    category_probabilities = sorted(
        zip(model.classes_, probabilities),
        key=lambda item: item[1],
        reverse=True
    )

    top_three = [
        (category_name, round(probability * 100, 2))
        for category_name, probability in category_probabilities[:3]
    ]

    print("Top 3 predicted categories:", top_three)

    # Priority detection
    text = complaint.lower()

    if any(word in text for word in [
            "emergency",
            "danger",
            "fire",
            "theft",
            "unsafe",
            "no water",
            "power failure",
            "urgent",
            "open wire",
            "electric shock",
            "electrical shock",
            "shock",
            "sparking",
            "short circuit",
            "stolen",
            "robbery",
            "unsafe"
        ]):
            priority = "High"

    elif any(word in text for word in [
            "not working",
            "stopped working",
            "broken",
            "problem",
            "issue",
            "leak",
            "slow",
            "damaged"
        ]):
            priority = "Medium"

    else:
            priority = "Low"

        # Generate a simple explanation for the AI prediction
    explanation = (
            f"The complaint appears to be related to {category} issues. "
            f"The system classified its priority as {priority} "
            f"with an AI confidence of {confidence:.2f}%."
        )

    return category, priority, confidence, explanation


# ---------------- DATABASE ----------------

def init_db():
    conn = sqlite3.connect("complaints.db")
    cursor = conn.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS complaints (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            complaint TEXT NOT NULL,
            category TEXT DEFAULT 'General',
            priority TEXT DEFAULT 'Low',
            confidence REAL DEFAULT 0,
            explanation TEXT DEFAULT '',
            status TEXT DEFAULT 'Pending'
        )
    """)

    # Add new columns if an older database already exists
    cursor.execute("PRAGMA table_info(complaints)")
    columns = [column[1] for column in cursor.fetchall()]

    if "category" not in columns:
        cursor.execute(
            "ALTER TABLE complaints ADD COLUMN category TEXT DEFAULT 'General'"
        )

    if "priority" not in columns:
        cursor.execute(
            "ALTER TABLE complaints ADD COLUMN priority TEXT DEFAULT 'Low'"
        )

    if "confidence" not in columns:
        cursor.execute(
            "ALTER TABLE complaints ADD COLUMN confidence REAL DEFAULT 0"
        )

    if "explanation" not in columns:
        cursor.execute(
           "ALTER TABLE complaints ADD COLUMN explanation TEXT DEFAULT ''"
        )

    conn.commit()
    conn.close()


# ---------------- HOME PAGE ----------------

@app.route("/")
def home():
    return render_template("index.html")


# ---------------- SUBMIT COMPLAINT ----------------

@app.route("/submit", methods=["POST"])
def submit():

    name = request.form.get("name", "").strip()
    complaint = request.form.get("complaint", "").strip()

    if not name or not complaint:
        return "Name and complaint are required.", 400

    if len(name) > 100:
        return "Name must be 100 characters or fewer.", 400

    if len(complaint) > 2000:
        return "Complaint must be 2000 characters or fewer.", 400

    # AI analyzes the complaint
    category, priority, confidence, explanation = analyze_complaint(complaint)

    # Save complaint into database
    conn = sqlite3.connect("complaints.db")
    cursor = conn.cursor()

    cursor.execute(
        """
        INSERT INTO complaints
        (name, complaint, category, priority, confidence, explanation)
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        (name, complaint, category, priority, confidence, explanation)
    )

    conn.commit()

    # Get the ID of the newly submitted complaint
    complaint_id = cursor.lastrowid

    conn.close()

    print("Name:", name)
    print("Complaint:", complaint)
    print("Category:", category)
    print("Priority:", priority)
    print("AI Match Score:", round(confidence, 2), "%")
    print("Complaint ID:", complaint_id)

    return render_template(
        "success.html",
        complaint_id=complaint_id
    )

# ---------------- TRACK COMPLAINT ----------------

@app.route("/track", methods=["GET", "POST"])
def track():

    complaint = None

    if request.method == "POST":

        complaint_id = request.form["complaint_id"]

        conn = sqlite3.connect("complaints.db")
        cursor = conn.cursor()

        cursor.execute(
            """
            SELECT id, name, complaint, category, priority, status
            FROM complaints
            WHERE id = ?
            """,
            (complaint_id,)
        )

        complaint = cursor.fetchone()

        conn.close()

    return render_template(
        "track.html",
        complaint=complaint
    )


# ---------------- ADMIN DASHBOARD ----------------

@app.route("/admin")
def admin():

    if not session.get("admin_logged_in"):
        return redirect("/admin/login")

    conn = sqlite3.connect("complaints.db")
    cursor = conn.cursor()

    # Get all complaints
    cursor.execute("""
        SELECT id, name, complaint, category, priority, status
        FROM complaints
        ORDER BY id ASC
    """)

    complaints = cursor.fetchall()

    # Total complaints
    cursor.execute("SELECT COUNT(*) FROM complaints")
    total_complaints = cursor.fetchone()[0]

    # Pending complaints
    cursor.execute(
        "SELECT COUNT(*) FROM complaints WHERE status = ?",
        ("Pending",)
    )
    pending_complaints = cursor.fetchone()[0]

    # In Progress complaints
    cursor.execute(
         "SELECT COUNT(*) FROM complaints WHERE status = ?",
         ("In Progress",)
    )
    in_progress_complaints = cursor.fetchone()[0]

    # Resolved complaints
    cursor.execute(
        "SELECT COUNT(*) FROM complaints WHERE status = ?",
        ("Resolved",)
    )
    resolved_complaints = cursor.fetchone()[0]

    # High priority complaints
    cursor.execute(
        "SELECT COUNT(*) FROM complaints WHERE priority = ?",
        ("High",)
    )
    high_priority = cursor.fetchone()[0]

    # Medium priority complaints
    cursor.execute(
        "SELECT COUNT(*) FROM complaints WHERE priority = ?",
        ("Medium",)
    )
    medium_priority = cursor.fetchone()[0]

    # Low priority complaints
    cursor.execute(
        "SELECT COUNT(*) FROM complaints WHERE priority = ?",
        ("Low",)
    )
    low_priority = cursor.fetchone()[0]

    # Category-wise complaint count
    cursor.execute("""
        SELECT category, COUNT(*)
        FROM complaints
        GROUP BY category
    """)

    category_data = cursor.fetchall()

    conn.close()

    return render_template(
        "admin.html",
        complaints=complaints,
        total_complaints=total_complaints,
        pending_complaints=pending_complaints,
        in_progress_complaints=in_progress_complaints,
        resolved_complaints=resolved_complaints,
        high_priority=high_priority,
        medium_priority=medium_priority,
        low_priority=low_priority,
        category_data=category_data
    )

# ---------------- COMPLAINT DETAILS ----------------

@app.route("/admin/complaint/<int:complaint_id>")
def complaint_details(complaint_id):

    if not session.get("admin_logged_in"):
        return redirect("/admin/login")

    conn = sqlite3.connect("complaints.db")
    cursor = conn.cursor()

    cursor.execute(
        """
        SELECT id, name, complaint, category, priority, confidence, explanation, status
        FROM complaints
        WHERE id = ?
        """,
        (complaint_id,)
    )

    complaint = cursor.fetchone()

    conn.close()

    if complaint is None:
        return "Complaint not found", 404

    return render_template(
        "complaint_details.html",
        complaint=complaint
    )

# ---------------- RUN APPLICATION ----------------

@app.route("/update_status/<int:complaint_id>", methods=["POST"])
def update_status(complaint_id):

    if not session.get("admin_logged_in"):
        return redirect("/admin/login")

    status = request.form.get("status", "").strip()

    allowed_statuses = {"Pending", "In Progress", "Resolved"}

    if status not in allowed_statuses:
        return "Invalid complaint status.", 400

    conn = sqlite3.connect("complaints.db")
    cursor = conn.cursor()

    cursor.execute(
        "UPDATE complaints SET status = ? WHERE id = ?",
        (status, complaint_id)
    )

    conn.commit()
    conn.close()

    return redirect("/admin")


if __name__ == "__main__":
    init_db()
    app.run(debug=True)