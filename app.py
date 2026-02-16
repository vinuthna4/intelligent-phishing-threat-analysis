from flask import Flask, render_template, request, jsonify
import sqlite3
import pandas as pd
import re

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

app = Flask(__name__)

# -------------------- LOAD DATASET --------------------
df = pd.read_csv("phishing_dataset.csv")

# Label encoding
df["label"] = df["label"].map({"legitimate": 0, "phishing": 1})

# -------------------- FEATURE EXTRACTION --------------------
def extract_features(url):
    return [
        len(url),
        1 if "@" in url else 0,
        1 if "-" in url else 0,
        sum(1 for k in ["login", "bank", "verify", "secure", "update"] if k in url.lower())
    ]

X = df["url"].apply(extract_features).tolist()
y = df["label"].tolist()

# -------------------- DATASET SCALING --------------------
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# -------------------- ML MODEL TRAINING --------------------
model = LogisticRegression()
model.fit(X_scaled, y)

# -------------------- DATABASE --------------------
def init_db():
    conn = sqlite3.connect("phishing.db")
    cur = conn.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS attacks (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            url TEXT,
            prediction TEXT,
            risk INTEGER,
            severity TEXT,
            time DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.commit()
    conn.close()

init_db()

# -------------------- RISK SCORING --------------------
def risk_score(prob):
    if prob > 0.75:
        return 90, "High"
    elif prob > 0.4:
        return 60, "Medium"
    else:
        return 20, "Low"

# -------------------- ROUTES --------------------
@app.route("/")
def home():
    return render_template("index.html")

@app.route("/analyze", methods=["POST"])
def analyze():
    url = request.json["url"]

    features = extract_features(url)
    features_scaled = scaler.transform([features])
    prob = model.predict_proba(features_scaled)[0][1]

    risk, severity = risk_score(prob)
    prediction = "Phishing" if prob > 0.5 else "Legitimate"

    conn = sqlite3.connect("phishing.db")
    cur = conn.cursor()
    cur.execute(
        "INSERT INTO attacks (url, prediction, risk, severity) VALUES (?, ?, ?, ?)",
        (url, prediction, risk, severity)
    )
    conn.commit()
    conn.close()

    return jsonify({
        "prediction": prediction,
        "risk": risk,
        "severity": severity,
        "confidence": round(prob, 2)
    })

@app.route("/history")
def history():
    conn = sqlite3.connect("phishing.db")
    cur = conn.cursor()
    cur.execute("SELECT url, risk, severity, time FROM attacks")
    data = cur.fetchall()
    conn.close()
    return jsonify(data)

if __name__ == "__main__":
    app.run(debug=True)
