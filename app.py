"""
DWP Public Service Request Prioritization & Routing - Streamlit App
Single-file Streamlit app intended for Streamlit Community deployment (free tier).

Features:
- Citizen submission form (text) -> stored in local SQLite DB
- Simple text preprocessing and embedding using SentenceTransformers ('all-MiniLM-L6-v2')
- Rule-based fallback classifier + optional train-from-CSV to build a logistic-regression on embeddings
- Priority scoring (rule + learned urgency)
- Routing using a simple YAML-like Python dict (editable in Admin)
- Citizen status check by request_id
- Admin dashboard to view & export data, and to trigger retraining

Notes for deployment:
- Put this file as `app.py` in a new GitHub repo, add `requirements.txt` (listed below) and deploy to Streamlit Community.
- The first run downloads the sentence-transformers model (~40-80MB) and will create a local SQLite DB `dwp_requests.db`.

Requirements (put in requirements.txt):
# ----- requirements.txt -----
# streamlit
# sentence-transformers
# scikit-learn
# sqlalchemy
# joblib
# pandas
# numpy
# uvicorn (optional)
# pyyaml
# faiss-cpu (optional, not used by default)
# ----------------------------

"""

import streamlit as st
from datetime import datetime, timezone
import uuid
import sqlite3
import os
import json
import re
import pandas as pd
import numpy as np
from typing import Optional, Tuple, Dict, Any

# ML libs
from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import joblib

# DB setup
DB_PATH = 'dwp_requests.db'
MODEL_DIR = 'models'
os.makedirs(MODEL_DIR, exist_ok=True)

EMBED_MODEL_NAME = 'all-MiniLM-L6-v2'  # small, CPU-friendly
EMBED_MODEL_PATH = os.path.join(MODEL_DIR, 'embed_model')
CLASSIFIER_PATH = os.path.join(MODEL_DIR, 'classifier.joblib')
REGRESSOR_PATH = os.path.join(MODEL_DIR, 'eta_regressor.joblib')

# Routing table (editable in Admin UI)
ROUTING_TABLE = [
    {"match": {"category": "Benefit Payment Stopped"}, "assign": "income_support_team"},
    {"match": {"category": "Universal Credit"}, "assign": "uc_team"},
    {"match": {"category": "Pension"}, "assign": "pension_team"},
    {"match": {"vulnerability": True}, "assign": "vulnerable_cases_team"},
]

# Simple teams metadata
TEAMS = {
    'income_support_team': {'name': 'Income Support', 'sla_hours': 48, 'capacity': 20},
    'uc_team': {'name': 'Universal Credit Team', 'sla_hours': 72, 'capacity': 40},
    'pension_team': {'name': 'Pensions', 'sla_hours': 96, 'capacity': 10},
    'vulnerable_cases_team': {'name': 'Vulnerable Cases', 'sla_hours': 24, 'capacity': 5},
    'intake_general': {'name': 'General Intake', 'sla_hours': 168, 'capacity': 100},
}

# Initialize DB
def init_db():
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    cur = conn.cursor()
    cur.execute('''
        CREATE TABLE IF NOT EXISTS requests (
            request_id TEXT PRIMARY KEY,
            received_at TEXT,
            channel TEXT,
            subject TEXT,
            full_text TEXT,
            contact TEXT,
            postcode TEXT,
            status TEXT,
            category TEXT,
            vulnerability INTEGER,
            priority INTEGER,
            assigned_team TEXT,
            eta_hours INTEGER,
            embedding BLOB
        )
    ''')
    conn.commit()
    return conn

conn = init_db()

# Utility: store numpy array as bytes
import io
import pickle

def np_to_blob(arr: np.ndarray) -> bytes:
    return pickle.dumps(arr)

def blob_to_np(b: bytes) -> np.ndarray:
    return pickle.loads(b)

# Load embedding model lazily
@st.cache_resource(show_spinner=False)
def load_embed_model():
    return SentenceTransformer(EMBED_MODEL_NAME)

# Preprocess text
def preprocess(text: str) -> str:
    t = text.strip()
    t = re.sub(r"\s+", " ", t)
    return t

# Simple rule-based classifier fallback
def rule_based_classify(text: str) -> Tuple[str, bool, float]:
    lower = text.lower()
    vuln = False
    if any(x in lower for x in ["homeless", "no shelter", "sleeping rough"]):
        vuln = True
    if any(x in lower for x in ["benefit stopped", "payment stopped", "money stopped"]):
        return "Benefit Payment Stopped", vuln, 0.95
    if any(x in lower for x in ["universal credit", "uc claim", "claim stopped"]):
        return "Universal Credit", vuln, 0.9
    if any(x in lower for x in ["pension", "state pension"]):
        return "Pension", vuln, 0.9
    if any(x in lower for x in ["complaint", "service failure", "not responded"]):
        return "Complaint", vuln, 0.7
    return "General Enquiry", vuln, 0.5

# Compute priority score
def compute_priority(urgency_score: float, vulnerability: bool, sla_hours: int) -> int:
    # urgency_score in [0,1]
    base = int(urgency_score * 60)  # up to 60
    if vulnerability:
        base += 25
    sla_factor = max(0, 48 - sla_hours) // 2  # shorter SLAs get higher priority
    score = base + sla_factor
    return min(100, max(0, score))

# Routing
def route_to_team(category: str, vulnerability: bool) -> str:
    for rule in ROUTING_TABLE:
        m = rule.get('match', {})
        # match all keys present in m
        ok = True
        for k, v in m.items():
            if k == 'vulnerability':
                if bool(v) != bool(vulnerability):
                    ok = False
            else:
                if v != category:
                    ok = False
        if ok:
            return rule.get('assign')
    return 'intake_general'

# Save request
def save_request(payload: Dict[str, Any], embedding: Optional[np.ndarray]=None):
    rid = payload.get('request_id') or str(uuid.uuid4())
    received_at = payload.get('received_at') or datetime.now(timezone.utc).isoformat()
    cur = conn.cursor()
    emb_blob = np_to_blob(embedding) if embedding is not None else None
    cur.execute('''INSERT INTO requests (request_id, received_at, channel, subject, full_text, contact, postcode, status, category, vulnerability, priority, assigned_team, eta_hours, embedding)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)''', (
        rid, received_at, payload.get('channel'), payload.get('subject'), payload.get('full_text'), payload.get('contact'), payload.get('postcode'), payload.get('status', 'new'), payload.get('category'), int(payload.get('vulnerability', False)), int(payload.get('priority', 0)), payload.get('assigned_team'), payload.get('eta_hours'), emb_blob
    ))
    conn.commit()
    return rid

# Query request
def get_request(request_id: str) -> Optional[Dict[str, Any]]:
    cur = conn.cursor()
    cur.execute('SELECT * FROM requests WHERE request_id=?', (request_id,))
    row = cur.fetchone()
    if not row:
        return None
    cols = [d[0] for d in cur.description]
    rec = dict(zip(cols, row))
    if rec.get('embedding'):
        rec['embedding'] = blob_to_np(rec['embedding'])
    return rec

# Simple ETA regressor (if trained)
def predict_eta_hours(features: Dict[str, Any]) -> Optional[int]:
    if os.path.exists(REGRESSOR_PATH):
        reg = joblib.load(REGRESSOR_PATH)
        X = np.array([features['priority'], features.get('team_capacity', 10)]).reshape(1, -1)
        pred = reg.predict(X)[0]
        return int(max(1, round(pred)))
    return None

# Classify using model if available else rules
def classify_text(text: str) -> Tuple[str, bool, float, Optional[np.ndarray]]:
    text = preprocess(text)
    embed_model = load_embed_model()
    emb = embed_model.encode([text])[0]
    if os.path.exists(CLASSIFIER_PATH):
        clf = joblib.load(CLASSIFIER_PATH)
        cat = clf.predict([emb])[0]
        prob = max(clf.predict_proba([emb])[0]) if hasattr(clf, 'predict_proba') else 0.6
        # vulnerability simple heuristic using keywords
        vuln = bool(re.search(r'\b(elderly|disabled|homeless|no income|vulnerable)\b', text.lower()))
        return cat, vuln, float(prob), emb
    else:
        cat, vuln, prob = rule_based_classify(text)
        return cat, vuln, prob, emb

# Template status text
def generate_status_text(rec: Dict[str, Any]) -> str:
    if rec is None:
        return "Request not found. Please check your Request ID."
    lines = []
    lines.append(f"Request ID: {rec['request_id']}")
    lines.append(f"Received: {rec['received_at']}")
    lines.append(f"Status: {rec['status']}")
    lines.append(f"Category: {rec.get('category')}")
    lines.append(f"Assigned team: {TEAMS.get(rec.get('assigned_team'), {}).get('name', rec.get('assigned_team'))}")
    if rec.get('eta_hours'):
        eta = int(rec.get('eta_hours'))
        lines.append(f"Estimated resolution time: ~{eta} hours")
    else:
        sla = TEAMS.get(rec.get('assigned_team'), {}).get('sla_hours')
        if sla:
            lines.append(f"Target SLA: {sla} hours")
    lines.append(f"Priority score: {rec.get('priority')}")
    return "\n".join(lines)

# Processing pipeline for a new request
def process_and_store(channel: str, subject: str, text: str, contact: str, postcode: str) -> str:
    cat, vuln, prob, emb = classify_text(text)
    assigned = route_to_team(cat, vuln)
    sla = TEAMS.get(assigned, {}).get('sla_hours', 168)
    priority = compute_priority(prob, vuln, sla)
    features = {'priority': priority, 'team_capacity': TEAMS.get(assigned, {}).get('capacity', 10)}
    eta = predict_eta_hours(features) or sla
    payload = {
        'channel': channel,
        'subject': subject,
        'full_text': text,
        'contact': contact,
        'postcode': postcode,
        'status': 'queued',
        'category': cat,
        'vulnerability': vuln,
        'priority': priority,
        'assigned_team': assigned,
        'eta_hours': int(eta)
    }
    rid = save_request(payload, embedding=emb)
    return rid

# Admin: train classifier from uploaded CSV
def train_classifier_from_csv(df: pd.DataFrame, text_col='full_text', label_col='category'):
    st.info('Training classifier...')
    embed_model = load_embed_model()
    texts = df[text_col].astype(str).tolist()
    X = embed_model.encode(texts, show_progress_bar=True)
    y = df[label_col].astype(str).tolist()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=42)
    clf = LogisticRegression(max_iter=2000)
    clf.fit(X_train, y_train)
    preds = clf.predict(X_test)
    report = classification_report(y_test, preds, output_dict=False)
    joblib.dump(clf, CLASSIFIER_PATH)
    st.success('Classifier trained and saved to ' + CLASSIFIER_PATH)
    return report

# Admin: train simple ETA regressor
def train_eta_regressor_from_csv(df: pd.DataFrame, label_col='resolution_hours'):
    st.info('Training ETA regressor...')
    # features: priority (computed via heuristic), team_capacity (lookup)
    # require columns: full_text, category, assigned_team, resolution_hours, vulnerability(optional)
    # compute priority via rule-based classifier for training
    embed_model = load_embed_model()
    texts = df['full_text'].astype(str).tolist()
    X_emb = embed_model.encode(texts, show_progress_bar=True)
    # compute heuristic priority using rule-based classifier prob
    priorities = []
    capacities = []
    for i, row in df.iterrows():
        cat, vuln, prob = rule_based_classify(str(row['full_text']))
        sla = TEAMS.get(row.get('assigned_team'), {}).get('sla_hours', 168)
        pr = compute_priority(prob, vuln, sla)
        priorities.append(pr)
        capacities.append(TEAMS.get(row.get('assigned_team'), {}).get('capacity', 10))
    X_feat = np.column_stack([priorities, capacities])
    y = df[label_col].astype(float).values
    X_train, X_test, y_train, y_test = train_test_split(X_feat, y, test_size=0.15, random_state=42)
    reg = GradientBoostingRegressor()
    reg.fit(X_train, y_train)
    joblib.dump(reg, REGRESSOR_PATH)
    st.success('ETA regressor trained and saved to ' + REGRESSOR_PATH)
    return reg

# Streamlit app layout
st.set_page_config(page_title='DWP Public Service Routing (Demo)', layout='wide')
st.title('DWP — Service Request Prioritisation & Routing (Open-source demo)')

tabs = st.tabs(['Citizen — Submit', 'Citizen — Check Status', 'Admin'])

# --- Tab 1: Submit ---
with tabs[0]:
    st.header('Submit a service request')
    with st.form('submit_form'):
        channel = st.selectbox('Channel', ['Web form', 'Email', 'Phone-to-text', 'SMS'])
        contact = st.text_input('Contact (email or phone)')
        postcode = st.text_input('Postcode (optional)')
        subject = st.text_input('Subject (short)')
        text = st.text_area('Describe the issue (as much detail as possible)', height=200)
        submitted = st.form_submit_button('Submit request')
        if submitted:
            if not text.strip():
                st.error('Please enter the issue description.')
            else:
                with st.spinner('Processing request...'):
                    rid = process_and_store(channel, subject, text, contact, postcode)
                st.success('Request submitted successfully!')
                st.info('Your Request ID: ' + rid)
                st.markdown('You can use this ID in the *Check Status* tab to get updates.')

# --- Tab 2: Check Status ---
with tabs[1]:
    st.header('Check request status')
    qid = st.text_input('Enter your Request ID')
    if st.button('Get status'):
        if not qid.strip():
            st.error('Please enter a Request ID')
        else:
            rec = get_request(qid.strip())
            if not rec:
                st.error('Request not found. Check your ID.')
            else:
                st.code(generate_status_text(rec))
                st.subheader('Conversation / updates')
                # For demo: show last 200 chars of full text and category
                st.write('Category:', rec.get('category'))
                st.write('Priority score:', rec.get('priority'))
                st.write('Assigned team:', TEAMS.get(rec.get('assigned_team'), {}).get('name'))
                st.write('Full text (masked contact):')
                st.text(rec.get('full_text'))

# --- Tab 3: Admin ---
with tabs[2]:
    st.header('Admin dashboard')
    st.subheader('Routing table (editable)')
    with st.expander('Show/edit routing rules'):
        st.write('Routing rules are matched in order. The first match is used.')
        # display python dict as json
        rt_json = st.text_area('Routing table (JSON)', value=json.dumps(ROUTING_TABLE, indent=2), height=200)
        if st.button('Update routing table'):
            try:
                new_rt = json.loads(rt_json)
                ROUTING_TABLE.clear()
                ROUTING_TABLE.extend(new_rt)
                st.success('Routing table updated (in memory). Save by exporting below if required.')
            except Exception as e:
                st.error('Failed to parse JSON: ' + str(e))

    st.subheader('Teams')
    st.json(TEAMS)

    st.subheader('View requests')
    with st.expander('Query & export'):
        q = st.text_input('Filter by category (leave blank for all)')
        cur = conn.cursor()
        if q.strip():
            cur.execute('SELECT request_id, received_at, status, category, priority, assigned_team FROM requests WHERE category LIKE ?', ('%'+q+'%',))
        else:
            cur.execute('SELECT request_id, received_at, status, category, priority, assigned_team FROM requests ORDER BY received_at DESC LIMIT 200')
        rows = cur.fetchall()
        df = pd.DataFrame(rows, columns=['request_id','received_at','status','category','priority','assigned_team'])
        st.dataframe(df)
        csv = df.to_csv(index=False).encode('utf-8')
        st.download_button('Download CSV', data=csv, file_name='requests_export.csv')

    st.subheader('Train classifier from CSV')
    st.markdown('CSV must have columns: `full_text`, `category`')
    uploaded = st.file_uploader('Upload CSV for classifier training', type=['csv'])
    if uploaded is not None:
        df = pd.read_csv(uploaded)
        report = train_classifier_from_csv(df)
        st.text(report)

    st.subheader('Train ETA regressor from CSV')
    st.markdown('CSV must have columns: `full_text`, `assigned_team`, `resolution_hours` (numeric)')
    uploaded2 = st.file_uploader('Upload CSV for ETA training', type=['csv'], key='eta')
    if uploaded2 is not None:
        df2 = pd.read_csv(uploaded2)
        reg = train_eta_regressor_from_csv(df2)
        st.write('Trained regressor saved.')

    st.subheader('Model files')
    st.write('Classifier path: ', CLASSIFIER_PATH)
    st.write('ETA regressor path: ', REGRESSOR_PATH)

    st.subheader('Seed demo data')
    if st.button('Create 5 demo requests'):
        samples = [
            ("Web form","alice@example.com","AB1 2CD","My benefit payment stopped last week, I have no money"),
            ("Email","bob@example.com","XY9 8ZZ","I need to update my address for universal credit"),
            ("Phone","carol@example.com","NM3 4RT","My state pension payment is missing"),
            ("SMS","dave@example.com","LM5 6OP","I am elderly and need urgent help, no heating"),
            ("Web form","erin@example.com","GH7 1JK","Complaint: not responded to previous emails")
        ]
        ids = []
        for ch,cnt,pc,txt in samples:
            rid = process_and_store(ch, txt[:30], txt, cnt, pc)
            ids.append(rid)
        st.success('Created demo requests: ' + ', '.join(ids))

    st.subheader('Notes & limitations')
    st.markdown('''
    - This demo uses local files and SQLite; for production use Postgres and a vector DB (Milvus / FAISS / Chroma).
    - Streamlit Community has resource limits. Large models or heavy training may fail. Use the smallest embedder and avoid training on Streamlit free tier.
    - For SMS, paid gateway required in production. Email via SMTP is possible.
    - All data stored locally in this demo. For DWP use, follow GDPR and DWP security guidance: encrypt data at rest and in transit, pseudonymise PII, review retention policy.
    ''')

st.sidebar.header('Quick actions')
if st.sidebar.button('Show DB file location'):
    st.sidebar.write(os.path.abspath(DB_PATH))

# End of app

