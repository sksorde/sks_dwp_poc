# app.py
import uuid
from datetime import datetime, timezone
import numpy as np
import pandas as pd
import streamlit as st
from sentence_transformers import SentenceTransformer
import faiss
from sqlalchemy import create_engine, Table, Column, Integer, String, Text, LargeBinary, DateTime, MetaData, select
from sqlalchemy.exc import OperationalError
import smtplib
from email.message import EmailMessage

# -------------------
# Page config
# -------------------
st.set_page_config(page_title="DWP AI Service Requests", layout="wide")

# -------------------
# Database setup
# -------------------
DATABASE_URL = st.secrets.get("DATABASE_URL")
if not DATABASE_URL:
    st.error("Add DATABASE_URL to Streamlit secrets.")
    st.stop()

engine = create_engine(DATABASE_URL, echo=False, future=True)
metadata = MetaData()

# Requests table
requests_table = Table(
    "requests", metadata,
    Column("id", String, primary_key=True),
    Column("received_at", DateTime),
    Column("channel", String),
    Column("contact", String),
    Column("postcode", String),
    Column("subject", String),
    Column("full_text", Text),
    Column("status", String),
    Column("category", String),
    Column("vulnerability", Integer),
    Column("priority", Integer),
    Column("assigned_team", String),
    Column("eta_hours", Integer)
)

# Embeddings table
embeddings_table = Table(
    "embeddings", metadata,
    Column("id", String, primary_key=True),
    Column("embedding", LargeBinary)
)

def create_tables():
    try:
        metadata.create_all(engine)
    except OperationalError as e:
        st.error(f"Database error: {e}")
        st.stop()

create_tables()

# -------------------
# Embedding model
# -------------------
@st.cache_resource
def load_embedder():
    return SentenceTransformer("all-MiniLM-L6-v2")

embedder = load_embedder()
EMBED_DIM = embedder.get_sentence_embedding_dimension()

# -------------------
# DWP Routing / Priority
# -------------------
DWP_TEAMS = [
    {"name": "Income Support", "keywords":["benefit stopped","payment stopped"], "sla_hours":48},
    {"name": "Universal Credit", "keywords":["universal credit","uc claim"], "sla_hours":72},
    {"name": "Pensions", "keywords":["pension"], "sla_hours":96},
    {"name": "Vulnerable Cases", "keywords":["homeless","sleeping rough"], "sla_hours":24, "vulnerability": True},
    {"name": "General Intake", "keywords":[], "sla_hours":168}
]

# Specific gov.uk guidance URLs
GUIDANCE_URLS = {
    "universal credit": "https://www.gov.uk/guidance/universal-credit-and-students",
    "child maintenance": "https://www.gov.uk/child-maintenance",
    "pension": "https://www.gov.uk/state-pension",
    "income support": "https://www.gov.uk/income-support",
    "homeless": "https://www.gov.uk/homelessness-information-advice"
}

def route_request(text: str):
    text_l = text.lower()
    # Check for specific guidance URLs first
    for key, url in GUIDANCE_URLS.items():
        if key in text_l:
            team = next((t["name"] for t in DWP_TEAMS if key in " ".join(t.get("keywords", []))), "General Intake")
            return team, False, 0.9, url
    # Default keyword-based routing
    for team in DWP_TEAMS:
        for kw in team.get("keywords", []):
            if kw in text_l:
                vuln = team.get("vulnerability", False)
                return team["name"], vuln, 0.9, None
    return "General Intake", False, 0.5, None

def compute_priority(prob:float,vulnerability:bool,sla:int)->int:
    base = int(prob*60)
    if vulnerability: base += 25
    sla_factor = max(0,48-sla)//2
    return min(100,max(0,base+sla_factor))

# -------------------
# DB helpers
# -------------------
def insert_request(row:dict):
    with engine.begin() as conn:
        conn.execute(requests_table.insert().values(**row))

def save_embedding(rid:str, vec:np.ndarray):
    with engine.begin() as conn:
        conn.execute(embeddings_table.insert().values(id=rid, embedding=vec.tobytes()))

def get_request(rid:str):
    with engine.connect() as conn:
        q = select(requests_table).where(requests_table.c.id==rid)
        row = conn.execute(q).mappings().fetchone()
        return dict(row) if row else None

def get_all_requests_df():
    with engine.connect() as conn:
        q = select(requests_table)
        rows = conn.execute(q).mappings().all()
        return pd.DataFrame(rows) if rows else pd.DataFrame()

def get_all_embeddings():
    with engine.connect() as conn:
        q = select(embeddings_table)
        rows = conn.execute(q).mappings().all()
        if not rows:
            return [], np.zeros((0,EMBED_DIM), dtype='float32')
        ids = [r["id"] for r in rows]
        vectors = np.array([np.frombuffer(r["embedding"],dtype='float32') for r in rows])
        return ids, vectors

# -------------------
# FAISS semantic search
# -------------------
def search_similar_requests(query:str, top_k=3):
    if len(st.session_state.embedding_ids)==0:
        return []
    vec = embedder.encode([query]).astype("float32")
    D,I = st.session_state.faiss_index.search(vec, top_k)
    results = []
    for idx in I[0]:
        rid = st.session_state.embedding_ids[idx]
        rec = get_request(rid)
        if rec: results.append(rec)
    return results

def search_requests(query: str, top_k=5):
    """Staff/Admin search for past requests"""
    if len(st.session_state.embedding_ids) == 0:
        return pd.DataFrame()
    
    vec = embedder.encode([query]).astype("float32")
    D, I = st.session_state.faiss_index.search(vec, top_k)
    
    results = []
    for idx in I[0]:
        rid = st.session_state.embedding_ids[idx]
        rec = get_request(rid)
        if rec:
            results.append({
                "id": rec["id"],
                "subject": rec["subject"],
                "assigned_team": rec["assigned_team"],
                "status": rec["status"],
                "priority": rec["priority"],
                "eta_hours": rec["eta_hours"]
            })
    return pd.DataFrame(results)

# -------------------
# Email notifications
# -------------------
def send_email(to_email:str, subject:str, body:str):
    EMAIL_USER = st.secrets.get("EMAIL_USER")
    EMAIL_PASS = st.secrets.get("EMAIL_PASS")
    if not EMAIL_USER or not EMAIL_PASS:
        return
    msg = EmailMessage()
    msg["Subject"] = subject
    msg["From"] = EMAIL_USER
    msg["To"] = to_email
    msg.set_content(body)
    try:
        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as smtp:
            smtp.login(EMAIL_USER, EMAIL_PASS)
            smtp.send_message(msg)
    except Exception as e:
        st.warning(f"Email not sent: {e}")

DEPT_EMAILS = {
    "Income Support":"income.support@example.com",
    "Universal Credit":"uc@example.com",
    "Pensions":"pensions@example.com",
    "Vulnerable Cases":"vulnerable@example.com",
    "General Intake":"general.intake@example.com"
}

# -------------------
# Session state
# -------------------
if "page" not in st.session_state: st.session_state.page="submit"
if "last_request_id" not in st.session_state: st.session_state.last_request_id=None
if "faiss_index" not in st.session_state:
    ids, vecs = get_all_embeddings()
    index = faiss.IndexFlatL2(EMBED_DIM)
    if len(vecs)>0: index.add(vecs)
    st.session_state.faiss_index = index
    st.session_state.embedding_ids = ids

# -------------------
# UI: Submit Request
# -------------------
st.title("DWP AI Service Request System")
if st.session_state.page=="submit":
    st.header("Submit a Request")
    with st.form("submit_form"):
        channel = st.selectbox("Channel", ["Web form","Email","Phone-to-text","SMS"])
        contact = st.text_input("Contact")
        postcode = st.text_input("Postcode")
        subject = st.text_input("Subject")
        full_text = st.text_area("Describe the issue", height=200)
        submitted = st.form_submit_button("Submit")
        if submitted and full_text.strip():
            rid = "REQ-"+uuid.uuid4().hex[:10].upper()
            team, vuln, prob, guidance_url = route_request(full_text)
            sla = next(t["sla_hours"] for t in DWP_TEAMS if t["name"]==team)
            priority = compute_priority(prob, vuln, sla)
            row = {
                "id": rid,
                "received_at": datetime.now(timezone.utc),
                "channel": channel,
                "contact": contact,
                "postcode": postcode,
                "subject": subject,
                "full_text": full_text,
                "status":"queued",
                "category": team,
                "vulnerability": int(vuln),
                "priority": priority,
                "assigned_team": team,
                "eta_hours": sla
            }
            insert_request(row)
            vec = embedder.encode([full_text]).astype("float32")
            save_embedding(rid, vec)
            st.session_state.faiss_index.add(vec)
            st.session_state.embedding_ids.append(rid)

            # Semantic search for similar requests
            similar = search_similar_requests(full_text)
            if similar:
                st.markdown("### Similar past requests:")
                for s in similar:
                    st.markdown(f"- **{s['subject']}** ({s['assigned_team']}) – Status: {s['status']}")

            # Email notification
            dept_email = DEPT_EMAILS.get(team)
            if dept_email:
                body = f"New request assigned to {team}:\n\nID: {rid}\nSubject: {subject}\nText: {full_text}\nETA: {sla}h"
                send_email(dept_email, f"New DWP Request: {rid}", body)

            st.session_state.last_request_id = rid
            st.session_state.page="department"
            st.session_state.guidance_url = guidance_url
            st.experimental_rerun()

# -------------------
# Department Dashboard
# -------------------
elif st.session_state.page=="department":
    rec = get_request(st.session_state.last_request_id)
    guidance_url = st.session_state.get("guidance_url")
    if rec:
        st.success(f"Your request has been routed to **{rec['assigned_team']}**")
        st.markdown(f"**Request ID:** {rec['id']}")
        st.markdown(f"**ETA (hours):** {rec['eta_hours']}")
        st.markdown(f"**Status:** {rec['status']}")
        st.write("**Full text:**", rec['full_text'])

        if guidance_url:
            st.markdown("### Recommended gov.uk guidance:")
            st.markdown(f"[Click here for guidance]({guidance_url})", unsafe_allow_html=True)

        df = get_all_requests_df()
        df_team = df[df['assigned_team']==rec['assigned_team']]
        st.markdown(f"### All requests in {rec['assigned_team']}")
        st.dataframe(df_team[['id','subject','status','priority','eta_hours']])

        if st.button("Submit another request"):
            st.session_state.page="submit"
            st.session_state.last_request_id=None
            st.session_state.guidance_url = None
            st.experimental_rerun()

# -------------------
# Admin Dashboard & Staff Search
# -------------------
with st.expander("Admin Dashboard / Monitoring", expanded=False):
    st.markdown("### Real-time Status & Priority Breakdown")
    df = get_all_requests_df()
    if not df.empty:
        st.markdown("#### Requests by Status")
        st.bar_chart(df['status'].value_counts())
        st.markdown("#### Requests by Priority")
        st.bar_chart(df.groupby('priority').size())
        st.markdown("#### Requests by Assigned Team")
        st.bar_chart(df['assigned_team'].value_counts())

with st.expander("Staff / Department Search", expanded=False):
    st.markdown("### Search Past Requests")
    query = st.text_input("Enter keywords or description")
    top_k = st.slider("Number of results", min_value=1, max_value=10, value=5)
    if st.button("Search") and query.strip():
        df_results = search_requests(query, top_k)
        if not df_results.empty:
            st.dataframe(df_results)
        else:
            st.info("No matching requests found.")
