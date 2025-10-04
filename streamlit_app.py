# app.py
# DWP Public Service Request Agent - Streamlit (Postgres + FAISS fallback)
# Deployable on Streamlit Community. Put this file as app.py in your repo.
#
# Requirements (requirements.txt):
# streamlit
# sentence-transformers
# sqlalchemy
# psycopg2-binary
# pandas
# numpy
# scikit-learn
# joblib
# faiss-cpu    # optional; if installation fails, code falls back to numpy search
#
# How to configure on Streamlit Community:
# - Add a secret named DATABASE_URL in the app's Secrets (e.g. postgres://user:pass@host:5432/dbname)
# - Deploy the repo to Streamlit Cloud and it will run app.py
#
# Limitations:
# - FAISS may fail to install on Streamlit Cloud depending on their build environment. The app will still run using a numpy fallback.
# - Milvus is not included here (requires external service). If you run Milvus externally, you can adapt the vector insert/search parts.
#
import streamlit as st
st.set_page_config(layout="wide", page_title="DWP Request Router (Demo)")

import os, io, uuid, time
from datetime import datetime, timezone
import numpy as np
import pandas as pd
from typing import Optional, Tuple, List
from sentence_transformers import SentenceTransformer
from sqlalchemy import create_engine, Column, Integer, String, Text, Float, LargeBinary, DateTime, Table, MetaData, select
from sqlalchemy.exc import OperationalError
from sklearn.ensemble import RandomForestRegressor
import joblib

# Try import faiss
_use_faiss = False
try:
    import faiss
    _use_faiss = True
except Exception:
    _use_faiss = False

# Config
EMBED_MODEL_NAME = "all-MiniLM-L6-v2"
EMBED_DIM = 384  # for all-MiniLM-L6-v2
INDEX_PATH = "faiss_index.bin"
IDS_PATH = "faiss_ids.npy"
DB_SQLITE_PATH = "dwp_demo.db"

# Read DB URL from secrets (Streamlit Cloud) or fallback to SQLite
DATABASE_URL = None
if st.secrets.get("DATABASE_URL"):
    DATABASE_URL = st.secrets["DATABASE_URL"]
else:
    DATABASE_URL = os.environ.get("DATABASE_URL")  # optional env fallback

USE_POSTGRES = False
if DATABASE_URL:
    engine = create_engine(DATABASE_URL, echo=False, future=True)
    USE_POSTGRES = True
else:
    engine = create_engine(f"sqlite:///{DB_SQLITE_PATH}", echo=False, future=True)
    USE_POSTGRES = False

metadata = MetaData()

requests_table = Table(
    "requests", metadata,
    Column("id", String, primary_key=True),
    Column("received_at", DateTime, nullable=False),
    Column("channel", String),
    Column("contact", String),
    Column("postcode", String),
    Column("subject", String),
    Column("full_text", Text),
    Column("status", String, default="new"),
    Column("category", String),
    Column("vulnerability", Integer, default=0),
    Column("priority", Integer, default=0),
    Column("assigned_team", String),
    Column("eta_hours", Integer, default=None),
)

embeddings_table = Table(
    "embeddings", metadata,
    Column("id", String, primary_key=True),
    Column("embedding", LargeBinary),
)

# Create tables if not present
def create_db():
    try:
        metadata.create_all(engine)
    except OperationalError as e:
        st.error("Database error: " + str(e))
        raise

create_db()

# Embedding model (cached)
@st.cache_resource
def load_embedder():
    return SentenceTransformer(EMBED_MODEL_NAME)

embedder = load_embedder()

# Simple heuristic routing table (editable)
DEFAULT_ROUTING = [
    {"match": {"keywords": ["benefit stopped", "payment stopped", "no money"]}, "team": "Income Support", "sla_hours": 48},
    {"match": {"keywords": ["universal credit", "uc claim", "uc issue"]}, "team": "Universal Credit", "sla_hours": 72},
    {"match": {"keywords": ["pension", "state pension"]}, "team": "Pensions", "sla_hours": 96},
    {"match": {"keywords": ["homeless", "no shelter", "sleeping rough"]}, "team": "Vulnerable Cases", "sla_hours": 24, "vulnerability": True},
]

TEAMS_META = {
    "Income Support": {"capacity": 20, "sla_hours": 48},
    "Universal Credit": {"capacity": 40, "sla_hours": 72},
    "Pensions": {"capacity": 10, "sla_hours": 96},
    "Vulnerable Cases": {"capacity": 5, "sla_hours": 24},
    "General Intake": {"capacity": 100, "sla_hours": 168},
}

# FAISS index management (with numpy fallback)
class VectorIndex:
    def __init__(self, dim: int):
        self.dim = dim
        self.ids = []  # ordered list of string IDs
        self.index = None
        self.use_faiss = _use_faiss
        if self.use_faiss:
            # we'll use IndexFlatIP (inner product) with normalized vectors for cosine
            self.index = faiss.IndexFlatIP(dim)
        else:
            self.vectors = None  # numpy array NxD

    def add(self, ids: List[str], vecs: np.ndarray):
        if self.use_faiss:
            # normalize
            faiss.normalize_L2(vecs)
            self.index.add(vecs.astype('float32'))
            self.ids.extend(ids)
        else:
            if self.vectors is None:
                self.vectors = vecs.copy()
                self.ids = list(ids)
            else:
                self.vectors = np.vstack([self.vectors, vecs])
                self.ids.extend(ids)
        self._persist()

    def _persist(self):
        # save ids
        np.save(IDS_PATH, np.array(self.ids, dtype=object))
        if self.use_faiss:
            faiss.write_index(self.index, INDEX_PATH)
        else:
            np.save("vectors.npy", self.vectors)

    def load(self):
        if os.path.exists(IDS_PATH):
            self.ids = list(np.load(IDS_PATH, allow_pickle=True))
        if self.use_faiss and os.path.exists(INDEX_PATH):
            self.index = faiss.read_index(INDEX_PATH)
        elif not self.use_faiss and os.path.exists("vectors.npy"):
            self.vectors = np.load("vectors.npy")

    def search(self, vec: np.ndarray, k=5):
        # returns list of (id, score)
        if self.use_faiss and self.index is not None:
            v = vec.astype('float32')
            faiss.normalize_L2(v)
            D, I = self.index.search(v, k)
            results = []
            for i, score in zip(I[0], D[0]):
                if i < len(self.ids):
                    results.append((self.ids[i], float(score)))
            return results
        else:
            if self.vectors is None:
                return []
            # cosine similarity
            q = vec / np.linalg.norm(vec, axis=1, keepdims=True)
            M = self.vectors / np.linalg.norm(self.vectors, axis=1, keepdims=True)
            sims = (M @ q.T).squeeze()
            idx = np.argsort(-sims)[:k]
            return [(self.ids[i], float(sims[i])) for i in idx]

index = VectorIndex(EMBED_DIM)
index.load()

# Helpers: DB insert / query
def insert_request(row: dict):
    with engine.begin() as conn:
        conn.execute(requests_table.insert().values(**row))

def get_request_by_id(rid: str) -> Optional[dict]:
    with engine.connect() as conn:
        q = select([requests_table]).where(requests_table.c.id == rid)
        rs = conn.execute(q).fetchone()
        if rs:
            return dict(rs)
    return None

def save_embedding(rid: str, vec: np.ndarray):
    b = vec.astype('float32').tobytes()
    with engine.begin() as conn:
        conn.execute(embeddings_table.insert().values(id=rid, embedding=b))

def load_all_embeddings() -> Tuple[List[str], np.ndarray]:
    with engine.connect() as conn:
        q = select([embeddings_table])
        rows = conn.execute(q).fetchall()
        ids = []
        vecs = []
        for r in rows:
            ids.append(r['id'])
            arr = np.frombuffer(r['embedding'], dtype='float32').reshape(1, -1)
            vecs.append(arr)
        if vecs:
            return ids, np.vstack(vecs)
    return [], np.empty((0, EMBED_DIM), dtype='float32')

def rebuild_index_from_db():
    ids, vecs = load_all_embeddings()
    if ids and vecs.size:
        # rebuild faiss or numpy store
        new_index = VectorIndex(EMBED_DIM)
        new_index.load()  # clear? we overwrite files anyway
        # replace vectors
        if new_index.use_faiss:
            new_index.index = faiss.IndexFlatIP(EMBED_DIM)
        else:
            new_index.vectors = None
            new_index.ids = []
        new_index.add(ids, vecs)
        return True
    return False

# Priority & ETA heuristics (simple)
def compute_priority(score_prob: float, vulnerability: bool, sla_hours: int) -> int:
    base = int(score_prob * 60)  # 0..60
    if vulnerability:
        base += 25
    sla_factor = max(0, 48 - sla_hours) // 2
    score = base + sla_factor
    return min(100, max(0, score))

# Simple classifier/routing using keywords + nearest neighbor to historical examples
def rule_route(text: str) -> Tuple[str, bool, float]:
    lower = text.lower()
    for r in DEFAULT_ROUTING:
        for kw in r["match"]["keywords"]:
            if kw in lower:
                vuln = bool(r.get("vulnerability", False))
                return r["team"], vuln, 0.9
    # fallback
    return "General Intake", False, 0.5

# Dashboard helpers
def get_summary_df():
    with engine.connect() as conn:
        q = select([requests_table])
        rows = conn.execute(q).fetchall()
        if not rows:
            return pd.DataFrame()
        df = pd.DataFrame(rows)
        return df

# UI: Sidebar instructions
st.sidebar.header("Setup & Notes")
st.sidebar.markdown("""
- Use **Streamlit Secrets** to set `DATABASE_URL` for a Postgres DB.
- FAISS is optional. If `faiss-cpu` installs in the environment, the app will use it. Otherwise, the app uses numpy fallback.
- Rebuild the vector index after bulk inserts via the Admin panel.
""")
st.sidebar.markdown("**Connection:** " + ("Postgres" if USE_POSTGRES else "SQLite (local)"))

# Main UI layout
col1, col2 = st.columns([2,1])

with col1:
    st.header("Citizen — Submit a Service Request")
    with st.form("submit_form"):
        channel = st.selectbox("Channel", ["Web form", "Email", "Phone-to-text", "SMS"])
        contact = st.text_input("Contact (email/phone)")
        postcode = st.text_input("Postcode (optional)")
        subject = st.text_input("Subject (short)")
        full_text = st.text_area("Describe the issue", height=200)
        submitted = st.form_submit_button("Submit")
        if submitted:
            if not full_text.strip():
                st.error("Please enter the issue description.")
            else:
                rid = "REQ-" + uuid.uuid4().hex[:8].upper()
                received_at = datetime.now(timezone.utc)
                # quick classification by rules + embedding
                team, vuln, prob = rule_route(full_text)
                emb = embedder.encode([full_text]).astype('float32')
                # compute priority and ETA (heuristic)
                sla = TEAMS_META.get(team, {}).get('sla_hours', 168)
                priority = compute_priority(prob, vuln, sla)
                eta_hours = int(sla)  # simple ETA
                row = {
                    "id": rid,
                    "received_at": received_at,
                    "channel": channel,
                    "contact": contact,
                    "postcode": postcode,
                    "subject": subject,
                    "full_text": full_text,
                    "status": "queued",
                    "category": team,
                    "vulnerability": int(vuln),
                    "priority": int(priority),
                    "assigned_team": team,
                    "eta_hours": int(eta_hours),
                }
                insert_request(row)
                # Save embedding & add to index
                save_embedding(rid, emb)
                try:
                    index.add([rid], emb)
                except Exception:
                    # if index not initialized, rebuild from db (safe)
                    rebuild_index_from_db()
                st.success("Request submitted. Your Request ID: " + rid)
                st.info(f"Assigned: {team} — Priority {priority} — ETA ~{eta_hours} hours")

    st.markdown("---")
    st.subheader("Check Status / Conversation (Citizen)")
    qid = st.text_input("Enter Request ID to check")
    if st.button("Get Status"):
        if not qid.strip():
            st.error("Enter Request ID")
        else:
            rec = get_request_by_id(qid.strip().upper())
            if not rec:
                st.error("Request not found")
            else:
                st.write("**Request ID:**", rec["id"])
                st.write("**Received:**", rec["received_at"])
                st.write("**Status:**", rec["status"])
                st.write("**Assigned team:**", rec["assigned_team"])
                st.write("**Priority:**", rec["priority"])
                st.write("**ETA (hours):**", rec["eta_hours"])
                st.write("**Description:**")
                st.write(rec["full_text"])

with col2:
    st.header("Admin Dashboard")
    df = get_summary_df()
    if df.empty:
        st.info("No requests yet. Use the form to submit or use 'Seed demo data' below.")
    else:
        # Key metrics
        total = len(df)
        open_count = len(df[df['status'] != 'Resolved'])
        avg_eta = int(df['eta_hours'].dropna().astype(int).mean())
        st.metric("Total requests", total)
        st.metric("Open requests", open_count)
        st.metric("Avg ETA (hours)", avg_eta)

        # Status distribution
        st.subheader("Status distribution")
        status_counts = df['status'].value_counts()
        st.bar_chart(status_counts)

        # Priority breakdown
        st.subheader("Priority breakdown")
        pr_bins = pd.cut(df['priority'].astype(int), bins=[-1,20,50,80,100], labels=["Low","Medium","High","Critical"])
        st.bar_chart(pr_bins.value_counts())

        # Requests per department
        st.subheader("Requests per Department")
        dept_counts = df['assigned_team'].value_counts()
        st.table(dept_counts)

        # Avg ETA per department
        st.subheader("Avg ETA per Department")
        avg_eta_dept = df.groupby('assigned_team')['eta_hours'].mean().rename('avg_eta_hours').reset_index()
        st.dataframe(avg_eta_dept)

    st.markdown("---")
    st.subheader("Admin Actions")
    if st.button("Rebuild vector index from DB"):
        ok = rebuild_index_from_db()
        if ok:
            st.success("Rebuilt index from DB")
            index.load()
        else:
            st.warning("No embeddings found to rebuild index")

    if st.button("Export requests CSV"):
        df = get_summary_df()
        if not df.empty:
            csv = df.to_csv(index=False).encode('utf-8')
            st.download_button("Download CSV", csv, file_name="requests_export.csv")

    st.markdown("### Search similar requests")
    search_text = st.text_input("Enter text to find similar historical requests (semantic search)")
    k = st.slider("k (results)", min_value=1, max_value=10, value=5)
    if st.button("Search similar"):
        if not search_text.strip():
            st.error("Enter query text")
        else:
            qvec = embedder.encode([search_text]).astype('float32')
            results = index.search(qvec, k=k)
            if not results:
                st.info("No historical matches")
            else:
                rows = []
                for rid, score in results:
                    rec = get_request_by_id(rid)
                    if rec:
                        rows.append({"id": rid, "score": score, "assigned_team": rec['assigned_team'], "status": rec['status'], "subject": rec['subject']})
                st.dataframe(pd.DataFrame(rows))

    st.markdown("---")
    st.subheader("Seed / Demo Data")
    if st.button("Seed 5 demo requests"):
        samples = [
            ("Web form", "alice@example.com", "AB1 2CD", "Benefit payment stopped", "My benefit payment stopped last week and I have no money"),
            ("Email", "bob@example.com", "XY9 8ZZ", "Universal Credit issue", "I need to update my universal credit details"),
            ("Phone", "carol@example.com", "NM3 4RT", "Pension missing", "My state pension didn't arrive this month"),
            ("SMS", "dave@example.com", "LM5 6OP", "I am homeless", "I am sleeping rough and need urgent help"),
            ("Web form", "erin@example.com", "GH7 1JK", "Complaint", "No response to previous claims, complaint")
        ]
        for ch,contact,pc,sub,txt in samples:
            rid = "REQ-" + uuid.uuid4().hex[:8].upper()
            rec = {
                "id": rid,
                "received_at": datetime.now(timezone.utc),
                "channel": ch,
                "contact": contact,
                "postcode": pc,
                "subject": sub,
                "full_text": txt,
                "status": "queued",
                "category": sub,
                "vulnerability": 1 if "homeless" in txt.lower() else 0,
                "priority": 90 if "homeless" in txt.lower() else 50,
                "assigned_team": rule_route(txt)[0],
                "eta_hours": TEAMS_META.get(rule_route(txt)[0], {}).get('sla_hours', 72),
            }
            insert_request(rec)
            emb = embedder.encode([txt]).astype('float32')
            save_embedding(rid, emb)
            index.add([rid], emb)
        st.success("Seeded demo requests")

st.markdown("---")
st.caption("Note: For production use, replace heuristics with trained models and use an external Milvus/FAISS service for large-scale vector search. Ensure GDPR compliance (PII handling, encryption).")
