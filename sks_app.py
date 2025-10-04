# app.py
# DWP Public Service Request Agent - Streamlit
# Postgres (via st.secrets["DATABASE_URL"]) + FAISS (if available, otherwise numpy fallback)
# Single-file deploy to Streamlit Community.

import os, uuid, time
from datetime import datetime, timezone
from typing import List, Tuple, Optional

import streamlit as st
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer

from sqlalchemy import create_engine, Table, Column, Integer, String, Text, LargeBinary, DateTime, MetaData, select
from sqlalchemy.exc import OperationalError

# Try import faiss
_use_faiss = False
try:
    import faiss
    _use_faiss = True
except Exception:
    _use_faiss = False

# -----------------------
# Config
# -----------------------
st.set_page_config(page_title="DWP Request Router (Streamlit)", layout="wide")
EMBED_MODEL_NAME = "all-MiniLM-L6-v2"
EMBED_DIM = 384
IDS_PATH = "faiss_ids.npy"
INDEX_PATH = "faiss_index.bin"
VECTORS_PATH = "vectors.npy"  # fallback persistence for numpy vectors
SQLITE_FALLBACK = "dwp_demo.db"

# -----------------------
# Database setup
# -----------------------
# prefer st.secrets, fallback to env var, else SQLite local file for dev
DATABASE_URL = None
if st.secrets and st.secrets.get("DATABASE_URL"):
    DATABASE_URL = st.secrets["DATABASE_URL"]
elif os.environ.get("DATABASE_URL"):
    DATABASE_URL = os.environ.get("DATABASE_URL")

USE_POSTGRES = False
if DATABASE_URL:
    engine = create_engine(DATABASE_URL, echo=False, future=True)
    USE_POSTGRES = True
else:
    engine = create_engine(f"sqlite:///{SQLITE_FALLBACK}", echo=False, future=True)

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

def create_db():
    try:
        metadata.create_all(engine)
    except OperationalError as e:
        st.error("Database error: " + str(e))
        raise

create_db()

# -----------------------
# Embedding model (cached)
# -----------------------
@st.cache_resource
def load_embedder():
    return SentenceTransformer(EMBED_MODEL_NAME)

embedder = load_embedder()

# -----------------------
# Vector index abstraction (FAISS with numpy fallback)
# -----------------------
class VectorIndex:
    def __init__(self, dim: int):
        self.dim = dim
        self.ids: List[str] = []
        self.use_faiss = _use_faiss
        if self.use_faiss:
            # IndexFlatIP expects float32 vectors; we will normalize to use cosine similarity
            self.index = faiss.IndexFlatIP(dim)
        else:
            self.vectors = None

    def add(self, ids: List[str], vecs: np.ndarray):
        vecs = vecs.astype('float32')
        if self.use_faiss:
            faiss.normalize_L2(vecs)
            self.index.add(vecs)
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
        np.save(IDS_PATH, np.array(self.ids, dtype=object))
        if self.use_faiss:
            faiss.write_index(self.index, INDEX_PATH)
        else:
            if self.vectors is not None:
                np.save(VECTORS_PATH, self.vectors)

    def load(self):
        if os.path.exists(IDS_PATH):
            self.ids = list(np.load(IDS_PATH, allow_pickle=True))
        if self.use_faiss and os.path.exists(INDEX_PATH):
            self.index = faiss.read_index(INDEX_PATH)
        elif not self.use_faiss and os.path.exists(VECTORS_PATH):
            self.vectors = np.load(VECTORS_PATH)

    def search(self, vec: np.ndarray, k=5) -> List[Tuple[str,float]]:
        vec = vec.astype('float32')
        if self.use_faiss and self.index is not None and self.index.ntotal>0:
            faiss.normalize_L2(vec)
            D, I = self.index.search(vec, k)
            results = []
            for idx, score in zip(I[0], D[0]):
                if idx < len(self.ids):
                    results.append((self.ids[idx], float(score)))
            return results
        else:
            if getattr(self, 'vectors', None) is None or len(self.ids)==0:
                return []
            q = vec / np.linalg.norm(vec, axis=1, keepdims=True)
            M = self.vectors / np.linalg.norm(self.vectors, axis=1, keepdims=True)
            sims = (M @ q.T).squeeze()
            idxs = np.argsort(-sims)[:k]
            return [(self.ids[i], float(sims[i])) for i in idxs]

index = VectorIndex(EMBED_DIM)
index.load()

# -----------------------
# Simple routing & priority heuristics
# -----------------------
DEFAULT_ROUTING = [
    {"match": ["benefit stopped", "payment stopped", "no money"], "team":"Income Support", "sla_hours":48},
    {"match": ["universal credit", "uc claim", "uc issue"], "team":"Universal Credit", "sla_hours":72},
    {"match": ["pension", "state pension"], "team":"Pensions", "sla_hours":96},
    {"match": ["homeless","no shelter","sleeping rough"], "team":"Vulnerable Cases", "sla_hours":24, "vulnerability":True},
]

TEAMS_META = {
    "Income Support": {"capacity":20, "sla_hours":48},
    "Universal Credit": {"capacity":40, "sla_hours":72},
    "Pensions": {"capacity":10, "sla_hours":96},
    "Vulnerable Cases": {"capacity":5, "sla_hours":24},
    "General Intake": {"capacity":100, "sla_hours":168},
}

def rule_route(text: str):
    lower = text.lower()
    for r in DEFAULT_ROUTING:
        for kw in r["match"]:
            if kw in lower:
                vuln = bool(r.get("vulnerability", False))
                return r["team"], vuln, 0.9
    return "General Intake", False, 0.5

def compute_priority(prob: float, vulnerability: bool, sla_hours: int) -> int:
    base = int(prob * 60)
    if vulnerability:
        base += 25
    sla_factor = max(0, 48 - sla_hours)//2
    score = base + sla_factor
    return min(100, max(0, score))

# -----------------------
# DB helper functions
# -----------------------
def insert_request(row: dict):
    with engine.begin() as conn:
        conn.execute(requests_table.insert().values(**row))

def save_embedding(rid: str, vec: np.ndarray):
    with engine.begin() as conn:
        conn.execute(embeddings_table.insert().values(id=rid, embedding=vec.tobytes()))

def get_request(rid: str) -> Optional[dict]:
    with engine.connect() as conn:
        q = select([requests_table]).where(requests_table.c.id == rid)
        row = conn.execute(q).fetchone()
        return dict(row) if row else None

def get_all_requests_df() -> pd.DataFrame:
    with engine.connect() as conn:
        q = select([requests_table])
        rows = conn.execute(q).fetchall()
        if not rows:
            return pd.DataFrame()
        df = pd.DataFrame(rows)
        return df

def load_all_embeddings_from_db():
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

def rebuild_index():
    ids, vecs = load_all_embeddings_from_db()
    if len(ids)>0:
        # create fresh index
        new_idx = VectorIndex(EMBED_DIM)
        if new_idx.use_faiss:
            new_idx.index = faiss.IndexFlatIP(EMBED_DIM)
            faiss.normalize_L2(vecs)
        new_idx.add(ids, vecs)
        new_idx.load()
        return True
    return False

# -----------------------
# Streamlit UI
# -----------------------
st.title("DWP — Public Service Request Prioritisation & Routing (Demo)")
st.sidebar.header("Setup")
st.sidebar.write("DB: " + ("Postgres (external)" if USE_POSTGRES else "SQLite fallback (local)"))
st.sidebar.write("Vector search: " + ("FAISS" if _use_faiss else "NumPy fallback"))

col_left, col_right = st.columns([2,1])

with col_left:
    st.header("Submit a Request")
    with st.form("submit"):
        channel = st.selectbox("Channel", ["Web form","Email","Phone-to-text","SMS"])
        contact = st.text_input("Contact (email/phone, optional)")
        postcode = st.text_input("Postcode (optional)")
        subject = st.text_input("Subject (short)")
        full_text = st.text_area("Describe the issue (detailed)", height=220)
        submitted = st.form_submit_button("Submit")
        if submitted:
            if not full_text.strip():
                st.error("Please enter the issue description.")
            else:
                rid = "REQ-" + uuid.uuid4().hex[:10].upper()
                received_at = datetime.now(timezone.utc)
                team, vuln, prob = rule_route(full_text)
                emb = embedder.encode([full_text]).astype('float32')
                sla = TEAMS_META.get(team, {}).get('sla_hours', 168)
                priority = compute_priority(prob, vuln, sla)
                eta = int(sla)
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
                    "eta_hours": eta,
                }
                insert_request(row)
                save_embedding(rid, emb)
                try:
                    index.add([rid], emb)
                except Exception:
                    rebuild_index()
                    index.load()
                st.success("Submitted — Request ID: " + rid)
                st.info(f"Assigned team: {team} — Priority {priority} — ETA ~{eta} hours")

    st.markdown("---")
    st.header("Check Status")
    qid = st.text_input("Enter Request ID")
    if st.button("Get Status"):
        if not qid.strip():
            st.error("Enter a valid Request ID")
        else:
            rec = get_request(qid.strip().upper())
            if not rec:
                st.error("Request not found")
            else:
                st.markdown(f"**Request ID:** {rec['id']}")
                st.markdown(f"**Received:** {rec['received_at']}")
                st.markdown(f"**Status:** {rec['status']}")
                st.markdown(f"**Assigned team:** {rec['assigned_team']}")
                st.markdown(f"**Priority:** {rec['priority']}")
                st.markdown(f"**ETA (hours):** {rec['eta_hours']}")
                st.write("**Full text:**")
                st.write(rec['full_text'])

with col_right:
    st.header("Admin Dashboard")
    df = get_all_requests_df()
    if df.empty:
        st.info("No requests yet.")
    else:
        total = len(df)
        open_count = len(df[df['status'] != 'Resolved'])
        avg_eta = int(df['eta_hours'].astype(int).mean())
        st.metric("Total Requests", total)
        st.metric("Open Requests", open_count)
        st.metric("Avg ETA (hours)", avg_eta)

        st.subheader("Status distribution")
        status_counts = df['status'].value_counts()
        st.bar_chart(status_counts)

        st.subheader("Priority breakdown (counts)")
        pr_bins = pd.cut(df['priority'].astype(int), bins=[-1,20,50,80,100], labels=["Low","Medium","High","Critical"])
        st.bar_chart(pr_bins.value_counts())

        st.subheader("Requests per Team")
        team_counts = df['assigned_team'].value_counts()
        st.table(team_counts)

        st.subheader("Avg ETA per Team")
        avg_eta_team = df.groupby('assigned_team')['eta_hours'].mean().reset_index().rename(columns={'eta_hours':'avg_eta_hours'})
        st.dataframe(avg_eta_team)

    st.markdown("---")
    st.subheader("Admin Actions")
    if st.button("Rebuild vector index from DB"):
        ok = rebuild_index()
        if ok:
            st.success("Rebuilt index")
            index.load()
        else:
            st.warning("No embeddings to rebuild")

    if st.button("Export CSV"):
        if not df.empty:
            csv = df.to_csv(index=False).encode('utf-8')
            st.download_button("Download requests CSV", csv, file_name="requests_export.csv")

    st.markdown("### Semantic search (find similar historical requests)")
    qtext = st.text_input("Search text for semantic match")
    k = st.slider("k (results)", 1, 10, 5)
    if st.button("Search similar"):
        if not qtext.strip():
            st.error("Please enter search text")
        else:
            qvec = embedder.encode([qtext]).astype('float32')
            results = index.search(qvec, k=k)
            if not results:
                st.info("No matches found")
            else:
                rows = []
                for rid, score in results:
                    rec = get_request(rid)
                    if rec:
                        rows.append({"id":rid, "score":round(score,3), "team":rec['assigned_team'], "status":rec['status'], "subject":rec['subject']})
                st.dataframe(pd.DataFrame(rows))

    st.markdown("---")
    st.subheader("Seed sample requests")
    if st.button("Seed demo 5 requests"):
        samples = [
            ("Web form","alice@example.com","AB1 2CD","Benefit payment stopped","My benefit payment stopped and I have no money"),
            ("Email","bob@example.com","XY9 8ZZ","UC details update","I need to update my universal credit details"),
            ("Phone","carol@example.com","NM3 4RT","Pension missing","My state pension didn't arrive this month"),
            ("SMS","dave@example.com","LM5 6OP","I am homeless","I am sleeping rough and need urgent help"),
            ("Web form","erin@example.com","GH7 1JK","Complaint","No response to earlier claim")
        ]
        for ch,contact,pc,sub,txt in samples:
            rid = "REQ-" + uuid.uuid4().hex[:8].upper()
            rec = {
                "id":rid,
                "received_at":datetime.now(timezone.utc),
                "channel":ch,
                "contact":contact,
                "postcode":pc,
                "subject":sub,
                "full_text":txt,
                "status":"queued",
                "category":sub,
                "vulnerability":1 if "homeless" in txt.lower() else 0,
                "priority":90 if "homeless" in txt.lower() else 50,
                "assigned_team": rule_route(txt)[0],
                "eta_hours": TEAMS_META.get(rule_route(txt)[0],{}).get('sla_hours',72)
            }
            insert_request(rec)
            emb = embedder.encode([txt]).astype('float32')
            save_embedding(rid, emb)
            index.add([rid], emb)
        st.success("Seeded demo requests")

st.markdown("---")
st.caption("Notes: For production, replace heuristics with trained classifiers/regressors, store vectors in Milvus for large scale, and secure PII according to GDPR.")
