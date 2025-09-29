import sqlite3, numpy as np, json
from pathlib import Path
from typing import Iterable

SCHEMA = """
CREATE TABLE IF NOT EXISTS meta (k TEXT PRIMARY KEY, v TEXT);
CREATE TABLE IF NOT EXISTS chunks (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  path TEXT, start_line INT, end_line INT, sha TEXT,
  kind TEXT, -- 'code' | 'doc' | 'pipeline'
  content TEXT,
  embedding BLOB
);
CREATE INDEX IF NOT EXISTS idx_path ON chunks(path);
"""

def connect(db_path: str):
    p = Path(db_path); p.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(p)
    con.execute("PRAGMA journal_mode=WAL;")
    con.executescript(SCHEMA)
    return con

def upsert_chunks(con, rows: Iterable[dict], embed_dim: int):
    con.execute("INSERT OR REPLACE INTO meta(k,v) VALUES('embed_dim',?)", (str(embed_dim),))
    cur = con.cursor()
    for r in rows:
        cur.execute("""
        INSERT INTO chunks(path,start_line,end_line,sha,kind,content,embedding)
        VALUES(?,?,?,?,?,?,?)
        """, (r["path"], r["start_line"], r["end_line"], r["sha"], r["kind"], r["content"],
              memoryview(np.asarray(r["embedding"], dtype="float32").tobytes())))
    con.commit()

def fetch_embeddings(con):
    cur = con.cursor()
    cur.execute("SELECT id, path, start_line, end_line, content, embedding FROM chunks")
    ids, meta, vecs = [], [], []
    for id_, path, s, e, content, emb in cur.fetchall():
        ids.append(id_); meta.append((path,s,e,content))
        vecs.append(np.frombuffer(emb, dtype="float32"))
    return ids, meta, (np.vstack(vecs) if vecs else np.zeros((0,768), dtype="float32"))

def top_k(con, query_vec: np.ndarray, k=8):
    ids, meta, mat = fetch_embeddings(con)
    if mat.shape[0] == 0: return []
    # cosine
    q = query_vec / (np.linalg.norm(query_vec)+1e-8)
    m = mat / (np.linalg.norm(mat, axis=1, keepdims=True)+1e-8)
    scores = m @ q
    order = np.argsort(-scores)[:k]
    return [(ids[i], meta[i], float(scores[i])) for i in order]
