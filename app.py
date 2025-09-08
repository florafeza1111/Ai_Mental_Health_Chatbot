import os, json, numpy as np
from flask import Flask, request, jsonify, send_from_directory, render_template, redirect
from flask_cors import CORS
import ollama
from dotenv import load_dotenv
import sqlite3
from werkzeug.security import generate_password_hash, check_password_hash
import time
import uuid
import tempfile
import pytesseract
from werkzeug.utils import secure_filename

load_dotenv()

# --- Constants ---
DATA_DIR = "data"  # knowledgebase directory containing source files
STORAGE_DIR = "storage"
EMBED_FILE = os.path.join(STORAGE_DIR, "embeddings.json")
CHAT_MODEL = os.getenv("CHAT_MODEL", "llama3.2:3b")
EMBED_MODEL = os.getenv("EMBED_MODEL", "nomic-embed-text")
# sentence-level embedder used for query / semantic search (can be same as EMBED_MODEL or a dedicated sentence model)
SENT_EMBED_MODEL = os.getenv("SENT_EMBED_MODEL", EMBED_MODEL)

# lazy-loaded SentenceTransformer instance (only used when SENT_EMBED_MODEL points to a sentence-transformers model)
SENT_MODEL = None
USE_SENT_TRANSFORMERS = SENT_EMBED_MODEL.startswith("sentence-transformers/")

# --- new DB config ---
DB_FILE = "storage/conversations.db"

def init_storage():
    os.makedirs(os.path.dirname(DB_FILE), exist_ok=True)
    # ensure embeddings storage dir exists too
    os.makedirs(os.path.dirname(EMBED_FILE), exist_ok=True)
    conn = sqlite3.connect(DB_FILE)
    try:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                conv_id TEXT NOT NULL,
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                ts REAL NOT NULL
            )
        """)
        # attachments table: stores extracted text per uploaded file
        conn.execute("""
            CREATE TABLE IF NOT EXISTS attachments (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                conv_id TEXT NOT NULL,
                filename TEXT NOT NULL,
                text TEXT NOT NULL,
                ts REAL NOT NULL
            )
        """)
        # sessions table: map an ip/account key to a conversation id
        conn.execute("""
            CREATE TABLE IF NOT EXISTS sessions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                key TEXT UNIQUE NOT NULL,
                conv_id TEXT NOT NULL,
                ts REAL NOT NULL
            )
        """)
        # users table: store username + password hash
        conn.execute("""
            CREATE TABLE IF NOT EXISTS users (
                username TEXT PRIMARY KEY,
                password_hash TEXT NOT NULL,
                created_ts REAL NOT NULL
            )
        """)
        # conversations table: metadata for user-visible conversation list
        conn.execute("""
            CREATE TABLE IF NOT EXISTS conversations (
                conv_id TEXT PRIMARY KEY,
                owner_key TEXT,
                preview TEXT,
                ts REAL
            )
        """)
        # Add archived column if missing
        try:
            cur = conn.execute("PRAGMA table_info(conversations)")
            cols = [r[1] for r in cur.fetchall()]
            if "archived" not in cols:
                conn.execute("ALTER TABLE conversations ADD COLUMN archived INTEGER DEFAULT 0")
            if "archive_pw_hash" not in cols:
                conn.execute("ALTER TABLE conversations ADD COLUMN archive_pw_hash TEXT")
        except Exception:
            pass
        conn.commit()
    finally:
        conn.close()

def create_conversation(owner_key: str = None, conv_id: str = None, preview: str = "New chat"):
    if not conv_id:
        conv_id = str(uuid.uuid4())
    conn = sqlite3.connect(DB_FILE)
    try:
        conn.execute(
            "INSERT OR IGNORE INTO conversations (conv_id, owner_key, preview, ts) VALUES (?, ?, ?, ?)",
            (conv_id, owner_key, preview, time.time()),
        )
        # if a row existed with no owner_key and we received one, update it
        if owner_key:
            conn.execute(
                "UPDATE conversations SET owner_key = ?, ts = ? WHERE conv_id = ? AND (owner_key IS NULL OR owner_key = '')",
                (owner_key, time.time(), conv_id),
            )
        conn.commit()
    finally:
        conn.close()
    return conv_id

# helper: map conv_id -> owner_key (if any) using sessions table
def get_owner_key_for_conv(conv_id: str):
    conn = sqlite3.connect(DB_FILE)
    try:
        cur = conn.execute("SELECT key FROM sessions WHERE conv_id = ? LIMIT 1", (conv_id,))
        row = cur.fetchone()
        return row[0] if row else None
    finally:
        conn.close()

def save_message(conv_id: str, role: str, content: str):
    conn = sqlite3.connect(DB_FILE)
    try:
        conn.execute(
            "INSERT INTO messages (conv_id, role, content, ts) VALUES (?, ?, ?, ?)",
            (conv_id, role, content, time.time()),
        )
        # update conversation preview/timestamp for owner-visible list
        try:
            if role == "user":
                snippet = _extract_question_from_prompt(content)
                snippet = (snippet.strip().replace("\n", " ") if snippet else "").strip()
                if snippet:
                    # find existing conversation row
                    cur = conn.execute("SELECT preview FROM conversations WHERE conv_id = ?", (conv_id,))
                    row = cur.fetchone()
                    # determine owner_key if needed
                    owner_key = get_owner_key_for_conv(conv_id)
                    if row is None:
                        # create conversation row if missing, attach owner_key when available
                        conn.execute(
                            "INSERT OR REPLACE INTO conversations (conv_id, owner_key, preview, ts) VALUES (?, ?, ?, ?)",
                            (conv_id, owner_key, snippet[:120], time.time()),
                        )
                    else:
                        existing_preview = row[0] or ""
                        if existing_preview.strip() in ("", "New chat"):
                            conn.execute(
                                "UPDATE conversations SET preview = ?, ts = ? WHERE conv_id = ?",
                                (snippet[:120], time.time(), conv_id),
                            )
                        else:
                            # update timestamp at least so listing sorts by recent activity
                            conn.execute(
                                "UPDATE conversations SET ts = ? WHERE conv_id = ?",
                                (time.time(), conv_id),
                            )
        except Exception:
            # don't break saving messages if preview update fails
            pass
        conn.commit()
    finally:
        conn.close()

def load_history(conv_id: str):
    conn = sqlite3.connect(DB_FILE)
    try:
        cur = conn.execute(
            "SELECT role, content FROM messages WHERE conv_id = ? ORDER BY id ASC",
            (conv_id,),
        )
        rows = cur.fetchall()
        return [{"role": r[0], "content": r[1]} for r in rows]
    finally:
        conn.close()

def reset_db():
    conn = sqlite3.connect(DB_FILE)
    try:
        # remove all conversation messages, attachments and session mappings
        conn.execute("DELETE FROM messages")
        conn.execute("DELETE FROM attachments")
        conn.execute("DELETE FROM sessions")
        conn.execute("DELETE FROM conversations")
        conn.execute("DELETE FROM users")
        conn.commit()
    finally:
        conn.close()
# --- end DB helpers ---

app = Flask(__name__)
CORS(app)

SYSTEM_PROMPT = """You are AIMHSA, a supportive mental-health companion for Rwanda.
- Be warm, brief, and evidence-informed. Use simple English (or Kinyarwanda if the user uses it).
- Do NOT diagnose or prescribe medications. Encourage professional care when appropriate.
- If the user mentions self-harm or immediate danger, express care and advise contacting local emergency services right away.
- Ground answers in the provided CONTEXT. If context is insufficient, say what is known and unknown, and offer general coping strategies.only answer in English!
also keep it brief except when details are required.
"""

def rebuild_vector_store():
    """
    Rebuild vector store from documents in /data directory.
    - Process all .txt files in /data
    - Split into chunks with overlap
    - Embed chunks using EMBED_MODEL
    - Save to storage/embeddings.json
    """
    app.logger.info("Rebuilding vector store from /data...")
    
    # ensure storage dir exists
    os.makedirs(STORAGE_DIR, exist_ok=True)
    
    chunks = []
    chunk_id = 0
    
    # process all .txt files in data directory
    for root, _, files in os.walk(DATA_DIR):
        for fname in files:
            if not fname.endswith('.txt'):
                continue
            
            fpath = os.path.join(root, fname)
            rel_path = os.path.relpath(fpath, DATA_DIR)
            
            with open(fpath, 'r', encoding='utf-8') as f:
                text = f.read()
            
            # split into chunks (~500 chars with 100 char overlap)
            words = text.split()
            chunk_words = []
            chunk_size = 500
            overlap = 100
            
            for i in range(0, len(words), chunk_size - overlap):
                chunk = ' '.join(words[i:i + chunk_size])
                if not chunk.strip():
                    continue
                    
                chunks.append({
                    "text": chunk,
                    "source": rel_path,
                    "chunk": chunk_id
                })
                chunk_id += 1
    
    if not chunks:
        app.logger.warning("No chunks found in /data directory")
        return
    
    # embed chunks using EMBED_MODEL
    try:
        app.logger.info(f"Embedding {len(chunks)} chunks...")
        texts = [c["text"] for c in chunks]
        
        # batch embed to avoid memory issues (32 chunks per batch)
        batch_size = 32
        all_embeddings = []
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            resp = ollama.embed(model=EMBED_MODEL, input=batch)
            all_embeddings.extend(resp["embeddings"])
            
        # add embeddings to chunks
        for chunk, embedding in zip(chunks, all_embeddings):
            chunk["embedding"] = embedding
            
        # save to storage/embeddings.json
        with open(EMBED_FILE, 'w', encoding='utf-8') as f:
            json.dump(chunks, f, ensure_ascii=False, indent=2)
            
        app.logger.info(f"Saved {len(chunks)} embedded chunks to {EMBED_FILE}")
        return chunks
            
    except Exception as e:
        app.logger.exception("Failed to embed chunks")
        raise

# --- Load embeddings into memory ---
chunks_data = None
if os.path.exists(EMBED_FILE):
    try:
        with open(EMBED_FILE, "r", encoding="utf-8") as f:
            chunks_data = json.load(f)
        app.logger.info(f"Loaded {len(chunks_data)} chunks from {EMBED_FILE}")
    except Exception:
        app.logger.exception(f"Failed to load {EMBED_FILE}")

if not chunks_data:
    # rebuild if no valid embeddings found
    chunks_data = rebuild_vector_store()
    if not chunks_data:
        raise RuntimeError("Failed to initialize vector store")

# prepare numpy arrays for retrieval
chunk_texts = [c["text"] for c in chunks_data]
chunk_sources = [{"source": c["source"], "chunk": c["chunk"]} for c in chunks_data]
chunk_embeddings = np.array([c["embedding"] for c in chunks_data], dtype=np.float32)

# --- Cosine similarity function ---
def cosine_similarity(a, b):
    a_norm = a / np.linalg.norm(a, axis=1, keepdims=True)
    b_norm = b / np.linalg.norm(b, axis=1, keepdims=True)
    return np.dot(a_norm, b_norm.T)

def _mmr_selection(doc_embs: np.ndarray, query_emb: np.ndarray, k: int = 4, lambda_param: float = 0.6):
    """
    Maximal Marginal Relevance selection for diversity+relevance.
    doc_embs: (n_docs, dim)
    query_emb: (1, dim) or (dim,)
    returns: list of selected indices (len <= k)
    """
    if doc_embs.size == 0:
        return []
    # normalize
    doc_norm = doc_embs / np.linalg.norm(doc_embs, axis=1, keepdims=True)
    q = query_emb.reshape(-1)
    q_norm = q / np.linalg.norm(q)

    # relevance scores to query
    sims_q = np.dot(doc_norm, q_norm)
    selected = []
    # pick highest relevance first
    first = int(np.argmax(sims_q))
    selected.append(first)
    if k == 1:
        return selected

    candidates = set(range(doc_embs.shape[0])) - set(selected)
    # precompute doc-doc similarities for speed
    doc_doc_sims = np.dot(doc_norm, doc_norm.T)

    while len(selected) < k and candidates:
        best_score = None
        best_idx = None
        for cand in candidates:
            # relevance
            rel = sims_q[cand]
            # redundancy = max similarity to already selected
            red = max(doc_doc_sims[cand, s] for s in selected) if selected else 0.0
            score = lambda_param * rel - (1.0 - lambda_param) * red
            if best_score is None or score > best_score:
                best_score = score
                best_idx = cand
        if best_idx is None:
            break
        selected.append(best_idx)
        candidates.remove(best_idx)
    return selected

def retrieve(query: str, k: int = 4, lambda_param: float = 0.6):
    """
    Semantic retrieval: embed the query with a sentence embedding model and
    select top-k chunks using MMR for a balance of relevance and diversity.

    Supports two modes:
      - If SENT_EMBED_MODEL is "sentence-transformers/<model-name>", uses the
        local sentence-transformers library (SentenceTransformer).
      - Otherwise falls back to ollama.embed with the configured model.
    """
    global SENT_MODEL

    # compute query embedding
    if USE_SENT_TRANSFORMERS:
        # model name format: sentence-transformers/<model-id>
        model_id = SENT_EMBED_MODEL.split("/", 1)[1]
        try:
            if SENT_MODEL is None:
                from sentence_transformers import SentenceTransformer
                SENT_MODEL = SentenceTransformer(model_id)
            # encode returns (dim,) or (1,dim) depending on args; ensure numpy array (1,dim)
            q_emb = SENT_MODEL.encode(query, convert_to_numpy=True)
            if q_emb.ndim == 1:
                q_emb = q_emb.reshape(1, -1)
            q_emb = q_emb.astype(np.float32)
        except Exception as e:
            # fallback to ollama if local model not available
            try:
                q_emb_resp = ollama.embed(model=EMBED_MODEL, input=[query])
                q_emb = np.array(q_emb_resp["embeddings"], dtype=np.float32)
            except Exception:
                raise
    else:
        # default: use ollama embed API
        try:
            q_emb_resp = ollama.embed(model=SENT_EMBED_MODEL, input=[query])
            q_emb = np.array(q_emb_resp["embeddings"], dtype=np.float32)
        except Exception as e:
            app.logger.error(f"Failed to embed query with {SENT_EMBED_MODEL}: {e}")
            # Return empty results if embedding fails
            return []

    # ensure chunk_embeddings shape OK
    if chunk_embeddings.size == 0:
        return []

    # select indices via MMR (works with doc embeddings and query embedding)
    idxs = _mmr_selection(chunk_embeddings, q_emb, k=k, lambda_param=lambda_param)
    return [(chunk_texts[i], chunk_sources[i]) for i in idxs]

def build_context(snippets):
    lines = []
    for i, (doc, meta) in enumerate(snippets, 1):
        src = f"{meta.get('source','unknown')}#chunk{meta.get('chunk')}"
        lines.append(f"[{i}] ({src}) {doc}")
    return "\n\n".join(lines)

@app.get("/healthz")
def healthz():
    return {"ok": True}

# initialize DB on startup
init_storage()

# --- helper to normalize older saved "user_prompt" shapes so we don't re-save CONTEXT ---
def _extract_question_from_prompt(content: str) -> str:
    """
    If content looks like the constructed user_prompt with "QUESTION:" and "CONTEXT:",
    extract and return only the QUESTION text. Otherwise return content unchanged.
    """
    if not isinstance(content, str):
        return content
    low = content
    q_marker = "QUESTION:"
    c_marker = "CONTEXT:"
    if q_marker in low and c_marker in low:
        try:
            q_start = low.index(q_marker) + len(q_marker)
            c_start = low.index(c_marker)
            question = low[q_start:c_start].strip()
            if question:
                return question
        except Exception:
            pass
    return content
# --- end helper ---

# --- conversation helpers ---
def create_conversation(owner_key: str = None, conv_id: str = None, preview: str = "New chat"):
    if not conv_id:
        conv_id = str(uuid.uuid4())
    conn = sqlite3.connect(DB_FILE)
    try:
        conn.execute(
            "INSERT OR IGNORE INTO conversations (conv_id, owner_key, preview, ts) VALUES (?, ?, ?, ?)",
            (conv_id, owner_key, preview, time.time()),
        )
        # if a row existed with no owner_key and we received one, update it
        if owner_key:
            conn.execute(
                "UPDATE conversations SET owner_key = ?, ts = ? WHERE conv_id = ? AND (owner_key IS NULL OR owner_key = '')",
                (owner_key, time.time(), conv_id),
            )
        conn.commit()
    finally:
        conn.close()
    return conv_id

def list_conversations(owner_key: str):
    conn = sqlite3.connect(DB_FILE)
    try:
        cur = conn.execute(
            "SELECT conv_id, preview, ts FROM conversations WHERE owner_key = ? AND IFNULL(archived,0) = 0 ORDER BY ts DESC",
            (owner_key,),
        )
        rows = cur.fetchall()
        return [{"id": r[0], "preview": r[1] or "New chat", "timestamp": r[2]} for r in rows]
    finally:
        conn.close()
# --- end conversation helpers ---

@app.post("/ask")
def ask():
    data = request.get_json(force=True)
    query = (data.get("query") or "").strip()
    if not query:
        return jsonify({"error": "Missing 'query'"}), 400

    # conversation id handling: if none provided, create one and return it
    conv_id = data.get("id")
    new_conv = False
    if not conv_id:
        conv_id = str(uuid.uuid4())
        new_conv = True

    # if new conv created server-side, make sure we have a conversations entry (owner inferred from account or ip)
    if new_conv:
        owner = None
        account = (data.get("account") or "").strip()
        if account:
            owner = f"acct:{account}"
        else:
            ip = request.remote_addr or "unknown"
            owner = f"ip:{ip}"
        create_conversation(owner_key=owner, conv_id=conv_id, preview="New chat")

    # client may supply recent history; ensure it's a list
    client_history = data.get("history", [])
    if not isinstance(client_history, list):
        client_history = []

    # load server-side history for this conv_id
    server_history = load_history(conv_id)

    # load attachments for this conv_id (won't be persisted into messages table;
    # attachments are provided as separate CONTEXT blocks to the model)
    attachments = load_attachments(conv_id)

    # build a set of existing (role, content) pairs to avoid duplicates; normalize saved user prompts
    existing_set = set()
    normalized_server = []
    for entry in server_history:
        role = entry.get("role", "user")
        content = entry.get("content", "")
        if role == "user":
            content = _extract_question_from_prompt(content)
        normalized_server.append({"role": role, "content": content})
        existing_set.add((role, content))

    # merge histories: system prompt, then attachments as SYSTEM CONTEXT, then server_history, then client_history
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    # include attachments as separate system-context blocks (kept short-ish)
    for att in attachments:
        att_text = att.get("text", "")
        if att_text:
            # truncate very long attachments to a safe limit to avoid blowing token budget
            SHORT = 40_000
            if len(att_text) > SHORT:
                att_text = att_text[:SHORT] + "\n\n...[truncated]"
            messages.append({"role": "system", "content": f"PDF CONTEXT ({att.get('filename')}):\n{att_text}"})

    for entry in normalized_server:
        role = entry.get("role", "user")
        if role not in ("user", "assistant"):
            role = "user"
        messages.append({"role": role, "content": entry.get("content", "")})

    # If client provided additional history, append it (and persist only if not already present)
    for entry in client_history:
        role = entry.get("role", "user")
        if role not in ("user", "assistant"):
            role = "user"
        content = entry.get("content", "")
        if content:
            # normalize client's user entries when comparing against existing saved entries
            cmp_content = _extract_question_from_prompt(content) if role == "user" else content
            if (role, cmp_content) not in existing_set:
                messages.append({"role": role, "content": content})
                save_message(conv_id, role, cmp_content)  # persist the normalized/raw client content
                existing_set.add((role, cmp_content))
            else:
                # already present server-side; still include in messages so model has recent context
                messages.append({"role": role, "content": content})

    # retrieval-based context
    top = retrieve(query, k=4)
    context = build_context(top)

    user_prompt = f"""Answer the user's question using ONLY the CONTEXT below.
If the context is insufficient, be honest and provide safe, general guidance.

QUESTION:
{query}

CONTEXT:
{context}
"""

    # add the current user question to messages (do NOT persist the whole user_prompt)
    messages.append({"role": "user", "content": user_prompt})

    try:
        reply = ollama.chat(model=CHAT_MODEL, messages=messages)
        answer = reply["message"]["content"]
    except Exception as e:
        app.logger.error(f"Failed to get chat response with {CHAT_MODEL}: {e}")
        # Provide a fallback response when the model is not available
        answer = f"I'm sorry, I'm having trouble accessing my AI model right now. However, I can still help you with mental health resources in Rwanda. Please contact the Mental Health Hotline at 105 or CARAES Ndera Hospital at +250 788 305 703 for immediate support. You can also try refreshing the page or contacting support if this issue persists."

    # persist the current user RAW query (not the constructed user_prompt) and assistant reply
    save_message(conv_id, "user", query)
    save_message(conv_id, "assistant", answer)

    sources = [{"source": m["source"], "chunk": m["chunk"]} for (_, m) in top]
    resp = {"answer": answer, "sources": sources, "id": conv_id}
    # if newly created conv, client will need to store/use this id
    if new_conv:
        resp["new"] = True
    return jsonify(resp)

@app.post("/reset")
def reset():
    # clear stored conversations, attachments and sessions
    reset_db()
    return jsonify({"ok": True})

# --- attachment helpers ---
def save_attachment(conv_id: str, filename: str, text: str):
    conn = sqlite3.connect(DB_FILE)
    try:
        conn.execute(
            "INSERT INTO attachments (conv_id, filename, text, ts) VALUES (?, ?, ?, ?)",
            (conv_id, filename, text, time.time()),
        )
        conn.commit()
    finally:
        conn.close()

def load_attachments(conv_id: str):
    conn = sqlite3.connect(DB_FILE)
    try:
        cur = conn.execute(
            "SELECT filename, text FROM attachments WHERE conv_id = ? ORDER BY id ASC",
            (conv_id,),
        )
        rows = cur.fetchall()
        return [{"filename": r[0], "text": r[1]} for r in rows]
    finally:
        conn.close()

# --- session helpers (new) ---
def get_or_create_session(key: str):
    """Return (conv_id, was_created_bool) for the given session key."""
    conn = sqlite3.connect(DB_FILE)
    try:
        cur = conn.execute("SELECT conv_id FROM sessions WHERE key = ?", (key,))
        row = cur.fetchone()
        if row:
            conv_id = row[0]
            conn.execute("UPDATE sessions SET ts = ? WHERE key = ?", (time.time(), key))
            # ensure conversations entry exists and is associated with this owner key
            try:
                # create conversation row if missing
                conn.execute(
                    "INSERT OR IGNORE INTO conversations (conv_id, owner_key, preview, ts) VALUES (?, ?, ?, ?)",
                    (conv_id, key, "New chat", time.time()),
                )
                # if conversation exists without owner_key, set it
                conn.execute(
                    "UPDATE conversations SET owner_key = ? WHERE conv_id = ? AND (owner_key IS NULL OR owner_key = '')",
                    (key, conv_id),
                )
            except Exception:
                pass
            conn.commit()
            return conv_id, False
        conv_id = str(uuid.uuid4())
        conn.execute(
            "INSERT INTO sessions (key, conv_id, ts) VALUES (?, ?, ?)",
            (key, conv_id, time.time()),
        )
        # also create a conversations row bound to this owner key
        try:
            conn.execute(
                "INSERT OR IGNORE INTO conversations (conv_id, owner_key, preview, ts) VALUES (?, ?, ?, ?)",
                (conv_id, key, "New chat", time.time()),
            )
        except Exception:
            pass
        conn.commit()
        return conv_id, True
    finally:
        conn.close()

# --- API: create/retrieve session by IP or account ---
@app.post("/session")
def session():
    """
    Request JSON: { "account": "<optional account id>" }
    If account is provided, session is bound to account:<account>.
    Otherwise session is bound to ip:<remote_addr>.
    Returns: { "id": "<conv_id>", "new": true|false }
    """
    try:
        data = request.get_json(silent=True) or {}
    except Exception:
        data = {}
    account = (data.get("account") or "").strip()
    if account:
        key = f"acct:{account}"
    else:
        # request.remote_addr may be proxied; frontends should pass account when available
        ip = request.remote_addr or "unknown"
        key = f"ip:{ip}"
    conv_id, new = get_or_create_session(key)
    return jsonify({"id": conv_id, "new": new})

# --- API: get conversation history (messages + attachments) ---
@app.get("/history")
def history():
    """
    Query params: ?id=<conv_id>
    Returns: { "id": "<conv_id>", "history": [ {role, content}, ... ], "attachments": [ {filename,text}, ... ] }
    """
    conv_id = request.args.get("id")
    password = (request.args.get("password") or "").strip()
    if not conv_id:
        return jsonify({"error": "Missing 'id' parameter"}), 400
    try:
        # if conversation is archived and locked, require password to view history
        try:
            conn = sqlite3.connect(DB_FILE)
            cur = conn.execute("SELECT IFNULL(archived,0), archive_pw_hash FROM conversations WHERE conv_id = ?", (conv_id,))
            row = cur.fetchone()
        finally:
            conn.close()
        if row and int(row[0]) == 1 and row[1]:
            if not password or not check_password_hash(row[1], password):
                return jsonify({"error": "password required"}), 403
        hist = load_history(conv_id)
        atts = load_attachments(conv_id)
        return jsonify({"id": conv_id, "history": hist, "attachments": atts})
    except Exception as e:
        app.logger.exception("history endpoint failed")
        return jsonify({"error": str(e)}), 500

# --- file upload endpoint (unchanged) ---
@app.post("/upload_pdf")
def upload_pdf():
    """
    Initial upload:
    Accepts multipart/form-data:
      - file: PDF file (required, .pdf only)
      - id: optional conversation id (if omitted, a new id is created)
    Returns JSON:
      { "id": "<conv_id>", "filename": "...", "new": true|false }

    Question about uploaded PDF will be handled by /ask endpoint using the stored text
    """
    if "file" not in request.files:
        return jsonify({"error": "Missing 'file'"}), 400
    f = request.files["file"]
    filename = secure_filename(f.filename or "")
    if not filename.lower().endswith(".pdf"):
        return jsonify({"error": "Only PDF files allowed"}), 400

    conv_id = request.form.get("id")
    new_conv = False
    if not conv_id:
        conv_id = str(uuid.uuid4())
        new_conv = True

    # if server created a conv for this upload, persist conversation metadata with owner
    if new_conv:
        account = (request.form.get("account") or "").strip()
        if account:
            owner = f"acct:{account}"
        else:
            owner = f"ip:{request.remote_addr or 'unknown'}"
        create_conversation(owner_key=owner, conv_id=conv_id, preview="New chat")

    # save uploaded PDF to a temp file
    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
        tmp_path = tmp.name
        f.save(tmp_path)

    extracted_text = ""
    extraction_errors = []

    try:
        # Try to render PDF pages to images using pdf2image -> pytesseract
        try:
            from pdf2image import convert_from_path
            pages = convert_from_path(tmp_path, dpi=300)
            texts = []
            for img in pages:
                try:
                    texts.append(pytesseract.image_to_string(img))
                except Exception as e_img:
                    extraction_errors.append(f"pytesseract on pdf2image image error: {e_img}")
                    app.logger.exception("pytesseract error on pdf2image image")
            extracted_text = "\n\n".join(t for t in texts if t).strip()
            if not extracted_text:
                extraction_errors.append("pdf2image+pytesseract produced empty text")
        except Exception as e_pdf2:
            extraction_errors.append(f"pdf2image error: {e_pdf2}")
            app.logger.exception("pdf2image extraction failed")

        # fallback to PyMuPDF (fitz) if first approach failed to produce text
        if not extracted_text:
            try:
                import fitz
                doc = fitz.open(tmp_path)
                texts = []
                for page in doc:
                    try:
                        pix = page.get_pixmap(dpi=300)
                        img = pix.tobytes("png")
                        from PIL import Image
                        import io
                        img_obj = Image.open(io.BytesIO(img))
                        texts.append(pytesseract.image_to_string(img_obj))
                    except Exception as e_page:
                        extraction_errors.append(f"pytesseract on fitz image error: {e_page}")
                        app.logger.exception("pytesseract error on fitz image")
                extracted_text = "\n\n".join(t for t in texts if t).strip()
                if not extracted_text:
                    extraction_errors.append("PyMuPDF+pytesseract produced empty text")
            except Exception as e_fitz:
                extraction_errors.append(f"PyMuPDF (fitz) error: {e_fitz}")
                app.logger.exception("PyMuPDF extraction failed")

        # fallback to text extraction using PyPDF2 (no OCR)
        if not extracted_text:
            try:
                from PyPDF2 import PdfReader
                reader = PdfReader(tmp_path)
                texts = []
                for p in reader.pages:
                    try:
                        texts.append(p.extract_text() or "")
                    except Exception as e_page_text:
                        extraction_errors.append(f"PyPDF2 page extract error: {e_page_text}")
                        app.logger.exception("PyPDF2 page extraction error")
                extracted_text = "\n\n".join(t for t in texts if t).strip()
                if not extracted_text:
                    extraction_errors.append("PyPDF2 produced empty text")
            except Exception as e_pypdf2:
                extraction_errors.append(f"PyPDF2 error: {e_pypdf2}")
                app.logger.exception("PyPDF2 extraction failed")

    finally:
        try:
            os.remove(tmp_path)
        except Exception:
            pass

    if not extracted_text:
        # Build user-friendly, actionable details from collected errors
        hints = []
        for err in extraction_errors:
            hints.append(err)
            # common issues -> suggested fixes
            if "Unable to get page count" in err or "pdf2image error" in err or "pdf2image" in err:
                hints.append(
                    "pdf2image needs poppler (pdftoppm). Install poppler and ensure it's in PATH "
                    "(e.g. 'apt-get install poppler-utils' or 'brew install poppler' on macOS)."
                )
            if "No module named 'fitz'" in err or "PyMuPDF (fitz) error" in err:
                hints.append("Install PyMuPDF: pip install pymupdf")
            if "No module named 'PyPDF2'" in err or "PyPDF2 error" in err:
                hints.append("Install PyPDF2: pip install PyPDF2")
            if "pytesseract" in err and ("No such file or directory" in err or "Tesseract" in err):
                hints.append(
                    "Tesseract binary not found. Install Tesseract OCR and ensure it's in PATH "
                    "(e.g. 'apt-get install tesseract-ocr' or 'brew install tesseract')."
                )

        details = " | ".join(hints) if hints else "unknown error"
        app.logger.warning("PDF extraction failed: %s", details)
        return jsonify({
            "error": "Could not extract text from PDF (no supported tool available or file empty)",
            "details": details
        }), 400

    # persist attachment
    save_attachment(conv_id, filename, extracted_text)

    resp = {"id": conv_id, "filename": filename}
    if new_conv:
        resp["new"] = True

    return jsonify(resp)

# new endpoints: create and list conversations
@app.post("/conversations")
def create_conversations_endpoint():
    """
    POST /conversations
    Body JSON: { "account": "<required account id>" }
    Returns: { "id": "<conv_id>", "new": true }
    """
    try:
        data = request.get_json(silent=True) or {}
    except Exception:
        data = {}
    account = (data.get("account") or "").strip()
    if not account:
        return jsonify({"error": "Account required to create server-backed conversations"}), 403
    key = f"acct:{account}"
    conv_id = create_conversation(owner_key=key, preview="New chat")
    return jsonify({"id": conv_id, "new": True})

@app.get("/conversations")
def get_conversations_endpoint():
    """
    GET /conversations?account=<required>
    Returns: { "conversations": [ {id, preview, timestamp}, ... ] }
    """
    account = (request.args.get("account") or "").strip()
    if not account:
        return jsonify({"error": "Account required to list conversations"}), 403
    key = f"acct:{account}"
    try:
        rows = list_conversations(key)
        return jsonify({"conversations": rows})
    except Exception as e:
        app.logger.exception("failed to list conversations")
        return jsonify({"error": str(e)}), 500

@app.post("/conversations/rename")
def rename_conversation():
    """
    POST /conversations/rename
    JSON: { "account": "...", "id": "<conv_id>", "preview": "<new title>" }
    """
    try:
        data = request.get_json(force=True)
    except Exception:
        return jsonify({"error": "Invalid JSON"}), 400
    account = (data.get("account") or "").strip()
    conv_id = (data.get("id") or "").strip()
    preview = (data.get("preview") or "").strip()
    if not account or not conv_id or not preview:
        return jsonify({"error": "account, id and preview required"}), 400
    owner_key = f"acct:{account}"
    conn = sqlite3.connect(DB_FILE)
    try:
        cur = conn.execute("SELECT owner_key, IFNULL(archived,0) FROM conversations WHERE conv_id = ?", (conv_id,))
        row = cur.fetchone()
        if not row:
            return jsonify({"error": "conversation not found"}), 404
        if (row[0] or "") != owner_key:
            return jsonify({"error": "forbidden"}), 403
        if int(row[1]) == 1:
            return jsonify({"error": "cannot rename archived conversation"}), 403
        conn.execute("UPDATE conversations SET preview = ?, ts = ? WHERE conv_id = ?", (preview[:120], time.time(), conv_id))
        conn.commit()
        return jsonify({"ok": True})
    except Exception as e:
        conn.rollback()
        return jsonify({"error": str(e)}), 500
    finally:
        conn.close()

@app.get("/conversations/archived")
def get_archived_conversations_endpoint():
    """
    GET /conversations/archived?account=<required>
    Returns archived conversations for this account
    """
    account = (request.args.get("account") or "").strip()
    if not account:
        return jsonify({"error": "Account required to list conversations"}), 403
    key = f"acct:{account}"
    conn = sqlite3.connect(DB_FILE)
    try:
        cur = conn.execute(
            "SELECT conv_id, preview, ts, CASE WHEN archive_pw_hash IS NULL OR archive_pw_hash = '' THEN 0 ELSE 1 END AS locked FROM conversations WHERE owner_key = ? AND IFNULL(archived,0) = 1 ORDER BY ts DESC",
            (key,),
        )
        rows = cur.fetchall()
        items = [{"id": r[0], "preview": r[1] or "New chat", "timestamp": r[2], "locked": bool(r[3])} for r in rows]
        return jsonify({"conversations": items})
    except Exception as e:
        app.logger.exception("failed to list archived conversations")
        return jsonify({"error": str(e)}), 500
    finally:
        conn.close()

@app.post("/conversations/archive")
def archive_conversation():
    """
    POST /conversations/archive
    JSON: { "account": "...", "id": "<conv_id>", "archived": true|false }
    """
    try:
        data = request.get_json(force=True)
    except Exception:
        return jsonify({"error": "Invalid JSON"}), 400
    account = (data.get("account") or "").strip()
    conv_id = (data.get("id") or "").strip()
    archived = bool(data.get("archived", True))
    password = (data.get("password") or "").strip()
    if not account or not conv_id:
        return jsonify({"error": "account and id required"}), 400
    owner_key = f"acct:{account}"
    conn = sqlite3.connect(DB_FILE)
    try:
        cur = conn.execute("SELECT owner_key FROM conversations WHERE conv_id = ?", (conv_id,))
        row = cur.fetchone()
        if not row:
            return jsonify({"error": "conversation not found"}), 404
        if (row[0] or "") != owner_key:
            return jsonify({"error": "forbidden"}), 403
        # when archiving, password is REQUIRED; when unarchiving, password MUST match
        if archived:
            if not password:
                return jsonify({"error": "password required to archive"}), 400
            pw_hash = generate_password_hash(password)
            conn.execute("UPDATE conversations SET archive_pw_hash = ? WHERE conv_id = ?", (pw_hash, conv_id))
        else:
            cur = conn.execute("SELECT archive_pw_hash FROM conversations WHERE conv_id = ?", (conv_id,))
            row = cur.fetchone()
            if row and row[0]:
                if not password or not check_password_hash(row[0], password):
                    return jsonify({"error": "invalid password"}), 403
            # clear hash on successful unarchive
            conn.execute("UPDATE conversations SET archive_pw_hash = NULL WHERE conv_id = ?", (conv_id,))
        conn.execute("UPDATE conversations SET archived = ? WHERE conv_id = ?", (1 if archived else 0, conv_id))
        conn.commit()
        return jsonify({"ok": True})
    except Exception as e:
        conn.rollback()
        return jsonify({"error": str(e)}), 500
    finally:
        conn.close()

@app.post("/register")
def register():
    """
    POST /register
    JSON: { "username": "...", "password": "..." }
    """
    try:
        data = request.get_json(force=True)
    except Exception:
        return jsonify({"error": "Invalid JSON"}), 400
    username = (data.get("username") or "").strip()
    password = (data.get("password") or "")
    if not username or not password:
        return jsonify({"error": "username and password required"}), 400
    pw_hash = generate_password_hash(password)
    conn = sqlite3.connect(DB_FILE)
    try:
        try:
            conn.execute(
                "INSERT INTO users (username, password_hash, created_ts) VALUES (?, ?, ?)",
                (username, pw_hash, time.time()),
            )
            conn.commit()
        except sqlite3.IntegrityError:
            return jsonify({"error": "username exists"}), 409
    finally:
        conn.close()
    return jsonify({"ok": True, "account": username})

@app.post("/login")
def login():
    """
    POST /login
    JSON: { "username": "...", "password": "..." }
    """
    try:
        data = request.get_json(force=True)
    except Exception:
        return jsonify({"error": "Invalid JSON"}), 400
    username = (data.get("username") or "").strip()
    password = (data.get("password") or "")
    if not username or not password:
        return jsonify({"error": "username and password required"}), 400
    conn = sqlite3.connect(DB_FILE)
    try:
        cur = conn.execute("SELECT password_hash FROM users WHERE username = ?", (username,))
        row = cur.fetchone()
        if not row:
            return jsonify({"error": "invalid credentials"}), 401
        stored = row[0]
        if not check_password_hash(stored, password):
            return jsonify({"error": "invalid credentials"}), 401
    finally:
        conn.close()
    return jsonify({"ok": True, "account": username})

@app.post("/clear_chat")
def clear_chat():
    """Clear messages and attachments for a conversation."""
    data = request.get_json(force=True)
    conv_id = data.get("id")
    if not conv_id:
        return jsonify({"error": "Missing conversation id"}), 400

    conn = sqlite3.connect(DB_FILE)
    try:
        # Delete messages and attachments for this conversation
        conn.execute("DELETE FROM messages WHERE conv_id = ?", (conv_id,))
        conn.execute("DELETE FROM attachments WHERE conv_id = ?", (conv_id,))
        # Reset conversation preview
        conn.execute(
            "UPDATE conversations SET preview = ? WHERE conv_id = ?",
            ("New chat", conv_id),
        )
        conn.commit()
        return jsonify({"ok": True})
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    finally:
        conn.close()

# --- delete a conversation (requires account owner) ---
@app.post("/conversations/delete")
def delete_conversation():
    """
    POST /conversations/delete
    JSON: { "account": "...", "id": "<conv_id>" }
    Only allows deletion when the conversation owner matches acct:<account>.
    """
    try:
        data = request.get_json(force=True)
    except Exception:
        return jsonify({"error": "Invalid JSON"}), 400

    account = (data.get("account") or "").strip()
    conv_id = (data.get("id") or "").strip()
    password = (data.get("password") or "").strip()
    if not account or not conv_id:
        return jsonify({"error": "account and id required"}), 400

    owner_key = f"acct:{account}"

    conn = sqlite3.connect(DB_FILE)
    try:
        cur = conn.execute("SELECT owner_key, IFNULL(archived,0), archive_pw_hash FROM conversations WHERE conv_id = ?", (conv_id,))
        row = cur.fetchone()
        if not row:
            return jsonify({"error": "conversation not found"}), 404
        if (row[0] or "") != owner_key:
            return jsonify({"error": "forbidden"}), 403
        # If archived and locked, require correct password to delete
        if int(row[1]) == 1 and row[2]:
            if not password or not check_password_hash(row[2], password):
                return jsonify({"error": "invalid password"}), 403

        # delete related rows
        conn.execute("DELETE FROM messages WHERE conv_id = ?", (conv_id,))
        conn.execute("DELETE FROM attachments WHERE conv_id = ?", (conv_id,))
        conn.execute("DELETE FROM sessions WHERE conv_id = ?", (conv_id,))
        conn.execute("DELETE FROM conversations WHERE conv_id = ?", (conv_id,))
        conn.commit()
        return jsonify({"ok": True})
    except Exception as e:
        conn.rollback()
        return jsonify({"error": str(e)}), 500
    finally:
        conn.close()

# Update run configuration to use port 5057 for API only
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5057, debug=True)