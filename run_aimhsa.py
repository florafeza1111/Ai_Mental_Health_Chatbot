#!/usr/bin/env python3
"""
AIMHSA Unified Launcher
Runs both backend API and frontend on a single port using Flask
"""

import os
import sys
import time
import webbrowser
import argparse
from flask import Flask, request, jsonify, send_from_directory, render_template_string
from flask_cors import CORS
import json
import numpy as np
from openai import OpenAI
from dotenv import load_dotenv
import sqlite3
from werkzeug.security import generate_password_hash, check_password_hash
import uuid
import tempfile
import pytesseract
from werkzeug.utils import secure_filename

# Load environment variables
load_dotenv()

# Configuration
EMBED_FILE = "storage/embeddings.json"
CHAT_MODEL = os.getenv("CHAT_MODEL", "llama3.2")
EMBED_MODEL = os.getenv("EMBED_MODEL", "nomic-embed-text")
DB_FILE = "storage/conversations.db"
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434/v1")
OLLAMA_API_KEY = os.getenv("OLLAMA_API_KEY", "ollama")  # Ollama doesn't require a real API key

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Initialize OpenAI client for Ollama
openai_client = OpenAI(
    base_url=OLLAMA_BASE_URL,
    api_key=OLLAMA_API_KEY
)

# System prompt for AIMHSA
SYSTEM_PROMPT = """You are AIMHSA, a supportive mental-health companion for Rwanda.
- Be warm, brief, and evidence-informed. Use simple English (or Kinyarwanda if the user uses it).
- Do NOT diagnose or prescribe medications. Encourage professional care when appropriate.
- If the user mentions self-harm or immediate danger, express care and advise contacting local emergency services right away.
- Ground answers in the provided CONTEXT. If context is insufficient, say what is known and unknown, and offer general coping strategies.
- Only answer in English!
- Also keep it brief except when details are required.
"""

# Global variables for embeddings
chunk_texts = []
chunk_sources = []
chunk_embeddings = None

def init_storage():
    """Initialize database and load embeddings"""
    os.makedirs(os.path.dirname(DB_FILE), exist_ok=True)
    os.makedirs(os.path.dirname(EMBED_FILE), exist_ok=True)
    
    conn = sqlite3.connect(DB_FILE)
    try:
        # Create tables
        conn.execute("""
            CREATE TABLE IF NOT EXISTS messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                conv_id TEXT NOT NULL,
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                ts REAL NOT NULL
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS attachments (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                conv_id TEXT NOT NULL,
                filename TEXT NOT NULL,
                text TEXT NOT NULL,
                ts REAL NOT NULL
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS sessions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                key TEXT UNIQUE NOT NULL,
                conv_id TEXT NOT NULL,
                ts REAL NOT NULL
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS users (
                username TEXT PRIMARY KEY,
                password_hash TEXT NOT NULL,
                created_ts REAL NOT NULL
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS conversations (
                conv_id TEXT PRIMARY KEY,
                owner_key TEXT,
                preview TEXT,
                ts REAL
            )
        """)
        conn.commit()
    finally:
        conn.close()
    
    # Load embeddings
    global chunk_texts, chunk_sources, chunk_embeddings
    try:
        with open(EMBED_FILE, "r", encoding="utf-8") as f:
            chunks_data = json.load(f)
        chunk_texts = [c["text"] for c in chunks_data]
        chunk_sources = [{"source": c["source"], "chunk": c["chunk"]} for c in chunks_data]
        chunk_embeddings = np.array([c["embedding"] for c in chunks_data], dtype=np.float32)
        print(f"✅ Loaded {len(chunk_texts)} embedding chunks")
    except FileNotFoundError:
        print("⚠️  Embeddings file not found. RAG features will be limited.")
        chunk_texts = []
        chunk_sources = []
        chunk_embeddings = None

def cosine_similarity(a, b):
    """Calculate cosine similarity between embeddings"""
    a_norm = a / np.linalg.norm(a, axis=1, keepdims=True)
    b_norm = b / np.linalg.norm(b, axis=1, keepdims=True)
    return np.dot(a_norm, b_norm.T)

def retrieve(query: str, k: int = 4):
    """Retrieve relevant chunks using embeddings"""
    if chunk_embeddings is None:
        return []
    
    try:
        # Use OpenAI client for embeddings
        response = openai_client.embeddings.create(
            model=EMBED_MODEL,
            input=query
        )
        q_emb = np.array([response.data[0].embedding], dtype=np.float32)
        sims = cosine_similarity(chunk_embeddings, q_emb)[:,0]
        top_idx = sims.argsort()[-k:][::-1]
        return [(chunk_texts[i], chunk_sources[i]) for i in top_idx]
    except Exception as e:
        print(f"Error in retrieve: {e}")
        return []

def build_context(snippets):
    """Build context from retrieved snippets"""
    lines = []
    for i, (doc, meta) in enumerate(snippets, 1):
        src = f"{meta.get('source','unknown')}#chunk{meta.get('chunk')}"
        lines.append(f"[{i}] ({src}) {doc}")
    return "\n\n".join(lines)

def save_message(conv_id: str, role: str, content: str):
    """Save message to database"""
    conn = sqlite3.connect(DB_FILE)
    try:
        conn.execute(
            "INSERT INTO messages (conv_id, role, content, ts) VALUES (?, ?, ?, ?)",
            (conv_id, role, content, time.time()),
        )
        conn.commit()
    finally:
        conn.close()

def load_history(conv_id: str):
    """Load conversation history"""
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

# ============================================================================
# FRONTEND ROUTES
# ============================================================================

@app.route('/')
def index():
    """Serve main chat interface"""
    return send_from_directory('chatbot', 'index.html')

@app.route('/landing')
@app.route('/landing.html')
def landing():
    """Serve landing page"""
    return send_from_directory('chatbot', 'landing.html')

@app.route('/login')
@app.route('/login.html')
def login():
    """Serve login page"""
    return send_from_directory('chatbot', 'login.html')

@app.route('/register')
@app.route('/register.html')
def register():
    """Serve registration page"""
    return send_from_directory('chatbot', 'register.html')

@app.route('/admin_dashboard.html')
def admin_dashboard():
    """Serve admin dashboard"""
    return send_from_directory('chatbot', 'admin_dashboard.html')

@app.route('/professional_dashboard.html')
def professional_dashboard():
    """Serve professional dashboard"""
    return send_from_directory('chatbot', 'professional_dashboard.html')

@app.route('/<path:filename>')
def static_files(filename):
    """Serve static files (CSS, JS, etc.)"""
    return send_from_directory('chatbot', filename)

# ============================================================================
# API ROUTES
# ============================================================================

@app.route('/healthz')
def healthz():
    """Health check endpoint"""
    return {"ok": True}

@app.route('/ask', methods=['POST'])
def ask():
    """Main chat endpoint"""
    data = request.get_json(force=True)
    query = (data.get("query") or "").strip()
    if not query:
        return jsonify({"error": "Missing 'query'"}), 400

    # Conversation ID handling
    conv_id = data.get("id")
    new_conv = False
    if not conv_id:
        conv_id = str(uuid.uuid4())
        new_conv = True

    # Load conversation history
    history = load_history(conv_id)
    
    # Build messages for AI
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    
    # Add conversation history
    for entry in history:
        role = entry.get("role", "user")
        if role in ("user", "assistant"):
            messages.append({"role": role, "content": entry.get("content", "")})
    
    # Add current query
    messages.append({"role": "user", "content": query})
    
    # Get context from embeddings if available
    context = ""
    if chunk_embeddings is not None:
        top = retrieve(query, k=4)
        context = build_context(top)
    
    # Build user prompt with context
    if context:
        user_prompt = f"""Answer the user's question using ONLY the CONTEXT below.
If the context is insufficient, be honest and provide safe, general guidance.

QUESTION:
{query}

CONTEXT:
{context}
"""
    else:
        user_prompt = query
    
    # Replace the last user message with the enhanced prompt
    messages[-1] = {"role": "user", "content": user_prompt}
    
    try:
        # Get AI response using OpenAI client
        response = openai_client.chat.completions.create(
            model=CHAT_MODEL,
            messages=messages,
            temperature=0.7,
            max_tokens=1000
        )
        answer = response.choices[0].message.content
        
        # Save conversation
        save_message(conv_id, "user", query)
        save_message(conv_id, "assistant", answer)
        
        # Prepare response
        resp = {"answer": answer, "id": conv_id}
        if new_conv:
            resp["new"] = True
        
        return jsonify(resp)
        
    except Exception as e:
        return jsonify({"error": f"Unexpected error: {str(e)}"}), 500

@app.route('/api/register', methods=['POST'])
def api_register():
    """User registration endpoint"""
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

@app.route('/api/login', methods=['POST'])
def api_login():
    """User login endpoint"""
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

@app.route('/api/history')
def api_history():
    """Get conversation history"""
    conv_id = request.args.get("id")
    if not conv_id:
        return jsonify({"error": "Missing 'id' parameter"}), 400
    
    try:
        hist = load_history(conv_id)
        return jsonify({"id": conv_id, "history": hist})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/reset', methods=['POST'])
def api_reset():
    """Reset database"""
    conn = sqlite3.connect(DB_FILE)
    try:
        conn.execute("DELETE FROM messages")
        conn.execute("DELETE FROM attachments")
        conn.execute("DELETE FROM sessions")
        conn.execute("DELETE FROM conversations")
        conn.execute("DELETE FROM users")
        conn.commit()
    finally:
        conn.close()
    
    return jsonify({"ok": True})

# ============================================================================
# MAIN FUNCTION
# ============================================================================

def test_ollama_connection():
    """Test connection to Ollama"""
    try:
        print("🔗 Testing Ollama connection...")
        response = openai_client.chat.completions.create(
            model=CHAT_MODEL,
            messages=[{"role": "user", "content": "Hello"}],
            max_tokens=10
        )
        print("✅ Ollama connection successful!")
        return True
    except Exception as e:
        print(f"❌ Ollama connection failed: {e}")
        print("💡 Make sure Ollama is running: ollama serve")
        return False

def main():
    parser = argparse.ArgumentParser(description="AIMHSA Unified Launcher - Single Port")
    parser.add_argument("--port", "-p", type=int, default=8000, help="Port to run on (default: 8000)")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to (default: 0.0.0.0)")
    parser.add_argument("--skip-ollama-test", action="store_true", help="Skip Ollama connection test")
    args = parser.parse_args()
    
    print("="*60)
    print("🧠 AIMHSA - AI Mental Health Support Assistant")
    print("="*60)
    print(f"🌐 Running on: http://{args.host}:{args.port}")
    print(f"🤖 Ollama URL: {OLLAMA_BASE_URL}")
    print(f"🧠 Chat Model: {CHAT_MODEL}")
    print(f"🔍 Embed Model: {EMBED_MODEL}")
    print("="*60)
    
    # Test Ollama connection
    if not args.skip_ollama_test:
        if not test_ollama_connection():
            print("⚠️  Continuing without Ollama connection test...")
    
    # Initialize storage and embeddings
    print("🚀 Initializing AIMHSA...")
    init_storage()
    
    print("Available routes:")
    print(f"  - http://localhost:{args.port}/ (main chat)")
    print(f"  - http://localhost:{args.port}/landing (landing page)")
    print(f"  - http://localhost:{args.port}/login (login page)")
    print(f"  - http://localhost:{args.port}/register (register page)")
    print("="*60)
    
    # Open browser
    try:
        webbrowser.open(f"http://localhost:{args.port}")
    except Exception:
        pass
    
    print("✅ AIMHSA is ready!")
    print("Press Ctrl+C to stop")
    
    # Run Flask app
    app.run(host=args.host, port=args.port, debug=False)

if __name__ == "__main__":
    main()