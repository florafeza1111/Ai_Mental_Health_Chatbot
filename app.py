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
import re
import math
from typing import Dict, List, Tuple, Optional
from translation_service import translation_service

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
        # password resets: username + token + expiry
        conn.execute("""
            CREATE TABLE IF NOT EXISTS password_resets (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT NOT NULL,
                token TEXT UNIQUE NOT NULL,
                expires_ts REAL NOT NULL,
                used INTEGER DEFAULT 0
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
            if "booking_prompt_shown" not in cols:
                conn.execute("ALTER TABLE conversations ADD COLUMN booking_prompt_shown INTEGER DEFAULT 0")
        except Exception:
            pass
        
        # --- NEW TABLES FOR THERAPY BOOKING SYSTEM ---
        # Professionals table (doctors, therapists, counselors)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS professionals (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT UNIQUE NOT NULL,
                password_hash TEXT NOT NULL,
                first_name TEXT NOT NULL,
                last_name TEXT NOT NULL,
                email TEXT NOT NULL,
                phone TEXT,
                license_number TEXT,
                specialization TEXT NOT NULL,
                expertise_areas TEXT NOT NULL,
                location_latitude REAL,
                location_longitude REAL,
                location_address TEXT,
                district TEXT,
                availability_schedule TEXT,
                max_patients_per_day INTEGER DEFAULT 10,
                consultation_fee REAL,
                languages TEXT,
                qualifications TEXT,
                experience_years INTEGER,
                bio TEXT,
                profile_picture TEXT,
                is_active BOOLEAN DEFAULT 1,
                created_ts REAL NOT NULL,
                updated_ts REAL NOT NULL
            )
        """)
        
        # Risk assessment table for conversation monitoring
        conn.execute("""
            CREATE TABLE IF NOT EXISTS risk_assessments (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                conv_id TEXT NOT NULL,
                user_query TEXT NOT NULL,
                risk_score REAL NOT NULL,
                risk_level TEXT NOT NULL,
                detected_indicators TEXT,
                assessment_timestamp REAL NOT NULL,
                processed BOOLEAN DEFAULT 0,
                booking_created BOOLEAN DEFAULT 0
            )
        """)
        
        # Automated bookings table
        conn.execute("""
            CREATE TABLE IF NOT EXISTS automated_bookings (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                booking_id TEXT UNIQUE NOT NULL,
                conv_id TEXT NOT NULL,
                user_account TEXT,
                user_ip TEXT,
                professional_id INTEGER NOT NULL,
                risk_level TEXT NOT NULL,
                risk_score REAL NOT NULL,
                detected_indicators TEXT,
                conversation_summary TEXT,
                booking_status TEXT DEFAULT 'pending',
                scheduled_datetime REAL,
                session_duration INTEGER DEFAULT 60,
                session_type TEXT DEFAULT 'emergency',
                location_preference TEXT,
                notes TEXT,
                created_ts REAL NOT NULL,
                updated_ts REAL NOT NULL,
                FOREIGN KEY (professional_id) REFERENCES professionals (id)
            )
        """)
        
        # Professional notifications
        conn.execute("""
            CREATE TABLE IF NOT EXISTS professional_notifications (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                professional_id INTEGER NOT NULL,
                booking_id TEXT NOT NULL,
                notification_type TEXT NOT NULL,
                title TEXT NOT NULL,
                message TEXT NOT NULL,
                is_read BOOLEAN DEFAULT 0,
                priority TEXT DEFAULT 'normal',
                created_ts REAL NOT NULL,
                FOREIGN KEY (professional_id) REFERENCES professionals (id)
            )
        """)
        
        # Session records
        conn.execute("""
            CREATE TABLE IF NOT EXISTS therapy_sessions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                booking_id TEXT NOT NULL,
                professional_id INTEGER NOT NULL,
                conv_id TEXT NOT NULL,
                session_start REAL,
                session_end REAL,
                session_notes TEXT,
                treatment_plan TEXT,
                follow_up_required BOOLEAN DEFAULT 0,
                follow_up_date REAL,
                session_rating INTEGER,
                session_feedback TEXT,
                created_ts REAL NOT NULL,
                FOREIGN KEY (professional_id) REFERENCES professionals (id)
            )
        """)
        
        # Admin users table
        conn.execute("""
            CREATE TABLE IF NOT EXISTS admin_users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT UNIQUE NOT NULL,
                password_hash TEXT NOT NULL,
                email TEXT NOT NULL,
                role TEXT DEFAULT 'admin',
                permissions TEXT,
                created_ts REAL NOT NULL
            )
        """)
        
        conn.commit()
    finally:
        conn.close()

def create_conversation(owner_key: str = None, conv_id: str = None, preview: str = "New chat"):
    if not conv_id:
        conv_id = str(uuid.uuid4())
    conn = sqlite3.connect(DB_FILE)
    try:
        conn.execute(
            "INSERT OR IGNORE INTO conversations (conv_id, owner_key, preview, ts, booking_prompt_shown) VALUES (?, ?, ?, ?, ?)",
            (conv_id, owner_key, preview, time.time(), 0),
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

# --- THERAPY BOOKING SYSTEM CLASSES ---
class RiskDetector:
    def __init__(self):
        # Risk indicators patterns
        self.critical_indicators = [
            r'\b(suicide|kill myself|end it all|not worth living)\b',
            r'\b(harm myself|hurt myself|self harm)\b',
            r'\b(overdose|poison|jump|hang)\b',
            r'\b(final goodbye|last time|never see)\b'
        ]
        
        self.high_risk_indicators = [
            r'\b(hopeless|worthless|burden|better off without)\b',
            r'\b(can\'t go on|can\'t take it|end this pain)\b',
            r'\b(no point|nothing matters|give up)\b',
            r'\b(severe depression|major depression)\b'
        ]
        
        self.medium_risk_indicators = [
            r'\b(depressed|sad|anxious|panic)\b',
            r'\b(can\'t sleep|insomnia|nightmares)\b',
            r'\b(stress|overwhelmed|burnout)\b',
            r'\b(isolation|lonely|withdraw)\b'
        ]
        
        # Specialized indicators for Rwanda context
        self.rwanda_specific_indicators = [
            r'\b(genocide|trauma|ptsd|flashback)\b',
            r'\b(orphan|widow|survivor)\b',
            r'\b(community violence|domestic violence)\b'
        ]

    def assess_risk(self, user_query: str, conversation_history: List[Dict]) -> Dict:
        """Comprehensive risk assessment"""
        risk_score = 0.0
        detected_indicators = []
        
        # Text-based pattern matching
        text_score, text_indicators = self._analyze_text_patterns(user_query)
        risk_score += text_score
        detected_indicators.extend(text_indicators)
        
        # AI-powered sentiment and context analysis
        ai_score, ai_indicators = self._ai_risk_analysis(user_query, conversation_history)
        risk_score += ai_score
        detected_indicators.extend(ai_indicators)
        
        # Conversation pattern analysis
        pattern_score, pattern_indicators = self._analyze_conversation_patterns(conversation_history)
        risk_score += pattern_score
        detected_indicators.extend(pattern_indicators)
        
        # Normalize score to 0-1 range
        risk_score = min(1.0, risk_score / 3.0)
        
        # Determine risk level
        if risk_score >= 0.8:
            risk_level = 'critical'
        elif risk_score >= 0.6:
            risk_level = 'high'
        elif risk_score >= 0.4:
            risk_level = 'medium'
        else:
            risk_level = 'low'
        
        return {
            'risk_score': risk_score,
            'risk_level': risk_level,
            'detected_indicators': list(set(detected_indicators)),
            'assessment_timestamp': time.time()
        }

    def _analyze_text_patterns(self, text: str) -> Tuple[float, List[str]]:
        """Analyze text for risk indicators"""
        score = 0.0
        indicators = []
        
        text_lower = text.lower()
        
        # Critical indicators (highest weight)
        for pattern in self.critical_indicators:
            if re.search(pattern, text_lower):
                score += 0.8
                indicators.append(f"critical_pattern: {pattern}")
        
        # High risk indicators
        for pattern in self.high_risk_indicators:
            if re.search(pattern, text_lower):
                score += 0.6
                indicators.append(f"high_risk_pattern: {pattern}")
        
        # Medium risk indicators
        for pattern in self.medium_risk_indicators:
            if re.search(pattern, text_lower):
                score += 0.3
                indicators.append(f"medium_risk_pattern: {pattern}")
        
        # Rwanda-specific indicators
        for pattern in self.rwanda_specific_indicators:
            if re.search(pattern, text_lower):
                score += 0.4
                indicators.append(f"rwanda_context: {pattern}")
        
        return score, indicators

    def _ai_risk_analysis(self, query: str, history: List[Dict]) -> Tuple[float, List[str]]:
        """Use AI to analyze emotional state and risk"""
        try:
            # Build context for AI analysis
            context = f"Analyze this mental health conversation for risk indicators:\n\n"
            context += f"Current message: {query}\n\n"
            
            if history:
                context += "Recent conversation:\n"
                for msg in history[-5:]:  # Last 5 messages
                    context += f"{msg['role']}: {msg['content']}\n"
            
            # AI prompt for risk assessment
            ai_prompt = f"""
            {context}
            
            Assess the risk level (0.0-1.0) and identify specific indicators.
            Consider: suicidal ideation, self-harm, severe depression, trauma, crisis situations.
            Respond in JSON format: {{"risk_score": 0.0-1.0, "indicators": ["indicator1", "indicator2"]}}
            """
            
            response = ollama.chat(model=CHAT_MODEL, messages=[
                {"role": "system", "content": "You are a mental health risk assessment AI. Analyze conversations for risk indicators and provide structured JSON responses."},
                {"role": "user", "content": ai_prompt}
            ])
            
            # Parse AI response
            ai_result = json.loads(response["message"]["content"])
            return ai_result.get("risk_score", 0.0), ai_result.get("indicators", [])
            
        except Exception as e:
            app.logger.error(f"AI risk analysis failed: {e}")
            return 0.0, []

    def _analyze_conversation_patterns(self, history: List[Dict]) -> Tuple[float, List[str]]:
        """Analyze conversation patterns for escalating risk"""
        if len(history) < 3:
            return 0.0, []
        
        score = 0.0
        indicators = []
        
        # Check for escalating negative sentiment
        recent_messages = history[-3:]
        negative_count = 0
        
        for msg in recent_messages:
            if msg['role'] == 'user':
                if any(word in msg['content'].lower() for word in ['worse', 'getting worse', 'can\'t handle', 'breaking down']):
                    negative_count += 1
        
        if negative_count >= 2:
            score += 0.5
            indicators.append("escalating_negative_sentiment")
        
        # Check for repeated crisis mentions
        crisis_mentions = 0
        for msg in history:
            if msg['role'] == 'user':
                if any(word in msg['content'].lower() for word in ['crisis', 'emergency', 'urgent', 'help now']):
                    crisis_mentions += 1
        
        if crisis_mentions >= 2:
            score += 0.4
            indicators.append("repeated_crisis_mentions")
        
        return score, indicators

class ProfessionalMatcher:
    def __init__(self):
        self.specialization_mapping = {
            'suicide': ['psychiatrist', 'psychologist'],
            'depression': ['psychiatrist', 'psychologist', 'counselor'],
            'anxiety': ['psychologist', 'counselor'],
            'ptsd': ['psychiatrist', 'psychologist', 'counselor'],
            'trauma': ['psychologist', 'counselor', 'social_worker'],
            'crisis': ['psychiatrist', 'psychologist'],
            'general': ['counselor', 'social_worker']
        }

    def find_best_professional(self, risk_assessment: Dict, user_location: Optional[Dict] = None) -> Optional[Dict]:
        """Find the most suitable professional based on risk and availability"""
        
        # Get detected indicators
        indicators = risk_assessment.get('detected_indicators', [])
        risk_level = risk_assessment.get('risk_level', 'low')
        
        # Determine required specializations
        required_specializations = self._get_required_specializations(indicators, risk_level)
        
        # Query available professionals
        available_professionals = self._get_available_professionals(required_specializations)
        
        if not available_professionals:
            return None
        
        # Score and rank professionals
        scored_professionals = []
        for prof in available_professionals:
            score = self._calculate_match_score(prof, indicators, risk_level, user_location)
            scored_professionals.append((prof, score))
        
        # Sort by score (highest first)
        scored_professionals.sort(key=lambda x: x[1], reverse=True)
        
        return scored_professionals[0][0] if scored_professionals else None

    def _get_required_specializations(self, indicators: List[str], risk_level: str) -> List[str]:
        """Determine required specializations based on risk indicators"""
        specializations = set()
        
        # Map indicators to specializations
        for indicator in indicators:
            if 'suicide' in indicator or 'critical' in indicator:
                specializations.update(['psychiatrist', 'psychologist'])
            elif 'depression' in indicator:
                specializations.update(['psychiatrist', 'psychologist', 'counselor'])
            elif 'anxiety' in indicator:
                specializations.update(['psychologist', 'counselor'])
            elif 'ptsd' in indicator or 'trauma' in indicator:
                specializations.update(['psychiatrist', 'psychologist', 'counselor'])
            elif 'crisis' in indicator:
                specializations.update(['psychiatrist', 'psychologist'])
        
        # For high/critical risk, prioritize psychiatrists
        if risk_level in ['high', 'critical']:
            specializations.add('psychiatrist')
        
        return list(specializations) if specializations else ['counselor']

    def _get_available_professionals(self, specializations: List[str]) -> List[Dict]:
        """Get available professionals matching specializations"""
        conn = sqlite3.connect(DB_FILE)
        try:
            placeholders = ','.join(['?' for _ in specializations])
            query = f"""
                SELECT * FROM professionals 
                WHERE specialization IN ({placeholders}) 
                AND is_active = 1
                ORDER BY experience_years DESC, created_ts ASC
            """
            cur = conn.execute(query, specializations)
            rows = cur.fetchall()
            
            # Convert to dict format
            professionals = []
            columns = [desc[0] for desc in cur.description]
            for row in rows:
                prof = dict(zip(columns, row))
                professionals.append(prof)
            
            return professionals
        finally:
            conn.close()

    def _calculate_match_score(self, professional: Dict, indicators: List[str], risk_level: str, user_location: Optional[Dict]) -> float:
        """Calculate matching score for a professional"""
        score = 0.0
        
        # Base score for specialization match
        score += 0.3
        
        # Experience bonus
        experience_years = professional.get('experience_years', 0)
        score += min(0.2, experience_years * 0.01)
        
        # Expertise areas match
        expertise_areas = json.loads(professional.get('expertise_areas', '[]'))
        matching_expertise = 0
        for indicator in indicators:
            for area in expertise_areas:
                if area.lower() in indicator.lower():
                    matching_expertise += 1
        
        if matching_expertise > 0:
            score += min(0.3, matching_expertise * 0.1)
        
        # Location proximity (if user location provided)
        if user_location and professional.get('location_latitude') and professional.get('location_longitude'):
            distance = self._calculate_distance(
                user_location['latitude'], user_location['longitude'],
                professional['location_latitude'], professional['location_longitude']
            )
            # Closer professionals get higher scores
            if distance < 10:  # Within 10km
                score += 0.2
            elif distance < 25:  # Within 25km
                score += 0.1
        
        # Availability bonus
        if self._is_professional_available_now(professional):
            score += 0.2
        
        return score

    def _calculate_distance(self, lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        """Calculate distance between two coordinates in kilometers"""
        R = 6371  # Earth's radius in kilometers
        
        dlat = math.radians(lat2 - lat1)
        dlon = math.radians(lon2 - lon1)
        
        a = (math.sin(dlat/2) * math.sin(dlat/2) + 
             math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * 
             math.sin(dlon/2) * math.sin(dlon/2))
        
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
        return R * c

    def _is_professional_available_now(self, professional: Dict) -> bool:
        """Check if professional is available for immediate booking"""
        # This would check their current schedule
        # For now, return True if they have less than max patients today
        return True  # Simplified for now

app = Flask(__name__)
CORS(app, origins=["http://localhost:8000", "http://127.0.0.1:8000"], 
     methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
     allow_headers=["Content-Type", "Authorization"])

# --- Public landing page routes (serve files from chatbot/ without affecting APIs) ---
_CHATBOT_STATIC_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'chatbot')

@app.route('/')
def landing_root():
    return send_from_directory(_CHATBOT_STATIC_DIR, 'landing.html')

@app.route('/landing')
@app.route('/landing.html')
def landing_page():
    return send_from_directory(_CHATBOT_STATIC_DIR, 'landing.html')

@app.route('/landing.css')
def landing_css():
    return send_from_directory(_CHATBOT_STATIC_DIR, 'landing.css')

@app.route('/landing.js')
def landing_js():
    return send_from_directory(_CHATBOT_STATIC_DIR, 'landing.js')

# --- Auth and dashboard static routes (serve files directly from chatbot/) ---
@app.route('/login')
def login_page():
    return send_from_directory(_CHATBOT_STATIC_DIR, 'login.html')

@app.route('/login.html')
def login_html():
    return send_from_directory(_CHATBOT_STATIC_DIR, 'login.html')

@app.route('/register')
def register_page():
    return send_from_directory(_CHATBOT_STATIC_DIR, 'register.html')

@app.route('/register.html')
def register_html():
    return send_from_directory(_CHATBOT_STATIC_DIR, 'register.html')

@app.route('/index.html')
def index_html():
    return send_from_directory(_CHATBOT_STATIC_DIR, 'index.html')

@app.route('/admin_dashboard.html')
def admin_dashboard_html():
    return send_from_directory(_CHATBOT_STATIC_DIR, 'admin_dashboard.html')

@app.route('/professional_dashboard.html')
def professional_dashboard_html():
    return send_from_directory(_CHATBOT_STATIC_DIR, 'professional_dashboard.html')

# Common JS/CSS assets referenced by the above pages
@app.route('/login.js')
def login_js_asset():
    return send_from_directory(_CHATBOT_STATIC_DIR, 'login.js')

@app.route('/register.js')
def register_js_asset():
    return send_from_directory(_CHATBOT_STATIC_DIR, 'register.js')

@app.route('/admin.js')
def admin_js_asset():
    return send_from_directory(_CHATBOT_STATIC_DIR, 'admin.js')

@app.route('/professional.js')
def professional_js_asset():
    return send_from_directory(_CHATBOT_STATIC_DIR, 'professional.js')

@app.route('/auth.css')
def auth_css_asset():
    return send_from_directory(_CHATBOT_STATIC_DIR, 'auth.css')

@app.route('/style.css')
def style_css_asset():
    return send_from_directory(_CHATBOT_STATIC_DIR, 'style.css')

@app.route('/app.js')
def app_js_asset():
    return send_from_directory(_CHATBOT_STATIC_DIR, 'app.js')

SYSTEM_PROMPT = """You are AIMHSA, a supportive mental-health companion for Rwanda.
- Be warm, brief, and evidence-informed. Respond in the user's language (English, Kinyarwanda, French, or Kiswahili).
- Do NOT diagnose or prescribe medications. Encourage professional care when appropriate.
- If the user mentions self-harm or immediate danger, express care and advise contacting local emergency services right away.
- Ground answers in the provided CONTEXT. If context is insufficient, say what is known and unknown, and offer general coping strategies.
- Keep responses brief except when details are required.
- Always respond in the same language the user wrote in.
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
            # Note: ollama.embeddings is for single prompt, for batch we need to call individually
            batch_embeddings = []
            for text in batch:
                resp = ollama.embeddings(model=EMBED_MODEL, prompt=text)
                batch_embeddings.append(resp["embedding"])
            all_embeddings.extend(batch_embeddings)
            
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
    
    # Force clear any loaded sentence-transformers model if not using it
    if not USE_SENT_TRANSFORMERS and SENT_MODEL is not None:
        app.logger.info("Clearing loaded sentence-transformers model")
        SENT_MODEL = None

    app.logger.info(f"USE_SENT_TRANSFORMERS: {USE_SENT_TRANSFORMERS}, SENT_EMBED_MODEL: {SENT_EMBED_MODEL}, EMBED_MODEL: {EMBED_MODEL}")
    app.logger.info(f"chunk_embeddings shape: {chunk_embeddings.shape}")

    # compute query embedding
    if USE_SENT_TRANSFORMERS:
        app.logger.info("Attempting to use sentence-transformers")
        # model name format: sentence-transformers/<model-id>
        model_id = SENT_EMBED_MODEL.split("/", 1)[1]
        try:
            if SENT_MODEL is None:
                app.logger.info(f"Loading SentenceTransformer model: {model_id}")
                from sentence_transformers import SentenceTransformer
                SENT_MODEL = SentenceTransformer(model_id)
            # encode returns (dim,) or (1,dim) depending on args; ensure numpy array (1,dim)
            q_emb = SENT_MODEL.encode(query, convert_to_numpy=True)
            if q_emb.ndim == 1:
                q_emb = q_emb.reshape(1, -1)
            q_emb = q_emb.astype(np.float32)
            app.logger.info("Successfully embedded query with sentence-transformers")
        except Exception as e:
            app.logger.error(f"Failed to use sentence-transformers: {e}")
            # fallback to ollama if local model not available
            try:
                app.logger.info(f"Falling back to ollama.embeddings with model: {EMBED_MODEL}")
                q_emb_resp = ollama.embeddings(model=EMBED_MODEL, prompt=query)
                q_emb = np.array([q_emb_resp["embedding"]], dtype=np.float32)
                app.logger.info("Successfully embedded query with ollama fallback")
            except Exception as e2:
                app.logger.error(f"Ollama fallback also failed: {e2}")
                raise
    else:
        app.logger.info(f"Using ollama embeddings API with model: {SENT_EMBED_MODEL}")
        # default: use ollama embeddings API
        try:
            q_emb_resp = ollama.embeddings(model=SENT_EMBED_MODEL, prompt=query)
            q_emb = np.array([q_emb_resp["embedding"]], dtype=np.float32)
            app.logger.info(f"Successfully embedded query with ollama, shape: {q_emb.shape}")
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

# --- THERAPY BOOKING SYSTEM HELPER FUNCTIONS ---
def create_automated_booking(conv_id: str, risk_assessment: Dict, user_account: str = None) -> Optional[Dict]:
    """Create automated booking for high-risk cases"""
    
    # Find best professional
    matcher = ProfessionalMatcher()
    professional = matcher.find_best_professional(risk_assessment)
    
    if not professional:
        app.logger.warning(f"No available professional found for high-risk case: {conv_id}")
        return None
    
    # Generate booking ID
    booking_id = str(uuid.uuid4())
    
    # Create conversation summary
    conversation_summary = generate_conversation_summary(conv_id)
    
    # Determine session timing (immediate for critical, within 24h for high)
    if risk_assessment['risk_level'] == 'critical':
        scheduled_datetime = time.time() + 3600  # 1 hour from now
        session_type = 'emergency'
    else:
        scheduled_datetime = time.time() + 86400  # 24 hours from now
        session_type = 'urgent'
    
    # Create booking record
    conn = sqlite3.connect(DB_FILE)
    try:
        conn.execute("""
            INSERT INTO automated_bookings 
            (booking_id, conv_id, user_account, user_ip, professional_id, risk_level, 
             risk_score, detected_indicators, conversation_summary, booking_status, 
             scheduled_datetime, session_type, created_ts, updated_ts)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            booking_id, conv_id, user_account, request.remote_addr,
            professional['id'], risk_assessment['risk_level'],
            risk_assessment['risk_score'], 
            json.dumps(risk_assessment['detected_indicators']),
            conversation_summary, 'pending', scheduled_datetime,
            session_type, time.time(), time.time()
        ))
        
        # Create notification for professional
        conn.execute("""
            INSERT INTO professional_notifications 
            (professional_id, booking_id, notification_type, title, message, priority, created_ts)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (
            professional['id'], booking_id, 'new_booking',
            f"URGENT: {risk_assessment['risk_level'].upper()} Risk Case",
            f"Automated booking created for {risk_assessment['risk_level']} risk case. "
            f"Risk indicators: {', '.join(risk_assessment['detected_indicators'][:3])}",
            'urgent' if risk_assessment['risk_level'] == 'critical' else 'high',
            time.time()
        ))
        
        conn.commit()
        
        return {
            'booking_id': booking_id,
            'professional_name': f"{professional['first_name']} {professional['last_name']}",
            'specialization': professional['specialization'],
            'scheduled_time': scheduled_datetime,
            'session_type': session_type,
            'risk_level': risk_assessment['risk_level']
        }
        
    finally:
        conn.close()

def generate_conversation_summary(conv_id: str) -> str:
    """Generate AI summary of conversation for professional"""
    try:
        # Load conversation history
        history = load_history(conv_id)
        
        if not history:
            return "No conversation history available."
        
        # Build context for AI summary
        context = "Recent conversation:\n"
        for msg in history[-10:]:  # Last 10 messages
            context += f"{msg['role']}: {msg['content']}\n"
        
        # AI prompt for summary
        ai_prompt = f"""
        {context}
        
        Create a brief professional summary of this mental health conversation.
        Focus on: main concerns, emotional state, risk factors, and key issues.
        Keep it concise and professional for a mental health professional.
        """
        
        response = ollama.chat(model=CHAT_MODEL, messages=[
            {"role": "system", "content": "You are a mental health AI assistant. Create professional summaries of conversations for mental health professionals."},
            {"role": "user", "content": ai_prompt}
        ])
        
        return response["message"]["content"]
        
    except Exception as e:
        app.logger.error(f"Failed to generate conversation summary: {e}")
        return "Summary generation failed."

def get_professional_by_id(professional_id: int) -> Optional[Dict]:
    """Get professional details by ID"""
    conn = sqlite3.connect(DB_FILE)
    try:
        cur = conn.execute("SELECT * FROM professionals WHERE id = ?", (professional_id,))
        row = cur.fetchone()
        
        if row:
            columns = [desc[0] for desc in cur.description]
            return dict(zip(columns, row))
        return None
    finally:
        conn.close()

@app.get("/healthz")
def healthz():
    return {"ok": True}

@app.get("/debug/login")
def debug_login():
    """Debug endpoint to check login status"""
    conn = sqlite3.connect(DB_FILE)
    try:
        cur = conn.execute("SELECT username FROM users LIMIT 5")
        users = [row[0] for row in cur.fetchall()]
        return {
            "ok": True,
            "users_available": users,
            "total_users": len(users),
            "message": "Login debug info"
        }
    finally:
        conn.close()

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
            "INSERT OR IGNORE INTO conversations (conv_id, owner_key, preview, ts, booking_prompt_shown) VALUES (?, ?, ?, ?, ?)",
            (conv_id, owner_key, preview, time.time(), 0),
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

    # Detect user language
    user_language = translation_service.detect_language(query)
    print(f"Detected user language: {user_language}")

    # Get conversation message count
    conn = sqlite3.connect(DB_FILE)
    try:
        message_count = conn.execute("""
            SELECT COUNT(*) FROM messages WHERE conv_id = ?
        """, (conv_id,)).fetchone()[0]
    finally:
        conn.close()
    
    # NEW: Risk Assessment Integration
    risk_detector = RiskDetector()
    risk_assessment = risk_detector.assess_risk(query, server_history)
    
    # Store risk assessment
    conn = sqlite3.connect(DB_FILE)
    try:
        conn.execute("""
            INSERT INTO risk_assessments 
            (conv_id, user_query, risk_score, risk_level, detected_indicators, assessment_timestamp)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (
            conv_id, 
            query, 
            risk_assessment['risk_score'],
            risk_assessment['risk_level'],
            json.dumps(risk_assessment['detected_indicators']),
            risk_assessment['assessment_timestamp']
        ))
        conn.commit()
    finally:
        conn.close()
    
    # NEW: Dual Booking Triggers
    booking_result = None
    ask_booking = None
    
    # Check if booking prompt was already shown for this conversation
    conn = sqlite3.connect(DB_FILE)
    try:
        booking_prompt_shown = conn.execute("""
            SELECT booking_prompt_shown FROM conversations WHERE conv_id = ?
        """, (conv_id,)).fetchone()
        booking_prompt_shown = booking_prompt_shown[0] if booking_prompt_shown else False
    finally:
        conn.close()
    
    # Trigger 1: After 5 messages - ask user if they want to book (only once per conversation)
    if message_count >= 5 and not booking_prompt_shown:
        ask_booking = {
            'message': 'I notice we\'ve been chatting for a while. Would you like me to connect you with a mental health professional for additional support?',
            'options': ['Yes, I\'d like to book a session', 'No, I\'m okay for now']
        }
        
        # Mark that booking prompt was shown
        conn = sqlite3.connect(DB_FILE)
        try:
            conn.execute("""
                UPDATE conversations SET booking_prompt_shown = 1 WHERE conv_id = ?
            """, (conv_id,))
            conn.commit()
        finally:
            conn.close()
    
    # Trigger 2: High risk assessment - automatically book
    if risk_assessment['risk_level'] in ['high', 'critical']:
        booking_result = create_automated_booking(conv_id, risk_assessment, data.get("account"))
        if booking_result:
            # Add emergency response to system prompt
            emergency_prompt = f"""
            URGENT: High-risk situation detected. Professional help has been automatically scheduled.
            Professional: {booking_result['professional_name']} ({booking_result['specialization']})
            Session Type: {booking_result['session_type']}
            Please provide immediate support and reassurance while professional help is arranged.
            """
            messages.append({"role": "system", "content": emergency_prompt})

    try:
        reply = ollama.chat(model=CHAT_MODEL, messages=messages)
        answer = reply["message"]["content"]
        
        # Translate response to user's language if not English
        if user_language != 'english':
            try:
                answer = translation_service.get_appropriate_response(answer, user_language)
                print(f"Translated response to {user_language}")
            except Exception as e:
                print(f"Translation error: {e}")
                # Keep original answer if translation fails
                
    except Exception as e:
        app.logger.error(f"Failed to get chat response with {CHAT_MODEL}: {e}")
        # Provide a fallback response when the model is not available
        fallback_en = f"I'm sorry, I'm having trouble accessing my AI model right now. However, I can still help you with mental health resources in Rwanda. Please contact the Mental Health Hotline at 105 or CARAES Ndera Hospital at +250 788 305 703 for immediate support. You can also try refreshing the page or contacting support if this issue persists."
        
        if user_language != 'english':
            try:
                answer = translation_service.get_appropriate_response(fallback_en, user_language)
            except Exception:
                answer = fallback_en
        else:
            answer = fallback_en

    # persist the current user RAW query (not the constructed user_prompt) and assistant reply
    save_message(conv_id, "user", query)
    save_message(conv_id, "assistant", answer)

    sources = [{"source": m["source"], "chunk": m["chunk"]} for (_, m) in top]
    resp = {"answer": answer, "sources": sources, "id": conv_id}
    
    # Add language detection info
    resp["detected_language"] = user_language
    resp["language_name"] = translation_service.get_language_name(user_language)
    
    # Add risk assessment and booking info to response
    resp["risk_assessment"] = {
        "risk_level": risk_assessment['risk_level'],
        "risk_score": risk_assessment['risk_score'],
        "detected_indicators": risk_assessment['detected_indicators'][:3]  # Show top 3 indicators
    }
    
    if ask_booking:
        resp["ask_booking"] = ask_booking
    
    if booking_result:
        resp["emergency_booking"] = booking_result
    
    # if newly created conv, client will need to store/use this id
    if new_conv:
        resp["new"] = True
    return jsonify(resp)

@app.post("/booking_response")
def booking_response():
    """
    Handle user response to booking question
    POST /booking_response
    Body: { "conversation_id": "...", "response": "yes"|"no", "account": "..." }
    """
    try:
        data = request.get_json(force=True)
    except Exception:
        return jsonify({"error": "Invalid JSON"}), 400
    
    conversation_id = data.get("conversation_id")
    response = data.get("response", "").lower()
    account = data.get("account")
    
    if not conversation_id or not response:
        return jsonify({"error": "conversation_id and response required"}), 400
    
    if response == "yes":
        # Create a booking for the user
        try:
            # Create a moderate risk assessment for booking
            risk_assessment = {
                'risk_level': 'medium',
                'risk_score': 0.5,
                'detected_indicators': ['user_requested_booking'],
                'assessment_timestamp': time.time()
            }
            
            booking_result = create_automated_booking(conversation_id, risk_assessment, account)
            if booking_result:
                return jsonify({
                    "ok": True,
                    "message": "Booking created successfully!",
                    "booking": booking_result
                })
            else:
                return jsonify({"error": "Failed to create booking"}), 500
        except Exception as e:
            app.logger.error(f"Failed to create booking: {e}")
            return jsonify({"error": "Failed to create booking"}), 500
    else:
        return jsonify({
            "ok": True,
            "message": "No problem! I'm here whenever you need support."
        })

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

# --- Forgot/Reset Password (Users) ---
@app.post("/forgot_password")
def forgot_password():
    """
    POST /forgot_password
    JSON: { "username": "..." }
    Creates a short-lived reset token. In this demo, the token is returned in the response.
    """
    try:
        data = request.get_json(force=True)
    except Exception:
        return jsonify({"error": "Invalid JSON"}), 400
    username = (data.get("username") or "").strip()
    if not username:
        return jsonify({"error": "username required"}), 400
    # verify user exists
    conn = sqlite3.connect(DB_FILE)
    try:
        cur = conn.execute("SELECT 1 FROM users WHERE username = ?", (username,))
        if not cur.fetchone():
            # do not reveal whether the user exists; still return ok
            return jsonify({"ok": True})
        token = uuid.uuid4().hex[:6].upper()  # 6-char code
        expires = time.time() + 15 * 60  # 15 minutes
        conn.execute(
            "INSERT INTO password_resets (username, token, expires_ts, used) VALUES (?, ?, ?, 0)",
            (username, token, expires),
        )
        conn.commit()
        # In production, email/SMS this code. For demo, return it.
        return jsonify({"ok": True, "token": token, "expires_in": 900})
    finally:
        conn.close()

@app.post("/reset_password")
def reset_password():
    """
    POST /reset_password
    JSON: { "username": "...", "token": "ABC123", "new_password": "..." }
    Validates token and updates the user's password.
    """
    try:
        data = request.get_json(force=True)
    except Exception:
        return jsonify({"error": "Invalid JSON"}), 400
    username = (data.get("username") or "").strip()
    token = (data.get("token") or "").strip().upper()
    new_password = (data.get("new_password") or "")
    if not username or not token or not new_password:
        return jsonify({"error": "username, token, and new_password required"}), 400
    if len(new_password) < 6:
        return jsonify({"error": "new_password too short"}), 400
    conn = sqlite3.connect(DB_FILE)
    try:
        cur = conn.execute(
            "SELECT id, expires_ts, used FROM password_resets WHERE username = ? AND token = ?",
            (username, token),
        )
        row = cur.fetchone()
        if not row:
            return jsonify({"error": "invalid token"}), 400
        reset_id, expires_ts, used = row
        if used:
            return jsonify({"error": "token already used"}), 400
        if time.time() > float(expires_ts):
            return jsonify({"error": "token expired"}), 400
        # Update password and mark token used
        pw_hash = generate_password_hash(new_password)
        conn.execute("UPDATE users SET password_hash = ? WHERE username = ?", (pw_hash, username))
        conn.execute("UPDATE password_resets SET used = 1 WHERE id = ?", (reset_id,))
        conn.commit()
        return jsonify({"ok": True})
    finally:
        conn.close()

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

# --- NEW API ENDPOINTS FOR THERAPY BOOKING SYSTEM ---

# Admin endpoints
@app.post("/admin/professionals")
def create_professional():
    """Create a new professional"""
    try:
        data = request.get_json(force=True)
    except Exception:
        return jsonify({"error": "Invalid JSON"}), 400
    
    required_fields = ['username', 'password', 'first_name', 'last_name', 'email', 'specialization', 'expertise_areas']
    for field in required_fields:
        if not data.get(field):
            return jsonify({"error": f"Missing required field: {field}"}), 400
    
    # Hash password
    password_hash = generate_password_hash(data['password'])
    
    # Prepare expertise areas as JSON
    expertise_areas = json.dumps(data.get('expertise_areas', []))
    languages = json.dumps(data.get('languages', ['english']))
    qualifications = json.dumps(data.get('qualifications', []))
    availability_schedule = json.dumps(data.get('availability_schedule', {}))
    
    conn = sqlite3.connect(DB_FILE)
    try:
        conn.execute("""
            INSERT INTO professionals 
            (username, password_hash, first_name, last_name, email, phone, license_number,
             specialization, expertise_areas, location_latitude, location_longitude, 
             location_address, district, availability_schedule, max_patients_per_day,
             consultation_fee, languages, qualifications, experience_years, bio,
             profile_picture, created_ts, updated_ts)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            data['username'], password_hash, data['first_name'], data['last_name'],
            data['email'], data.get('phone'), data.get('license_number'),
            data['specialization'], expertise_areas, data.get('location_latitude'),
            data.get('location_longitude'), data.get('location_address'), data.get('district'),
            availability_schedule, data.get('max_patients_per_day', 10),
            data.get('consultation_fee'), languages, qualifications,
            data.get('experience_years', 0), data.get('bio'), data.get('profile_picture'),
            time.time(), time.time()
        ))
        conn.commit()
        return jsonify({"ok": True, "message": "Professional created successfully"})
    except sqlite3.IntegrityError:
        return jsonify({"error": "Username already exists"}), 409
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    finally:
        conn.close()

@app.get("/admin/professionals")
def list_professionals():
    """List all professionals with filtering"""
    specialization = request.args.get('specialization')
    is_active = request.args.get('is_active', '1')
    
    conn = sqlite3.connect(DB_FILE)
    try:
        query = "SELECT * FROM professionals WHERE is_active = ?"
        params = [is_active]
        
        if specialization:
            query += " AND specialization = ?"
            params.append(specialization)
        
        query += " ORDER BY created_ts DESC"
        
        cur = conn.execute(query, params)
        rows = cur.fetchall()
        
        professionals = []
        columns = [desc[0] for desc in cur.description]
        for row in rows:
            prof = dict(zip(columns, row))
            # Parse JSON fields
            prof['expertise_areas'] = json.loads(prof.get('expertise_areas', '[]'))
            prof['languages'] = json.loads(prof.get('languages', '[]'))
            prof['qualifications'] = json.loads(prof.get('qualifications', '[]'))
            prof['availability_schedule'] = json.loads(prof.get('availability_schedule', '{}'))
            professionals.append(prof)
        
        return jsonify({"professionals": professionals})
    finally:
        conn.close()

@app.get("/admin/bookings")
def list_bookings():
    """List all automated bookings"""
    status = request.args.get('status')
    risk_level = request.args.get('risk_level')
    
    conn = sqlite3.connect(DB_FILE)
    try:
        query = """
            SELECT ab.*, p.first_name, p.last_name, p.specialization, p.email, p.phone
            FROM automated_bookings ab
            JOIN professionals p ON ab.professional_id = p.id
        """
        params = []
        conditions = []
        
        if status:
            conditions.append("ab.booking_status = ?")
            params.append(status)
        
        if risk_level:
            conditions.append("ab.risk_level = ?")
            params.append(risk_level)
        
        if conditions:
            query += " WHERE " + " AND ".join(conditions)
        
        query += " ORDER BY ab.created_ts DESC"
        
        cur = conn.execute(query, params)
        rows = cur.fetchall()
        
        bookings = []
        columns = [desc[0] for desc in cur.description]
        for row in rows:
            booking = dict(zip(columns, row))
            booking['detected_indicators'] = json.loads(booking.get('detected_indicators', '[]'))
            bookings.append(booking)
        
        return jsonify({"bookings": bookings})
    finally:
        conn.close()

@app.get("/admin/risk-assessments")
def list_risk_assessments():
    """List recent risk assessments"""
    limit = int(request.args.get('limit', 50))
    
    conn = sqlite3.connect(DB_FILE)
    try:
        cur = conn.execute("""
            SELECT * FROM risk_assessments 
            ORDER BY assessment_timestamp DESC 
            LIMIT ?
        """, (limit,))
        rows = cur.fetchall()
        
        assessments = []
        columns = [desc[0] for desc in cur.description]
        for row in rows:
            assessment = dict(zip(columns, row))
            assessment['detected_indicators'] = json.loads(assessment.get('detected_indicators', '[]'))
            assessments.append(assessment)
        
        return jsonify({"assessments": assessments})
    finally:
        conn.close()

# Professional endpoints
@app.post("/professional/login")
def professional_login():
    """Professional login"""
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
        cur = conn.execute("SELECT id, password_hash, first_name, last_name FROM professionals WHERE username = ? AND is_active = 1", (username,))
        row = cur.fetchone()
        if not row:
            return jsonify({"error": "invalid credentials"}), 401
        
        prof_id, stored_hash, first_name, last_name = row
        if not check_password_hash(stored_hash, password):
            return jsonify({"error": "invalid credentials"}), 401
        
        return jsonify({
            "ok": True, 
            "professional_id": prof_id,
            "name": f"{first_name} {last_name}",
            "username": username
        })
    finally:
        conn.close()

@app.post("/admin/login")
def admin_login():
    """Admin login"""
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
        cur = conn.execute("SELECT id, password_hash, email, role FROM admin_users WHERE username = ?", (username,))
        row = cur.fetchone()
        if not row:
            return jsonify({"error": "invalid credentials"}), 401
        
        admin_id, stored_hash, email, role = row
        if not check_password_hash(stored_hash, password):
            return jsonify({"error": "invalid credentials"}), 401
        
        return jsonify({
            "ok": True, 
            "admin_id": admin_id,
            "username": username,
            "email": email,
            "role": role
        })
    finally:
        conn.close()




@app.put("/professional/sessions/<booking_id>/status")
def update_session_status(booking_id):
    """Update session status (accept/decline)"""
    try:
        data = request.get_json(force=True)
    except Exception:
        return jsonify({"error": "Invalid JSON"}), 400
    
    new_status = data.get('status')
    professional_id = data.get('professional_id')
    
    if not new_status or not professional_id:
        return jsonify({"error": "status and professional_id required"}), 400
    
    if new_status not in ['confirmed', 'declined', 'completed']:
        return jsonify({"error": "Invalid status"}), 400
    
    conn = sqlite3.connect(DB_FILE)
    try:
        # Verify professional owns this booking
        cur = conn.execute("SELECT professional_id FROM automated_bookings WHERE booking_id = ?", (booking_id,))
        row = cur.fetchone()
        if not row or row[0] != professional_id:
            return jsonify({"error": "Unauthorized"}), 403
        
        # Update booking status
        conn.execute("UPDATE automated_bookings SET booking_status = ?, updated_ts = ? WHERE booking_id = ?", 
                    (new_status, time.time(), booking_id))
        
        # If confirmed, create session record
        if new_status == 'confirmed':
            conn.execute("""
                INSERT INTO therapy_sessions 
                (booking_id, professional_id, conv_id, created_ts)
                SELECT booking_id, professional_id, conv_id, ?
                FROM automated_bookings WHERE booking_id = ?
            """, (time.time(), booking_id))
        
        conn.commit()
        return jsonify({"ok": True})
    finally:
        conn.close()

@app.post("/professional/sessions/<booking_id>/notes")
def add_session_notes(booking_id):
    """Add notes to a session"""
    try:
        data = request.get_json(force=True)
    except Exception:
        return jsonify({"error": "Invalid JSON"}), 400
    
    notes = data.get('notes', '')
    professional_id = data.get('professional_id')
    
    if not professional_id:
        return jsonify({"error": "professional_id required"}), 400
    
    conn = sqlite3.connect(DB_FILE)
    try:
        # Verify professional owns this booking
        cur = conn.execute("SELECT professional_id FROM automated_bookings WHERE booking_id = ?", (booking_id,))
        row = cur.fetchone()
        if not row or row[0] != professional_id:
            return jsonify({"error": "Unauthorized"}), 403
        
        # Update session notes
        conn.execute("""
            UPDATE therapy_sessions 
            SET session_notes = ?, session_start = COALESCE(session_start, ?)
            WHERE booking_id = ?
        """, (notes, time.time(), booking_id))
        
        conn.commit()
        return jsonify({"ok": True})
    finally:
        conn.close()

# Real-time monitoring endpoints
@app.get("/monitor/risk-stats")
def get_risk_stats():
    """Get real-time risk statistics"""
    conn = sqlite3.connect(DB_FILE)
    try:
        # Get counts by risk level for last 24 hours
        cur = conn.execute("""
            SELECT risk_level, COUNT(*) as count
            FROM risk_assessments 
            WHERE assessment_timestamp > ?
            GROUP BY risk_level
        """, (time.time() - 86400,))
        rows = cur.fetchall()
        
        stats = {'critical': 0, 'high': 0, 'medium': 0, 'low': 0}
        for row in rows:
            stats[row[0]] = row[1]
        
        return jsonify({"risk_stats": stats})
    finally:
        conn.close()

@app.get("/monitor/recent-assessments")
def get_recent_assessments():
    """Get recent risk assessments"""
    limit = int(request.args.get('limit', 10))
    
    conn = sqlite3.connect(DB_FILE)
    try:
        cur = conn.execute("""
            SELECT ra.*, c.owner_key
            FROM risk_assessments ra
            LEFT JOIN conversations c ON ra.conv_id = c.conv_id
            ORDER BY ra.assessment_timestamp DESC 
            LIMIT ?
        """, (limit,))
        rows = cur.fetchall()
        
        assessments = []
        columns = [desc[0] for desc in cur.description]
        for row in rows:
            assessment = dict(zip(columns, row))
            assessment['detected_indicators'] = json.loads(assessment.get('detected_indicators', '[]'))
            assessments.append(assessment)
        
        return jsonify({"recent_assessments": assessments})
    finally:
        conn.close()

# Update run configuration to use port 5057 for API only
# --- PROFESSIONAL DASHBOARD API ENDPOINTS ---

@app.get("/professional/dashboard-stats")
def get_professional_dashboard_stats():
    """Get dashboard statistics for professional"""
    try:
        conn = sqlite3.connect(DB_FILE)
        
        # Get professional ID from session or request
        professional_id = request.headers.get('X-Professional-ID', '6')  # Default to Jean Ntwari for testing
        
        # Total sessions
        total_sessions = conn.execute("""
            SELECT COUNT(*) FROM automated_bookings WHERE professional_id = ?
        """, (professional_id,)).fetchone()[0]
        
        # Active users (users with recent sessions)
        active_users = conn.execute("""
            SELECT COUNT(DISTINCT user_account) FROM automated_bookings 
            WHERE professional_id = ? AND booking_status IN ('confirmed', 'completed')
        """, (professional_id,)).fetchone()[0]
        
        # High risk cases
        high_risk_cases = conn.execute("""
            SELECT COUNT(*) FROM automated_bookings 
            WHERE professional_id = ? AND risk_level IN ('high', 'critical')
        """, (professional_id,)).fetchone()[0]
        
        # Unread notifications
        unread_notifications = conn.execute("""
            SELECT COUNT(*) FROM professional_notifications 
            WHERE professional_id = ? AND is_read = 0
        """, (professional_id,)).fetchone()[0]
        
        conn.close()
        
        return jsonify({
            'totalSessions': total_sessions,
            'activeUsers': active_users,
            'highRiskCases': high_risk_cases,
            'unreadNotifications': unread_notifications
        })
        
    except Exception as e:
        app.logger.error(f"Error getting dashboard stats: {e}")
        return jsonify({'error': 'Failed to get dashboard stats'}), 500

@app.get("/professional/sessions")
def get_professional_sessions():
    """Get sessions for professional"""
    try:
        limit = request.args.get('limit', 50)
        professional_id = request.headers.get('X-Professional-ID', '6')  # Default to Jean Ntwari for testing
        
        conn = sqlite3.connect(DB_FILE)
        
        sessions = conn.execute("""
            SELECT ab.booking_id, ab.conv_id, ab.user_account, ab.user_ip, ab.risk_level, ab.risk_score,
                   ab.detected_indicators, ab.conversation_summary, ab.booking_status, 
                   ab.scheduled_datetime, ab.session_type, ab.created_ts, ab.updated_ts
            FROM automated_bookings ab
            WHERE ab.professional_id = ?
            ORDER BY ab.created_ts DESC
            LIMIT ?
        """, (professional_id, limit)).fetchall()
        
        conn.close()
        
        sessions_data = []
        for session in sessions:
            sessions_data.append({
                'bookingId': session[0],
                'convId': session[1],
                'userAccount': session[2],
                'userName': session[2],  # Use account as name for now
                'userIp': session[3],
                'riskLevel': session[4],
                'riskScore': session[5],
                'detectedIndicators': session[6],
                'conversationSummary': session[7],
                'bookingStatus': session[8],
                'scheduledDatetime': session[9],
                'sessionType': session[10],
                'createdTs': session[11],
                'updatedTs': session[12]
            })
        
        return jsonify(sessions_data)
        
    except Exception as e:
        app.logger.error(f"Error getting sessions: {e}")
        return jsonify({'error': 'Failed to get sessions'}), 500

@app.get("/professional/users")
def get_professional_users():
    """Get users for professional"""
    try:
        professional_id = request.headers.get('X-Professional-ID', '6')  # Default to Jean Ntwari for testing
        conn = sqlite3.connect(DB_FILE)
        
        # Get users who have sessions with this professional
        users = conn.execute("""
            SELECT DISTINCT ab.user_account, 
                   COUNT(*) as total_sessions,
                   MAX(ab.created_ts) as last_active,
                   MAX(ab.risk_level) as highest_risk_level,
                   COUNT(DISTINCT ab.conv_id) as total_conversations
            FROM automated_bookings ab
            WHERE ab.professional_id = ?
            GROUP BY ab.user_account
            ORDER BY last_active DESC
        """, (professional_id,)).fetchall()
        
        conn.close()
        
        users_data = []
        for user in users:
            users_data.append({
                'username': user[0],
                'email': f"{user[0]}@example.com",  # Placeholder
                'totalSessions': user[1],
                'lastActive': user[2],
                'highestRiskLevel': user[3],
                'totalConversations': user[4],
                'status': 'active'
            })
        
        return jsonify(users_data)
        
    except Exception as e:
        app.logger.error(f"Error getting users: {e}")
        return jsonify({'error': 'Failed to get users'}), 500

@app.get("/professional/notifications")
def get_professional_notifications():
    """Get notifications for professional"""
    try:
        limit = request.args.get('limit', 50)
        professional_id = request.headers.get('X-Professional-ID', '6')  # Default to Jean Ntwari for testing
        
        conn = sqlite3.connect(DB_FILE)
        
        notifications = conn.execute("""
            SELECT id, title, message, notification_type, is_read, created_at
            FROM professional_notifications 
            WHERE professional_id = ?
            ORDER BY created_at DESC
            LIMIT ?
        """, (professional_id, limit)).fetchall()
        
        conn.close()
        
        notifications_data = []
        for notification in notifications:
            notifications_data.append({
                'id': notification[0],
                'title': notification[1],
                'message': notification[2],
                'type': notification[3],
                'isRead': bool(notification[4]),
                'createdAt': notification[5]
            })
        
        return jsonify(notifications_data)
        
    except Exception as e:
        app.logger.error(f"Error getting notifications: {e}")
        return jsonify({'error': 'Failed to get notifications'}), 500

@app.get("/professional/sessions/<booking_id>")
def get_session_details(booking_id):
    """Get detailed session information"""
    try:
        conn = sqlite3.connect(DB_FILE)
        
        session = conn.execute("""
            SELECT booking_id, conv_id, user_account, user_ip, risk_level, risk_score,
                   detected_indicators, conversation_summary, booking_status, 
                   scheduled_datetime, session_type, created_ts, updated_ts
            FROM automated_bookings 
            WHERE booking_id = ?
        """, (booking_id,)).fetchone()
        
        conn.close()
        
        if not session:
            return jsonify({'error': 'Session not found'}), 404
        
        session_data = {
            'bookingId': session[0],
            'convId': session[1],
            'userAccount': session[2],
            'userName': session[2],  # Use account as name for now
            'userIp': session[3],
            'riskLevel': session[4],
            'riskScore': session[5],
            'detectedIndicators': session[6],
            'conversationSummary': session[7],
            'bookingStatus': session[8],
            'scheduledDatetime': session[9],
            'sessionType': session[10],
            'createdTs': session[11],
            'updatedTs': session[12]
        }
        
        return jsonify(session_data)
        
    except Exception as e:
        app.logger.error(f"Error getting session details: {e}")
        return jsonify({'error': 'Failed to get session details'}), 500

@app.get("/professional/users/<username>")
def get_user_profile(username):
    """Get detailed user profile"""
    try:
        conn = sqlite3.connect(DB_FILE)
        
        # Get user's sessions
        sessions = conn.execute("""
            SELECT booking_id, risk_level, risk_score, detected_indicators, 
                   scheduled_datetime, booking_status, session_type
            FROM automated_bookings 
            WHERE user_account = ?
            ORDER BY created_ts DESC
        """, (username,)).fetchall()
        
        # Get user's conversations
        conversations = conn.execute("""
            SELECT conv_id, preview, ts
            FROM conversations 
            WHERE owner_key = ?
            ORDER BY ts DESC
            LIMIT 10
        """, (username,)).fetchall()
        
        conn.close()
        
        # Calculate stats
        total_sessions = len(sessions)
        total_conversations = len(conversations)
        highest_risk_level = max([s[1] for s in sessions], default='low')
        last_active = max([s[4] for s in sessions], default=0) if sessions else 0
        
        # Build risk history
        risk_history = []
        for session in sessions[:10]:  # Last 10 sessions
            risk_history.append({
                'level': session[1],
                'score': session[2],
                'indicators': json.loads(session[3]) if session[3] else [],
                'timestamp': session[4]
            })
        
        user_profile = {
            'username': username,
            'email': f"{username}@example.com",  # Placeholder
            'totalSessions': total_sessions,
            'totalConversations': total_conversations,
            'highestRiskLevel': highest_risk_level,
            'lastActive': last_active,
            'recentConversations': [
                {
                    'title': conv[1] or 'Conversation',
                    'preview': conv[1] or 'No preview available',
                    'timestamp': conv[2]
                } for conv in conversations
            ],
            'riskHistory': risk_history
        }
        
        return jsonify(user_profile)
        
    except Exception as e:
        app.logger.error(f"Error getting user profile: {e}")
        return jsonify({'error': 'Failed to get user profile'}), 500

@app.post("/professional/sessions/<booking_id>/accept")
def accept_session(booking_id):
    """Accept a session"""
    try:
        conn = sqlite3.connect(DB_FILE)
        
        conn.execute("""
            UPDATE automated_bookings 
            SET booking_status = 'confirmed', updated_ts = ?
            WHERE booking_id = ?
        """, (time.time(), booking_id))
        
        conn.commit()
        conn.close()
        
        return jsonify({'success': True, 'message': 'Session accepted'})
        
    except Exception as e:
        app.logger.error(f"Error accepting session: {e}")
        return jsonify({'error': 'Failed to accept session'}), 500

@app.post("/professional/notifications/mark-all-read")
def mark_all_notifications_read():
    """Mark all notifications as read"""
    try:
        professional_id = request.headers.get('X-Professional-ID', '1')
        
        conn = sqlite3.connect(DB_FILE)
        
        conn.execute("""
            UPDATE professional_notifications 
            SET is_read = 1 
            WHERE professional_id = ?
        """, (professional_id,))
        
        conn.commit()
        conn.close()
        
        return jsonify({'success': True, 'message': 'All notifications marked as read'})
        
    except Exception as e:
        app.logger.error(f"Error marking notifications as read: {e}")
        return jsonify({'error': 'Failed to mark notifications as read'}), 500

@app.post("/professional/notifications/<notification_id>/read")
def mark_notification_read(notification_id):
    """Mark a specific notification as read"""
    try:
        conn = sqlite3.connect(DB_FILE)
        
        conn.execute("""
            UPDATE professional_notifications 
            SET is_read = 1 
            WHERE id = ?
        """, (notification_id,))
        
        conn.commit()
        conn.close()
        
        return jsonify({'success': True, 'message': 'Notification marked as read'})
        
    except Exception as e:
        app.logger.error(f"Error marking notification as read: {e}")
        return jsonify({'error': 'Failed to mark notification as read'}), 500

@app.post("/professional/reports/generate")
def generate_professional_report():
    """Generate comprehensive report for professional"""
    try:
        data = request.get_json()
        period = data.get('period', 30)
        professional_id = request.headers.get('X-Professional-ID', '1')
        
        conn = sqlite3.connect(DB_FILE)
        
        # Calculate date range
        end_date = time.time()
        start_date = end_date - (int(period) * 24 * 60 * 60)
        
        # Get session statistics
        sessions = conn.execute("""
            SELECT user_account, risk_level, booking_status, scheduled_datetime, session_type
            FROM automated_bookings 
            WHERE professional_id = ? AND created_ts >= ?
            ORDER BY created_ts DESC
        """, (professional_id, start_date)).fetchall()
        
        conn.close()
        
        # Calculate statistics
        total_sessions = len(sessions)
        unique_users = len(set(s[0] for s in sessions))
        high_risk_cases = len([s for s in sessions if s[1] in ['high', 'critical']])
        average_response_time = 15  # Placeholder - would need actual calculation
        
        # Build session breakdown
        session_breakdown = []
        for session in sessions[:20]:  # Last 20 sessions
            session_breakdown.append({
                'userName': session[0],
                'sessionType': session[4],
                'status': session[2],
                'date': session[3],
                'duration': 60,  # Placeholder
                'riskLevel': session[1]
            })
        
        report = {
            'totalSessions': total_sessions,
            'uniqueUsers': unique_users,
            'highRiskCases': high_risk_cases,
            'averageResponseTime': average_response_time,
            'sessionBreakdown': session_breakdown
        }
        
        return jsonify(report)
        
    except Exception as e:
        app.logger.error(f"Error generating report: {e}")
        return jsonify({'error': 'Failed to generate report'}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5057, debug=True)