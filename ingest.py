import os, uuid, json
from pathlib import Path
import ollama
from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from dotenv import load_dotenv

load_dotenv()

DATA_DIR = Path("data")
EMBED_FILE = Path("storage/embeddings.json")
EMBED_MODEL = os.getenv("EMBED_MODEL", "nomic-embed-text")

# --- Load or initialize embeddings ---
if EMBED_FILE.exists():
    with open(EMBED_FILE, "r", encoding="utf-8") as f:
        chunks_data = json.load(f)
else:
    chunks_data = []

# --- Helper functions ---
def load_text_from_file(path: Path) -> str:
    if path.suffix.lower() in [".txt", ".md"]:
        return path.read_text(encoding="utf-8", errors="ignore")
    if path.suffix.lower() == ".pdf":
        pdf = PdfReader(str(path))
        return "\n".join((page.extract_text() or "") for page in pdf.pages)
    return ""

def chunk_text(text: str):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=900, chunk_overlap=150,
        separators=["\n\n", "\n", " ", ""]
    )
    return splitter.split_text(text)

# --- Track existing sources ---
existing_files = {c["source"] for c in chunks_data}

new_chunks = []
for fp in DATA_DIR.glob("**/*"):
    if fp.suffix.lower() not in [".pdf", ".txt", ".md"]:
        continue
    if fp.name in existing_files:
        continue  # skip already processed files

    raw = load_text_from_file(fp)
    if not raw.strip():
        continue

    for idx, piece in enumerate(chunk_text(raw)):
        new_chunks.append({
            "id": str(uuid.uuid4()),
            "text": piece,
            "source": fp.name,
            "chunk": idx,
            "embedding": None  # to fill below
        })

# --- Generate embeddings with Ollama ---
if new_chunks:
    texts = [c["text"] for c in new_chunks]
    emb_res = ollama.embed(model=EMBED_MODEL, input=texts)
    embeddings = emb_res["embeddings"]

    for c, e in zip(new_chunks, embeddings):
        c["embedding"] = e

    chunks_data.extend(new_chunks)

    # Save updated embeddings
    EMBED_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(EMBED_FILE, "w", encoding="utf-8") as f:
        json.dump(chunks_data, f, ensure_ascii=False, indent=2)

    print(f"Added {len(new_chunks)} new chunks to {EMBED_FILE}")
else:
    print("No new documents found.")
