import os
import sys
import json
import re
import csv
import traceback
import difflib
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Tuple, Optional, Any
import time

try:
    from dotenv import load_dotenv  # type: ignore
except Exception:  # pragma: no cover
    load_dotenv = None

try:
    from google import genai  # type: ignore
except Exception:  # pragma: no cover
    genai = None

try:
    from groq import Groq  # type: ignore
except ImportError:  # pragma: no cover
    Groq = None

try:
    from bias_engine import calculate_bias_score, ENGINE_AVAILABLE as ENSEMBLE_ENGINE_AVAILABLE  # type: ignore
except Exception:  # pragma: no cover
    calculate_bias_score = None  # type: ignore
    ENSEMBLE_ENGINE_AVAILABLE = False

try:
    from implicit_bias_scorer import analyze_implicit_bias  # type: ignore
except Exception:  # pragma: no cover
    analyze_implicit_bias = None  # type: ignore

try:
    from llm_verification import verify_bias_async  # type: ignore
except Exception:  # pragma: no cover
    verify_bias_async = None  # type: ignore

try:  # pragma: no cover - optional heavy deps
    from sentence_transformers import SentenceTransformer, util, CrossEncoder  # type: ignore
except Exception as e:  # pragma: no cover
    SentenceTransformer = None  # type: ignore
    util = None  # type: ignore
    CrossEncoder = None  # type: ignore

try:
    import spacy  # type: ignore
    try:
        _spacy_nlp = spacy.load("en_core_web_md")
    except OSError:
        import spacy.cli  # type: ignore
        spacy.cli.download("en_core_web_md")
        _spacy_nlp = spacy.load("en_core_web_md")
except Exception:  # pragma: no cover
    spacy = None  # type: ignore
    _spacy_nlp = None

# --- NLTK WordNet for Tier 1 synonym expansion ---
try:
    import nltk  # type: ignore
    nltk.download('wordnet', quiet=True)
    nltk.download('omw-1.4', quiet=True)
    nltk.download('words', quiet=True)
    from nltk.corpus import wordnet  # type: ignore
    from nltk.corpus import words  # type: ignore
    _valid_english_words = set(w.lower() for w in words.words())
    print("NLTK WordNet and Words loaded successfully.")
except Exception as _nltk_err:
    nltk = None  # type: ignore
    wordnet = None  # type: ignore
    _valid_english_words = set()
    print(f"WARNING: NLTK load failed: {_nltk_err}")

# --- FAISS for Tier 2 semantic index ---
try:
    import faiss  # type: ignore
    import numpy as np  # type: ignore
    print("FAISS loaded successfully.")
except Exception as _faiss_err:
    faiss = None  # type: ignore
    np = None  # type: ignore
    print(f"WARNING: FAISS load failed: {_faiss_err}")

# --- Zero-Shot Context Evaluator for coded HR terms ---
_context_evaluator = None
try:
    from transformers import pipeline as hf_pipeline  # type: ignore
    _context_evaluator = hf_pipeline(
        "zero-shot-classification",
        model="typeform/distilbert-base-uncased-mnli"
    )
    print("Zero-Shot Context Evaluator loaded: typeform/distilbert-base-uncased-mnli")
except Exception as _zsc_err:
    print(f"WARNING: Zero-Shot Context Evaluator load failed: {_zsc_err}")
    _context_evaluator = None

# --- General System Spellchecker ---
try:
    from spellchecker import SpellChecker
    _spell = SpellChecker()
    print("pyspellchecker loaded successfully.")
except Exception as _spell_err:
    print(f"WARNING: pyspellchecker load failed: {_spell_err}")
    _spell = None

# --- Cross-Encoder model for Layer 2 semantic re-ranking ---
# Upgraded from stsb-TinyBERT-L-4 to a proper NLI model for much stronger
# bias/safe discrimination scores.
CROSS_ENCODER_MODEL_NAME = "cross-encoder/nli-deberta-v3-small"
_cross_encoder_model = None
try:
    if CrossEncoder is not None:
        _cross_encoder_model = CrossEncoder(CROSS_ENCODER_MODEL_NAME)
        print(f"Cross-Encoder loaded: {CROSS_ENCODER_MODEL_NAME}")
except Exception as _ce_err:
    print(f"WARNING: Primary Cross-Encoder ({CROSS_ENCODER_MODEL_NAME}) failed: {_ce_err}")
    # Fallback to distilroberta if DeBERTa has dependency issues
    try:
        CROSS_ENCODER_MODEL_NAME = "cross-encoder/nli-distilroberta-base"
        if CrossEncoder is not None:
            _cross_encoder_model = CrossEncoder(CROSS_ENCODER_MODEL_NAME)
            print(f"Cross-Encoder fallback loaded: {CROSS_ENCODER_MODEL_NAME}")
    except Exception as _ce_err2:
        print(f"WARNING: Fallback Cross-Encoder also failed: {_ce_err2}")
        _cross_encoder_model = None

try:
    import chromadb  # type: ignore
    _chroma_client = chromadb.EphemeralClient()
except Exception:
    chromadb = None
    _chroma_client = None

_bias_rules_collection = None
def _init_chroma_rules():
    global _bias_rules_collection
    if _chroma_client is None or _st_model is None:
        return
    if _bias_rules_collection is not None:
        return
    try:
        # Delete any stale collection (from Flask debug restart) to ensure cosine metric
        try:
            _chroma_client.delete_collection("bias_rules")  # type: ignore
        except Exception:
            pass
        _bias_rules_collection = _chroma_client.create_collection(  # type: ignore
            "bias_rules", metadata={"hnsw:space": "cosine"}
        )
        rules_path = os.path.join(os.path.dirname(__file__), "rules.json")
        if os.path.exists(rules_path):
            with open(rules_path, "r", encoding="utf-8") as f:
                rules = json.load(f)
            docs = []
            metadatas = []
            ids = []
            for i, rule in enumerate(rules):
                docs.append(rule["explanation"])
                metadatas.append(rule)
                ids.append(f"rule_{i}")
            
            embeddings = _st_model.encode(docs, convert_to_numpy=True).tolist()  # type: ignore
            _bias_rules_collection.add(
                embeddings=embeddings,
                documents=docs,
                metadatas=metadatas,
                ids=ids
            )
            print(f"ChromaDB rules initialized with {len(rules)} rules (cosine metric).")
        else:
            print(f"WARNING: rules.json not found at {rules_path}")
    except Exception as e:
        print(f"Failed to init chroma rules: {e}")
        import traceback
        traceback.print_exc()


# --- Interpreter / environment bootstrap ---
# Goal: make `python app.py` "just work".
# If a local virtualenv exists at `backend/venv` but the user runs the
# *system* Python, we transparently re-launch into the venv's python so that
# Flask and other deps are available without forcing a specific command.
#
# On Windows, os.execv does NOT truly replace the process (it uses _spawnv
# internally), which breaks terminal process tracking.  We use subprocess
# instead: spawn the venv python, inherit stdio, wait, and exit with its

import subprocess as _bootstrap_subprocess  # noqa: E402 — must be available before venv check

BASE_DIR = Path(__file__).resolve().parent
ROOT_VENV_PYTHON = BASE_DIR.parent / "venv" / "Scripts" / "python.exe"
BACKEND_VENV_PYTHON = BASE_DIR / "venv" / "Scripts" / "python.exe"

# --- DEBUG instrumentation (temporary) ---
# Writes NDJSON lines to project root debug-4f8e4d.log (no secrets).
# Only auto-reexec when launched as a script file (not `python -c`, not `python -m`),
# otherwise re-exec will lose the original command and can break tooling/imports.
argv0 = sys.argv[0] if sys.argv else ""
is_script_launch = bool(argv0) and argv0 not in ("-c", "-m") and argv0.lower().endswith(".py")

def _relaunch_in_venv(venv_python: Path) -> None:
    """Spawn *venv_python* running the same script, wait, and exit.

    Uses Popen (not call) so we can intercept KeyboardInterrupt and
    explicitly terminate the child.  Without this, Ctrl+C on Windows
    can kill the parent while leaving the child Flask server orphaned
    and holding port 5000.
    """
    print(
        f"\n[BOOTSTRAP] System Python detected — relaunching with venv interpreter:\n"
        f"  {venv_python}\n"
    )
    process = _bootstrap_subprocess.Popen([str(venv_python)] + sys.argv)
    try:
        process.wait()
    except KeyboardInterrupt:
        # Ctrl+C received — forward the termination to the child process
        process.terminate()
        process.wait()
    sys.exit(process.returncode)

# Enforce venv: if we are NOT inside a virtual environment and a venv exists,
# relaunch into it.
is_in_venv = sys.prefix != getattr(sys, "base_prefix", sys.prefix) or "VIRTUAL_ENV" in os.environ
if not is_in_venv and is_script_launch:
    if ROOT_VENV_PYTHON.exists():
        _relaunch_in_venv(ROOT_VENV_PYTHON)
    elif BACKEND_VENV_PYTHON.exists():
        _relaunch_in_venv(BACKEND_VENV_PYTHON)

try:
    import groq as _groq  # noqa: F401  # type: ignore
    from groq import Groq as _Groq  # noqa: F401  # type: ignore
except Exception as e:
    pass

# Load environment variables from `.env` if python-dotenv is installed.
if load_dotenv is not None:
    load_dotenv()


# --- Dependency boot checks ---
# This project is meant to run with Flask installed. If it is not available
# even after the bootstrap above, we emit a clear error and log evidence.
try:
    from flask import Flask, request, jsonify  # type: ignore
    from flask_cors import CORS  # type: ignore
except ModuleNotFoundError as e:
    if str(e).startswith("No module named 'flask'"):
        print(
            "\nERROR: Flask is not installed for this Python interpreter.\n"
            f"Python in use: {os.sys.executable}\n\n"
            "Fix (recommended):\n"
            "  1) cd C:\\BiasBuster\\backend\n"
            "  2) .\\venv\\Scripts\\python app.py\n\n"
            "If venv does not exist, create it and install deps:\n"
            "  python -m venv .\\venv\n"
            "  .\\venv\\Scripts\\python -m pip install -r requirements.txt\n"
        )
        raise
    raise

"""
BiasBuster backend (hackathon mode)

This backend intentionally runs as a purely local dictionary-based evaluator.
No LLM/ML providers are required for bias evaluation.
"""

app = Flask(__name__)
CORS(app)
# Accept both `/path` and `/path/` to avoid frontend/back-end slash mismatches.
app.url_map.strict_slashes = False

# -------------------------------
# User Feedback / Active Learning
# -------------------------------
FALSE_POSITIVES_CACHE = {
    "chairperson",
    "double-blind",
    "blind review",
    "own",
    "push",
    "results",
    "human",
    "humans",
}
BLACKLIST = set()

def _init_feedback_db():
    db_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "feedback.db"))
    try:
        conn = sqlite3.connect(db_path)
        cur = conn.cursor()
        cur.execute('''
            CREATE TABLE IF NOT EXISTS feedback (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                word TEXT NOT NULL,
                label INTEGER NOT NULL,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        cur.execute("SELECT word, label FROM feedback")
        rows = cur.fetchall()
        for word, label in rows:
            normalized = word.strip().lower()
            if label == 0:
                FALSE_POSITIVES_CACHE.add(normalized)
            elif label == 1:
                BLACKLIST.add(normalized)
        conn.commit()
        conn.close()
    except Exception as e:
        print(f"Failed to init feedback DB: {e}")

_init_feedback_db()

# -------------------------------
# Phrase-based allowlist (configurable)
# -------------------------------
# If a trigger term is found as part of one of these phrases in the user's text,
# we will NOT flag it as biased. Keep phrases lowercase; matching is case-insensitive.
IGNORED_PHRASES = [
    "lead vocalist",
    "lead singer",
    "lead the",
    "lead to",
]

# -------------------------------
# Semantic context guard configuration
# -------------------------------
SEMANTIC_MODEL_NAME = "all-MiniLM-L6-v2"

_st_model: Optional["SentenceTransformer"] = None
# Per-term semantic anchors: term -> (biased_embedding, safe_embedding)
_term_anchor_embeddings: dict[str, Any] | None = None
_canonical_term_embeddings = None  # cached tensor for semantic dedup of canonicals

# --- Health check (for System Status indicator) ---
@app.route('/api/health', methods=['GET'])
def health_check():
    return jsonify({"status": "active"}), 200

@app.route('/api/feedback', methods=['POST'])
def submit_feedback():
    data = request.get_json(silent=True) or {}
    word = (data.get("word") or "").strip().lower()
    label = data.get("label")

    if not word or label not in [0, 1]:
        return jsonify({"error": "Invalid payload"}), 400

    db_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "feedback.db"))
    try:
        conn = sqlite3.connect(db_path)
        cur = conn.cursor()
        cur.execute("INSERT INTO feedback (word, label) VALUES (?, ?)", (word, label))
        conn.commit()
        conn.close()
    except Exception as e:
        return jsonify({"error": str(e)}), 500

    if label == 0:
        FALSE_POSITIVES_CACHE.add(word)
        BLACKLIST.discard(word)
    elif label == 1:
        BLACKLIST.add(word)
        FALSE_POSITIVES_CACHE.discard(word)

    return jsonify({"success": True, "word": word, "label": label})

# -------------------------------
# BiasBuster: Bias evaluation
# -------------------------------
# This API MUST be:
# - Local-only evaluation (no LLM calls)
# - Driven by a CSV "model" (`bias_database.csv`)
# - Regex-based with strict word-boundary matching for the trigger terms
#
# Why CSV?
# - It provides deterministic, auditable detections.
#
# NOTE: The generation endpoints may still use LLMs (Gemini/Groq), but the evaluator never will.

def _bias_db_path() -> str:
    """
    Resolve the bias database CSV path.

    `backend/app.py` lives one directory below the project root where we store
    `bias_database.csv`, so we walk up one level.
    """
    return os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "bias_database.csv"))


def _bias_dictionary_path() -> str:
    """
    Resolve the bias dictionary path.

    In dictionary-only mode we prefer a `bias_dictionary.csv` in the project root.
    If it doesn't exist, fall back to the legacy `bias_database.csv`.
    """
    root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    preferred = os.path.join(root, "bias_dictionary.csv")
    fallback = os.path.join(root, "bias_database.csv")
    return preferred if os.path.exists(preferred) else fallback


def _debug_log(hypothesis_id: str, location: str, message: str, data: dict):
    pass


def _load_bias_database_rows():
    """
    Load `bias_database.csv` rows safely.

    Returns a list of dict rows with canonical keys:
    - Trigger_Word
    - Dimension
    - Affected_Group
    - Severity
    - Explanation
    """
    path = _bias_db_path()
    if not os.path.exists(path):
        raise FileNotFoundError(f"bias_database.csv not found at: {path}")

    rows = []
    # Use utf-8-sig to safely strip Excel's BOM (Byte Order Mark) that can
    # silently rename the first header (e.g. "\ufeffTrigger_Word") and break
    # DictReader key lookups.
    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)

        # Tolerate formatting variations in the header row:
        # - extra whitespace
        # - different casing (trigger_word vs Trigger_Word)
        # - accidental BOM already handled by utf-8-sig, but we still normalize.
        #
        # We normalize each input row into our canonical keys.
        def _normalize_row_keys(r):
            normalized = {}
            for k, v in (r or {}).items():
                key = (k or "").strip().lower()
                normalized[key] = v
            return normalized

        for row in reader:
            r = _normalize_row_keys(row)
            # Strict safeguard:
            # - Strip whitespace
            # - Skip empty triggers
            # - Skip triggers shorter than 3 characters (prevents stop-word noise)
            trigger_word = (r.get("trigger_word") or r.get("trigger word") or r.get("trigger") or "").strip()
            if (not trigger_word) or (len(trigger_word) < 3):
                continue

            rows.append(
                {
                    "Trigger_Word": trigger_word,
                    "Dimension": (r.get("dimension") or "").strip(),
                    "Affected_Group": (r.get("affected_group") or r.get("affected group") or "").strip(),
                    "Severity": (r.get("severity") or "").strip(),
                    "Explanation": (r.get("explanation") or "").strip(),
                }
            )
    return rows


def build_robust_regex(trigger_word: str) -> str:
    flexible = _build_flexible_term_pattern(trigger_word, pluralize_last=True)
    if not flexible:
        return ""

    # Strict word-boundary matching:
    # - Ensures we match standalone words like "man" but not "manager" or "human"
    # - With multi-word triggers, boundaries apply to the outer edges of the phrase
    return r"\b" + flexible + r"\b"


def _maybe_singularize_last_token(token: str) -> str:
    """
    If the CSV contains simple plurals (e.g., "rockstars", "guys"), normalize the
    last token to its singular root so we can add optional (s|es) safely.

    IMPORTANT: Do NOT singularize words that naturally end with "s" (e.g., "boss").
    """
    t = (token or "").strip()
    if len(t) < 4:
        return t

    lower = t.lower()
    # Avoid common non-plural endings that end with 's'
    if lower.endswith(("ss", "us", "is")):
        return t

    # Only strip a trailing 's' when the word isn't likely a natural-s ending.
    if lower.endswith("s"):
        return t[:-1]  # type: ignore
    return t


def _build_flexible_term_pattern(term: str, pluralize_last: bool) -> str:
    """
    Transform a dictionary term/phrase into a regex fragment:
    - Hyphen/space tolerant: spaces become `[-\\s]+`
    - Optional pluralization for the final token: `(?:s|es)?` (when appropriate)

    Returns a regex fragment WITHOUT outer word boundaries.
    """
    raw = (term or "").strip()
    if not raw:
        return ""

    tokens = [t for t in re.split(r"[-\s]+", raw) if t]
    if not tokens:
        return ""

    if pluralize_last:
        tokens[-1] = _maybe_singularize_last_token(tokens[-1])

    escaped = [re.escape(t) for t in tokens]

    if pluralize_last:
        last = tokens[-1]
        # Only pluralize alphabetic-ish tokens of reasonable length (avoids "to" -> "tos").
        if re.fullmatch(r"[A-Za-z]{3,}", last):
            escaped[-1] = escaped[-1] + r"(?:s|es)?"

    # Hyphen/space tolerance
    return r"[-\s]+".join(escaped)


def _compile_phrase_regex(phrase: str) -> re.Pattern | None:
    flexible = _build_flexible_term_pattern(phrase, pluralize_last=True)
    if not flexible:
        return None
    return re.compile(rf"\b{flexible}\b", re.IGNORECASE)


def _normalize_for_contains(text: str) -> str:
    """
    Normalize text for simple phrase containment checks:
    - lowercase
    - collapse hyphens/whitespace to single spaces
    - strip surrounding whitespace
    """
    t = (text or "").lower()
    t = re.sub(r"[-\s]+", " ", t).strip()
    return t


def _split_sentences(text: str) -> list[Tuple[str, int, int]]:
    """
    Sentence splitter using spaCy (or fallback to regex).
    Each item: (sentence_text, start_index, end_index).
    """
    raw = text or ""
    if not raw:
        return []

    sentences: list[Tuple[str, int, int]] = []
    if _spacy_nlp is not None:
        doc = _spacy_nlp(raw)
        for sent in doc.sents:
            s_text = sent.text.strip()
            if s_text:
                sentences.append((s_text, sent.start_char, sent.end_char))
    else:
        for m in re.finditer(r"[^\.!\?\n]+[\.!\?]?", raw):
            s = (m.group(0) or "").strip()
            if not s:
                continue
            start = m.start()
            end = m.end()
            sentences.append((s, start, end))
    return sentences



def _lazy_init_semantic_guard() -> bool:
    """
    Lazily load the sentence-transformers model and anchor embeddings used
    for context-dependent disambiguation (e.g., 'lead' in music vs management).
    """
    global _st_model, _term_anchor_embeddings


    if SentenceTransformer is None or util is None:
        return False

    if _st_model is None:
        try:
            _st_model = SentenceTransformer(SEMANTIC_MODEL_NAME)  # type: ignore
        except Exception as e:
            return False

    if _term_anchor_embeddings is not None:
        return True

    # Each entry is (biased_context, safe_context). We discard a candidate when
    # the sentence is closer to the SAFE context than the BIASED context.
    anchors: dict[str, tuple[str, str]] = {
        # Existing behavior for "lead": corporate vs music/art
        "lead": (
            "Manager, leadership, boss, corporate, supervise, team lead, manager of people.",
            "Music, band, singer, vocal, song, performance, lead singer, lead guitarist.",
        ),
        # New: dinosaur disambiguation (literal paleontology vs ageism metaphor)
        "dinosaur": (
            "Workplace, corporate, office, IT department, older employees, replace workers, outdated management, legacy systems, ageism, out of touch, firing staff.",
            "Paleontology, history, fossils, jurassic, extinct literal reptiles, museum, triassic, biology, lived millions of years ago, prehistoric.",
        ),
        # Fallback pair for other ambiguous terms (kept for backward compatibility)
        "default": (
            "Bias in workplace, corporate, hiring, discrimination, exclusionary language.",
            "Literal/historical/scientific context, education, neutral description, factual discussion.",
        ),
    }

    _term_anchor_embeddings = {}
    try:
        for term, (biased_txt, safe_txt) in anchors.items():
            emb = _st_model.encode([biased_txt, safe_txt], convert_to_tensor=True, normalize_embeddings=True)
            _term_anchor_embeddings[term] = (emb[0], emb[1])
    except Exception as e:
        _term_anchor_embeddings = None
        return False
    return True


def _lazy_encode_canonicals(canonical_terms: list[str]):
    """
    Cache embeddings for canonical terms we might return so we can semantically
    dedupe near-duplicate canonicals (beyond simple plural/hyphen normalization).
    """
    global _canonical_term_embeddings
    if _st_model is None:
        return None
    if not canonical_terms:
        return None
    # Encode on-demand; small list per request, but cache the last computed set.
    _canonical_term_embeddings = _st_model.encode(canonical_terms, convert_to_tensor=True, normalize_embeddings=True)
    return _canonical_term_embeddings


def _is_safe_context(term: str, sentence_emb) -> bool:
    """
    Returns True when the sentence is closer to the SAFE context than the BIASED context
    for the given term.
    """
    if _term_anchor_embeddings is None or util is None:
        return False
    assert _term_anchor_embeddings is not None and util is not None
    t = (term or "").strip().lower()
    biased_emb, safe_emb = _term_anchor_embeddings.get(t) or _term_anchor_embeddings.get("default")
    biased = float(util.cos_sim(sentence_emb, biased_emb).item())
    safe = float(util.cos_sim(sentence_emb, safe_emb).item())
    
    if t == "dinosaur":
        print(f"\n--- SCORE EVALUATION (dinosaur) ---")
        print(f"Safe Score:   {safe:.4f}")
        print(f"Biased Score: {biased:.4f}")
        print(f"Margin met?:  {safe > (biased + 0.05)}")
        print(f"-----------------------------------\n")
        
    return safe > (biased + 0.05)


def _canonical_key_from_trigger(trigger_word: str) -> str:
    """
    Normalize a trigger into a canonical key that is stable across simple plural
    variants, so "dinosaur"/"dinosaurs" both map to "dinosaur".
    """
    raw = (trigger_word or "").strip()
    if not raw:
        return ""
    tokens = [t for t in re.split(r"[-\s]+", raw) if t]
    if not tokens:
        return ""
    tokens[-1] = _strip_simple_plural(tokens[-1])
    return _normalize_for_contains(" ".join(tokens))


def _strip_simple_plural(token: str) -> str:
    """
    Strip simple plural endings from a token for canonical/guard keying:
    - dinosaurs -> dinosaur
    - rockstars -> rockstar
    - niches -> nich (not perfect; this is only used for guard key lookup)

    We keep it conservative to avoid breaking words that naturally end with s.
    """
    t = (token or "").strip()
    if len(t) < 4:
        return t

    lower = t.lower()
    # Avoid common non-plural endings
    if lower.endswith(("ss", "us", "is")):
        return t

    # Try stripping "es" first for very basic cases
    if lower.endswith("es") and len(t) >= 5:
        # Avoid stripping from words like "theses"/"diseases" etc. (still heuristic)
        if not lower.endswith(("ses", "xes", "zes", "ches", "shes")):  # type: ignore
            return t[:-2]

    # Fallback: strip trailing s
    if lower.endswith("s"):  # type: ignore
        return t[:-1]

    return t


def _canonical_key_from_match(matched_text: str) -> str:
    """
    Canonical key resolution based on the *matched surface form*.
    This ensures semantic guard lookups don't miss plurals like "dinosaurs".
    """
    raw = (matched_text or "").strip()
    if not raw:
        return ""
    tokens = [t for t in re.split(r"[-\s]+", raw) if t]
    if not tokens:
        return ""
    tokens[-1] = _strip_simple_plural(tokens[-1])
    return _normalize_for_contains(" ".join(tokens))


def _ignored_phrase_index(ignored_phrases: list[str]) -> dict[str, list[re.Pattern]]:
    """
    Build an index: trigger_word -> [compiled phrase regexes].

    We keep it lightweight and scalable:
    - Each phrase is compiled once.
    - Each phrase is associated with every contiguous token n-gram it contains
      (e.g., "digital native speaker" indexes:
        "digital", "native", "speaker", "digital native", "native speaker", "digital native speaker")
      so both single-word and multi-word triggers can be efficiently checked.
    """
    index: dict[str, list[re.Pattern]] = {}
    for phrase in ignored_phrases or []:
        rx = _compile_phrase_regex(phrase)
        if rx is None:
            continue
        words = [w for w in re.split(r"[-\s]+", (phrase or "").strip().lower()) if w]
        if not words:
            continue

        # Index all contiguous n-grams (1..N)
        n = len(words)
        for i in range(n):
            for j in range(i + 1, n + 1):  # type: ignore
                key = " ".join(words[i:j])
                index.setdefault(key, []).append(rx)
    return index


def _is_match_within_ignored_phrase(
    text: str,
    match_span: tuple[int, int],
    patterns: list[re.Pattern],
) -> bool:
    """
    Returns True if the match span is contained within any ignored-phrase match.
    """
    if not text or not patterns:
        return False
    ms, me = match_span
    if ms < 0 or me <= ms:
        return False
    for rx in patterns:
        for pm in rx.finditer(text):
            if pm.start() <= ms and pm.end() >= me:
                return True
    return False


def _extract_suggestion(explanation: str) -> str:
    """
    Extract a short suggestion from the CSV Explanation field.

    Expected formats in the CSV:
    - '... Try "expert".'
    - '... Try "team", "community", or "network".'
    - '... Try values fit.'
    """
    text = (explanation or "").strip()
    if not text:
        return ""

    # Prefer quoted alternatives after "Try".
    m = re.search(r"\btry\b\s+([\"“”'])(.+?)\1", text, flags=re.IGNORECASE)
    if m:
        return (m.group(2) or "").strip()

    # Fallback: capture a short unquoted suggestion after "Try".
    m2 = re.search(r"\btry\b\s+([^.\n;]+)", text, flags=re.IGNORECASE)
    if m2:
        # Keep first alternative chunk and strip conjunction clutter.
        candidate = (m2.group(1) or "").strip()
        candidate = re.split(r"\bor\b|\band\b|,", candidate, maxsplit=1, flags=re.IGNORECASE)[0].strip()
        return candidate

    return ""


def is_definitional_context(prompt_text: str, flagged_word: str) -> bool:
    """
    Lightweight false-positive filter for "innocent" questions about a term itself.

    Examples we want to treat as safe:
    - "What is a X?"
    - "What were X?"
    - "How old are X?"
    - "Define X"
    - "Meaning of X"
    - "Tell me about X"
    """
    lower_prompt = (prompt_text or "").lower()
    lower_word = (flagged_word or "").strip().lower()
    if not lower_prompt or not lower_word:
        return False

    # Allow flexible separators in the word (spaces/hyphens/newlines).
    word_pat = re.escape(lower_word).replace(r"\ ", r"[-\s]+").replace(r"\-", r"[-\s]+")

    safe_patterns = [
        re.compile(rf"\bwhat\s+(?:is|are|was|were)(?:\s+(?:a|an))?\s+{word_pat}\b", re.IGNORECASE),
        re.compile(rf"\bhow\s+old\s+(?:is|are|was|were)\s+{word_pat}\b", re.IGNORECASE),
        re.compile(rf"\bdefine\s+{word_pat}\b", re.IGNORECASE),
        re.compile(rf"\bmeaning\s+of\s+{word_pat}\b", re.IGNORECASE),
        re.compile(rf"\btell\s+me\s+about\s+{word_pat}\b", re.IGNORECASE),
    ]

    return any(p.search(lower_prompt) for p in safe_patterns)


def is_safe_context(full_text: str, trigger_word: str, dimension: str) -> bool:
    text = (full_text or "")
    # Normalize simple "held key" typos so context rules still match:
    # e.g., "perioddddd" -> "period"
    normalized_text = re.sub(r"(.)\1{2,}", r"\1", text)
    lower_text = normalized_text.lower()
    word = (trigger_word or "").strip()
    lower_word = word.lower()
    if not lower_text or not lower_word:
        return False

    # --- Rule 1: Definitional questions (Tightened to prevent false positives) ---
    if is_definitional_context(normalized_text, word):
        return True

    definitional_strict = re.compile(
        rf"\b(?:define|meaning of|what is a|what is an|history of)\s+{re.escape(lower_word)}\b",
        re.IGNORECASE,
    )
    if definitional_strict.search(lower_text):
        return True

    # --- Rule 2: Massive proximity-based safe-context rules ---
    escaped_word = re.escape(lower_word)

    # Historical / scientific contexts: geology, paleontology, archaeology, evolution, academic research.
    historical_and_scientific_markers = [
        "fossil",
        "fossils",
        "ancient",
        "prehistoric",
        "jurassic",
        "triassic",
        "cretaceous",
        "mesozoic",
        "paleozoic",
        "cenozoic",
        "archaeology",
        "archaeological",
        "archaeologist",
        "archaeologists",
        "paleontology",
        "paleontologist",
        "paleontologists",
        "geology",
        "geological",
        "geologist",
        "geologists",
        "geophysics",
        "stratigraphy",
        "strata",
        "rock layer",
        "rock layers",
        "sediment",
        "sedimentary",
        "igneous",
        "metamorphic",
        "tectonic",
        "tectonics",
        "plate tectonics",
        "continental drift",
        "fault line",
        "seafloor",
        "oceanic crust",
        "mantle",
        "crust",
        "core",
        "eruption",
        "volcano",
        "volcanic",
        "lava",
        "magma",
        "seismic",
        "earthquake",
        "aftershock",
        "glacier",
        "ice age",
        "holocene",
        "pleistocene",
        "neolithic",
        "bronze age",
        "iron age",
        "middle ages",
        "renaissance",
        "industrial revolution",
        "artifacts",
        "artifact",
        "relics",
        "relic",
        "excavation",
        "excavations",
        "dig site",
        "dig sites",
        "fieldwork",
        "museum",
        "museums",
        "exhibit",
        "exhibition",
        "curator",
        "specimen",
        "specimens",
        "femur",
        "skull",
        "skeleton",
        "skeletons",
        "fossilized",
        "timeline",
        "era",
        "epoch",
        "period",
        "time period",
        "age group",
        "geologic time",
        "geologic period",
        "evolution",
        "evolutionary",
        "species",
        "genus",
        "phylogeny",
        "ecosystem",
        "habitat",
        "inhabit",
        "inhabited",
        "inhabiting",
        "roamed",
        "range",
        "population",
        "speciation",
        "extinct",
        "extinction",
        "mass extinction",
        "climate",
        "climate change",
        "pollen record",
        "carbon dating",
        "radiocarbon",
        "laboratory",
        "research",
        "study",
        "studies",
        "paper",
        "journal",
        "academic",
        "university",
        "lecture",
        "professor",
        "students",
        "course",
        "curriculum",
    ]

    # Familial / anthropological contexts: everyday human relationships, ancestry, community, rituals.
    familial_and_anthropological_markers = [
        "family",
        "families",
        "household",
        "households",
        "home",
        "homes",
        "grandfather",
        "grandmother",
        "grandparents",
        "grandson",
        "granddaughter",
        "parent",
        "parents",
        "mother",
        "father",
        "mom",
        "dad",
        "sister",
        "brother",
        "siblings",
        "cousin",
        "cousins",
        "aunt",
        "uncle",
        "niece",
        "nephew",
        "children",
        "child",
        "kid",
        "kids",
        "offspring",
        "descendant",
        "descendants",
        "ancestor",
        "ancestors",
        "ancestry",
        "lineage",
        "heritage",
        "inheritance",
        "marriage",
        "married",
        "spouse",
        "husband",
        "wife",
        "partner",
        "partners",
        "couple",
        "couples",
        "community",
        "communities",
        "neighborhood",
        "neighbourhood",
        "village",
        "town",
        "tribe",
        "tribes",
        "clan",
        "clans",
        "indigenous",
        "native people",
        "native peoples",
        "first nations",
        "aboriginal",
        "ethnic group",
        "ethnic groups",
        "culture",
        "cultures",
        "cultural",
        "ritual",
        "rituals",
        "ceremony",
        "ceremonies",
        "festival",
        "festivals",
        "tradition",
        "traditions",
        "custom",
        "customs",
        "folklore",
        "myth",
        "myths",
        "legend",
        "legends",
        "storytelling",
        "settlement",
        "settlements",
        "camp",
        "encampment",
        "migration",
        "migrated",
        "migrating",
        "diaspora",
        "kinship",
        "household head",
        "caretaker",
        "guardian",
        "caregiver",
        "neighbor",
        "neighbour",
        "neighbors",
        "society",
        "civilization",
        "civilisation",
        "population",
        "demographic",
        "demographics",
    ]

    # Professional / technical contexts: jobs, tools, testing, neutral business or academic jargon.
    professional_and_technical_markers = [
        "degree",
        "bachelor",
        "masters",
        "doctorate",
        "phd",
        "university",
        "college",
        "campus",
        "course",
        "credits",
        "curriculum",
        "major",
        "minor",
        "laboratory",
        "lab",
        "experiment",
        "experiments",
        "study",
        "clinical",
        "trial",
        "double blind",
        "single blind",
        "blind test",
        "control group",
        "variable",
        "statistical",
        "dataset",
        "database",
        "server",
        "backend",
        "frontend",
        "full stack",
        "scrum",
        "sprint",
        "retrospective",
        "standup",
        "roadmap",
        "backlog",
        "ticket",
        "issue tracker",
        "repository",
        "version control",
        "webmaster",
        "administrator",
        "sysadmin",
        "network engineer",
        "mechanic",
        "carpenter",
        "plumber",
        "electrician",
        "technician",
        "operator",
        "manager",
        "director",
        "coordinator",
        "analyst",
        "consultant",
        "specialist",
        "architect",
        "designer",
        "developer",
        "programmer",
        "engineer",
        "scientist",
        "researcher",
        "accountant",
        "bookkeeper",
        "auditor",
        "lawyer",
        "attorney",
        "paralegal",
        "clerk",
        "pilot",
        "captain",
        "officer",
        "chief of staff",
        "chief executive officer",
        "chief operating officer",
        "chief financial officer",
        "cfo",
        "ceo",
        "coo",
        "cmo",
        "cto",
        "cio",
        "product manager",
        "project manager",
        "qa engineer",
        "quality assurance",
        "tester",
        "test plan",
        "suite",
        "test suite",
        "bedroom",
        "studio",
        "office",
        "workspace",
        "workshop",
        "factory",
        "plant",
        "warehouse",
        "clinic",
        "hospital",
        "ward",
        "labor and delivery",
        "intensive care",
        "operations",
        "logistics",
        "supply chain",
        "manufacturing",
        "assembly line",
        "maintenance",
        "support desk",
        "help desk",
        "customer service",
        "service desk",
    ]

    dim = (dimension or "").strip().lower()
    markers_to_check = []

    # Map specific flagged words to marker categories.
    if lower_word in ["dinosaur", "dinosaurs"]:
        markers_to_check = historical_and_scientific_markers + familial_and_anthropological_markers
    elif lower_word in ["tribe", "tribes", "native", "natives"]:
        markers_to_check = historical_and_scientific_markers + familial_and_anthropological_markers
    elif lower_word in ["grandfather", "grandfathers", "grandmother", "grandmothers", "grandfathered"]:
        markers_to_check = familial_and_anthropological_markers
    elif lower_word in ["master", "masters"]:
        markers_to_check = professional_and_technical_markers
    elif lower_word in ["blind", "crazy"]:
        markers_to_check = professional_and_technical_markers
    else:
        # If the bias dimension explicitly indicates historical / cultural axes, include more context.
        if any(key in dim for key in ("age", "history", "culture", "native", "ethnicity", "race")):
            markers_to_check = (
                historical_and_scientific_markers
                + familial_and_anthropological_markers
            )

    if markers_to_check:
        # Sort so multi-word phrases are matched before shorter single words.
        markers_to_check = sorted(set(markers_to_check), key=len, reverse=True)
        escaped_markers = [re.escape(m).replace(r"\ ", r"\s+") for m in markers_to_check]
        combined_markers_pattern = "|".join(escaped_markers)

        proximity_pattern = re.compile(
            rf"\b{escaped_word}\b(?:\W+\w+){{0,5}}\W+\b(?:{combined_markers_pattern})\b"
            rf"|\b(?:{combined_markers_pattern})\b(?:\W+\w+){{0,5}}\W+\b{escaped_word}\b",
            re.IGNORECASE,
        )

        if proximity_pattern.search(lower_text):
            return True

    # --- Rule 3: Professional Titles (e.g., Chief Executive Officer) ---
    if lower_word == "chief":
        title_patterns = [
            re.compile(
                r"\bchief\s+(?:executive|operating|financial|technology|information|medical|marketing|revenue|officer|of\s+staff)\b",
                re.IGNORECASE,
            ),
            re.compile(r"\b(?:commander\s+in|police|fire)\s+chief\b", re.IGNORECASE),
        ]
        if any(p.search(lower_text) for p in title_patterns):
            return True

    # --- Rule 4: General Familial, Literal, & Anthropological Contexts ---
    # Prevents false positives for everyday statements and literal historical usage
    if lower_word in [
        "grandfather",
        "grandfathers",
        "grandfathered",
        "grandmother",
        "master",
        "native",
        "blind",
        "crazy",
        "tribe",
        "tribes",
    ]:
        literal_markers = [
            "child",
            "children",
            "kid",
            "kids",
            "family",
            "parent",
            "carpenter",
            "house",
            "home",
            "loved",
            "passed away",
            "dog",
            "cat",
            "friend",
            "human",
            "ancient",
            "century",
            "history",
            "first appear",
            "people",
            "group",
        ]
        escaped_word = re.escape(lower_word)
        for marker in literal_markers:
            # Handle multi-word markers safely
            escaped_marker = re.escape(marker).replace(r"\ ", r"\s+")

            # Check within 4 words to catch phrases like "ancient human tribes"
            pattern = re.compile(
                rf"\b{escaped_word}\b(?:\W+\w+){{0,4}}\W+\b{escaped_marker}\b|\b{escaped_marker}\b(?:\W+\w+){{0,4}}\W+\b{escaped_word}\b",
                re.IGNORECASE,
            )
            if pattern.search(lower_text):
                return True

    return False



def resolve_canonical_root(matched_text: str, valid_keys: set) -> str:
    """
    Finds the dictionary root of a matched text by checking exact match,
    then stripping common plural suffixes.
    """
    lower_text = (matched_text or "").strip().lower()
    
    if lower_text in valid_keys:
        return lower_text
        
    if lower_text.endswith("s"):
        candidate = lower_text[:-1]  # type: ignore[index]
        if candidate in valid_keys:
            return candidate
            
    if lower_text.endswith("es"):
        candidate = lower_text[:-2]  # type: ignore[index]
        if candidate in valid_keys:
            return candidate
            
    # Fallback to simple plural stripping for downstream payload consistency
    if lower_text.endswith("s") and not lower_text.endswith("ss"):
        return lower_text[:-1]  # type: ignore[index]
        
    return lower_text

def _is_corrective_sentence(sentence: str) -> bool:
    """
    Heuristic filter: returns True if the sentence appears to be
    educational, corrective, or explanatory rather than exhibiting bias.
    
    Catches patterns like:
      - "To rewrite this text objectively..."
      - "Instead of 'dinosaurs', use 'experienced professionals'"
      - "Replaced ageist language with inclusive terms"
      - "Calling employees 'dinosaurs' is ageist and exclusionary"
      - "The term 'confined to a wheelchair' is considered ableist"
    """
    lower = sentence.lower()

    # ── Corrective / rewriting action phrases ──
    corrective_prefixes = [
        "to rewrite",
        "to rephrase",
        "to make this more inclusive",
        "to remove bias",
        "to address",
        "to fix",
        "to correct",
        "to avoid",
        "rewritten version",
        "here is a revised",
        "here is the revised",
        "here's a revised",
        "revised version",
        "reframed",
        "rewrote",
        "we rewrote",
    ]
    for prefix in corrective_prefixes:
        if prefix in lower:
            return True

    # ── Chain of Thought (CoT) conversational reasoning markers ──
    cot_markers = [
        "okay, the user wants",
        "ok, the user wants",
        "first, they mention",
        "wait, the user used",
        "let me start by",
        "i need to",
        "i should check",
        "the user is asking",
        "thinking process:",
        "here's a thinking process"
    ]
    for marker in cot_markers:
        if lower.startswith(marker) or marker in lower:
            return True

    # ── Substitution / replacement language ──
    substitution_markers = [
        "instead of",
        "replaced with",
        "replace with",
        "replaced by",
        "replaced ",
        "substituted",
        "changed to",
        "changed from",
        "swapped with",
        "use the term",
        "consider using",
        "a better alternative",
        "more inclusive alternative",
        "more inclusive term",
        "more inclusive language",
        "more respectful",
    ]
    for marker in substitution_markers:
        if marker in lower:
            return True

    # ── Explanatory / educational phrases ──
    # Patterns like "X is ageist", "is considered ableist", "is exclusionary"
    explanatory_markers = [
        "is ageist",
        "is sexist",
        "is racist",
        "is ableist",
        "is exclusionary",
        "is discriminatory",
        "is biased",
        "is offensive",
        "is problematic",
        "can be exclusionary",
        "can be discriminatory",
        "can be ageist",
        "can be sexist",
        "can be racist",
        "can be ableist",
        "is considered",
        "is a stereotype",
        "is stereotypical",
        "perpetuates",
        "reinforces",
        "implies that",
        "suggests that",
        "can be seen as",
        "may come across as",
        "could be perceived as",
        "is discouraged",
        "excludes",
    ]
    for marker in explanatory_markers:
        if marker in lower:
            return True

    # ── Meta-discussion about bias types ──
    meta_markers = [
        "age-based stereotype",
        "gender stereotype",
        "racial stereotype",
        "ableist language",
        "gendered language",
        "coded language",
        "microaggression",
        "unconscious bias",
        "implicit bias",
        "inclusive language",
        "inclusive writing",
        "bias detection",
        "bias-free",
        "bias free",
        "ableist",
        "age-inclusive",
        "discussing",
    ]
    for marker in meta_markers:
        if marker in lower:
            return True

    # ── Markdown-formatted correction headers (AI output) ──
    # e.g. "**From \"Dinosaurs\" to \"Legacy skill sets\"**"
    if "from \"" in lower and "to \"" in lower:
        return True
    if 'from "' in lower and 'to "' in lower:
        return True

    # ── Educational / historical analysis heuristic ──
    # If the sentence is clearly academic or analytical in nature,
    # suppress bias flags (e.g. "Analyze the history of Jim Crow laws").
    _EDU_ANALYSIS_KEYWORDS = {
        "analyze", "analyse", "analysis", "history", "historical",
        "impact", "laws", "segregation", "essay", "examine",
        "evaluate", "discuss", "describe", "explain", "outline",
        "summarize", "summarise", "overview", "review", "critique",
        "compare", "contrast", "assess", "investigate", "explore",
        "academic", "scholarly", "research", "study", "thesis",
        "dissertation", "lecture", "curriculum", "textbook",
        "civil rights", "human rights", "legislation", "amendment",
        "reconstruction", "abolition", "emancipation",
    }
    _EDU_ANALYSIS_LABELS = [
        "historical analysis", "educational prompt", "academic discussion",
        "analyze the history", "impact of", "history of",
        "write an essay", "discuss the", "examine the",
    ]
    edu_keyword_hits = sum(1 for kw in _EDU_ANALYSIS_KEYWORDS if kw in lower)
    edu_label_hits = sum(1 for lbl in _EDU_ANALYSIS_LABELS if lbl in lower)
    if edu_keyword_hits >= 2 or edu_label_hits >= 1:
        return True

    return False


# ---------------------------------------------------------------------------
# THREE-TIERED OOV DETECTION ENSEMBLE
# ---------------------------------------------------------------------------

# ── Tier 1: NLTK WordNet Synonym Expansion ──
def _expand_with_wordnet(db_rows):
    """
    Loop through CSV trigger words, fetch WordNet synonyms, and return a
    dict mapping each valid synonym → its parent CSV row.
    """
    if wordnet is None:
        return {}

    expansions = {}
    stop_words = {
        "this", "that", "with", "from", "your", "what", "have", "they",
        "will", "just", "like", "some", "very", "much", "more", "than",
        "most", "only", "women", "older", "young", "black", "white", "asian",
        # Generic Word Clamp for Long-Form Evaluation
        "delay", "delays", "delayed", "parent", "parents", "use", "using", "used",
        "caring", "respect", "strong", "leadership", "initiative", "dominant",
        "support", "supporting", "supported", "perform", "performing", "agile",
        "master", "slave", "whitelist", "blacklist", "blind",
        # Technical / database architecture terms
        "primary", "replica", "secondary", "principal", "primitive",
    }

    for row in db_rows:
        trigger = str(row.get("Trigger_Word") or "").strip().lower().rstrip("*")
        if not trigger or " " in trigger or len(trigger) < 4:
            continue
        if trigger in stop_words:
            continue

        try:
            for syn in wordnet.synsets(trigger):
                for lemma in syn.lemmas():
                    name = lemma.name().replace("_", " ").lower()
                    # Only accept single-word, sufficiently long synonyms
                    if " " in name or len(name) < 4:
                        continue
                    if name == trigger:
                        continue
                    if name in stop_words:
                        continue
                    if name not in expansions:
                        expansions[name] = row
        except Exception:
            continue

    print(f"Tier 1 (WordNet): Expanded lexicon with {len(expansions)} synonym(s).")
    return expansions


# ── Tier 2: FAISS Semantic Index ──
_faiss_index = None
_faiss_words = []      # index → word string
_faiss_rows = []       # index → CSV row dict
_faiss_ready = False


def _init_faiss_index(db_rows):
    """
    Build a FAISS inner-product (cosine) index from all single-word CSV
    trigger embeddings using the existing SentenceTransformer model.
    Called once at startup.
    """
    global _faiss_index, _faiss_words, _faiss_rows, _faiss_ready

    if faiss is None or np is None:
        print("WARNING: FAISS not available — Tier 2 disabled.")
        return

    # Reuse the already-loaded bi-encoder
    try:
        model = globals().get("_st_model")
        if model is None:
            print("WARNING: SentenceTransformer model not loaded — Tier 2 disabled.")
            return
    except Exception:
        return

    words = []
    rows = []
    stop_words = {
        "this", "that", "with", "from", "your", "what", "have", "they",
        "will", "just", "like", "some", "very", "much", "more", "than",
        "most", "only", "women", "older", "young", "black", "white", "asian",
    }

    for row in db_rows:
        trigger = str(row.get("Trigger_Word") or "").strip().lower().rstrip("*")
        if not trigger or len(trigger) < 4 or trigger in stop_words:
            continue
        words.append(trigger)
        rows.append(row)

    if not words:
        print("WARNING: No words to index — Tier 2 disabled.")
        return

    try:
        embeddings = model.encode(words, convert_to_numpy=True, normalize_embeddings=True)
        dim = embeddings.shape[1]
        index = faiss.IndexFlatIP(dim)  # inner product on normalized = cosine
        index.add(embeddings.astype(np.float32))

        _faiss_index = index
        _faiss_words = words
        _faiss_rows = rows
        _faiss_ready = True
        print(f"Tier 2 (FAISS): Indexed {len(words)} bias terms ({dim}-dim).")
    except Exception as e:
        print(f"WARNING: FAISS index build failed: {e}")


# ── Tunable parameter: FAISS cosine-similarity threshold ──
FAISS_THRESHOLD = float(os.environ.get("FAISS_THRESHOLD", "0.85"))


def _faiss_semantic_check(token, threshold=FAISS_THRESHOLD):
    """
    Embed the input token and query the FAISS index for the nearest
    neighbor. Returns (matched_word, row, score) if similarity > threshold,
    else None.
    """
    if not _faiss_ready or _faiss_index is None:
        return None

    try:
        model = globals().get("_st_model")
        if model is None:
            return None

        embedding = model.encode([token], convert_to_numpy=True, normalize_embeddings=True)
        scores, indices = _faiss_index.search(embedding.astype(np.float32), 1)

        if len(scores) > 0 and len(scores[0]) > 0:
            score = float(scores[0][0])
            idx = int(indices[0][0])
            if score > threshold and 0 <= idx < len(_faiss_words):
                return (_faiss_words[idx], _faiss_rows[idx], score)
    except Exception:
        pass

    return None


# ── Tier 3: spaCy Vector Validation ──
def _spacy_vector_validate(token, bias_word):
    """
    Use spaCy word vectors (en_core_web_md) to validate semantic
    similarity between a candidate token and the matched bias word.
    Returns the similarity score (float), or 0.0 if unavailable.
    """
    if _spacy_nlp is None:
        return 0.0

    try:
        token_doc = _spacy_nlp(token)
        bias_doc = _spacy_nlp(bias_word)

        # Guard against zero-vectors (OOV in spaCy)
        if not token_doc.has_vector or not bias_doc.has_vector:
            return 0.0

        return float(token_doc.similarity(bias_doc))
    except Exception:
        return 0.0


def _find_bias_issues_explicit(text: str, db_rows: list[dict], is_ai_response: bool = False) -> dict:
    """
    Legacy explicit exact-match engine for the Hybrid Architecture.
    Used exclusively for the user prompt to ensure exact words are highlighted.
    """
    from typing import Any
    issues: dict[str, dict[str, Any]] = {}
    text = str(text or "")
    if not text or not db_rows:
        return issues

    # ── Tier 1: Merge WordNet synonyms into db_rows for exact-match ──
    # We do this once natively so it benefits from the robust regex builder
    # and plurals management without short-circuiting downstream.
    synthetic_rows = []
    wordnet_expansions = _expand_with_wordnet(db_rows)
    for syn_word, syn_row in wordnet_expansions.items():
        synth = syn_row.copy()
        synth["Trigger_Word"] = syn_word
        synthetic_rows.append(synth)
    
    combined_rows = db_rows + synthetic_rows

    # Re-build the ignored phrases index to avoid triggering on safe compound words
    all_ignored = []
    for row in combined_rows:
        ignored_raw = str(row.get("ignored_phrases") or "")
        if ignored_raw:
            all_ignored.extend([p.strip() for p in ignored_raw.split("|") if p.strip()])
    ignored_idx = _ignored_phrase_index(all_ignored)

    for row in combined_rows:
        trigger_word = str(row.get("Trigger_Word") or row.get("trigger_word") or "").strip()
        if not trigger_word:
            continue

        raw_trigger = trigger_word
        if raw_trigger.endswith("*"):
            pluralize_last = False
            raw_trigger = raw_trigger[:-1]  # type: ignore
        else:
            pluralize_last = True

        pattern_str = _build_flexible_term_pattern(raw_trigger, pluralize_last)
        if not pattern_str:
            continue
            
        pattern_str = r"\b" + pattern_str + r"\b"
            
        try:
            rx = re.compile(pattern_str, re.IGNORECASE)
        except Exception:
            continue

        for m in rx.finditer(text):
            match_text = m.group(0)
            start, end = m.span()
            
            canonical_trigger = _canonical_key_from_trigger(trigger_word)
            if canonical_trigger in ignored_idx:
                if _is_match_within_ignored_phrase(text, (start, end), ignored_idx[canonical_trigger]):  # type: ignore
                    continue
                    
            concept = canonical_trigger or trigger_word
            
            entry = issues.get(concept)
            if entry is None:
                entry = {
                    "biased_word": match_text,
                    "canonical_key": concept,
                    "dimension": str(row.get("Dimension") or row.get("dimension") or ""),
                    "category": str(row.get("Dimension") or row.get("dimension") or ""),
                    "severity": str(row.get("Severity") or row.get("severity") or ""),
                    "affected_group": str(row.get("Affected_Group") or row.get("affected_group") or ""),
                    "suggestion": _extract_suggestion(str(row.get("Explanation") or row.get("explanation") or "")),
                    "matched_sentence": match_text,
                    "type": "bias",
                    "matches": []
                }
                issues[concept] = entry

            entry["matches"].append({  # type: ignore
                "text": match_text,
                "start": start,
                "end": end
            })

    # ── Fix 1 (Simplified): Pass clean text through FAISS ──
    # Since typo correction now happens upstream, we only need to run
    # the FAISS Cross-Encoder checks on the clean text tokens that
    # are not explicit regex matches.
    valid_bias_words = {}
    stop_words = {"this", "that", "with", "from", "your", "what", "have", "they", "will", "just", "like", "some", "very", "much", "more", "than", "most", "only", "women", "older", "young", "black", "white", "asian"}

    for row in combined_rows:
        trigger = str(row.get("Trigger_Word") or row.get("trigger_word") or "").strip().lower()
        if not trigger:
            continue
        raw = trigger.rstrip("*")

        if " " not in raw and len(raw) >= 3 and raw not in stop_words:
            if raw not in valid_bias_words:
                valid_bias_words[raw] = row

        for w in raw.split():
            clean_w = "".join(c for c in w if c.isalnum())
            if len(clean_w) >= 3 and clean_w not in stop_words:
                if clean_w not in valid_bias_words:
                    valid_bias_words[clean_w] = row

    target_words = list(valid_bias_words.keys())

    for token_m in re.finditer(r"\b[a-zA-Z]{3,}\b", text):
        token = str(token_m.group(0))
        lower_token = token.lower()

        # Normalization for things like 'highlyyy' -> 'highly' just in case
        bias_token = re.sub(r'(.)\1{2,}', r'\1', lower_token)

        # Skip tokens that are already exact matches
        if bias_token in target_words:
            continue

        # ══════════════════════════════════════════════════════════
        # TRACK 2: Bias check via FAISS/spaCy (uses normalized bias_token)
        # ══════════════════════════════════════════════════════════
        faiss_result = _faiss_semantic_check(bias_token)
        if faiss_result:
            matched_word, matched_row, faiss_score = faiss_result
            # Tier 3: spaCy vector validation as secondary confirmation
            spacy_score = _spacy_vector_validate(bias_token, matched_word)
            if spacy_score > 0.55:
                # Both engines agree — inject as standard bias issue

                # 1. Find the full database row for the matched FAISS word
                matched_db_row = next((r for r in db_rows if r.get("Trigger_Word", "").lower() == matched_word.lower()), {})

                # 2. Extract concept via canonical key with strict fallback
                final_concept = _canonical_key_from_trigger(matched_word) or matched_word

                # 3. Build payload — STRICT mirror of the Tier 1 Exact Match schema
                original_raw_token = str(token_m.group(0))
                start, end = token_m.span()
                if final_concept not in issues:
                    issues[final_concept] = {
                        "biased_word": original_raw_token,
                        "canonical_key": final_concept,
                        "dimension": str(matched_db_row.get("Dimension") or matched_db_row.get("dimension") or ""),
                        "category": str(matched_db_row.get("Dimension") or matched_db_row.get("dimension") or ""),
                        "severity": str(matched_db_row.get("Severity") or matched_db_row.get("severity") or ""),
                        "affected_group": str(matched_db_row.get("Affected_Group") or matched_db_row.get("affected_group") or ""),
                        "suggestion": _extract_suggestion(str(matched_db_row.get("Explanation") or matched_db_row.get("explanation") or "")),
                        "matched_sentence": original_raw_token,
                        "type": "bias",
                        "bypass_l2": True,
                        "matches": []
                    }

                issues[final_concept]["matches"].append({
                    "text": original_raw_token,
                    "start": start,
                    "end": end
                })

                print(f"[DEBUG FAISS HYDRATED] Token: {original_raw_token}, Concept: {final_concept}, Start: {start}, End: {end}")

    # ── Fix 2: Full-Sentence RAG Fallback ──
    # If the token-level extraction yields 0 issues, run the full raw text
    # through the ChromaDB Semantic RAG engine to catch non-noun-anchored microaggressions.
    # ONLY perform this forceful bypass on user prompts, NOT AI responses, to avoid defeating L2 on long texts.
    if not issues and not is_ai_response:
        rag_issues = _find_bias_issues(text, db_rows)
        for k, v in rag_issues.items():
            issues[k] = v
            issues[k]["bypass_l2"] = True
            print(f"[DEBUG FULL-SENTENCE RAG] Matched via Semantic Search: {k}")

    return issues


def _find_bias_issues(text: str, db_rows=None):
    from typing import Any
    issues: dict[str, dict[str, Any]] = {}
    text = text or ""
    if not text:
        return issues
        
    _lazy_init_semantic_guard() # Ensure _st_model is loaded
    _init_chroma_rules()
    
    if _bias_rules_collection is None or _st_model is None:
        return issues
        
    sentences = _split_sentences(text)
    if not sentences:
        return issues
        
    query_texts = [s[0] for s in sentences]
    
    try:
        embeddings = _st_model.encode(query_texts, convert_to_numpy=True).tolist()  # type: ignore
        results = _bias_rules_collection.query(  # type: ignore
            query_embeddings=embeddings,
            n_results=1
        )
    except Exception as e:
        print(f"Chroma query failed: {e}")
        return issues

    threshold = 0.75  # Cosine distance: balanced to catch microaggressions without false positives

    for i, (s_text, s_start, s_end) in enumerate(sentences):
        if not results["distances"] or not results["distances"][i]:
            continue
            
        distance = results["distances"][i][0]
        if distance < threshold:
            # ── Context blindness filter ──
            # Skip sentences that are educational/corrective rather than biased
            if _is_corrective_sentence(s_text):
                continue

            # ── Educational Firewall (full-text) ──
            # Even if this single sentence matched RAG semantically,
            # This catches cases where the user's prompt is academic
            # but a single sentence fragment matches a bias rule.
            if _is_corrective_sentence(text):
                continue

            metadata = results["metadatas"][i][0]
            concept = metadata["concept"]
            
            # --- Extract exact literal word from the text for UI highlighting ---
            # RAG uses semantic matching, but the UI needs the exact string. We intersect
            # the literal sentence words with the rule's explanation text to find the trigger.
            s_words = [w for w in re.findall(r'\b[a-zA-Z-]+\b', s_text) if len(w) > 3]
            expl_text = (str(metadata.get("explanation", "")) + " " + concept).lower()
            overlap_words = [w for w in s_words if w.lower() in expl_text]
            extracted_word = overlap_words[0] if overlap_words else concept
            
            entry = issues.get(concept)
            if entry is None:
                entry = {
                    "biased_word": extracted_word,
                    "canonical_key": concept,
                    "dimension": metadata["dimension"],
                    "category": metadata["dimension"],
                    "severity": metadata["severity"],
                    "affected_group": metadata["dimension"],
                    "suggestion": metadata["explanation"],
                    "matched_sentence": s_text,
                    "matches": []
                }
                issues[concept] = entry

            entry["matches"].append({  # type: ignore
                "text": s_text,
                "start": s_start,
                "end": s_end
            })

    return issues



def _levenshtein_distance(a: str, b: str) -> int:
    """
    Simple Levenshtein distance for short words.
    Used to distinguish genuine typos from loosely related words
    (e.g. "honestly" vs "honey").
    """
    if a == b:
        return 0
    la, lb = len(a), len(b)
    if la == 0:
        return lb
    if lb == 0:
        return la

    dp = list(range(lb + 1))
    for i in range(1, la + 1):  # type: ignore
        prev = dp[0]  # type: ignore
        dp[0] = i
        for j in range(1, lb + 1):  # type: ignore
            tmp = dp[j]  # type: ignore
            cost = 0 if a[i - 1] == b[j - 1] else 1
            dp[j] = min(  # type: ignore[index]
                dp[j] + 1,      # deletion  # type: ignore[operator]
                dp[j - 1] + 1,  # insertion
                prev + cost,    # substitution
            )
            prev = tmp
    return dp[lb]


def _find_typos(text: str, db_rows):
    """
    Fuzzy-match user words against known trigger words to detect likely typos.

    Returns a list of dicts:
    - {"typo": actual_typed_word, "suggestion": matched_trigger_word}
    """
    raw_text = text or ""
    if not raw_text:
        return []

    # Tokenize into simple word tokens from the ORIGINAL text so we preserve casing
    # for display and replacement, but use lowercase for matching.
    original_words = re.findall(r"\b\w+\b", raw_text)
    if not original_words:
        return []

    # Map lowercase -> list of original-cased variants seen in the text.
    lower_to_originals = {}
    for w in original_words:
        lw = w.lower()
        if lw not in lower_to_originals:
            lower_to_originals[lw] = []
        lower_to_originals[lw].append(w)

    # Collect canonical trigger words from the database.
    trigger_words = []
    for row in db_rows or []:
        tw = (row.get("Trigger_Word") or "").strip().lower()
        if len(tw) >= 3:
            trigger_words.append(tw)

    trigger_words = list(set(trigger_words))
    if not trigger_words:
        return []

    typos = []
    seen_typos = set()

    for lower_word, originals in lower_to_originals.items():
        # Universal Acronym Bypass for Spellcheck
        if originals[0].isupper():
            continue

        # Length guard: avoid noisy matches like "time" -> "tribe"
        if len(lower_word) < 5:
            continue

        # THE TYPO CLAMP: Do not fuzzy match mathematically valid English words
        # (e.g. prevents 'pants' from turning into 'pansy')
        if lower_word in _valid_english_words:
            continue

        # Slightly stricter cutoff to prevent unrelated matches like "honestly" -> "honey"
        # while still catching close typos like "dinasour" and "dinosaursss".
        matches = difflib.get_close_matches(lower_word, trigger_words, n=1, cutoff=0.80)
        if not matches:
            continue

        best = matches[0]
        # If the lowercase version is a valid trigger, do NOT flag capitalization
        # differences (e.g., "Honestly" vs "honestly") as typos.
        if best == lower_word:
            continue  # Exact match, not a typo.

        # Substring/plural/phrase guard with support for "stretched" typos:
        # - Don't flag simple plurals like "dinosaurs" -> "dinosaur"
        # - Do flag repeated-letter typos like "dinosaursss" -> "dinosaur"
        if (lower_word in best) or (best in lower_word):
            # Simple plural/suffix cases are not typos.
            if lower_word in (best + "s", best + "es"):
                continue

            # Allow "stretched" typos where the user repeats the trailing character(s).
            if lower_word.startswith(best) and len(lower_word) > len(best):
                extra = lower_word[len(best) :]
                if len(extra) >= 2 and len(set(extra)) == 1:
                    pass  # accept as typo (e.g., dinosaursss)
                else:
                    continue
            else:
                continue

        # Distance guard:
        # Use Levenshtein distance to avoid loose matches like "honestly" -> "honey"
        # while still catching close typos such as "dinasour" / "dinosaursss".
        # Allow up to 3 single-character edits for words of reasonable length.
        if _levenshtein_distance(lower_word, best) > 3:
            continue

        key = (lower_word, best)
        if key in seen_typos:
            continue

        seen_typos.add(key)
        # Frontend expects `original` (user-typed) and `suggested` (correction).
        # Use the first observed original-cased variant for display, and
        # match the suggestion's casing to it (e.g., "Honnestly" -> "Honestly").
        original_word = originals[0]
        if original_word.isupper():
            suggested = best.upper()
        elif original_word.istitle():
            suggested = best.capitalize()
        else:
            suggested = best

        typos.append({"original": original_word, "suggested": suggested})

    return typos


# ---------------------------------------------------------------------------
# Semantic false-positive filter for implicit bias (agentic/communal) terms
# ---------------------------------------------------------------------------
# Soft-skill words that frequently appear in non-workplace contexts
_SOFT_SKILL_TERMS = {
    "respect", "respectful", "understanding", "authoritative", "authority",
    "vision", "visionary", "strong", "strength", "confident", "confidence",
    "support", "supportive", "collaborative", "bold", "decisive",
    "initiative", "strategic", "analytical", "objective", "logical",
    "compassionate", "empathetic", "empathy", "patient", "patience",
    "thoughtful", "considerate", "kind", "kindness", "warm", "warmth",
    "encouraging", "diplomatic", "caring", "care", "helpful", "sensitive",
    "inclusive", "independent", "drive", "driven", "excellence", "excel",
    "direct", "leadership", "leader", "encourage", "encouragement",
    "communicate", "communication", "community", "emotional",
}

_CODED_HR_TERMS = {
    "rockstar", "ninja", "guru", "digital native", "energetic"
}

# Nouns that indicate the adjective is describing a thing/concept, not a person
_SAFE_OBJECT_LEMMAS = {
    "resource", "resources", "data", "source", "evidence", "study",
    "research", "report", "analysis", "paper", "article", "book",
    "document", "framework", "method", "approach", "strategy", "plan",
    "system", "tool", "technique", "guide", "reference", "material",
    "information", "knowledge", "insight", "finding", "result",
    "solution", "model", "theory", "concept", "idea", "principle",
    "value", "standard", "policy", "practice", "process", "response",
    "answer", "argument", "statement", "claim", "position", "view",
    "perspective", "opinion", "voice", "tone", "message", "effort",
    "work", "action", "step", "measure", "bond", "connection", "link",
    "force", "field", "acid", "base", "current", "signal", "beam",
    "structure", "foundation", "pillar", "support", "presence",
}

# Nouns that indicate the adjective IS describing a person → keep the flag
_PERSON_LEMMAS = {
    "leader", "manager", "employee", "worker", "candidate", "applicant",
    "he", "she", "they", "person", "individual", "woman", "man",
    "boss", "executive", "director", "officer", "supervisor", "colleague",
    "teammate", "hire", "recruit", "staff", "team",
}

# ---------------------------------------------------------------------------
# Hardcoded human nouns for the Cascading Verification Pipeline.
# If the flagged adjective modifies one of these, the flag is KEPT.
# If it modifies anything NOT in this set, the flag is DROPPED (false positive).
# ---------------------------------------------------------------------------
HUMAN_NOUNS = {
    # Core person words
    "person", "people", "individual", "human", "man", "woman", "child", "kid",
    # Pronouns (spaCy sometimes resolves the subject to a pronoun)
    "he", "she", "they", "someone", "anyone", "everyone", "nobody",
    # Relative pronouns (in "developer who is nurturing", nsubj of 'is' is 'who')
    "who", "whom", "whose", "that",
    # Workplace / professional roles
    "developer", "engineer", "programmer", "coder", "designer", "architect",
    "candidate", "applicant", "employee", "worker", "intern", "contractor",
    "manager", "boss", "supervisor", "director", "executive", "officer",
    "leader", "colleague", "coworker", "teammate", "peer",
    "hire", "recruit", "staff", "team", "crew",
    "user", "admin", "administrator", "client", "customer",
    "student", "teacher", "professor", "instructor", "mentor", "trainee",
    "player", "athlete", "coach",
    "doctor", "nurse", "patient", "therapist",
    "analyst", "consultant", "specialist", "technician", "operator",
    "volunteer", "member", "participant", "attendee", "speaker",
    "founder", "owner", "partner", "stakeholder", "investor",
    "thinker", "chairman", "chairwoman", "chairperson",
    "stewardess", "steward", "policeman", "fireman", "mailman",
}

# Context keywords indicating non-workplace discourse
_NON_WORKPLACE_CONTEXT_KEYWORDS = {
    # Ethics / human rights
    "racism", "anti-racism", "antiracism", "discrimination", "equity",
    "justice", "human rights", "civil rights", "equality", "dignity",
    "oppression", "marginalized", "privilege", "systemic",
    "prejudice", "races", "race",
    # Academic / scientific
    "research", "study", "scientific", "academic", "peer-reviewed",
    "data", "evidence", "findings", "methodology", "hypothesis",
    "experiment", "observation", "theory", "literature", "journal",
    "publication", "citation", "bibliography",
    "science", "knowledge", "resources", "resource",
    # Medical / clinical
    "clinical", "diagnosis", "treatment", "patient", "therapy",
    "symptoms", "medical", "healthcare", "health",
    # Historical / cultural
    "historical", "history", "ancient", "cultural", "tradition",
    "civilization", "heritage", "anthropology",
    "museum", "fossil", "triassic", "jurassic", "paleontology",
    # AI / technology
    "ai", "prompt", "users", "equally",
    # General / everyday
    "family", "friend", "child", "parent", "community", "society",
    "education", "school", "university", "classroom",
}


def _extract_sentence_for_word(text: str, word: str) -> str:
    """Extract the sentence containing the given word using spaCy or regex."""
    lower_word = word.lower()
    if _spacy_nlp is not None:
        doc = _spacy_nlp(text)
        for sent in doc.sents:
            if lower_word in sent.text.lower():
                return sent.text
    else:
        # Regex fallback: split on sentence-ending punctuation
        for m in re.finditer(r"[^\.!\?\n]+[\.!\?]?", text):
            if lower_word in m.group(0).lower():
                return m.group(0).strip()
    return text  # fallback to full text


def verify_flag_with_spacy(text: str, flagged_word: str) -> bool:
    """
    Cascading Verification Pipeline – Layer 1 (syntactic dependency parsing).

    Diagrams the sentence with spaCy and inspects the dependency of *flagged_word*.
    If the word is an adjective (amod or acomp) modifying a noun that is NOT in
    HUMAN_NOUNS (i.e. an inanimate object / abstract concept), return **False**
    → the flag is a false positive.
    Otherwise return **True** → keep the flag.
    """
    if _spacy_nlp is None:
        return True  # spaCy unavailable, conservatively keep the flag

    sentence = _extract_sentence_for_word(text, flagged_word)
    doc = _spacy_nlp(sentence)
    lower_word = flagged_word.lower()

    for token in doc:
        if token.text.lower() != lower_word:
            continue

        # Allow adjectives (amod, acomp) AND gerunds/participles/adjectives
        # (VBG, JJ, JJR, JJS) through the filter so words like
        # 'nurturing' are not silently dropped by the POS gate.
        if token.dep_ not in ("amod", "acomp") and token.tag_ not in ("VBG", "JJ", "JJR", "JJS"):
            continue

        head = token.head
        noun = head  # default: the syntactic head IS the noun

        # For acomp the head is usually a copula/verb (e.g. 'is').
        # Walk to the subject to find the real noun.
        if token.dep_ == "acomp" and head.pos_ in ("AUX", "VERB"):
            for child in head.children:
                if child.dep_ in ("nsubj", "nsubjpass"):
                    noun = child
                    break

        # --- Decision: is the modified noun a human? ---
        noun_lemma = noun.lemma_.lower()

        # 1. spaCy NER tagged it as a PERSON → human
        if noun.ent_type_ == "PERSON":
            continue  # keep the flag

        # 2. Lemma is in our hardcoded HUMAN_NOUNS set → human
        if noun_lemma in HUMAN_NOUNS:
            continue  # keep the flag

        # 3. Not a recognised human noun → inanimate / abstract → false positive
        return False

    return True  # word not found as amod/acomp, conservatively keep the flag


# ---------------------------------------------------------------------------
# Cross-Encoder Layer 2: Dual-anchor semantic re-ranking
# ---------------------------------------------------------------------------
# For each bias concept we maintain a (biased_description, safe_description)
# pair.  The sentence is scored against BOTH anchors via the Cross-Encoder.
# If biased_score > safe_score  →  keep the flag (genuine bias signal).
# If safe_score >= biased_score →  drop the flag (false positive).
# ---------------------------------------------------------------------------

_CE_ANCHORS: dict[str, tuple[str, str]] = {
    # (biased_context, safe_context)
    "ageism": (
        "A derogatory metaphor using words like dinosaur or fossil to insult an older human being, employee, or person.",
        "Referring to literal extinct reptiles, literal bones in the dirt, or replacing old inanimate computer hardware.",
    ),
    "age-based bias": (
        "A derogatory metaphor using words like dinosaur or fossil to insult an older human being, employee, or person.",
        "Referring to literal extinct reptiles, literal bones in the dirt, or replacing old inanimate computer hardware.",
    ),
    "ableism": (
        "Using ableist language, slurs, or derogatory terms targeting mental or physical disabilities, such as calling someone crazy or blind.",
        "Using terms like crazy, blind, or dumb in a colloquial, literal, or scientific way, such as blind study or crazy good.",
    ),
    "sexism": (
        "Sexist language that discriminates against women or men in the workplace, gender stereotypes.",
        "Educational discussion about gender equality, academic research, inclusivity training.",
    ),
    "racism": (
        "Racist language or racial discrimination, derogatory stereotypes about ethnicity.",
        "Historical or anthropological discussion, cultural studies, academic context.",
    ),
    "gender": (
        "Gender-biased or stereotypically gendered workplace language, hiring discrimination.",
        "Educational discussion about gender, biology class, neutral factual context.",
    ),
    "implicit bias (agentic language)": (
        "Stereotypically masculine workplace language: aggressive, dominant, competitive, rockstar, ninja.",
        "Educational discussion about language, inclusivity awareness, bias education, discussing problematic terms.",
    ),
    "implicit bias (communal language)": (
        "Stereotypically feminine workplace language: nurturing, supportive, collaborative, warm.",
        "Educational discussion about language, inclusivity awareness, bias education, discussing problematic terms.",
    ),
}

# Fallback anchors used when the concept doesn't match any key above.
_CE_DEFAULT_ANCHORS = (
    "Biased, discriminatory, exclusionary workplace language targeting a specific group.",
    "Educational, scientific, historical, or corrective discussion in a neutral context.",
)


def _get_ce_anchors(concept: str) -> tuple[str, str]:
    """Return (biased_desc, safe_desc) anchors for a concept label."""
    key = (concept or "").strip().lower()
    if key in _CE_ANCHORS:
        return _CE_ANCHORS[key]
    # Substring match
    for label, pair in _CE_ANCHORS.items():
        if label in key or key in label:
            return pair
    return _CE_DEFAULT_ANCHORS


def verify_flag_with_cross_encoder(text: str, matched_concept: str) -> bool:
    """
    Cascading Verification Pipeline – Layer 2 (semantic re-ranking).

    Uses a Cross-Encoder with dual anchors to decide whether a sentence
    genuinely contains bias.  The sentence is scored against a *biased*
    context description AND a *safe* context description.

    For NLI models (nli-deberta, nli-distilroberta), .predict() returns
    arrays of shape (N, 3) with [contradiction, entailment, neutral] logits.
    We use the entailment score (index 1) from each pair.

    For STS models (stsb-TinyBERT), .predict() returns scalar floats.

    - If prob_biased >= 0.60 →  True (keep)
    - Otherwise              →  False (drop)

    Returns True conservatively when the model is unavailable.
    """
    if _cross_encoder_model is None:
        return True  # model unavailable, conservatively keep the flag

    sentence = _extract_sentence_for_word(text, matched_concept)
    biased_desc, safe_desc = _get_ce_anchors(matched_concept)

    try:
        raw_scores = _cross_encoder_model.predict(
            [(sentence, biased_desc), (sentence, safe_desc)],
            apply_softmax=False,
        )

        import numpy as _np

        raw_scores = _np.array(raw_scores)

        if raw_scores.ndim == 2 and raw_scores.shape[1] >= 3:
            # NLI model: shape (2, 3) -> [contradiction, entailment, neutral]
            # Use entailment logit (index 1) as the relevance score
            biased_score = float(raw_scores[0][1])  # entailment for biased desc
            safe_score = float(raw_scores[1][1])    # entailment for safe desc
        else:
            # STS model: shape (2,) -> scalar similarity
            biased_score = float(raw_scores[0])
            safe_score = float(raw_scores[1])

    except Exception as e:
        print(f"Cross-Encoder prediction failed: {e}")
        return True  # conservatively keep the flag

    # Softmax probability: convert raw logits to a 0-1 confidence scale.
    # This correctly handles negative logits (common with DistilRoBERTa)
    # without the flawed CE_MARGIN / negative-override heuristics.
    import math
    exp_biased = math.exp(biased_score)
    exp_safe = math.exp(safe_score)
    prob_biased = exp_biased / (exp_biased + exp_safe)

    # Keep the flag if the AI is >= 60% confident it leans biased over safe
    keep = prob_biased >= 0.60
    print(f"  [Cross-Encoder L2] concept='{matched_concept}' | "
          f"biased={biased_score:.4f} safe={safe_score:.4f} prob_biased={prob_biased:.4f} | keep={keep}")

    return keep


def _spacy_adjective_modifies_object(sentence: str, word: str) -> bool | None:
    """
    Use spaCy dependency parsing to check what a flagged adjective modifies.

    Returns:
      True  → the word modifies a safe object (e.g., 'authoritative resources')
      False → the word modifies a person (e.g., 'authoritative leader')
      None  → inconclusive (word not found as adjective, or spaCy unavailable)
    """
    if _spacy_nlp is None:
        return None
    doc = _spacy_nlp(sentence)
    lower_word = word.lower()

    for token in doc:
        if token.text.lower() != lower_word:
            continue

        # Only apply this logic to adjectives and adverbs
        if token.pos_ not in ("ADJ", "ADV"):
            continue

        # Check what the adjective modifies via its head
        head = token.head
        head_lemma = head.lemma_.lower()

        if head_lemma in _SAFE_OBJECT_LEMMAS:
            return True  # modifies a thing → safe
        if head_lemma in _PERSON_LEMMAS:
            return False  # modifies a person → keep flag

        # (e.g., "she is authoritative" — 'she' is nsubj of 'is', 'authoritative' is acomp)
        if token.dep_ in ("acomp", "attr", "oprd") and head.pos_ == "AUX":
            for child in head.children:
                if child.dep_ == "nsubj" and child.lemma_.lower() in _PERSON_LEMMAS:
                    return False

    return None  # inconclusive


def _is_non_workplace_context(sentence: str) -> bool:
    """Check if the sentence context is non-workplace (ethics, academic, general)."""
    lower = sentence.lower()
    matches = sum(1 for kw in _NON_WORKPLACE_CONTEXT_KEYWORDS if kw in lower)
    # A single safe-context keyword is enough to drop the flag
    return matches >= 1


def filter_false_positives(
    issues: dict[str, dict],
    full_text: str,
) -> dict[str, dict]:
    """
    Semantic context filter that suppresses false positives for soft-skill words
    appearing in non-workplace contexts (e.g., ethics, scientific research,
    AI guidelines, historical discussion).
    """
    if not issues:
        return issues

    print("\n===== filter_false_positives: START =====")
    print(f"  Total issues entering filter: {len(issues)}")

    keys_to_remove: list[str] = []

    # --- Step -1: Educational SENTENCE Firewall ---
    # Operate at sentence level (not paragraph) so that mixed-context text
    # like "To remove bias, ignore X. However, David is surprisingly articulate."
    # only suppresses issues inside the educational sentence, not the whole block.
    educational_spans = []
    edu_markers = [
        "to remove corporate bias",
        "stripped of gender-specific markers",
        "inclusive version",
        "to rewrite", "to rephrase",
        "to make this more inclusive",
        "objective version", "neutral version",
        "bias-free", "bias free",
        "inclusive language",
        "revised version", "here is a revised",
        "substituted", "instead of"
    ]
    for s_text, s_start, s_end in _split_sentences(full_text):
        lower_s = s_text.lower()
        is_edu = any(m in lower_s for m in edu_markers) or _is_corrective_sentence(s_text)
        if is_edu:
            educational_spans.append((s_start, s_end))

    for key, issue in issues.items():
        biased_word = str(issue.get("biased_word") or key).strip().lower()

        # Context blindness check - Drop ANY issue in a corrective/educational sentence or paragraph
        valid_matches = []
        is_fully_educational = False
        
        if "matches" in issue and issue["matches"]:
            for m in issue["matches"]:
                m_start, m_end = m.get("start", -1), m.get("end", -1)
                is_edu_match = False
                for span_start, span_end in educational_spans:
                    if m_start >= span_start and m_end <= span_end:
                        is_edu_match = True
                        break
                if not is_edu_match:
                    valid_matches.append(m)
            
            if not valid_matches:
                is_fully_educational = True
        
        sentence = _extract_sentence_for_word(full_text, biased_word)
        if is_fully_educational or _is_corrective_sentence(sentence):
            print(f"    DROPPING FALSE POSITIVE: '{biased_word}' -- sentence/paragraph is corrective/educational")
            keys_to_remove.append(key)
            continue
            
        if "matches" in issue:
            issue["matches"] = valid_matches

        # --- Tech context allowlist ---
        # Suppress tech jargon FPs like slave/master in DB replication, blacklist in IP context, etc.
        _TECH_TRIGGERS = {
            "slave": {"database", "replication", "architecture", "server", "cluster", "node", "protocol", "redis", "primary", "secondary", "replica"},
            "master": {"database", "replication", "architecture", "server", "cluster", "node", "protocol", "redis", "branch", "git", "merge", "replica"},
            "blacklist": {"ip", "ips", "dns", "firewall", "domain", "server", "filter", "endpoint", "api", "whitelist"},
            "whitelist": {"ip", "ips", "dns", "firewall", "domain", "server", "filter", "endpoint", "api", "blacklist"},
            "sanity check": {"code", "script", "test", "debug", "function", "variable", "deploy", "build", "pipeline", "inject", "dummy"},
            "sanity": {"code", "script", "test", "debug", "function", "variable", "deploy", "build", "pipeline", "check", "inject", "dummy"},
            "check": {"sanity", "health", "dummy", "code", "script", "test"},
        }
        tech_keywords = _TECH_TRIGGERS.get(biased_word, set())
        if tech_keywords:
            lower_full = full_text.lower()
            if any(tk in lower_full for tk in tech_keywords):
                print(f"    DROPPING FALSE POSITIVE: '{biased_word}' -- tech context detected")
                keys_to_remove.append(key)
                continue

        # --- Context bypass for Chief ---
        if biased_word == "chief":
            safe_chief_phrases = {"chief executive officer", "ceo", "chief financial officer", "chief operating officer", "chief of staff"}
            if any(p in full_text.lower() for p in safe_chief_phrases):
                print(f"    DROPPING FALSE POSITIVE: '{biased_word}' -- safe professional title context")
                keys_to_remove.append(key)
                continue

        # --- Context bypass for Reign ---
        if biased_word == "reign":
            safe_reign_words = {"dinosaurs", "history", "era", "monarch", "king"}
            if any(sw in sentence.lower() for sw in safe_reign_words):
                print(f"    DROPPING FALSE POSITIVE: '{biased_word}' -- safe historical context")
                keys_to_remove.append(key)
                continue

        # Only continue semantic filtering for soft-skill terms
        if biased_word not in _SOFT_SKILL_TERMS:
            continue

        print(f"\n  [FILTER] Evaluating soft-skill word: '{biased_word}' (key='{key}')")
        display_sentence = sentence if len(sentence) <= 120 else f"{sentence[:120]}..."  # pyre-ignore
        print(f"    Sentence: '{display_sentence}'")

        dep_result = _spacy_adjective_modifies_object(sentence, biased_word)
        if dep_result is True:
            # Word modifies a safe object (e.g., 'authoritative resources') -> suppress
            print(f"    DROPPING FALSE POSITIVE: '{biased_word}' -- spaCy: modifies safe object")
            keys_to_remove.append(key)
            continue
        if dep_result is False:
            # Word modifies a person (e.g., 'authoritative leader') -> keep
            print(f"    KEEPING: '{biased_word}' -- spaCy: modifies a person")
            continue

        if _is_non_workplace_context(sentence):
            print(f"    DROPPING FALSE POSITIVE: '{biased_word}' -- non-workplace context keywords found")
            keys_to_remove.append(key)
            continue

        print(f"    KEEPING: '{biased_word}' -- no safe context detected")

    # Remove suppressed issues
    for key in keys_to_remove:
        issues.pop(key, None)
        print(f"  [REMOVED] '{key}' from issues dict")

    # If we removed all implicit issues, also remove the summary warning card
    remaining_implicit = any(
        str(v.get("type") or "").lower() in ("agentic", "communal")
        for v in issues.values()
    )
    if not remaining_implicit:
        issues.pop("implicit bias warning", None)

    print(f"  Total issues after filter: {len(issues)}")
    print("===== filter_false_positives: END =====\n")

    return issues


def _spellcheck_and_build_clean(text: str, db_rows) -> tuple[str, list[dict]]:
    """
    1. Runs _find_typos (difflib against bias triggers)
    2. Separates fused prefix words (e.g. verybossy -> very bossy)
    3. Runs pyspellchecker directly on the text for general misspellings
    4. Builds a clean_text string with typos substituted
    """
    if not text:
        return text, []

    typos = []
    clean_text = text

    # Phase 0: Leet-speak / symbol obfuscation normalizer
    # Catches evasion like bl!nd, cr@zy, sch!zophrenic, g*psy, r3tarded
    _LEET_MAP = {'!': 'i', '@': 'a', '3': 'e', '0': 'o', '$': 's', '1': 'l', '4': 'a', '5': 's'}
    # Build trigger set early for Phase 0
    _phase0_triggers = set()
    for _row in db_rows:
        _tw = (str(_row.get("Trigger_Word") or _row.get("trigger_word") or "")).strip().lower()
        if len(_tw) >= 3:
            _phase0_triggers.add(_tw)
    for raw_word in clean_text.split():
        # Strip surrounding punctuation (periods, commas, quotes)
        stripped = raw_word.strip('.,!?;:()[]{}"\'-')
        if not stripped or len(stripped) < 3:
            continue
        # Only process tokens with non-alpha characters embedded (the obfuscation)
        if stripped.isalpha():
            continue
        # Build normalized version via leet-speak map
        normalized = ''
        has_sub = False
        for ch in stripped.lower():
            if ch in _LEET_MAP:
                normalized += _LEET_MAP[ch]
                has_sub = True
            elif ch == '*':
                normalized += '*'
                has_sub = True
            else:
                normalized += ch
        if not has_sub:
            continue
        already = any(x['original'].lower() == stripped.lower() for x in typos)
        if already:
            continue
        # Handle * as vowel wildcard — try each vowel
        if '*' in normalized:
            for vowel in 'aeiouy':
                candidate = normalized.replace('*', vowel)
                if candidate in _phase0_triggers:
                    typos.append({"original": stripped, "suggested": candidate})
                    clean_text = clean_text.replace(stripped, candidate, 1)
                    break
        elif normalized in _phase0_triggers:
            typos.append({"original": stripped, "suggested": normalized})
            clean_text = clean_text.replace(stripped, normalized, 1)

    # Phase 1: Difflib for specific bias triggers
    difflib_typos = _find_typos(text, db_rows)
    for t in difflib_typos:
        typos.append(t)
        pattern = re.compile(rf"\b{re.escape(t['original'])}\b", re.IGNORECASE)
        clean_text = pattern.sub(t['suggested'], clean_text)

    # Phase 2: Fused prefix separation (e.g. verybossy -> very bossy)
    _trigger_set = set()
    for _row in db_rows:
        _tw = (str(_row.get("Trigger_Word") or _row.get("trigger_word") or "")).strip().lower()
        if len(_tw) >= 3:
            _trigger_set.add(_tw)

    # ── Fix: Add implicit lexicon terms to trigger set ──
    try:
        from implicit_bias_scorer import agentic_words, communal_words
        for _tw in agentic_words + communal_words:
            if len(_tw) >= 3:
                _trigger_set.add(_tw.lower())
    except ImportError:
        pass

    _COMMON_PREFIXES = {
        "too", "so", "very", "non", "not", "super", "ultra", "over",
        "un", "anti", "pre", "more", "most", "less", "real", "really",
        "quite", "extra", "mega", "semi", "being", "just", "way",
    }

    for _token_m in re.finditer(r"\b[a-zA-Z]{6,}\b", clean_text):
        _token = str(_token_m.group(0))
        # Universal Acronym Bypass for Spellcheck
        if _token.isupper():
            continue

        _lower_token = _token.lower()
        if _lower_token in _trigger_set:
            continue
        for _split_pos in range(2, len(_lower_token) - 2):
            _prefix = str(_lower_token[:_split_pos])
            _suffix = str(_lower_token[_split_pos:])
            if _prefix in _COMMON_PREFIXES and _suffix in _trigger_set:
                _suggestion = _prefix + " " + _suffix
                already_found = any(x["original"].lower() == _lower_token for x in typos)
                if not already_found:
                    typos.append({
                        "original": _token,
                        "suggested": _suggestion
                    })
                    pattern = re.compile(rf"\b{re.escape(_token)}\b", re.IGNORECASE)
                    clean_text = pattern.sub(_suggestion, clean_text)
                break

    # Phase 3: Repeated-character slang normalizer + Pyspellchecker
    # Internet slang like 'crazyyy' or 'manpowerrr' defeats pyspellchecker.
    # and fall back to pyspellchecker for anything else.
    for token_m in re.finditer(r"\b[a-zA-Z]{3,}\b", clean_text):
        token = token_m.group(0)
        # Universal Acronym Bypass for Spellcheck
        if token.isupper():
            continue

        lower_token = token.lower()

        # Skip tokens already handled in earlier phases
        already_found = any(x["original"].lower() == lower_token for x in typos)
        if already_found or lower_token in _trigger_set:
            continue

        # Strip repeated characters: 'crazyyy' -> 'crazy', 'manpowerrr' -> 'manpower'
        normalized = re.sub(r'(.)\1{2,}', r'\1', lower_token)

        if normalized != lower_token:
            if normalized in _trigger_set:
                typos.append({
                    "original": token,
                    "suggested": normalized
                })
                pattern = re.compile(rf"\b{re.escape(token)}\b", re.IGNORECASE)
                clean_text = pattern.sub(normalized, clean_text)
                continue

            # Also try pyspellchecker on the normalized form
            if _spell is not None and len(normalized) <= 25:
                if not _spell.known([normalized]):
                    correction = _spell.correction(normalized)
                    if correction and correction != normalized:
                        typos.append({
                            "original": token,
                            "suggested": correction
                        })
                        pattern = re.compile(rf"\b{re.escape(token)}\b", re.IGNORECASE)
                        clean_text = pattern.sub(correction, clean_text)
                        continue
                else:
                    # Normalized form IS a valid word — use it as the correction
                    typos.append({
                        "original": token,
                        "suggested": normalized
                    })
                    pattern = re.compile(rf"\b{re.escape(token)}\b", re.IGNORECASE)
                    clean_text = pattern.sub(normalized, clean_text)
                    continue

        # Standard pyspellchecker fallback for non-repeated-char typos
        if _spell is not None and len(lower_token) <= 25 and not _spell.known([lower_token]):
            correction = _spell.correction(lower_token)
            if correction and correction != lower_token:
                typos.append({
                    "original": token,
                    "suggested": correction
                })
                pattern = re.compile(rf"\b{re.escape(token)}\b", re.IGNORECASE)
                clean_text = pattern.sub(correction, clean_text)

    # Phase 4: Spelling corrections for "variant" DB triggers
    # Words like 'bimboo' or 'wackoooo' may be literally in the DB as triggers
    # (for the Nuclear Option bypass), but the frontend still needs a spelling
    # correction payload so it can display the blue "typo" card alongside the
    # yellow "bias" card.  For each token that was NOT already handled, check
    # if difflib finds a CLOSER canonical trigger that differs from the token.
    for token_m in re.finditer(r"\b[a-zA-Z]{3,}\b", text):
        token = token_m.group(0)
        # Universal Acronym Bypass for Spellcheck
        if token.isupper():
            continue

        lower_token = token.lower()

        # Skip tokens already corrected in earlier phases
        already_found = any(x["original"].lower() == lower_token for x in typos)
        if already_found:
            continue

        # Only consider tokens that ARE in the trigger set (exact DB hit)
        if lower_token not in _trigger_set:
            continue

        # Find closest trigger that is NOT identical
        close = difflib.get_close_matches(lower_token, list(_trigger_set), n=3, cutoff=0.75)
        for candidate in close:
            if candidate != lower_token and len(candidate) < len(lower_token):
                typos.append({
                    "original": token,
                    "suggested": candidate
                })
                break

    return clean_text, typos


@app.route('/evaluate_bias', methods=['POST'])
@app.route('/api/evaluate', methods=['POST'])  # compatibility with existing frontend
def evaluate_bias():
    data = request.get_json(silent=True) or {}
    raw_text = (data.get("text") or "").strip()

    is_ai_response = data.get("is_ai_response", False)

    if not raw_text:
        return jsonify({"is_biased": False, "issues": {}, "typos": []})

    # HYBRID EVALUATION ARCHITECTURE
    db_rows = _load_bias_database_rows()
    
    # ── Pipeline Refactor: Run Spellcheck FIRST to build a Clean Input ──
    clean_text, typos = _spellcheck_and_build_clean(raw_text, db_rows)

    # 1. User Prompt AND AI Response -> Explicit Lexicon Match (Run on clean_text)
    issues = _find_bias_issues_explicit(clean_text, db_rows, is_ai_response=is_ai_response)

    # ── Fix: Inject typing issues directly into the `issues` map so the frontend renders spelling corrections ──
    for t in typos:
        orig = t["original"]
        sugg = t["suggested"]
        spelling_key = f"{orig.lower()}_spelling"
        if spelling_key not in issues:
            # Find the index of the typo in the original raw text for accurate frontend spans
            start_idx = raw_text.lower().find(orig.lower())
            start = start_idx if start_idx != -1 else -1
            end = start_idx + len(orig) if start_idx != -1 else -1

            issues[spelling_key] = {
                "biased_word": orig,
                "canonical_key": spelling_key,
                "dimension": "Spelling / Formatting",
                "category": "Spelling",
                "severity": "Low",
                "affected_group": "",
                "suggestion": sugg,
                "matched_sentence": orig,
                "typo_embedded": True,
                "type": "spelling",
                "original": orig,
                "replacement": sugg,
                "matches": [
                    {
                        "text": orig,
                        "start": start,
                        "end": end
                    }
                ]
            }

    if is_ai_response:
        # 2. AI Response gets ADDITIONAL RAG Semantic Audit
        rag_issues = _find_bias_issues(clean_text, db_rows)
        for k, v in rag_issues.items():
            if k not in issues:
                issues[k] = v
            else:
                issues[k]["matches"].extend(v.get("matches", []))

    # Lexicon-based implicit bias scorer (agentic vs communal workplace language).
    # Map found agentic/communal terms into the canonical issue map so duplicates are impossible.
    if analyze_implicit_bias is not None:
        try:
            implicit = analyze_implicit_bias(clean_text)
            if implicit:
                agentic_found = implicit.get("agentic_found") or []
                communal_found = implicit.get("communal_found") or []

                # Add per-term implicit issues for highlighting (deduped by lowercase).
                for w in sorted({str(x).strip() for x in agentic_found if str(x).strip()}):
                    k = _normalize_for_contains(w)
                    if not k:
                        continue
                    issues.setdefault(
                        k,
                        {
                            "biased_word": w,
                            "canonical_key": k,
                            "matches": [],
                            "dimension": "Implicit Bias (Agentic Language)",
                            "category": "Implicit Bias (Agentic Language)",
                            "severity": "Low",
                            "affected_group": "Gender-coded workplace language",
                            "type": "agentic",
                            "suggestion": "",
                            "bypass_l2": True,
                        },
                    )
                for w in sorted({str(x).strip() for x in communal_found if str(x).strip()}):
                    k = _normalize_for_contains(w)
                    if not k:
                        continue
                    issues.setdefault(
                        k,
                        {
                            "biased_word": w,
                            "canonical_key": k,
                            "matches": [],
                            "dimension": "Implicit Bias (Communal Language)",
                            "category": "Implicit Bias (Communal Language)",
                            "severity": "Low",
                            "affected_group": "Gender-coded workplace language",
                            "type": "communal",
                            "suggestion": "",
                            "bypass_l2": True,
                        },
                    )

                # Add a single summary warning card (if skewed enough).
                if implicit.get("warning"):
                    issues["implicit bias warning"] = {
                        "biased_word": "Implicit Bias Warning",
                        "canonical_key": "implicit bias warning",
                        "matches": [],
                        "dimension": "Implicit Bias",
                        "category": "Implicit Bias",
                        "severity": "Medium",
                        "affected_group": "Gender-coded workplace language",
                        "type": "implicit_warning",
                        "warning": implicit.get("warning"),
                        "details": {
                            "agentic_count": implicit.get("agentic_count"),
                            "communal_count": implicit.get("communal_count"),
                            "agentic_ratio": implicit.get("agentic_ratio"),
                            "communal_ratio": implicit.get("communal_ratio"),
                        },
                        "suggestion": "",
                    }
        except Exception:
            pass

    # Optional ensemble bias engine (spaCy + embeddings). The engine is fully
    # local and only runs when its dependencies are installed.
    ensemble_result = None
    if calculate_bias_score and ENSEMBLE_ENGINE_AVAILABLE:
        try:
            ensemble_result = calculate_bias_score(clean_text, issues)
        except Exception:
            ensemble_result = None

    # Overlap cleanup is no longer necessary here because:
    # - Explicit CSV issues are canonical-keyed
    # - The frontend banner is set-deduped
    # - Ignored phrases are removed by span containment

    # Semantic false-positive filter: suppress soft-skill words in non-workplace contexts
    issues = filter_false_positives(issues, clean_text)

    # ── Fix 2: Typo bypass for L1 spaCy verifier and L2 Cross-Encoder ──
    # For very short prompts (1-2 words) or typo-embedded matches, the sentence-level
    # models lack context and would incorrectly reject valid bias terms.
    prompt_word_count = len(clean_text.split())
    is_short_prompt = prompt_word_count <= 2

    keys_to_remove = []
    if is_short_prompt:
        print(f"  [L2 BYPASS] Short prompt ({prompt_word_count} words) — skipping spaCy and Cross-Encoder verification")
    else:
        # Mark typo-embedded issues for L2 bypass
        typo_suggestions = {t["suggested"].lower() for t in typos}
        for k, issue in issues.items():
            if issue.get("typo_embedded"):
                issue["bypass_l2"] = True
                print(f"  [L2 BYPASS] '{k}' is typo-embedded — bypassing Cross-Encoder")
            elif str(issue.get("severity") or "").lower() == "high":
                issue["bypass_l2"] = True
                print(f"  [L2 BYPASS] '{k}' is high-severity — bypassing Cross-Encoder")
            elif "bro" in str(issue.get("dimension") or "").lower():
                issue["bypass_l2"] = True
                print(f"  [L2 BYPASS] '{k}' is Bro Culture — bypassing Cross-Encoder")
            elif str(issue.get("biased_word", "")).lower() in typo_suggestions or str(issue.get("canonical_key", "")).lower() in typo_suggestions:
                issue["bypass_l2"] = True
                print(f"  [L2 BYPASS] '{k}' originated from a typo obfuscation — bypassing Cross-Encoder")
            elif str(issue.get("type") or "").lower() in ("agentic", "communal"):
                # Pronoun Check: Only keep agentic/communal if a gendered pronoun exists in the sentence
                # OR if a highly stereotyped role is present (e.g. nurse, surgeon)
                word_to_verify = issue.get("biased_word", k)
                sentence = str(issue.get("matched_sentence") or _extract_sentence_for_word(clean_text, word_to_verify)).lower()
                pronouns = {"he", "him", "his", "himself", "she", "her", "hers", "herself", "man", "woman", "boy", "girl", "lady", "gentleman"}
                words_in_sentence = set(re.findall(r'\b\w+\b', sentence))
                
                # Check for stereotyped role noun phrases in the full text to handle cross-sentence pronoun resolution
                stereotyped_roles = {"nurse", "surgeon", "assistant", "executive", "leader", "pediatric nurse", "trauma surgeon", "manager", "project manager", "director", "executive director"}
                has_stereotyped_role = any(role in clean_text.lower() for role in stereotyped_roles)
                
                if not any(p in words_in_sentence for p in pronouns) and not has_stereotyped_role:
                    print(f"  [L2 BYPASS] '{k}' is implicit bias but NO gendered pronouns or stereotyped roles found in sentence — dropping")
                    keys_to_remove.append(k)
                    continue

                # Implicit bias words are already verified by the implicit bias scorer.
                # The L2 Cross-Encoder NLI model is not trained to classify soft-skill
                # language and would incorrectly drop valid implicit bias flags.
                issue["bypass_l2"] = True
                print(f"  [L2 BYPASS] '{k}' is implicit bias ({issue.get('type')}) — bypassing Cross-Encoder")

    # Cascading Verification Pipeline step:
    if not is_short_prompt:
        for k, issue in issues.items():
            if k == "implicit bias warning":
                continue
            if issue.get("typo_embedded") or issue.get("bypass_l2") or str(issue.get("severity") or "").lower() == "high":
                continue
            word_to_verify = issue.get("biased_word", k)
            
            # Sentence Boundary Enforcement for Spacy
            sentence = str(issue.get("matched_sentence") or _extract_sentence_for_word(clean_text, word_to_verify))
            # Verify the flag using spacy against the local sentence
            if not verify_flag_with_spacy(sentence, word_to_verify):
                keys_to_remove.append(k)

    for k in keys_to_remove:
        issues.pop(k, None)

    # --- Layer 2: Cross-Encoder semantic re-ranking ---
    ce_keys_to_remove = []
    if not is_short_prompt:
        for k, issue in issues.items():
            if k == "implicit bias warning":
                continue
            
            # Contextual Whitelist for 'dinosaur'
            w = str(issue.get("biased_word", k)).strip().lower()
            if w in ("dinosaur", "dinosaurs"):
                safe_dino_words = {
                    "bone", "bones", "fossil", "museum", "paleontology", "species", "extinct",
                    "time", "period", "era", "history", "lived", "jurassic", "cretaceous",
                    "mesozoic", "reptile", "earth",
                }
                if any(sw in clean_text.lower() for sw in safe_dino_words):
                    print(f"  [CONTEXT WHITELIST] '{w}' safe context detected -> dropping flag")
                    ce_keys_to_remove.append(k)
                    continue

            if issue.get('bypass_l2'):
                continue
            
            # The Cultural AI blindspot L2 bypass
            iss_dim = str(issue.get("dimension", "")).lower()
            if "cultural appropriation" in iss_dim or "cultural" in iss_dim or "indigenous" in iss_dim:
                print(f"  [L2 BYPASS] '{k}' is cultural phrasing — bypassing Cross-Encoder")
                continue
            
            concept = issue.get("dimension", k)
            word_to_verify = issue.get("biased_word", k)
            
            # Sentence Boundary Enforcement for Cross-Encoder
            sentence = str(issue.get("matched_sentence") or _extract_sentence_for_word(clean_text, word_to_verify))
            
            l2_result = verify_flag_with_cross_encoder(sentence, concept)
            print(f"[DEBUG L2] Token: {k}, Concept: {concept}, Keep: {l2_result}")
            if not l2_result:
                ce_keys_to_remove.append(k)

    for k in ce_keys_to_remove:
        issues.pop(k, None)

    # --- Active Learning / User Feedback Integration ---
    # 1. Apply BLACKLIST -> MUST flag
    for b_word in BLACKLIST:
        pattern = re.compile(rf"\b{re.escape(b_word)}\b", re.IGNORECASE)
        for m in pattern.finditer(clean_text):
            match_str = m.group(0)
            if b_word not in issues:
                issues[b_word] = {
                    "biased_word": match_str,
                    "canonical_key": b_word,
                    "dimension": "User Reported Bias",
                    "category": "User Reported Bias",
                    "severity": "High",
                    "affected_group": "User Feedback",
                    "suggestion": "This term was flagged by users as a false negative.",
                    "matches": []
                }
            existing_match = False
            for ext_m in issues[b_word]["matches"]:  # type: ignore
                if ext_m.get("start") == m.start() and ext_m.get("end") == m.end():
                    existing_match = True
                    break
            if not existing_match:
                issues[b_word]["matches"].append({  # type: ignore
                    "text": match_str,
                    "start": m.start(),
                    "end": m.end()
                })

    # 2. Apply FALSE_POSITIVES_CACHE -> MUST bypass flag
    keys_to_remove_fp = []
    for k, issue in issues.items():
        if k == "implicit bias warning":
            continue
        w = str(issue.get("biased_word") or "").strip().lower()
        k_lower = str(k).strip().lower()
        if w in FALSE_POSITIVES_CACHE or k_lower in FALSE_POSITIVES_CACHE:
            print(f"  [WHITELIST] Dropping '{k}' (biased_word='{w}') — in FALSE_POSITIVES_CACHE")
            keys_to_remove_fp.append(k)

    for k in keys_to_remove_fp:
        issues.pop(k, None)

    # Re-evaluate implicit bias warning card if necessary
    remaining_implicit = any(
        str(v.get("type") or "").lower() in ("agentic", "communal")
        for k, v in issues.items() if k != "implicit bias warning"
    )
    if "implicit bias warning" in issues and not remaining_implicit:
        issues.pop("implicit bias warning", None)

    # --- Deduplication: remove substring flags ---
    keys_to_drop = set()
    for k1, v1 in issues.items():
        if k1 == "implicit bias warning": continue
        w1 = str(v1.get("biased_word") or k1).strip().lower()
        for k2, v2 in issues.items():
            if k1 == k2 or k2 == "implicit bias warning" or k2 in keys_to_drop: continue
            w2 = str(v2.get("biased_word") or k2).strip().lower()
            if w1 in w2 and len(w1) < len(w2):
                keys_to_drop.add(k1)
                break
                
    for k in keys_to_drop:
        issues.pop(k, None)

    # ── Silent LLM Verification — fire-and-forget background thread ──
    # The response is sent to the client BEFORE the LLM call begins.
    if verify_bias_async is not None:
        try:
            # Try to grab explicit parameters if the frontend starts sending them,
            # otherwise derive them from clean_text based on the context flag.
            user_prompt = data.get("user_prompt", clean_text if not is_ai_response else "")
            ai_resp = data.get("ai_response", clean_text if is_ai_response else "")
            
            verify_bias_async(user_prompt, ai_resp, issues)
        except Exception:
            pass  # Never block the response

    return jsonify(
        {
            "is_biased": len([k for k, v in issues.items() if k != "implicit bias warning" and str(v.get("type") or "") != "spelling"]) > 0,
            # Canonical map: keys are canonical terms, values are details
            "issues": issues,
            "typos": typos,
            "ensemble_bias": ensemble_result,
            "clean_text": clean_text
        }
    )


# -------------------------------
# BiasBuster: Text generation (disabled)
# -------------------------------

@app.route('/api/generate/gemini', methods=['POST'])
def generate_gemini():
    """Generate text via Google Gemini.  Robust wrapper with full terminal diagnostics."""

    # --- Pre-flight checks (clear error messages) ---
    if genai is None:
        msg = "google-genai SDK not installed.  Run: pip install google-genai"
        print(f"\n[GEMINI ERROR] {msg}")
        return jsonify({"error": msg}), 500

    api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY") or ""
    if not api_key:
        msg = "Missing API key.  Set GEMINI_API_KEY (or GOOGLE_API_KEY) in your .env file."
        print(f"\n[GEMINI ERROR] {msg}")
        return jsonify({"error": msg}), 500

    data = request.get_json(silent=True) or {}
    user_text = (data.get("text") or data.get("prompt") or data.get("input") or "").strip()
    if not user_text:
        return jsonify({"error": "Missing 'text' in request payload."}), 400

    try:
        # Explicitly pass the API key — genai.Client() defaults to GOOGLE_API_KEY,
        # but our .env stores it as GEMINI_API_KEY.
        client = genai.Client(api_key=api_key)
        response = client.models.generate_content(
            model="gemini-2.0-flash",
            contents=(
                "You are a helpful, intelligent assistant. "
                "Provide a direct, factual, and useful answer to the user's prompt.\n\n"
                f"{user_text}"
            ),
        )

        rewritten = getattr(response, "text", "") or ""
        if not rewritten:
            # Some responses embed text inside candidates
            try:
                rewritten = response.candidates[0].content.parts[0].text or ""
            except Exception:
                rewritten = ""

        if not rewritten:
            print("[GEMINI WARNING] API returned an empty response body.")
            return jsonify({"error": "Gemini returned an empty response.  The prompt may have been blocked by safety filters."}), 502

        # Frontend API layer expects `text`; keep `rewritten_text` for clarity/back-compat.
        return jsonify({"text": rewritten, "rewritten_text": rewritten})

    except Exception as exc:
        err_str = str(exc).lower()
        if "quota" in err_str or "rate" in err_str or "429" in err_str or "exhausted" in err_str:
            return jsonify({"error": "AI Generation quota exceeded. Please wait a minute or use a different prompt."}), 429

        # ── Full diagnostic dump to terminal ──
        print("\n" + "=" * 60)
        print("[GEMINI 500] Exception during /api/generate/gemini")
        print(f"  Type : {type(exc).__name__}")
        print(f"  Message: {exc}")
        print("-" * 60)
        traceback.print_exc()
        print("=" * 60 + "\n")

        # Categorize for the frontend so users get actionable feedback
        if "api key" in err_str or "authenticate" in err_str or "permission" in err_str or "403" in err_str:
            category = "API Key / Auth Error"
        elif "model" in err_str or "not found" in err_str or "404" in err_str:
            category = "Model Not Found"
        else:
            category = "Generation Error"

        return jsonify({"error": f"[{category}] {exc}"}), 500


def _groq_rewrite(model: str):
    # If Groq is not available locally, fail gracefully with a clear message instead
    # of surfacing a low-level SDK error into the UI.
    if Groq is None:
        return jsonify(
            {
                "error": "Groq SDK (`groq`) is not installed. Install it with `pip install groq` to enable Groq-backed models.",
            }
        ), 500

    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        return jsonify(
            {
                "error": "GROQ_API_KEY is not configured. Add GROQ_API_KEY to your `.env` file or environment to enable Groq-backed models.",
            }
        ), 500

    data = request.get_json(silent=True) or {}
    text = (data.get("text") or data.get("prompt") or data.get("input") or "").strip()
    if not text:
        return jsonify({"error": "Missing 'text' in request payload."}), 400

    client = Groq(api_key=api_key)
    completion = client.chat.completions.create(
        model=model,
        stream=False,
        messages=[
            {
                "role": "system",
                "content": "You are a helpful, intelligent assistant. Provide a direct, factual, and useful answer to the user's prompt.",
            },
            {"role": "user", "content": text},
        ],
    )

    rewritten = ""
    try:
        rewritten = completion.choices[0].message.content
    except Exception:
        rewritten = ""

    rewritten = rewritten or ""
    return jsonify({"text": rewritten, "rewritten_text": rewritten})


@app.route('/api/generate/llama', methods=['POST'])
def generate_llama():
    try:
        return _groq_rewrite(model="llama-3.1-8b-instant")
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/api/generate/qwen', methods=['POST'])
def generate_qwen():
    try:
        # Qwen 3 32B model name as exposed by Groq
        return _groq_rewrite(model="qwen/qwen3-32b")
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/api/generate/gptoss', methods=['POST'])
def generate_gptoss():
    try:
        # Model availability can vary on Groq; if this exact name isn't supported,
        # change it to the closest supported GPT-OSS model.
        return _groq_rewrite(model="openai/gpt-oss-120b")
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    # ── Initialize Tier 2 FAISS index at startup ──
    try:
        _lazy_init_semantic_guard()  # Ensure _st_model is loaded first
        _startup_db_rows = _load_bias_database_rows()
        _init_faiss_index(_startup_db_rows)
    except Exception as _init_err:
        print(f"WARNING: FAISS startup init failed: {_init_err}")
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port, debug=True)
