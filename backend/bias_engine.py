import math
import re
from typing import List, Dict, Any

try:  # pragma: no cover - optional heavy deps
    import spacy
except Exception:  # pragma: no cover
    spacy = None  # type: ignore

try:  # pragma: no cover - optional heavy deps
    from sentence_transformers import SentenceTransformer, util
except Exception:  # pragma: no cover
    SentenceTransformer = None  # type: ignore
    util = None  # type: ignore


# Expose a simple capability flag so the Flask app can decide whether to call us.
ENGINE_AVAILABLE = False

_nlp = None
_embedding_model = None
_biased_concepts = [
    "workplace sexism",
    "ageism against older employees",
    "racial microaggressions",
    "ableist language",
    "hostile work environment",
    "gender stereotyping in hiring",
    "discrimination based on pregnancy",
    "racist jokes in the office",
    "mocking accents or dialects",
    "derogatory language about disabilities",
]
_biased_concept_embeddings = None


def _lazy_init_models() -> None:
    """
    Lazily load spaCy and SentenceTransformers so importing the module is cheap and
    the app can still boot even if the heavy NLP libraries are not installed.
    """
    global ENGINE_AVAILABLE, _nlp, _embedding_model, _biased_concept_embeddings

    if ENGINE_AVAILABLE:
        return

    if spacy is None or SentenceTransformer is None:
        # Required libraries not installed; keep ENGINE_AVAILABLE = False
        return

    try:
        if _nlp is None:
            _nlp = spacy.load("en_core_web_sm")
        if _embedding_model is None:
            _embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
        if _biased_concept_embeddings is None:
            _biased_concept_embeddings = _embedding_model.encode(_biased_concepts, convert_to_tensor=True)
        ENGINE_AVAILABLE = True
    except Exception:
        # If anything fails (e.g., model not downloaded), fail closed.
        ENGINE_AVAILABLE = False


def _csv_factor_score(csv_matches: List[Dict[str, Any]]) -> int:
    """
    Factor 1 - CSV/Regex Match (+40 points if we have at least one match).
    """
    return 40 if csv_matches else 0


def _nlp_context_score(sentence: str, csv_matches: List[Dict[str, Any]]) -> int:
    """
    Factor 2 - NLP Context (-30 or +20 points).

    Heuristic:
    - If a trigger word appears as an adjective modifying a neutral noun (e.g., "female patient"),
      subtract 30 (context likely descriptive/clinical).
    - If a trigger word appears as a standalone noun referring to a person/group, add 20.
    - Otherwise, 0.
    """
    _lazy_init_models()
    if not ENGINE_AVAILABLE or not _nlp:
        return 0

    doc = _nlp(sentence)

    # Collect candidate trigger tokens from CSV matches
    trigger_lemmas = {str((m.get("biased_word") or "")).lower() for m in csv_matches if m.get("biased_word")}
    trigger_lemmas = {t for t in trigger_lemmas if len(t) >= 3}
    if not trigger_lemmas:
        return 0

    adjective_like_hit = False
    derogatory_noun_hit = False

    for token in doc:
        lower_tok = token.text.lower()
        if lower_tok not in trigger_lemmas:
            continue

        # Adjective modifying a neutral head noun (e.g., "female patient")
        if token.pos_ == "ADJ" and token.head.pos_ in {"NOUN", "PROPN"}:
            adjective_like_hit = True

        # Standalone noun referring to people or groups (e.g., "females", "natives", "elderly")
        if token.pos_ == "NOUN":
            # Simple heuristic: plural humans/groups more likely to be problematic.
            if token.tag_ in {"NNS", "NNPS"} or token.dep_ in {"nsubj", "dobj", "pobj", "attr"}:
                derogatory_noun_hit = True

    if adjective_like_hit and not derogatory_noun_hit:
        return -30
    if derogatory_noun_hit:
        return 20
    return 0


def _semantic_similarity_score(sentence: str) -> int:
    """
    Factor 3 - Semantic Similarity (+0 to +40 points).

    Embed the user's sentence and compute cosine similarity vs biased_concepts.
    Highest similarity * 40 (rounded).
    """
    _lazy_init_models()
    if not ENGINE_AVAILABLE or not _embedding_model or _biased_concept_embeddings is None or util is None:
        return 0

    try:
        sent_emb = _embedding_model.encode(sentence, convert_to_tensor=True)
        cos_scores = util.cos_sim(sent_emb, _biased_concept_embeddings)[0]
        # Convert to plain python float
        best_sim = float(cos_scores.max().item())
    except Exception:
        return 0

    best_sim = max(0.0, min(1.0, best_sim))
    return int(round(best_sim * 40))


def calculate_bias_score(sentence: str, csv_matches: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Ensemble Bias Detection Engine.

    Returns:
        {
            "is_biased": bool,
            "score": int,
            "breakdown": {
                "csv_factor": int,
                "nlp_factor": int,
                "semantic_factor": int,
            },
            "engine_available": bool,
        }
    """
    text = (sentence or "").strip()
    matches = csv_matches or []

    _lazy_init_models()

    base_score = 0
    csv_factor = _csv_factor_score(matches)
    nlp_factor = _nlp_context_score(text, matches)
    semantic_factor = _semantic_similarity_score(text)

    final_score = base_score + csv_factor + nlp_factor + semantic_factor
    is_biased = final_score >= 70

    return {
        "is_biased": bool(is_biased),
        "score": int(final_score),
        "breakdown": {
            "csv_factor": int(csv_factor),
            "nlp_factor": int(nlp_factor),
            "semantic_factor": int(semantic_factor),
        },
        "engine_available": bool(ENGINE_AVAILABLE),
    }

