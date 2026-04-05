import re
from typing import Dict, Any, List


# These lexicons are inspired by common gender-coded language research used in
# hiring / performance review analyses (agentic vs communal framing).
agentic_words = [
    "drive",
    "driven",
    "courage",
    "courageous",
    "outpace",
    "champion",
    "champions",
    "foresight",
    "competitive",
    "compete",
    "assertive",
    "assert",
    "ambitious",
    "dominant",
    "forceful",
    "independent",
    "decisive",
    "decision",
    "logical",
    "objective",
    "analytical",
    "leader",
    "lead",
    "leading",
    "leadership",
    "confident",
    "confidence",
    "selfreliant",
    "self-reliant",
    "selfreliance",
    "self-reliance",
    "take charge",
    "command",
    "commands",
    "direct",
    "directs",
    "directive",
    "strong",
    "strength",
    "bold",
    "fearless",
    "aggressive",
    "aggressively",
    "results",
    "result-driven",
    "win",
    "wins",
    "winning",
    "outperform",
    "excel",
    "excellence",
    "strategic",
    "strategy",
    "visionary",
    "vision",
    "initiative",
    "initiatives",
    "autonomous",
    "ownership",
    "own",
    "authority",
    "authoritative",
    "decisiveness",
    "dominance",
    "clarity, brevity, and directness",
    "concise and objective",
    "precise instructions",
    "clinical rigor",
    "unwavering focus",
]

communal_words = [
    "support",
    "supports",
    "supportive",
    "diplomatic",
    "rapport",
    "trustworthy",
    "liaison",
    "collaborative",
    "collaborate",
    "collaboration",
    "compassionate",
    "compassion",
    "understanding",
    "nurturing",
    "sympathetic",
    "helpful",
    "pleasant",
    "cooperative",
    "cooperate",
    "cooperation",
    "sensitive",
    "sensitivity",
    "inclusive",
    "inclusion",
    "empathic",
    "empathetic",
    "empathy",
    "interpersonal",
    "teamwork",
    "team player",
    "together",
    "share",
    "sharing",
    "mentor",
    "mentoring",
    "mentors",
    "develop others",
    "listening",
    "listen",
    "listens",
    "patient",
    "patience",
    "warm",
    "warmth",
    "friendly",
    "kind",
    "kindness",
    "considerate",
    "thoughtful",
    "respectful",
    "respect",
    "care",
    "caring",
    "community",
    "communicate",
    "communication",
    "relationship",
    "relationships",
    "harmonious",
    "harmonize",
    "tactful",
    "encourage",
    "encouraging",
    "encouragement",
    "togetherness",
    "empathetic updates",
    "foster a sense of trust",
]


def _tokenize(text: str) -> List[str]:
    # Keep it simple and deterministic: lowercase, strip punctuation to spaces, split.
    t = (text or "").lower()
    t = re.sub(r"[^a-z0-9\s'-]+", " ", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t.split(" ") if t else []


def analyze_implicit_bias(text: str, threshold: int = 3, skew_ratio: float = 0.75) -> Dict[str, Any]:
    """
    Lexicon-based implicit bias scorer (agentic vs communal).

    Returns:
      {
        "total_words": int,
        "agentic_count": int,
        "communal_count": int,
        "agentic_ratio": float,
        "communal_ratio": float,
        "agentic_found": [str],
        "communal_found": [str],
        "warning": str | None
      }
    """
    tokens = _tokenize(text)
    total_words = len(tokens)

    # Build fast lookup sets (including multi-word phrases handled via substring scan below)
    agentic_set = {w for w in agentic_words if " " not in w}
    communal_set = {w for w in communal_words if " " not in w}
    agentic_phrases = [w for w in agentic_words if " " in w]
    communal_phrases = [w for w in communal_words if " " in w]

    agentic_found: List[str] = []
    communal_found: List[str] = []

    for tok in tokens:
        if tok in agentic_set:
            agentic_found.append(tok)
        if tok in communal_set:
            communal_found.append(tok)

    # Phrase scan on the normalized text
    normalized_spaced = " " + " ".join(tokens) + " "
    for p in agentic_phrases:
        p_norm = " " + " ".join(_tokenize(p)) + " "
        if p_norm in normalized_spaced:
            agentic_found.append(p)
    for p in communal_phrases:
        p_norm = " " + " ".join(_tokenize(p)) + " "
        if p_norm in normalized_spaced:
            communal_found.append(p)

    agentic_count = len(agentic_found)
    communal_count = len(communal_found)
    coded_total = agentic_count + communal_count

    agentic_ratio = (agentic_count / coded_total) if coded_total else 0.0
    communal_ratio = (communal_count / coded_total) if coded_total else 0.0

    # Warning logic is independent from returning found words.
    # We ALWAYS return any found agentic/communal terms; the warning only appears
    # when there is enough signal and the ratio is strongly skewed.
    warning = None
    if coded_total > threshold:
        if agentic_ratio >= skew_ratio:
            warning = (
                f"Implicit Bias Warning: This text is heavily skewed toward Agentic language "
                f"({agentic_count} words vs {communal_count} words). Ensure these descriptions are based on "
                f"objective role requirements and not unconscious stereotyping."
            )
        elif communal_ratio >= skew_ratio:
            warning = (
                f"Implicit Bias Warning: This text is heavily skewed toward Communal language "
                f"({communal_count} words vs {agentic_count} words). Ensure these descriptions are based on "
                f"objective role requirements and not unconscious stereotyping."
            )

    # Always return findings.

    return {
        "total_words": int(total_words),
        "agentic_count": int(agentic_count),
        "communal_count": int(communal_count),
        "agentic_ratio": float(agentic_ratio),
        "communal_ratio": float(communal_ratio),
        "agentic_found": agentic_found,
        "communal_found": communal_found,
        "warning": warning,
    }

