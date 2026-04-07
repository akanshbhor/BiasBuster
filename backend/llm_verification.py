"""
Silent LLM Bias Verification Service
=====================================

Background verification layer that cross-references locally-detected bias
keywords against a large language model to confirm accuracy and catch
subtle proxies the lexicon engine may have missed.

Design constraints:
- Synchronous execution — blocks the /api/evaluate response until complete
- Returns false_positives list so the caller can remove them from the payload
- Dual-model failover — GPT-OSS-120B primary → Llama 3.3 70B fallback
- Graceful failure — swallows all errors, logs warnings, never crashes
"""

import json
import os
import time
from typing import Optional

try:
    from groq import Groq  # type: ignore
except ImportError:
    Groq = None

try:
    from dotenv import load_dotenv  # type: ignore
    load_dotenv()
except Exception:
    pass


# ─── Configuration ───────────────────────────────────────────────────────────

PRIMARY_MODEL = "openai/gpt-oss-120b"
FALLBACK_MODEL = "llama-3.3-70b-versatile"
REQUEST_TIMEOUT = 15  # seconds per model attempt

SYSTEM_PROMPT = """\
You are an Algorithmic Bias Auditor — a strict, expert-level reviewer \
specializing in detecting discriminatory language, proxy variables, coded \
terminology, and subtle exclusionary patterns in professional and workplace text.

You receive THREE inputs:
1. A User Prompt (human input).
2. An AI Response (machine output).
3. A list of words/phrases that a local bias-detection engine has already flagged.

Your task is to cross-reference the local engine's flags against your own \
analysis and independently evaluate BOTH the user's input and the AI's output \
for any subtle proxies or coded language the engine missed.

Respond ONLY with a valid JSON object. No markdown fences, no commentary, \
no explanation outside the JSON:
{
  "confirmed": ["word1", "word2"],
  "false_positives": ["word3"],
  "missed": [
    {"word": "culture fit", "dimension": "Exclusionary Language", "severity": "Medium", "source": "user_prompt"},
    {"word": "aggressive", "dimension": "Gender-coded Language", "severity": "Medium", "source": "ai_response"}
  ],
  "reasoning": "Brief 1-2 sentence summary of your audit."
}

Definitions:
- "confirmed": Words from the flagged list that ARE genuinely biased in this context.
- "false_positives": Any word from the LOCALLY FLAGGED WORDS list that you determine is NOT biased MUST be copied exactly into this array. Do not leave it empty if a flagged word is benign. (e.g., literal uses, scientific context, informal metaphors for inanimate objects like 'crazy rollercoaster', or safe professional parlance).
- "missed": Subtle bias proxies, coded language, dog-whistles, or exclusionary \
  patterns the local engine did NOT flag but SHOULD have. MUST include dimension, severity, \
  and the exact source ("user_prompt" or "ai_response"). For every item in the "missed" array, you MUST include a "source" key. The value MUST be exactly "user_prompt" or "ai_response". Do not use any other values.
- "reasoning": A concise explanation of your overall assessment.

Focus areas for "missed" detection:
- Proxy terms for protected classes (age, gender, race, disability, religion, orientation).
- Tokenism or racial focus (e.g., "diverse hire").
- Caregiver status/Marital status gatekeeping (e.g., "single and hungry", "unencumbered", "free to travel 100%", "no family obligations"). Please extract the full phrase.
- Ableist metaphors (e.g., "tone deaf", "insane", "blind spot", "crazy").
- Coded corporate jargon that acts as demographic filters (e.g., "rockstar", "ninja", "10x engineer", "culture fit", "digital native", "wizard").
- Gendered assumptions embedded in role descriptions.
- Ageist metaphors, youth-biased language, or expectation of extreme lack of boundaries (e.g., "adaptable", "frantic startup hours"). Ensure you independently flag "adaptable" alongside structural issues.
- Ableist framing or assumptions.
- Socioeconomic gatekeeping language ("pedigree", "top-tier", "Ivy league only").
- Name/origin bias indicators.

CRITICAL DIRECTIVE — Metaphorical Demographic Bias and Workplace Tropes:
Terms like "dinosaurs", "fossils", "greybeards", "old guard", "deadwood", "relics", \
"over the hill", or "past their prime" used in corporate, workplace, HR, or \
professional contexts MUST be assumed to refer to PEOPLE (Ageism) unless the \
surrounding text EXPLICITLY and UNAMBIGUOUSLY refers ONLY to hardware, software, \
or non-human systems. The mere presence of technology-adjacent words like \
"department", "team", "IT", "systems", or "infrastructure" is NOT sufficient to \
disambiguate — these words commonly describe groups of PEOPLE. You must apply the \
"reasonable HR auditor" standard: if a phrase could plausibly describe employees \
or colleagues in a demeaning age-related way, it IS ageist bias and MUST be \
confirmed, NOT marked as a false positive. Examples:
  - "dinosaurs in the IT department" → CONFIRMED (Ageism — refers to older employees)
  - "fossil systems from the 90s" → FALSE POSITIVE (explicitly about software)
  - "greybeards running the servers" → CONFIRMED (Ageism — refers to people)
  - "this codebase is a fossil" → FALSE POSITIVE (explicitly about code)
  - "deadwood on the engineering team" → CONFIRMED (Ageism — refers to people)

Be strict: flag ALL exclusionary proxies, dog whistles, and gated requirements, but allow neutral professional language.\
"""


# ─── Terminal Output Formatting ──────────────────────────────────────────────

def _print_report(
    model_used: str,
    latency: float,
    confirmed: list,
    false_positives: list,
    missed: list,
    reasoning: str,
) -> None:
    """Print a beautiful, structured verification report to the terminal."""

    border = "═" * 62
    thin   = "─" * 62

    print(f"\n╔{border}╗")
    print(f"║  {'🔍 LLM BIAS VERIFICATION REPORT':^60}║")
    print(f"╠{border}╣")
    print(f"║  Model: {model_used:<30} Latency: {latency:.2f}s     ║")
    print(f"╠{border}╣")

    # ── Confirmed ──
    print(f"║                                                              ║")
    print(f"║  ✅ CONFIRMED ({len(confirmed)}):{'':>42}║")
    if confirmed:
        for word in confirmed:
            label = f"     • {word}"
            print(f"║  {label:<60}║")
    else:
        print(f"║       {'(none)':<55}║")

    # ── Missed ──
    print(f"║                                                              ║")
    print(f"║  ⚠️  MISSED ({len(missed)}):{'':>43}║")
    if missed:
        for item in missed:
            if isinstance(item, dict):
                word = item.get("word", "?")
                dim = item.get("dimension", "Unknown")
                sev = item.get("severity", "?")
                src = item.get("source", "unknown").lower()
                src_badge = "[HUMAN]" if "user" in src else "[AI]" if "ai" in src else "[?]"
                label = f'     • {src_badge} "{word}" → {dim} [{sev}]'
            else:
                label = f"     • {item}"
            # Truncate if too long for the box
            if len(label) > 58:
                label = label[:55] + "..."
            print(f"║  {label:<60}║")
    else:
        print(f"║       {'(none)':<55}║")

    # ── False Positives ──
    print(f"║                                                              ║")
    print(f"║  ❌ FALSE POSITIVES ({len(false_positives)}):{'':>36}║")
    if false_positives:
        for word in false_positives:
            label = f"     • {word}"
            print(f"║  {label:<60}║")
    else:
        print(f"║       {'(none)':<55}║")

    # ── Reasoning ──
    print(f"║                                                              ║")
    print(f"║  💬 Reasoning:{'':>47}║")
    if reasoning:
        # Word-wrap reasoning into ~56-char lines
        words = reasoning.split()
        line = "     "
        for w in words:
            if len(line) + len(w) + 1 > 58:
                print(f"║  {line:<60}║")
                line = "     " + w
            else:
                line = line + " " + w if line.strip() else "     " + w
        if line.strip():
            print(f"║  {line:<60}║")
    else:
        print(f"║       {'(no reasoning provided)':<55}║")

    print(f"║                                                              ║")
    print(f"╚{border}╝\n")


def _print_warning(message: str) -> None:
    """Print a styled terminal warning for verification failures."""
    print(f"\n┌{'─' * 62}┐")
    print(f"│  ⚠️  LLM VERIFICATION WARNING{' ':>33}│")
    print(f"├{'─' * 62}┤")
    # Word-wrap the message
    words = message.split()
    line = "  "
    for w in words:
        if len(line) + len(w) + 1 > 60:
            print(f"│  {line:<60}│")
            line = "  " + w
        else:
            line = line + " " + w if line.strip() else "  " + w
    if line.strip():
        print(f"│  {line:<60}│")
    print(f"└{'─' * 62}┘\n")


# ─── LLM Call with Failover ─────────────────────────────────────────────────

def _call_llm(user_message: str) -> tuple[Optional[str], str]:
    """
    Call the LLM with automatic failover.

    Returns: (response_text | None, model_used)
    """
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        _print_warning("GROQ_API_KEY not set — LLM verification skipped.")
        return None, "none"

    if Groq is None:
        _print_warning("Groq SDK not installed — LLM verification skipped.")
        return None, "none"

    client = Groq(api_key=api_key, timeout=REQUEST_TIMEOUT)

    models = [PRIMARY_MODEL, FALLBACK_MODEL]

    for model in models:
        try:
            completion = client.chat.completions.create(
                model=model,
                stream=False,
                temperature=0.1,  # Low temp for deterministic auditing
                max_tokens=1024,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_message},
                ],
            )
            text = ""
            try:
                text = completion.choices[0].message.content or ""
            except Exception:
                text = ""

            if text.strip():
                return text.strip(), model

            # Empty response — try fallback
            print(f"  [LLM Verify] {model} returned empty response, trying fallback...")

        except Exception as e:
            err_type = type(e).__name__
            print(f"  [LLM Verify] {model} failed ({err_type}: {e}), trying fallback...")
            continue

    return None, "none"


# ─── Prompt Builder ──────────────────────────────────────────────────────────

def _build_user_message(user_prompt: str, ai_response: str, detected_issues: dict) -> str:
    """
    Construct the user message for the LLM from the user prompt, AI response,
    and the locally-detected issues dictionary.
    """
    # Extract the flagged word list from the issues dict
    flagged_words = []
    for key, issue in detected_issues.items():
        if key == "implicit bias warning":
            continue
        # Skip spelling-only entries
        if str(issue.get("type", "")).lower() == "spelling":
            continue

        word = str(issue.get("biased_word", key)).strip()
        dimension = str(issue.get("dimension", "")).strip()
        severity = str(issue.get("severity", "")).strip()

        if word:
            entry = word
            if dimension:
                entry += f" ({dimension}"
                if severity:
                    entry += f", {severity}"
                entry += ")"
            flagged_words.append(entry)

    flagged_list = "\n".join(f"  - {w}" for w in flagged_words) if flagged_words else "  (none detected)"

    return (
        f"USER PROMPT:\n"
        f'"""\n{user_prompt}\n"""\n\n'
        f"AI RESPONSE:\n"
        f'"""\n{ai_response}\n"""\n\n'
        f"LOCALLY FLAGGED WORDS:\n{flagged_list}\n\n"
        f"Please audit the conversation above. Cross-reference the flagged words and "
        f"independently identify any subtle bias proxies the local engine missed in either text."
    )


# ─── Response Parser ────────────────────────────────────────────────────────

def _parse_response(raw: str) -> Optional[dict]:
    """
    Parse the LLM's JSON response. Handles markdown code fences gracefully.
    """
    text = raw.strip()

    # Strip markdown code fences if present
    if text.startswith("```"):
        lines = text.split("\n")
        # Remove first and last lines (```json and ```)
        lines = [l for l in lines if not l.strip().startswith("```")]
        text = "\n".join(lines).strip()

    try:
        result = json.loads(text)
        if isinstance(result, dict):
            return result
    except json.JSONDecodeError:
        pass

    # Try to extract JSON from within the text
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        try:
            result = json.loads(text[start:end + 1])
            if isinstance(result, dict):
                return result
        except json.JSONDecodeError:
            pass

    return None


# ─── Main Entry Point ────────────────────────────────────────────────────────

def _run_verification(user_prompt: str, ai_response: str, detected_issues: dict) -> tuple[list[str], list[dict]]:
    """
    Internal verification runner. Executes synchronously.
    Returns (false_positives, missed) — both lists.
    All exceptions are caught and logged — never propagated.
    """
    try:
        # Don't verify if both are empty
        if not (user_prompt.strip() or ai_response.strip()):
            return [], []

        start_time = time.time()

        # Build the prompt
        user_message = _build_user_message(user_prompt, ai_response, detected_issues)

        # Call LLM with failover
        raw_response, model_used = _call_llm(user_message)

        if raw_response is None:
            _print_warning(
                f"Both models failed or unavailable. "
                f"Verification skipped — core detection results are unaffected."
            )
            return [], []

        latency = time.time() - start_time

        # Parse the structured response
        parsed = _parse_response(raw_response)

        if parsed is None:
            _print_warning(
                f"LLM returned unparseable response ({model_used}). "
                f"Raw output logged below:\n{raw_response[:300]}"
            )
            return [], []

        # Extract fields with safe defaults
        llm_confirmed = parsed.get("confirmed", [])
        false_positives = parsed.get("false_positives", [])
        missed = parsed.get("missed", [])
        reasoning = parsed.get("reasoning", "")

        # Ensure types are correct
        if not isinstance(llm_confirmed, list):
            llm_confirmed = []
        if not isinstance(false_positives, list):
            false_positives = []
        if not isinstance(missed, list):
            missed = []
        if not isinstance(reasoning, str):
            reasoning = str(reasoning)

        # ── Programmatically compute confirmed terms ──
        # The LLM uses a subtractive filter and often returns an empty/incomplete
        # "confirmed" array.  We compute confirmed = original_flagged − false_positives
        # so the terminal report always shows which terms survived the filter.
        original_flagged: list[str] = []
        for _k, _iss in detected_issues.items():
            if _k == "implicit bias warning":
                continue
            if str(_iss.get("type", "")).lower() == "spelling":
                continue
            word = str(_iss.get("biased_word", _k)).strip()
            if word:
                original_flagged.append(word)

        fp_lower_set = {str(fp).strip().lower() for fp in false_positives}
        confirmed = [w for w in original_flagged if w.strip().lower() not in fp_lower_set]

        # Print the beautiful terminal report
        _print_report(
            model_used=model_used,
            latency=latency,
            confirmed=confirmed,
            false_positives=false_positives,
            missed=missed,
            reasoning=reasoning,
        )
        return false_positives, missed

    except Exception as e:
        # Absolute last-resort catch — never crash, never block
        try:
            _print_warning(f"Unexpected error in verification thread: {type(e).__name__}: {e}")
        except Exception:
            pass  # Even print failed — silently swallow
        return [], []


def verify_bias_sync(user_prompt: str, ai_response: str, detected_issues: dict) -> tuple[list[str], list[dict]]:
    """
    Public entry point. Runs the LLM verification synchronously.
    
    Returns:
        A tuple of (false_positives, missed).
        - false_positives: list of term strings the LLM marked as not biased.
        - missed: list of dicts with keys: word, dimension, severity, source.

    Args:
        user_prompt: The clean user prompt (or empty if n/a).
        ai_response: The generated AI response (or empty if n/a).
        detected_issues: The issues dict from evaluate_bias() (post-filtering).
    """
    # Deep-copy the issues dict so the thread has its own snapshot
    try:
        issues_snapshot = json.loads(json.dumps(detected_issues, default=str))
    except Exception:
        issues_snapshot = {}

    return _run_verification(user_prompt, ai_response, issues_snapshot)
