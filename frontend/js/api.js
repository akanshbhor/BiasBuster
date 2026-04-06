/**
 * BiasBuster Frontend API Layer
 *
 * Goals:
 * - Keep *all* network requests in one place
 * - Normalize error handling + JSON parsing
 * - Support:
 *   1) Generation via Gemini or Groq-backed models
 *   2) Bias evaluation via the backend's Custom Regex CSV Model (/api/evaluate)
 *
 * IMPORTANT:
 * - Bias evaluation is NOT done by an LLM anymore.
 * - The evaluator returns:
 *   { is_biased: boolean, issues: [{ biased_word, dimension, severity, affected_group }] }
 */

// Detect local development reliably (includes file:// for quick demos).
const isLocalhost =
  window.location.hostname === 'localhost' ||
  window.location.hostname === '127.0.0.1' ||
  window.location.protocol === 'file:';

// If you deploy your backend, replace this with your deployed URL.
const API_BASE_URL = 'https://biasbuster-backend-9jqiy.ondigitalocean.app';

/**
 * Small helper to:
 * - Send JSON
 * - Parse JSON
 * - Provide a consistent error object shape for UI handling
 */
async function postJson(path, body) {
  window.dispatchEvent(new CustomEvent('bb:net', { detail: { phase: 'start', path } }));
  const response = await fetch(`${API_BASE_URL}${path}`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  });

  let data;
  try {
    data = await response.json();
  } catch (e) {
    data = { error: 'Invalid JSON from server.' };
  }

  if (!response.ok) {
    const message = data?.error || `HTTP ${response.status}`;
    window.dispatchEvent(new CustomEvent('bb:net', { detail: { phase: 'error', path, status: response.status } }));
    return { ok: false, error: message, data };
  }

  window.dispatchEvent(new CustomEvent('bb:net', { detail: { phase: 'success', path, status: response.status } }));
  return { ok: true, data };
}

// Lightweight connectivity probe used by the UI status indicator.
window.pingBackend = async function pingBackend() {
  try {
    const res = await fetch(`${API_BASE_URL}/api/health`, { method: 'GET' });
    return res.ok;
  } catch {
    return false;
  }
};

/**
 * Generate text using the selected engine.
 * Returns a string OR a user-visible error string prefixed with "Error:".
 */
window.fetchUnifiedAIResponse = async function fetchUnifiedAIResponse(promptText, model) {
  try {
    const endpoint =
      model === 'gemini' ? '/api/generate/gemini'
        : model === 'llama' ? '/api/generate/llama'
          : model === 'qwen' ? '/api/generate/qwen'
            : model === 'gptoss' ? '/api/generate/gptoss'
              : '/api/generate/gemini';

    const result = await postJson(endpoint, { prompt: promptText });
    if (!result.ok) return `Error: ${model.toUpperCase()} - ${result.error}`;

    const text = result.data?.text;
    if (typeof text !== 'string') return `Error: ${model.toUpperCase()} - Missing 'text' in response.`;
    return text;
  } catch (error) {
    return `Error: Network - ${error.message}`;
  }
};

/**
 * Evaluate generated text for bias using the backend CSV+Regex evaluator.
 * Always returns an object shaped like:
 *   { is_biased: boolean, issues: {} }
 */
window.evaluateForBias = async function evaluateForBias(text, is_ai_response = false) {
  try {
    const result = await postJson('/api/evaluate', { text, is_ai_response });
    if (!result.ok) return { is_biased: false, issues: {}, error: result.error };

    const is_biased = result.data?.is_biased || false;
    const issues = result.data?.issues || {};
    const typos = result.data?.typos || []; // ADD THIS LINE

    return { is_biased, issues, typos }; // UPDATE THIS LINE
  } catch (error) {
    return { is_biased: false, issues: {}, error: error.message };
  }
};

/**
 * Report user feedback (Active Learning)
 * label: 0 for False Positive, 1 for False Negative
 */
window.reportFeedback = async function reportFeedback(word, label) {
  try {
    const result = await postJson('/api/feedback', { word, label });
    if (!result.ok) {
      console.error("Failed to report feedback:", result.error);
      return false;
    }
    return true;
  } catch (error) {
    console.error("Network error reporting feedback:", error);
    return false;
  }
};