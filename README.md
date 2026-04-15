# BiasBuster

> **Unmasking Coded Bias in Algorithmic Decision Making**
> 
> A gamified diagnostic toolkit and API service designed to detect, highlight, and remediate implicit proxies and discriminatory patterns in AI systems.

![Live Demo](https://img.shields.io/badge/Live%20Demo-Vercel-black?logo=vercel)
![API Endpoint](https://img.shields.io/badge/Backend-DigitalOcean-blue?logo=digitalocean)

## Live Links

* **Live Demo:** [https://biasbuster-codequest26.vercel.app](https://biasbuster-codequest26.vercel.app)
* **Backend API Base:** [https://biasbuster-backend-9jqiy.ondigitalocean.app/](https://biasbuster-backend-9jqiy.ondigitalocean.app/)
* **Documentation:** [View User Manual (PDF)](frontend/assets/user_manual.pdf)

## The Problem & Solution

Artificial intelligence models are trained on historical data encompassing decades of societal prejudice. When deployed in critical sectors like hiring or finance, these models often rely on **proxy variables** (e.g., graduation year, vocabulary, or geographic location) to inadvertently replicate discriminatory behavior without explicitly filtering by protected classes.

**BiasBuster** solves this by enforcing a strong **Human-in-the-Loop** architecture. We combine a gamified "Resume Auditor" module with a real-time bias detection laboratory, bridging the gap between theoretical AI fairness and practical, actionable oversight.

## Key Features

* **Resume Auditor Experience:** A high-stakes, 60-second interactive game loop where users confront AI ageism and must catch algorithmic mistakes, featuring cinema-to-game auto-start mechanics and seamless UX.
* **Real-time Bias Laboratory:** A dynamic evaluation interface allowing users to audit text. Features include inline color-coded highlights, severity scores, and a missed-word reporting pipeline.
* **Lexicon-Based Implicit Bias Scorer:** Evaluates text for agentic vs. communal framing (operating on a mathematically sound 66% skew threshold) to expose deeply ingrained gender-coded language often missed by traditional LLMs.

---

## How the Backend Engine Works: A 3-Tier Architecture

BiasBuster’s backend is not a simple keyword scanner. It is a robust, asynchronous, multi-layered NLP application designed for computationally heavy verification workloads. When a user submits text, it passes through a rigorous 3-Tier gauntlet to eliminate false positives and "AI hallucinations."

### Tier 1: Local Verification (Regex & Heuristics)
The first layer is a lightning-fast heuristic net operating entirely locally. 
* **Dynamic Lexicon Expansion:** It uses WordNet to instantly expand our core bias dictionary (e.g., dynamically adding hundreds of synonyms to catch variations).
* **Regex & Typo Obfuscation:** It uses Regular Expressions and `PySpellChecker` to catch users attempting to bypass the system by slightly misspelling biased terms (e.g., flagging `strategic_spelling` as a typo-embedded proxy).
* **The "L2 Bypass" Router:** To save computational power and prevent generative AI from overriding established sociological facts, Tier 1 uses strict logic to bypass higher-tier checks for known, severe implicit biases (e.g., routing heavily "agentic" words like *driven* or *competitive* straight to the final report).

### Tier 2: Vector Embeddings & Semantic Guards

If a flagged word's context is ambiguous, it moves to the Machine Learning tier.
* **Vector Embeddings & ChromaDB:** Modern bias detection cannot rely on dictionaries. Instead, our engine converts text into mathematical coordinates in a high-dimensional space (Vector Embeddings). We use **ChromaDB** as our persistent vector database to store "safe" and "biased" reference sentences. 
* **Cosine Similarity & Cross-Encoders:** When a word like "dinosaur" is flagged, a local Sentence-Transformer (DeBERTa Cross-Encoder) calculates the semantic distance between the user's sentence and our ChromaDB rules. It mathematically determines if "dinosaur" refers to a literal fossil (safe) or is being used as a slur for an older IT worker (biased). 

### Tier 3: Synchronous LLM Verification & RAG

For the most complex, nuanced flags that survive Tier 1 and Tier 2, the system invokes its **Synchronous Contextual Verification Loop**.
* **RAG (Retrieval-Augmented Generation):** Before asking an LLM to evaluate the text, we query our FAISS/ChromaDB database for sociolinguistic context rules related to the specific flag. We inject this grounded, retrieved context into the prompt.
* **Dual-Engine LLM Verification:** We query `GPT-OSS-120B` (with a failover to `Llama 3.3 70B` via Groq). Equipped with the RAG context, the LLM analyzes the sentence and synchronously returns a highly structured report detailing `CONFIRMED`, `MISSED`, or `FALSE POSITIVES`. For instance, it can reason that the word *head* in "head of the department" is a false positive and drop it from the final UI payload.

---

## System Architecture & Tech Stack

### Frontend Modules
A lightweight, fast, vanilla web interface optimized for edge deployment. It is divided into three primary operational modules:

* **The Experience:** Our interactive human-in-the-loop simulation environment.

* **The Laboratory:** This serves as the primary workspace and workbench for analyzing and evaluating text for bias. 
  * **Evaluation Workflow:** Users input prompts such as candidate reviews or performance evaluations into the main text area (everyday prompts along with prompts that can trigger AI bias) and click Send to engage the dual-stage backend engine.
  * **Annotated Reports:** The interface dynamically interprets the backend JSON to provide **Highlighted Text** for potential biases, **Severity Scores** based on historical heuristic weight, and objective, neutralized **Actionable Suggestions**.
  * **Continuous Learning Pipeline:** If the engine misses a locally biased term, the **Report Bias Words** button triggers a modal where users can submit the anomalous word and its context. This data is securely logged and fed directly into the system's active learning loop to harden the heuristic engine over time.
  * **Auxiliary Help:** An integrated help overlay provides quick-reference tooltips on scoring methodology and bias categories.

* **The Information Page:** This module is dedicated to system transparency. It hosts the system documentation, release notes, and operational telemetry. It works in tandem with the persistent bottom status bar, which displays real-time operational readiness and uplink/latency monitoring to ensure the connection to the L2 Heuristic Engine remains secure and responsive.

### Frontend Tech
* **Tech:** Vanilla HTML5, CSS3, JavaScript
* **Components:** Splash Gateway (`index.html`), Resume Auditor (`experience.html`), Bias Laboratory (`laboratory.html`), Information Hub (`information.html`)
* **Routing & Hosting:** Vercel (Dynamic routing configured in `api.js` detecting local vs production environments)

### Backend Tech
* **Tech:** Python, FastAPI/Flask, Gunicorn, SQLite, Docker
* **Deployment:** DigitalOcean (App Platform / Droplet)
* **Structure:** Highly modularized engine (`app.py` router, `tier1_heuristics.py`, `tier2_ml.py`, `tier3_llm.py`)
* **NLP & ML Libraries:** SpaCy, Sentence-Transformers, NLTK, FAISS, PySpellChecker, ChromaDB
* **Generative AI Providers:** GenAI SDK (Gemini), Groq SDK (Llama/GPT-OSS)

## Local Installation & Setup

### Prerequisites
* Python 3.13.12
* Node.js (Optional, for frontend serving)
* Docker (Optional, to mirror production build)

### Backend Setup
1. Navigate to the backend directory:
   ```bash
   cd backend
   ```
2. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   .\venv\Scripts\activate  # On Mac: source venv/bin/activate
   ```
3. Install Python dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Create a `.env` file in the `backend/` directory. *(Note: For production on DigitalOcean, input these into your App Platform environment variables panel).*
   ```env
   # Required for LLM Verification Service and Failover (Llama / GPT-OSS)
   GROQ_API_KEY=your_groq_api_key_here
   
   # Required for generative AI capabilities (Gemini 3.0 Flash)
   GEMINI_API_KEY=your_gemini_api_key_here
   ```
5. Run the local backend server:
   ```bash
   python app.py
   ```

### Frontend Setup
1. Open the `frontend/` directory.
2. The frontend uses a dynamic API base URL. If your backend is running locally, `api.js` will automatically route traffic to your local server port.
3. Serve the static files using your preferred local server:
   ```bash
   npx serve .
   # or
   python -m http.server 3000
   ```

## API Reference

| Endpoint | Method | Description |
| :--- | :---: | :--- |
| `/api/health` | `GET` | Lightweight system connectivity probe. |
| `/api/evaluate` | `POST` | Core NLP-driven bias detection engine (Regex + Semantic Checking + Sync LLM Verification). |
| `/api/generate/gemini`| `POST` | Generates a response using the Gemini 3.0 Flash model. |
| `/api/generate/llama` | `POST` | Generates a response using the Meta Llama 3.1 model. |
| `/api/generate/qwen`  | `POST` | Generates a response using the Qwen 3.0 32B model. |
| `/api/generate/gptoss`| `POST` | Generates a response using the GPT-OSS 120B model. |
| `/api/feedback` | `POST` | Active learning endpoint; flags user-submitted false positives/negatives to SQLite. |

---

## License & Copyright

The source code for this project is licensed under the **Apache License 2.0**. You may use, distribute, and modify the code under the terms of this license. 

### Trademark Disclaimer
"BiasBuster" and the BiasBuster logo are intended trademarks of the repository owner. The Apache License 2.0 explicitly **does not** grant any rights to use these trade names, trademarks, service marks, or product names. Any commercial product or service utilizing this software must be rebranded and cannot use the "BiasBuster" name or imply endorsement without explicit, written permission.
