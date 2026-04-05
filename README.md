\# BiasBuster



> \*\*Unmasking Coded Bias in Algorithmic Decision Making\*\*

> A gamified diagnostic toolkit and API service designed to detect, highlight, and remediate implicit proxies and discriminatory patterns in AI systems.



!\[Live Demo](https://img.shields.io/badge/Live%20Demo-Vercel-black?logo=vercel)

!\[API Endpoint](https://img.shields.io/badge/API-Render-black?logo=render)



\## Live Links

\* \*\*Live Demo:\*\* \[https://biasbuster-codequest26.vercel.app](https://biasbuster-codequest26.vercel.app)

\* \*\*Render Pipeline:\*\* \[https://biasbuster-backend-9xtt.onrender.com](https://biasbuster-backend-9xtt.onrender.com)



\## The Problem \& Solution

Artificial intelligence models are trained on historical data encompassing decades of societal prejudice. When deployed in critical sectors like hiring or finance, these models often rely on \*\*proxy variables\*\* (e.g., graduation year, vocabulary, or geographic location) to inadvertently replicate discriminatory behavior without explicitly filtering by protected classes.



\*\*BiasBuster\*\* solves this by enforcing a strong \*\*Human-in-the-Loop\*\* architecture. We combine a gamified "Resume Auditor" module with a real-time bias detection laboratory, bridging the gap between theoretical AI fairness and practical, actionable oversight.



\## Key Features

\* \*\*Resume Auditor Experience:\*\* A high-stakes, 60-second interactive game loop where users confront AI ageism and must catch algorithmic mistakes, demonstrating the critical need for human oversight.

\* \*\*Diagnostic Dashboard:\*\* A validation gate that trains users to identify and patch biased proxies, unlocking the "AI Fairness Playbook" upon successful completion.

\* \*\*Real-time Bias Laboratory:\*\* A dynamic chat interface allowing users to challenge the AI, complete with multi-layered architectural safeguards like an Educational Firewall, RAG, Vector Embeddings, Contextual Tech Allowlist, and PySpellChecker Obfuscation Engine.

\* \*\*Dual-Engine Failover System:\*\* An automated, silent LLM verification layer utilizing `GPT-OSS-120B` with a graceful failover to `Llama 3.3 70B` via Groq, ensuring high availability and robust secondary audits of text.

\* \*\*Lexicon-Based Implicit Bias Scorer:\*\* Evaluates text for agentic vs. communal framing, exposing deeply ingrained gender-coded language often missed by traditional LLMs.



\## System Architecture \& Tech Stack



\### Frontend

A lightweight, fast, vanilla web interface optimized for edge deployment.

\* \*\*Tech:\*\* Vanilla HTML5, CSS3, JavaScript

\* \*\*Components:\*\* Splash Gateway (`index.html`), Resume Auditor (`experience.html`), Bias Laboratory (`laboratory.html`)

\* \*\*Routing \& Hosting:\*\* Vercel (Dynamic routing configured in `api.js` detecting `localhost` vs `onrender.com`)



\### Backend

A robust, asynchronous Flask application designed for computationally heavy NLP and AI verification workloads.

\* \*\*Tech:\*\* Python, Flask, Gunicorn, SQLite

\* \*\*Deployment:\*\* Render

\* \*\*NLP \& Detection Layer:\*\* SpaCy, Sentence-Transformers, NLTK, FAISS, PySpellChecker, ChromaDB

\* \*\*Generative \& Auditing AI Engines:\*\* GenAI SDK (Gemini), Groq SDK (Llama/GPT-OSS)



\## Local Installation \& Setup



\### Prerequisites

\* Python 3.9+

\* Node.js (Optional, for frontend serving)



\### Backend Setup

1\. Navigate to the backend directory:

&#x20;  ```bash

&#x20;  cd backend

&#x20;  ```

2\. Create and activate a virtual environment:

&#x20;  ```bash

&#x20;  python -m venv venv

&#x20;  source venv/bin/activate  # On Windows: .\\venv\\Scripts\\activate

&#x20;  ```

3\. Install Python dependencies:

&#x20;  ```bash

&#x20;  pip install -r requirements.txt

&#x20;  ```

4\. Create a `.env` file in the `backend/` directory. Based on the system architecture, you must define:

&#x20;  ```env

&#x20;  # Required for LLM Verification Service and Failover (Llama / GPT-OSS)

&#x20;  GROQ\_API\_KEY=your\_groq\_api\_key\_here

&#x20;  

&#x20;  # Required for generative AI capabilities (Gemini 3.0 Flash)

&#x20;  GEMINI\_API\_KEY=your\_gemini\_api\_key\_here

&#x20;  ```

5\. Run the local Flask server (auto-bootstraps via virtual environment):

&#x20;  ```bash

&#x20;  python app.py

&#x20;  ```



\### Frontend Setup

1\. Open the `frontend/` directory.

2\. The frontend uses a dynamic API base URL. If your backend is running locally on port `8000`, `api.js` will automatically route all traffic securely to `http://localhost:8000`.

3\. Serve the static files using your preferred local server:

&#x20;  ```bash

&#x20;  npx serve .

&#x20;  # or

&#x20;  python -m http.server 3000

&#x20;  ```



\## 📡 API Reference

| Endpoint | Method | Description |

| :--- | :---: | :--- |

| `/api/health` | `GET` | Lightweight system connectivity probe. |

| `/api/evaluate` | `POST` | Core NLP-driven bias detection engine (CSV Regex + Semantic Checking). |

| `/api/generate/gemini`| `POST` | Generates a response using the Gemini 3.0 Flash model. |

| `/api/generate/llama` | `POST` | Generates a response using the Meta Llama 3.1 model. |

| `/api/generate/qwen`  | `POST` | Generates a response using the Qwen 3.0 32B model. |

| `/api/generate/gptoss`| `POST` | Generates a response using the GPT-OSS 20B model. |

| `/api/feedback` | `POST` | Active learning endpoint; flags user-submitted false positives/negatives to SQLite. |

```

=======

\---

title: BiasBuster Backend

colorFrom: blue

colorTo: gray

sdk: docker

pinned: false

\---



Check out the configuration reference at https://huggingface.co/docs/hub/spaces-config-reference



