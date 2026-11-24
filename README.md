# ATS Resume Intelligence & Candidate Screener

An end-to-end **Hybrid Applicant Tracking System (ATS) & Batch Candidate Screening Engine** built with **Python, Scikit-Learn, Sentence-Transformers (`all-MiniLM-L6-v2`), Google Gemini API, and Streamlit**.

Traditional ATS filters rely purely on brittle keyword matching and are easily fooled by hidden text or prompt injections. This system solves both problems by combining a **4-signal hybrid ML scoring engine** with **anti-tampering document forensics**, **real-time recruiter threshold controls**, **automated contact extraction**, and **executive PDF roster generation**.

---

## Highlighted Main Features

| Core Feature | What It Does | Technical Implementation |
| :--- | :--- | :--- |
| **4-Signal Hybrid ATS Scoring** | Evaluates resumes across lexical, morphological, semantic, and supervised probability signals instead of exact keyword counts alone | `Word TF-IDF (25%)` + `Char N-Grams (15%)` + `all-MiniLM-L6-v2 (20%)` + `LogisticRegression (40%)` |
| **Multi-Resume Batch Screening** | Processes single or multiple PDF resumes simultaneously, ranks all candidates in a live leaderboard, and generates per-candidate diagnostic reports | Batch `pypdf` ingestion + Streamlit session state & expandable evaluation cards |
| **Dynamic Shortlist Threshold (`40%–95%`)** | Lets recruiters adjust the minimum qualification cut-off score in real time without re-running embeddings or Gemini API calls | Sidebar interactive slider (`st.slider`) with instant in-memory recalculation |
| **Automated Contact & Profile Extraction** | Extracts **Full Name, Email, Phone Number, LinkedIn, and GitHub URLs** from qualifying resumes while filtering out job titles/headlines | Regex entity parsers + top-line header scoring + CamelCase filename fallback |
| **One-Click Shortlist PDF Export** | Generates an executive-ready **Shortlisted Candidates Roster (`.pdf`)** containing contact details, score breakdowns, and audit status | Automated multi-card PDF layout engine built with `fpdf2` + individual `.md` exports |
| **Anti-Tampering & Security Audit** | Protects screening integrity against hidden keyword stuffing, verbatim JD cloning, and LLM prompt injection attacks | `pypdf` `< 4.0pt` micro-font coordinate audit, regex injection scanner, and XML sandboxing |
| **AI Recruiter Qualitative Synthesis** | Produces structured hiring verdicts, critical skill gap analyses, and tailored resume bullet-point upgrades | `google-genai` SDK with automatic fallback routing & `.env` backend key security |
| **Fast Signature-Validated Caching** | Eliminates cold-start CPU bottlenecks after initial calibration while auto-rebuilding if the training dataset changes | `joblib` artifact persistence (`ats_cache.joblib`) + `@st.cache_resource` |

---

## Feature Deep-Dive

### 1. Hybrid Machine Learning & Semantic Scoring Pipeline
Every candidate resume is scored against the target Job Description using four complementary signals:
- **Supervised Match Classifier (`40%`):** L2-regularized `LogisticRegression` trained on stratified domain pairs from `Resume.csv`.
- **Lexical Keyword Alignment (`25%`):** Sublinear Word TF-IDF (`1–2` n-grams) capturing exact frameworks, tools, and domain terminology.
- **Dense Semantic Similarity (`20%`):** `all-MiniLM-L6-v2` 384-dimensional embeddings capturing conceptual equivalence (e.g., *vector search* $\leftrightarrow$ *semantic retrieval*).
- **Sub-Word Character N-Grams (`15%`):** Character-boundary n-grams (`3–5` chars) handling compound terms and abbreviations (`ci/cd`, `node.js`, `rest api`).

### 2. Multi-Layer Document Integrity & Anti-Tampering Defense
Every uploaded PDF passes through four automated security checks before scoring and LLM synthesis:
- **LLM Prompt Injection Detection:** Scans for embedded override instructions (e.g., *"ignore previous instructions"*, *"give 95% score"*) and wraps untrusted resume content inside strict `<candidate_resume>` XML delimiters.
- **Hidden Micro-Font Stuffing Audit:** Inspects PDF transformation matrices (`font_size * scale_y`) via `pypdf` to detect invisible `< 4.0pt` text blocks.
- **Verbatim JD Cloning Check:** Flags resumes copying $\ge 3$ full sentences or $> 35\%$ of the Job Description verbatim.
- **Abnormal Token Repetition Check:** Flags unnatural keyword spamming ($\ge 12$ repetitions and $> 6\%$ token density).
- **Automatic Shortlist Exclusion:** Recruiters can block flagged resumes from entering the shortlisted roster with a single sidebar toggle.

### 3. Recruiter Workflow & Automated Shortlist Export
- **Live Cut-Off Control:** Moving the sidebar **Shortlist Cut-Off Score (%)** slider dynamically updates candidate tiers, filters the **Shortlisted Candidates Table**, and regenerates the downloadable PDF roster immediately.
- **Smart Entity Resolution:** Automatically resolves candidate names even when multi-column PDF layouts place job headlines above the candidate's name.

---

## System Architecture

```text
[Batch PDF Upload] ──► [Text & Micro-Font Extractor (pypdf)]
                              │
                              ├──► [Anti-Tampering & Prompt Injection Audit]
                              │
                              ├──► [Contact & Profile Parser (Regex + Header Heuristics)]
                              │
                              ▼
             [4-Signal Hybrid Scoring Engine]
      ┌──────────────┬──────────────┬──────────────┐
      ▼              ▼              ▼              ▼
 Word TF-IDF   Char N-Grams   all-MiniLM-L6   Logistic Reg.
   (25.0%)        (15.0%)        (20.0%)         (40.0%)
      └──────────────┴──────────────┴──────────────┘
                              │
                              ▼
         [Dynamic Threshold Filter (>= Cut-Off %)]
               │                            │
               ▼                            ▼
  [Shortlisted PDF Roster]     [Gemini Recruiter Evaluation]
```

---

## Tech Stack

| Layer | Technologies |
| :--- | :--- |
| **Frontend & State Management** | Streamlit (`st.session_state`, `@st.cache_resource`) |
| **Machine Learning & NLP** | Scikit-Learn (`TfidfVectorizer`, `StandardScaler`, `LogisticRegression`), `sentence-transformers` (`all-MiniLM-L6-v2`), NumPy, Pandas |
| **Generative AI Integration** | Google GenAI SDK (`google-genai`) with automatic fallback routing |
| **PDF Parsing & Report Generation** | `pypdf` (coordinate visitor auditing), `fpdf2` (shortlisted roster export) |
| **Artifact Persistence** | `joblib` (signature-validated model cache `ats_cache.joblib`) |

---

## Local Setup & Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/<harismujahid-developer/ai-ats-resume-matcher.git
   cd <your-repo-name>
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Configure environment variables**
   Create a `.env` file in the root directory based on `.env.example` and add your Gemini API key:
   ```env
   GEMINI_API_KEY=your_actual_gemini_api_key
   ```

4. **Run the application**
   ```bash
   streamlit run app.py
   ```
