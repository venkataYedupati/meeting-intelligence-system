# 🧠 Meeting Intelligence System

An end-to-end NLP + ML Engineering project that transforms raw meeting transcripts into structured insights including action items, decisions, topics, semantic search, and summaries.

---

## 🚀 Features

* 📄 Transcript preprocessing and cleaning
* ✅ Action item extraction
* 📌 Decision detection
* 🧩 Topic segmentation
* 🔍 Semantic search (Sentence Transformers + reranking)
* 📝 Automated meeting summary generation
* ⚡ FastAPI backend (production-style API)
* 🌐 Streamlit frontend (interactive UI)
* 🧪 Unit + API testing with pytest
* 🤖 Agentic meeting analysis with specialist agents
* ⚠️ Risk, blocker, and dependency detection
* 🧭 Execution plan with owners, deadlines, priority, and evidence
* ❓ Follow-up question generation and output quality scoring
* 🧱 Offline retrieval fallback for reliable local demos and tests

---

## 🏗️ System Architecture

```
Transcript Input
      ↓
Preprocessing (cleaning, parsing)
      ↓
Structured Records (speaker + text)
      ↓
-----------------------------------
| Action Items | Decisions | Topics |
-----------------------------------
      ↓
Semantic Search (Sentence Transformers)
      ↓
Summary Generation
      ↓
Agentic Orchestrator
      ↓
---------------------------------------------------------
| Participant | Action Plan | Decisions | Risks | Quality |
| Profiler    | Agent       | Agent     | Agent | Agent   |
---------------------------------------------------------
      ↓
Final Structured JSON Output
```

---

## 🤖 Agentic Upgrade Approach

The advanced version keeps the original deterministic NLP pipeline as the baseline and adds an agentic orchestration layer on top.

### Why this design

* The original project is easy to understand, but it runs as one linear pipeline.
* The upgraded version separates responsibilities into specialist agents.
* Each agent has a role, output key, confidence score, evidence count, and trace entry.
* The default implementation is local and reproducible, so it works without an API key.
* The structure is ready for an LLM-backed planner or tool-using agents later.

### Agents added

* **Participant Profiler Agent**: speaker contribution, action ownership, decision contribution
* **Action Planning Agent**: converts raw action items into owner, due date, priority, status, confidence, and evidence
* **Decision Register Agent**: normalizes decisions with topic, status, downstream actions, and evidence
* **Risk and Dependency Agent**: detects blockers, risks, dependencies, unclear items, and severity
* **Query Answer Agent**: synthesizes a cited answer from structured outputs and retrieval results
* **Follow-up Question Agent**: creates next questions from missing owners, deadlines, and unresolved risks
* **Quality Gate Agent**: scores output completeness and lists analysis gaps

### Agentic output sections

```json
{
  "agentic": {
    "mode": "deterministic_multi_agent",
    "participants": [],
    "answer": {},
    "execution_plan": [],
    "decision_register": [],
    "risks": [],
    "follow_up_questions": [],
    "quality": {},
    "agent_trace": []
  }
}
```

---

## 📂 Project Structure

```
meeting-intelligence-system/
│
├── app/
│   ├── api.py              # FastAPI backend
│   ├── streamlit_app.py   # Streamlit UI
│   ├── schemas.py         # Pydantic models
│   └── __init__.py
│
├── src/
│   ├── agentic.py
│   ├── pipeline.py
│   ├── preprocess.py
│   ├── search.py
│   ├── topics.py
│   ├── summary.py
│   ├── action_items.py
│   ├── decisions.py
│   └── output_formatter.py
│
├── tests/
│   ├── test_preprocess.py
│   ├── test_api.py
│   └── conftest.py
│
├── data/
│   ├── raw/
│   └── processed/
│
├── requirements.txt
├── main.py
└── README.md
```

---

## ⚙️ Setup Instructions

### 1. Clone the repository

```
git clone https://github.com/venkatasai0234/meeting-intelligence-system.git
cd meeting-intelligence-system
```

### 2. Create virtual environment

```
python3 -m venv venv
source venv/bin/activate
```

### 3. Install dependencies

```
pip install -r requirements.txt
```

---

## ▶️ Run the Project

### Run main agentic pipeline

```
python main.py
```

The CLI now saves the full agentic output to:

```
data/processed/meeting1_full_output.json
```

---

### Run FastAPI server

```
uvicorn app.api:app --reload
```

Open:

```
http://127.0.0.1:8000/docs
```

---

### Run Streamlit UI

```
streamlit run app/streamlit_app.py
```

---

## 📡 API Usage

### POST `/analyze`

Request:

```
{
  "transcript": "John: We should send the proposal...",
  "query": "What was decided?",
  "top_k": 3
}
```

### POST `/analyze-agentic`

Request:

```
{
  "transcript": "John: We should send the proposal...",
  "query": "What actions are open?",
  "top_k": 5
}
```

Returns the original summary, action items, decisions, topics, and search results plus the `agentic` section.

---

### POST `/analyze-file`

* Upload a `.txt` transcript file
* Optional query parameter

### POST `/analyze-file-agentic`

* Upload a `.txt` transcript file
* Optional query and `top_k` parameters

---

## 🔍 Retrieval Backend

The project defaults to local TF-IDF retrieval so tests and demos work offline.

To use Sentence Transformers:

```
MEETING_INTEL_SEARCH_BACKEND=sentence-transformers uvicorn app.api:app --reload
```

If the transformer backend cannot load, the app falls back to local retrieval.

---

## 🧪 Run Tests

```
pytest
```

---

## Example Output

```
{
  "summary": {
    "overview": "The meeting focused on proposal, client review, and demo.",
    "key_decisions": [
      "The team agreed to delay the product demo until Thursday."
    ],
    "key_action_items": [
      "John: We should send the updated proposal by Friday."
    ]
  },
  "action_items": [...],
  "decisions": [...],
  "topics": [...],
  "search_results": [...]
}
```

---

## 🔥 Key ML Concepts Used

* NLP preprocessing (text cleaning, parsing)
* Rule-based baseline systems
* Sentence embeddings (Sentence-BERT)
* Semantic similarity (cosine similarity)
* Retrieval + reranking strategy
* Structured information extraction
* API design with Pydantic models
* End-to-end ML pipeline design

---

## 🎯 Future Improvements

* Replace rule-based extraction with ML models
* Add real-time audio transcription (speech-to-text)
* Use FAISS for scalable vector search
* Add user feedback loop for model improvement
* Dockerize and deploy to cloud (AWS/GCP/Azure)

---

## 💼 Why this project is strong

This project demonstrates:

* End-to-end ML system design
* NLP + retrieval + API integration
* Production-style architecture
* Real-world use case (meeting intelligence systems like Otter.ai)

## 👨‍💻 Author

**Venkata Siva Sai Krishna Prasad Yedupati**
Master’s in Computer Science — San Jose State University
