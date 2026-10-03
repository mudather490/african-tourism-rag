# African Tourism RAG System

A Retrieval-Augmented Generation (RAG) chatbot that answers questions about tourism in **Egypt**, **Rwanda**, and **South Africa**, using a custom CSV knowledge base, local embeddings, and a local LLM.

Built with **LangChain**, **Ollama**, and **ChromaDB**.

---

## 📌 Why I Built This RAG System Myself

Large language models like `llama3.2` are trained on general internet data. If you ask one:

> "What is the best time to visit Rwanda for gorilla trekking?"

…it might answer from memory, guess, or hallucinate. It has **no access** to my curated dataset, and its knowledge may be outdated, vague, or simply wrong for my use case.

Instead of fine-tuning a model (which is expensive and slow), I built a **RAG system**. RAG means:

> **Retrieve** relevant facts from my own documents → **Augment** the prompt with them → **Generate** an answer grounded in those facts.

### The problems RAG solves for me

| Problem | How RAG fixes it |
|---|---|
| LLM hallucinates facts | The answer is grounded in retrieved CSV rows |
| LLM knowledge is outdated | I control the data — update the CSV, update the answers |
| Fine-tuning is expensive | No training required, just retrieval |
| Answers are generic | Answers are specific to my tourism dataset |
| No source of truth | Every answer traces back to a row in my CSV |

### Why I did it *myself* (not with a ready-made tool)

1. **Learning** — building retrieval, chunking, and prompting by hand teaches how RAG actually works.
2. **Control** — I decide chunk size, embedding model, retrieval strategy, and prompt.
3. **Local & private** — everything runs on my machine via Ollama. No API keys, no data leaving my laptop.
4. **Custom domain** — my CSV is a small, focused dataset about three countries. A general chatbot would never answer these specific questions well.

### The architecture
User Question
│
▼
┌──────────────────┐
│ Retriever │ ← Chroma vector DB
│ (top-k chunks) │
└──────────────────┘
│
▼
┌──────────────────┐
│ Prompt Template │ ← context + question
└──────────────────┘
│
▼
┌──────────────────┐
│ Ollama LLM │ ← llama3.2
│ (generation) │
└──────────────────┘
│
▼
Final Answer

text

### The two files

| File | Role |
|---|---|
| `vector.py` | Loads CSV → splits into chunks → embeds → stores in ChromaDB → exposes retriever |
| `mean.py` | Builds the RAG chain (prompt + LLM + parser) and runs the interactive chat |

---

## ✂️ What Is Chunking (and Why It Matters)

### The problem

My CSV has **48 rows**, each describing one tourism item (a place, activity, or tip). Some rows are short, some are long. I cannot:

- Send the **entire CSV** to the LLM every time (too many tokens, slow, expensive).
- Send **one row per query** (too rigid — a question might span multiple rows).

So I split the data into **small, overlapping pieces** called **chunks**. Each chunk is stored as a vector and can be retrieved independently.

### How chunking works in my project

```python
from langchain_text_splitters import RecursiveCharacterTextSplitter

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,      # max characters per chunk
    chunk_overlap=100,   # characters shared between neighbouring chunks
)

chunks = text_splitter.split_documents(documents)
chunk_size=500 → each chunk is roughly 500 characters (about 80–100 words).

chunk_overlap=100 → the last 100 characters of one chunk are repeated at the start of the next.

Why overlapping matters
Imagine a sentence is split in the middle:

text
Chunk A: "...The best time to visit Rwanda is the dry season from"
Chunk B: "June to September. Permits must be booked in advance..."
If a user asks "When is the dry season in Rwanda?", neither chunk alone has the full answer. With overlap, chunk A ends with "...dry season from June to September" and chunk B starts with "June to September..." — so the answer survives in at least one chunk.

How I chose the numbers
Parameter	Too small	Too large	My choice
chunk_size	Loses context, answers feel broken	Retrieval becomes noisy, mixes topics	500 — matches one CSV row nicely
chunk_overlap	Splits sentences and loses meaning	Wastes storage, duplicate results	100 — about 20% of chunk size
Rule of thumb: overlap should be 10–20% of chunk size.

The chunking pipeline
text
CSV row  ──▶  Document object  ──▶  RecursiveCharacterTextSplitter
                                            │
                    ┌───────────────────────┴───────────────────────┐
                    ▼                                               ▼
                Chunk 1  ──▶ embedding ──▶ vector                   Chunk 2  ──▶ embedding ──▶ vector
                                            │                                               │
                                            └───────────▶ ChromaDB  ◀───────────────────────┘
What the splitter actually does
RecursiveCharacterTextSplitter tries to split on natural boundaries, in this order:

Paragraph break (\n\n)

Single newline (\n)

Space

Character (last resort)

This means chunks stay semantically meaningful instead of cutting words in half.

Tuning chunking
If retrieval results are poor, try:

python
# Smaller chunks → more precise, but may lose context
RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=50)

# Larger chunks → more context, but noisier retrieval
RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=150)
Always re-run evaluation after changing chunk size — see the next section.

📋 Logging Every Experiment
Track each run in a results.csv:

csv
run,chunk_size,k,embedding_model,faithfulness,relevance,correctness,hit_rate
1,500,4,embeddinggemma,0.78,0.85,0.62,0.70
2,300,4,embeddinggemma,0.82,0.88,0.71,0.80
3,300,6,nomic-embed-text,0.86,0.90,0.79,0.90
After 5–10 runs, the winning configuration becomes obvious.

🚀 How to Run
1. Prerequisites
Install Ollama and pull the models:

bash
ollama pull llama3.2
ollama pull embeddinggemma
2. Install Python dependencies
bash
pip install -r requirements.txt
3. Build the vector database
bash
python vector.py
4. Run the chatbot
bash
python mean.py
5. (Optional) Evaluate
bash
python evaluate_manual.py
python evaluate_retrieval.py
python evaluate_llm_judge.py
📂 Project Structure
text
PythonProject/
├── mean.py                     # RAG chain + interactive chat
├── vector.py                   # CSV → chunks → embeddings → ChromaDB
├── evaluate_manual.py          # Human-in-the-loop evaluation
├── evaluate_retrieval.py       # Retrieval hit-rate test
├── evaluate_llm_judge.py       # Automated LLM-as-a-judge evaluation
├── evaluate_ragas.py           # RAGAS metrics (optional)
├── African Tourism Dataset.csv # Knowledge base (48 rows)
├── eval_questions.csv          # Test questions + expected answers
├── results.csv                 # Experiment log (optional)
├── requirements.txt
├── .gitignore
├── chroma_db/                  # Auto-generated (NOT committed)
└── README.md
🧠 Summary
Why RAG? To ground LLM answers in my own tourism dataset and avoid hallucination.

Why chunking? The LLM cannot receive the whole CSV; chunks let me retrieve only the most relevant rows, with overlap preserving context across boundaries.

Why evaluate? Without measurement, I cannot know if retrieval, the prompt, or the LLM is failing. Evaluation turns guesses into numbers.

The loop: test → measure → change one thing → re-measure → keep or revert.

📜 License
Personal learning project. Free to reuse and adapt.

text

---

## ✅ How to use this file

1. Create the file `D:\PythonProject\README.md`
2. Paste everything above
3. Commit and push:

```bash
cd D:\PythonProject
git add README.md
git commit -m "Add README explaining RAG, chunking, and evaluation"
git push
Open your repo on GitHub — the README will render automatically on the front page.



