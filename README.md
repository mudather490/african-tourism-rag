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

🧪 How to Evaluate the Model
Evaluating a RAG system means evaluating two things separately:

Retrieval — did we fetch the right chunks?

Generation — did the LLM answer correctly using those chunks?

If either fails, the answer is bad. Here is how I evaluate my system, from simplest to most advanced.

Level 1 — Manual Evaluation (start here)
Build a small test set of questions with known answers, then check answers by hand.

eval_questions.csv:

csv
question,expected_answer,country
What is the best time to visit Rwanda for gorilla trekking?,Dry seasons: June to September and December to February.,Rwanda
How much does a gorilla permit cost in Rwanda?,US$1,500 with a 30% discount in the offseason.,Rwanda
What are the top attractions in Egypt?,Giza Pyramid Complex Luxor Temple Abu Simbel Grand Egyptian Museum.,Egypt
What is the best place in Egypt to visit?,Giza Pyramid Complex and the Grand Egyptian Museum.,Egypt
What is the best time to visit South Africa for safaris?,June to October is peak safari season.,South Africa
Which South African park has the Big Five?,Kruger National Park.,South Africa
Where can you see African penguins in South Africa?,Boulders Beach near Cape Town.,South Africa
What is the Garden Route?,A scenic drive from Mossel Bay through Knysna to Tsitsikamma National Park.,South Africa
evaluate_manual.py:

python
import csv
from mean import rag_chain

def evaluate():
    with open("eval_questions.csv", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    results = []
    for r in rows:
        q = r["question"]
        expected = r["expected_answer"]
        answer = rag_chain.invoke(q)

        print("=" * 60)
        print(f"Q: {q}")
        print(f"Expected: {expected}")
        print(f"Got:      {answer}")
        verdict = input("Correct? (y/n/partial): ").strip().lower()
        results.append(verdict)

    correct = results.count("y")
    partial = results.count("partial")
    total = len(results)

    print("\n===== SUMMARY =====")
    print(f"Correct:  {correct}/{total}")
    print(f"Partial:  {partial}/{total}")
    print(f"Accuracy: {(correct + 0.5 * partial) / total:.2%}")

if __name__ == "__main__":
    evaluate()
How to read the result:

Accuracy	Meaning
Below 60%	Something is broken — check retrieval first
60–75%	Working but weak — tune chunk size or k
75–85%	Good — usable system
85–95%	Very good
Above 95%	Excellent
Level 2 — Evaluate Retrieval Separately
The retriever is usually the real problem. Test it alone.

evaluate_retrieval.py:

python
from vector import retriever

tests = [
    ("What is the best time to visit Rwanda?", "June to September"),
    ("How much does a gorilla permit cost?", "1,500"),
    ("Where can I see penguins in South Africa?", "Boulders Beach"),
    ("What is the best place in Egypt to visit?", "Giza"),
    ("Which park has the Big Five?", "Kruger"),
]

hits = 0
for question, must_contain in tests:
    docs = retriever.invoke(question)
    combined = "\n".join(d.page_content for d in docs)

    if must_contain.lower() in combined.lower():
        print(f"HIT  | {question}")
        hits += 1
    else:
        print(f"MISS | {question}  (looking for '{must_contain}')")

print(f"\nRetrieval hit rate: {hits}/{len(tests)} = {hits/len(tests):.2%}")
Key retrieval metrics:

Metric	What it means
Hit Rate	Did any correct chunk appear? (simplest)
Precision@k	Of top-k chunks, how many are relevant?
Recall@k	Of all relevant chunks, how many did we find?
MRR	How high did the first correct chunk rank?
For a beginner project, Hit Rate is enough.

Fixes if hit rate is low:

python
# 1. Increase k
retriever = vectorstore.as_retriever(search_kwargs={"k": 8})

# 2. Use MMR (diverse results)
retriever = vectorstore.as_retriever(
    search_type="mmr",
    search_kwargs={"k": 4, "fetch_k": 10},
)

# 3. Better embedding model
embeddings = OllamaEmbeddings(model="nomic-embed-text")

# 4. Smaller chunks
RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=50)
Level 3 — LLM-as-a-Judge (Automated)
Let an LLM grade each answer instead of you doing it by hand. I score three things:

Metric	Question it answers	Score
Faithfulness	Is the answer supported by the retrieved chunks?	0–1
Relevance	Does the answer actually address the question?	0–1
Correctness	Does the answer match the expected answer?	0–1
evaluate_llm_judge.py:

python
import csv
from langchain_ollama.llms import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from mean import rag_chain, retriever

judge = OllamaLLM(model="llama3.2")

judge_prompt = ChatPromptTemplate.from_template("""
You are a strict evaluator of a RAG system about African tourism.

Question: {question}
Retrieved context: {context}
Model answer: {answer}
Expected answer: {expected}

Grade on THREE criteria, each from 0 to 1:
1. Faithfulness — is the model answer fully supported by the context? (no hallucination)
2. Relevance — does the model answer address the question?
3. Correctness — does the model answer match the expected answer?

Reply EXACTLY in this format:
Faithfulness: <number>
Relevance: <number>
Correctness: <number>
Reason: <one sentence>
""")

judge_chain = judge_prompt | judge | StrOutputParser()


def parse_scores(text):
    scores = {"Faithfulness": 0, "Relevance": 0, "Correctness": 0}
    for line in text.splitlines():
        for key in scores:
            if line.strip().startswith(key + ":"):
                try:
                    scores[key] = float(line.split(":")[1].strip())
                except ValueError:
                    pass
    return scores


def run():
    with open("eval_questions.csv", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    totals = {"Faithfulness": 0, "Relevance": 0, "Correctness": 0}

    for r in rows:
        q, expected = r["question"], r["expected_answer"]

        docs = retriever.invoke(q)
        context = "\n\n".join(d.page_content for d in docs)
        answer = rag_chain.invoke(q)

        verdict = judge_chain.invoke({
            "question": q, "context": context,
            "answer": answer, "expected": expected,
        })
        scores = parse_scores(verdict)
        for k in totals:
            totals[k] += scores[k]

        print("=" * 60)
        print(f"Q: {q}")
        print(f"A: {answer[:200]}")
        print(verdict)

    n = len(rows)
    print("\n===== AVERAGE SCORES =====")
    for k, v in totals.items():
        print(f"{k}: {v/n:.2f} / 1.00")


if __name__ == "__main__":
    run()
How to read the scores:

If low...	The problem is...	Fix
Faithfulness	The LLM hallucinates	Tighten the prompt; reduce chunk overlap
Relevance	The answer misses the point	Better prompt; stronger LLM
Correctness	Retrieval or reasoning is wrong	Fix retrieval first, then prompt
Level 4 — RAGAS (Standard Framework)
RAGAS automates the metrics above with industry-standard implementations.

bash
pip install ragas datasets
python
# evaluate_ragas.py
import csv
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_precision, context_recall
from langchain_ollama import OllamaEmbeddings
from langchain_ollama.llms import OllamaLLM
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from mean import rag_chain, retriever

judge_llm = LangchainLLMWrapper(OllamaLLM(model="llama3.2"))
judge_emb = LangchainEmbeddingsWrapper(OllamaEmbeddings(model="embeddinggemma"))

with open("eval_questions.csv", encoding="utf-8") as f:
    rows = list(csv.DictReader(f))

questions, answers, contexts, ground_truths = [], [], [], []

for r in rows:
    q = r["question"]
    docs = retriever.invoke(q)
    questions.append(q)
    answers.append(rag_chain.invoke(q))
    contexts.append([d.page_content for d in docs])
    ground_truths.append(r["expected_answer"])

dataset = Dataset.from_dict({
    "question": questions, "answer": answers,
    "contexts": contexts, "ground_truth": ground_truths,
})

result = evaluate(
    dataset,
    metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
    llm=judge_llm,
    embeddings=judge_emb,
)
print(result)
What each RAGAS metric means:

Metric	Measures	Fix if low
faithfulness	Answer grounded in context	Reduce overlap; tighten prompt
answer_relevancy	Answer addresses the question	Better prompt; stronger LLM
context_precision	Retrieved chunks are relevant	Better retriever; smaller k
context_recall	All needed info was retrieved	Larger k; better embeddings
🔁 The Evaluation Loop
This is the real skill — not just running evaluation once.

text
1. Build test set (20–50 Q&A pairs)
        │
        ▼
2. Run manual eval → baseline accuracy
        │
        ▼
3. Run retrieval eval → find weak spots
        │
        ▼
4. Run LLM-judge or RAGAS → per-metric scores
        │
        ▼
5. Change ONE variable (k, chunk size, model, prompt)
        │
        ▼
6. Re-run all evals → compare
        │
        ▼
7. Keep if better, revert if worse → repeat
Rule: change only one variable at a time. Otherwise you won't know what helped.

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



