# 📚 Naive RAG PDF Chatbot

A **Retrieval-Augmented Generation (RAG)** application that lets you chat with your PDF documents.
Upload PDFs, ask questions in natural language, and get grounded answers with source citations.

![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python)
![Streamlit](https://img.shields.io/badge/Streamlit-1.30+-red?logo=streamlit)
![ChromaDB](https://img.shields.io/badge/ChromaDB-Vector_DB-orange)
![Groq](https://img.shields.io/badge/Groq-LLM_API-green)
![License](https://img.shields.io/badge/License-MIT-yellow)

---

## 📋 Table of Contents

- [Overview](#-overview)
- [Features](#-features)
- [How It Works](#-how-it-works)
- [Tech Stack](#-tech-stack)
- [Project Structure](#-project-structure)
- [Setup and Installation](#-setup-and-installation)
- [Usage](#-usage)
- [Example Queries](#-example-queries)
- [Limitations](#-limitations-honest)
- [Roadmap](#-roadmap-advanced-rag)
- [What I Learned](#-what-i-learned)
- [License](#-license)

---

## 🎯 Overview

This is a **Naive RAG** (Retrieval-Augmented Generation) implementation — the foundational approach to building AI systems that answer questions from your own documents.

**The core idea:**

> Give an LLM a search engine, so it looks things up before answering instead of relying only on its training memory.

**Why RAG matters:**

- LLMs have outdated knowledge (training cutoff)
- LLMs hallucinate (confidently make things up)
- LLMs have no access to your private data (docs, emails, databases)

**RAG solves this** by retrieving relevant chunks from your documents and feeding them to the LLM as context before generating an answer.

---

## ✨ Features

- 📄 **Upload multiple PDFs** — process them into a searchable knowledge base
- 💬 **Natural language Q&A** — ask questions in plain English
- 📎 **Source citations** — every answer shows which chunks it came from
- 🔍 **Retrieved chunks transparency** — see exactly what the retriever found
- 📊 **Distance scores** — understand similarity between query and chunks
- ⚙️ **Adjustable Top-K** — control how many chunks to retrieve per query
- 🎨 **Clean UI** — built with Streamlit, custom CSS styling
- 💾 **Persistent storage** — ChromaDB saves embeddings to disk

---

## 🏗️ How It Works

This is a **naive RAG** pipeline with two phases:

### Phase 1: Indexing (Offline, one-time)

PDFs ➜ Chunking ➜ Embeddings ➜ ChromaDB

1. **Load** — Extract text from PDFs using `pypdf`
2. **Chunk** — Split text into 500-character chunks with 50-char overlap
3. **Embed** — Convert each chunk to a 384-dim vector using `all-MiniLM-L6-v2`
4. **Store** — Save chunks + embeddings + metadata in ChromaDB

### Phase 2: Querying (Online, per question)

Query ➜ Embedding ➜ Top-K Search ➜ LLM ➜ Answer

1. **Embed query** — Same embedding model as chunks
2. **Retrieve** — Vector similarity search in ChromaDB, get top-K chunks
3. **Augment** — Combine chunks + query into a prompt
4. **Generate** — Groq LLM generates answer using ONLY the context

### Visual Flow

┌─────────────────── PHASE 1: INDEXING ───────────────────┐
│                                                          │
│   Documents  ➜  Chunking  ➜  Embeddings  ➜  ChromaDB     │
│    (PDFs)      (500 chars)   (384-dim)      (Vector DB)  │
│                                                          │
└──────────────────────────────────────────────────────────┘

┌─────────────────── PHASE 2: QUERYING ───────────────────┐
│                                                          │
│   Question  ➜  Embed  ➜  Retrieve  ➜  LLM  ➜  Answer     │
│    ("Who?")   (vector)   (top-5)     (Groq)   (text)     │
│                                                          │
└──────────────────────────────────────────────────────────┘

---

## 🛠️ Tech Stack

| Component | Tool | Why |
|---|---|---|
| **Embeddings** | sentence-transformers (`all-MiniLM-L6-v2`) | Free, local, fast, 384-dim |
| **Vector DB** | ChromaDB | Local, persistent, easy setup |
| **LLM** | Groq API (`openai/gpt-oss-20b`) | Free tier, very fast inference |
| **Chunking** | LangChain `RecursiveCharacterTextSplitter` | Smart splitting |
| **PDF parsing** | `pypdf` | Simple, reliable |
| **UI** | Streamlit | Fast Python UI for demos |
| **Env management** | `python-dotenv` | API key security |

**Why Groq?** Free tier is generous, inference is blazing fast (~1000 tokens/sec), and no local GPU is needed. Perfect for a portfolio project.

---

## 📁 Project Structure

naive-rag-app/
├── app.py                  # Streamlit UI
├── rag.py                  # RAG logic (NaiveRAG class)
├── requirements.txt        # Python dependencies
├── README.md               # This file
├── .env                    # API keys (NOT in git)
├── .gitignore              # Git ignore rules
│
├── data/                   # Uploaded PDFs (NOT in git)
│   └── *.pdf
│
├── chroma_db/              # Vector DB storage (NOT in git)
│   └── ...
│
└── screenshots/            # README images
    ├── main.png
    └── chunks.png

---

## 🚀 Setup and Installation

### Prerequisites

- **Python 3.10+**
- **Groq API key** (free) — [Get one here](https://console.groq.com)

### Step 1: Clone the repository

git clone https://github.com/YOUR_USERNAME/naive-rag-app.git
cd naive-rag-app

### Step 2: Create a virtual environment

# Windows
python -m venv venv
venv\Scripts\activate

# Mac/Linux
python3 -m venv venv
source venv/bin/activate

### Step 3: Install dependencies

pip install -r requirements.txt

### Step 4: Configure API key

Create a `.env` file in the project root:

GROQ_API_KEY=gsk_your_actual_key_here

> ⚠️ **Important:** Never commit `.env` to GitHub. It's already listed in `.gitignore`.

### Step 5: Run the app

streamlit run app.py

The browser will open at **http://localhost:8501**

---

## 💡 Usage

1. **Upload PDFs** using the sidebar file uploader.
2. Click **Process Documents** to index them into ChromaDB.
3. Type your question in the chat input.
4. View the **answer**, **source chunks**, and **distance scores** below.
5. Adjust **Top-K** in the sidebar to control retrieval depth.

---

## 🔎 Example Queries

- *"What is the main argument of this paper?"*
- *"Summarize section 3 in simple terms."*
- *"List all the key findings mentioned in the document."*
- *"What does the author say about X?"*
- *"Give me a quote about Y."*

---

## ⚠️ Limitations (Honest)

This is a **naive RAG** — intentionally simple. Known weaknesses:

- **No reranking** — retrieval is pure vector similarity, no cross-encoder refinement
- **Fixed chunking** — 500 chars may split sentences or ideas awkwardly
- **Single embedding model** — no hybrid (BM25 + dense) search
- **No query rewriting** — vague questions stay vague
- **No multi-hop reasoning** — cannot chain multiple retrievals
- **Context window limits** — very large Top-K may overflow the LLM prompt
- **PDF-only** — no support for DOCX, HTML, or web pages yet

These are **features of the naive approach**, not bugs — they're the baseline to improve upon.

---

## 🗺️ Roadmap (Advanced RAG)

Planned improvements to evolve this into an **Advanced RAG** system:

- [ ] **Hybrid search** — combine BM25 (keyword) + dense (vector) retrieval
- [ ] **Reranking** — add a cross-encoder (e.g., `bge-reranker`) after retrieval
- [ ] **Query rewriting** — use the LLM to expand/clarify user queries
- [ ] **Parent-document retrieval** — retrieve small chunks, return larger parents
- [ ] **Multi-query RAG** — generate multiple query variants, merge results
- [ ] **Metadata filtering** — filter by source file, page, or date
- [ ] **Evaluation harness** — measure retrieval hit-rate and answer faithfulness
- [ ] **Multi-format support** — DOCX, HTML, Markdown, URLs
- [ ] **Streaming answers** — token-by-token output in the UI

---

## 🧠 What I Learned

Building this project taught me:

- **RAG fundamentals** — why retrieval beats fine-tuning for private data
- **Embeddings** — how semantic similarity actually works in vector space
- **Vector databases** — indexing, persistence, and similarity search with ChromaDB
- **Prompt engineering** — grounding an LLM with context and forcing citations
- **Pipeline design** — separating indexing from querying for efficiency
- **Honest limitations** — knowing *why* naive RAG fails is the first step to fixing it

---

## 📜 License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.

---

<div align="center">

**⭐ If you found this useful, give it a star!**

Made with ❤️ using Streamlit, ChromaDB, and Groq

</div>
