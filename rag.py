# rag.py
# Naive RAG — PDF Chatbot (Groq version, no Ollama)

import os
import chromadb
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
from langchain_text_splitters import RecursiveCharacterTextSplitter
from groq import Groq
from dotenv import load_dotenv

# .env file se API key load karo
load_dotenv()


class NaiveRAG:
    def __init__(self, collection_name="documents"):
        # ---------- 1. EMBEDDING MODEL ----------
        print("Loading embedding model...")
        self.embedder = SentenceTransformer('all-MiniLM-L6-v2')
        print("✅ Embedding model loaded")

        # ---------- 2. VECTOR DB ----------
        print("Connecting to ChromaDB...")
        self.client = chromadb.PersistentClient(path="./chroma_db")
        self.collection = self.client.get_or_create_collection(
            name=collection_name
        )
        print(f"✅ Connected to collection: {collection_name}")

        # ---------- 3. CHUNKING ----------
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
        print("✅ Splitter ready")

        # ---------- 4. LLM CLIENT (Groq) ----------
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise ValueError(
                "GROQ_API_KEY not found. Check your .env file."
            )
        self.groq_client = Groq(api_key=api_key)
        print("✅ Groq client ready")

    # ---------- PDF LOADING ----------
    def load_pdf(self, pdf_path):
        """PDF se poora text nikaalo"""
        reader = PdfReader(pdf_path)
        text = ""
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
        return text

    # ---------- INGEST (PDF → Chunks → Embeddings → Vector DB) ----------
    def ingest(self, pdf_path):
        """PDF ko process karke vector DB mein store karo"""
        # 1. Text nikaalo
        print(f"📖 Reading {pdf_path}...")
        text = self.load_pdf(pdf_path)

        if not text.strip():
            print("⚠️  No text found in PDF")
            return 0

        # 2. Chunks banao
        chunks = self.splitter.split_text(text)
        print(f"✂️  Created {len(chunks)} chunks")

        # 3. Embeddings banao
        print("🔢 Creating embeddings...")
        embeddings = self.embedder.encode(chunks).tolist()

        # 4. Vector DB mein store karo
        doc_name = os.path.basename(pdf_path)
        ids = [f"{doc_name}_chunk_{i}" for i in range(len(chunks))]
        metadatas = [
            {"source": doc_name, "chunk_id": i}
            for i in range(len(chunks))
        ]

        # Pehle se exist karta hai to delete karo (re-upload ke liye)
        try:
            existing = self.collection.get(where={"source": doc_name})
            if existing and existing['ids']:
                self.collection.delete(ids=existing['ids'])
                print(f"🗑️  Removed {len(existing['ids'])} old chunks")
        except Exception:
            pass

        self.collection.add(
            ids=ids,
            embeddings=embeddings,
            documents=chunks,
            metadatas=metadatas
        )

        print(f"✅ Stored {len(chunks)} chunks from {doc_name}")
        return len(chunks)

    # ---------- RETRIEVE (Query → Relevant Chunks) ----------
    def retrieve(self, query, top_k=5):
        """Query ke liye relevant chunks dhundho"""
        # 1. Query ka embedding
        query_embedding = self.embedder.encode([query]).tolist()

        # 2. Vector DB mein search
        results = self.collection.query(
            query_embeddings=query_embedding,
            n_results=top_k
        )

        documents = results['documents'][0] if results['documents'] else []
        metadatas = results['metadatas'][0] if results['metadatas'] else []
        distances = results['distances'][0] if results['distances'] else []

        return documents, metadatas, distances

    # ---------- GENERATE (Retrieve + Augment + Generate) ----------
    def generate(self, query, top_k=5):
        """Poora RAG pipeline: retrieve + augment + generate"""
        # 1. RETRIEVE
        documents, metadatas, distances = self.retrieve(query, top_k)

        if not documents:
            return {
                'answer': "I don't have any documents to search from.",
                'sources': [],
                'chunks': [],
                'distances': []
            }

        # 2. AUGMENT
        context = "\n\n---\n\n".join(documents)

        prompt = f"""Answer the question using ONLY the context below.
If the answer is not in the context, respond with: "I don't know based on the provided documents."
Do not use any outside knowledge.

CONTEXT:
{context}

QUESTION: {query}

ANSWER:"""

        # 3. GENERATE (Groq)
        print("🤔 Generating answer via Groq...")
        response = self.groq_client.chat.completions.create(
            model="openai/gpt-oss-20b",  # ✅ Naya model
            messages=[
                {"role": "user", "content": prompt}
            ],
            temperature=0.1,
            max_tokens=1024
        )

        answer = response.choices[0].message.content

        return {
            'answer': answer,
            'sources': metadatas,
            'chunks': documents,
            'distances': distances
        }
        """Poora RAG pipeline"""
        # 1. RETRIEVE
        documents, metadatas, distances = self.retrieve(query, top_k)

        if not documents:
            return {
                'answer': "I don't have any documents to search from. Please upload a PDF first.",
                'sources': [],
                'chunks': [],
                'distances': []
            }

        # 2. AUGMENT
        context = "\n\n---\n\n".join(documents)

        prompt = f"""Answer the question using ONLY the context below.
If the answer is not in the context, respond with: "I don't know based on the provided documents."
Do not use any outside knowledge.

CONTEXT:
{context}

QUESTION: {query}

ANSWER:"""

        # 3. GENERATE (Groq)
        print("🤔 Generating answer via Groq...")
        response = self.groq_client.chat.completions.create(
            model="openai/gpt-oss-20b",
            messages=[
                {"role": "user", "content": prompt}
            ],
            temperature=0.1,
            max_tokens=1024
        )

        answer = response.choices[0].message.content

        return {
            'answer': answer,
            'sources': metadatas,
            'chunks': documents,
            'distances': distances
        }