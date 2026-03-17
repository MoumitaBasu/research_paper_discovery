# 📚 Research Paper Discovery AI

A production-grade RAG (Retrieval-Augmented Generation) pipeline for deep analysis of academic research papers. This tool combines hybrid search (Semantic + BM25) with high-performance LLMs to ground AI answers in verified research context.

## 🚀 Features

- **Hybrid Retrieval**: Combines FAISS-based semantic search with BM25 keyword matching for maximum recall.
- **Cross-Encoder Reranking**: Utilizes `ms-marco-MiniLM` to prioritize the most relevant paper chunks before generation.
- **Dual AI Engine**:
  - **Local**: `Flan-T5-Base` for offline, private analysis.
  - **Cloud**: `Gemini 1.5 Flash-Latest` for high-reasoning, long-context synthesis.
- **Adaptive Context windowing**: Automatically scales context size based on the model's capabilities (up to 3,000 chars per passage for Gemini).
- **Metadata Anchoring**: Always anchors AI answers on the paper's Abstract and Title to prevent hallucinations.
- **Parse-and-Purge Storage**: Automatically deletes local PDFs after ingestion to save disk space while keeping analyzed text in memory.

## 🛠️ Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/MoumitaBasu/research_paper_discovery.git
   cd research_paper_discovery
   ```

2. **Setup virtual environment**:
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # Mac/Linux
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

## 🖥️ Usage

1. **Start the app**:
   ```bash
   streamlit run app.py
   ```
2. **Configure AI**:
   - For high-quality results, toggle **"Use Gemini"** in the sidebar.
   - Enter your Gemini API key (get it at [aistudio.google.com](https://aistudio.google.com)).
3. **Search & Analyze**:
   - Enter a topic (e.g., "Generative AI in healthcare").
   - Click **"Extract & Run Deep AI"** on any paper.
   - Use the **Neural Insight** chat to ask specific research questions.

## 📦 Tech Stack

- **Frontend**: Streamlit
- **LLMs**: Google Gemini 1.5, Google Flan-T5
- **Vector DB**: FAISS
- **Embeddings**: BAAI/bge-small-en-v1.5
- **Retrieval**: BM25, Cross-Encoders
- **Parsing**: PyMuPDF4LLM

## 📄 License

MIT
