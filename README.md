# 🧠 RAG with Gemini (Simple Implementation)

> 📚 **Original source:** Adapted from an educational RAG project by [Asimov Academy](https://github.com/asimov-academy/rag-in-practice)  
> 🧠 This version was modified to use **Google Gemini API** directly (without LangChain)  
> 📄 Licensed under MIT License (see [LICENSE](LICENSE) file)

**Read in**: [English](README.md) | [Portuguese](README.pt-BR.md)

---

## 🇺🇸 English Version

### 🧩 1. Requirements

- **Python**: >=3.12
- **Libraries**:

```bash
pip install -U google-genai faiss-cpu python-dotenv numpy
```

---

### 🔐 2. Environment Setup

Create a `.env` file in the project root with your Google API key:

```env
GOOGLE_API_KEY=your-api-key-here
```

Or use:

```env
GEMINI_API_KEY=your-api-key-here
```

🔑 **Get your key at**: [Google AI Studio](https://aistudio.google.com/app/apikey)

---

### 📁 3. Document Structure

Place your Markdown (`.md`) files inside the `docs/` folder. The script will:
- Load all `.md` files recursively
- Split them into chunks
- Generate embeddings using `gemini-embedding-001`
- Index with FAISS for semantic search

---

### ⚙️ 4. How to Run

**Activate virtual environment (Windows):**

```bash
python -m venv .venv
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\activate
```

**Install dependencies:**

```bash
pip install -U google-genai faiss-cpu python-dotenv numpy
```

**Run the script:**

```bash
python rag_simple_genai.py
```

**Interact with the assistant:**

```
You: What does this document explain?
Agent AI: [Answer based on retrieved context with source citations]
```

Type `quit`, `exit`, or `sair` to stop.

---

### 💡 5. How It Works

1. **Document Loading**: Reads all `.md` files from `docs/` folder
2. **Chunking**: Splits documents into overlapping chunks (default: 1200 chars with 200 overlap)
3. **Embedding Generation**: Uses `gemini-embedding-001` model to vectorize chunks
4. **FAISS Indexing**: Creates a vector index for fast similarity search
5. **Query Processing**: 
   - Embeds user query
   - Retrieves top-K similar chunks (default: 4)
   - Builds context-aware prompt with chat history
6. **Answer Generation**: Uses `gemini-2.5-flash` model to generate responses
7. **Source Attribution**: Shows which document chunks were used

---

### 🔧 6. Configuration

Edit these constants in `rag_simple_genai.py`:

```python
DOCS_PATH = "./docs"              # Path to markdown files
CHUNK_SIZE = 1200                 # Characters per chunk
CHUNK_OVERLAP = 200               # Overlap between chunks
EMBEDDING_MODEL = "gemini-embedding-001"
GENERATION_MODEL = "gemini-2.5-flash"
TOP_K = 4                         # Number of snippets to retrieve
```

---

### 📖 7. Credits & References

- **Original project**: [Asimov Academy - RAG in Practice](https://github.com/asimov-academy/rag-in-practice)
- **Adapted by**: Using Google Gemini API directly via `google-genai` SDK
- **Key technologies**:
  - `google-genai`: Official Google Generative AI Python client
  - `faiss-cpu`: Vector similarity search
  - `numpy`: Numerical operations

---

### 📄 8. License

This project is licensed under the **MIT License** - see the [LICENSE](LICENSE) file for details.

Original work Copyright (c) 2025 Asimov Academy  
Modifications and adaptations are also covered under MIT License.

---

### 🚀 9. Next Steps

- Add more documents to `docs/` folder
- Adjust chunk size and overlap for your use case
- Experiment with different Gemini models
- Implement persistent storage for embeddings
- Add web interface with Streamlit or Gradio
