# Para manipular arquivos
import os
from pathlib import Path
import json 
# Para calcular vetores
import numpy as np 
# Para busca vetorial
import faiss
# Para o .env
from dotenv import load_dotenv

# Gemini client
from google import genai

# Carrega .env
load_dotenv() 

# Pega a Api Key
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise SystemExit("Error: GEMINI_API_KEY not found in .env")

# Inicializa o cliente
client = genai.Client(api_key=GEMINI_API_KEY)

DEBUG = False 

DOCS_PATH = "./docs" # Pasta com .md
CHUNK_SIZE = 1200 # Caracteres por chunk (ajuste conforme necessario)
CHUNK_OVERLAP = 200 # Overlap entre chunks
EMBEDDING_MODEL = "gemini-embedding-001"
GENERATION_MODEL = "gemini-2.5-flash"
TOP_K = 2 # Quantos trechos do documento usar

def debug_print(message: str):
    if DEBUG:
        print(f"[DEBUG] {message}")

# Cria funcao para carregar os .md
def load_markdown_files(root_path: str):
    root = Path(root_path)
    # Transforma o paht em string e busca recursivamente
    files = list(root.rglob("*.md"))
    docs = []

    for f in files:
        text = f.read_text(encoding="utf-8")
        debug_print(f"Loaded file: {f} ({len(text)} chars)")
        # Cria dicionarios
        docs.append({"path": str(f), "text": text})

    return docs

def chunk_text(text: str, size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP):
    chunks = []
    start = 0
    # Comprimento do vetor
    L = len(text)

    while start < L:
        end = start + size

        # Pega do start ate o end (sem incluir o end)
        chunk = text[start:end]
        chunks.append(chunk)
        # O proximo chunck comeca um pouco antes do final do anterior
        start = end - overlap

        # Evita index negativo
        if start < 0:
            start = 0

    debug_print(f"Created {len(chunks)} chunks from text of {L} chars")
    return chunks

def embed_texts(texts: list[str], model=EMBEDDING_MODEL):
    debug_print(f"Generating embeddings for {len(texts)} texts using {model}")

    # Usa o endpoint da API (embed_content)
    resp = client.models.embed_content(model=model, contents=texts)
    embeddings = []

    # Transforma embeddings array em NumPy array
    for e in resp.embeddings:
        vetor = np.array(e.values, dtype=np.float32)
        embeddings.append(vetor)

    debug_print(f"Generated {len(embeddings)} embeddings with dimension {embeddings[0].shape[0]}")
    return embeddings

def build_faiss_index(embeddings: list[np.ndarray]):
    # Vefica quantos valores o primeiro array tem
    dim = embeddings[0].shape[0]
    debug_print(f"Building FAISS index with {len(embeddings)} vectors of dimension {dim}")
    
    # Cria um index para usar produto interno (Inner product)
    index = faiss.IndexFlatIP(dim)  

    norm_vectors = []
    # Normaliza os vetores
    for v in embeddings:
        # Calcula magnitude, usando norma euclidiana
        norm = np.linalg.norm(v)

        # Divide cada elemento do vetor pelo seu tamanho
        # Resultando em um vetor unitario
        v_norm = v / norm

        # Adiciona na lista
        norm_vectors.append(v_norm)

    # Converte array normalizado em array NumPy 2D
    em_np = np.array(norm_vectors, dtype=np.float32)
    
    index.add(em_np)
    debug_print(f"FAISS index created with {index.ntotal} vectors")

    return index

def normalize_vector(v: np.ndarray):
    # Calcula magnitude do vetor
    norm = np.linalg.norm(v)

    # Normaliza vetor
    if norm > 0:
        # Divide o elemento pela norma
        v_normalized = v / norm 
        # Converte para float32
        v_normalized = v_normalized.astype(np.float32)

        return v_normalized
    else:
        # Apenas converte para float32
        return v.astype(np.float32)

# Preparacao dos documentos
raw_docs = load_markdown_files(DOCS_PATH)
# Lista de textos
all_chunks = []
# Metadados
all_meta = [] 

print(f"Found {len(raw_docs)} markdown files. Chunking...")

# Lista os dicionarios
for doc in raw_docs:  
    # Divide o texto
    chunks = chunk_text(doc["text"])

    # Percorre os chunks
    for i, ch in enumerate(chunks):
        # Remove espacos extras
        snippet = ch.strip()

        # Evita chunks vazios
        if snippet: 
            # Base dos metadados
            meta_data = {
                "source_full": doc["path"],
                "source_short": Path(doc["path"]).name,
                "chunk_id": i,
                "text_preview": snippet[:200]
            }
            
            # Salva chunks e metadados
            all_chunks.append(snippet)
            all_meta.append(meta_data)

print(f"{len(all_chunks)} chunks ready! Generating embeddings (this can take a while)...")

# Quantidade de textos enviados para o embedding
BATCH = 64
embeddings = []

# Gera indices com espacos de 64
for i in range(0, len(all_chunks), BATCH):
    # Seleciona um pedaco da lista
    batch_texts = all_chunks[i:i+BATCH]
    
    debug_print(f"Processing batch {i//BATCH + 1}/{(len(all_chunks) + BATCH - 1)//BATCH}")
    
    # Gera embeddings do batch
    batch_embs = embed_texts(batch_texts)

    # Adiciona a lista
    embeddings.extend(batch_embs)

# Normaliza transformando em valores unitarios
embeddings = [normalize_vector(e) for e in embeddings]
# Cria um index
index = build_faiss_index(embeddings)
print("FAISS index built! Ready to answer queries")

# Conversa
chat_history = []

def retrieve_similar(query: str, top_k=TOP_K):
    debug_print(f"Retrieving top {top_k} similar chunks for query: '{query[:50]}...'")
    
    # Gera embedding da query
    q_emb = embed_texts([query])[0]
    # Normaliza
    q_emb = normalize_vector(q_emb)
    # Pesquisa vetores mais proximos da query
    D, I = index.search(np.array([q_emb]), top_k)
    
    debug_print(f"Similarity scores: {D[0]}")
    debug_print(f"Retrieved indices: {I[0]}")
    
    results = []

    # Pega os indices    
    for idx in I[0]:
        # Ignoras indices invalidos
        if idx < 0 or idx >= len(all_chunks):
            continue

        # Para os indices validos recura chunk e metadados
        results.append({"chunk": all_chunks[idx], "meta": all_meta[idx]})

    debug_print(f"Successfully retrieved {len(results)} chunks")
    return results

def build_prompt(query: str, retrieved: list, history: list):
    context_parts = []

    # Cria um header interativo por chunk
    for i, r in enumerate(retrieved):
        if DEBUG:
            # Modo DEBUG: mostra informacoes completas
            header = f"[Source: {r['meta']['source_full']} | chunk:{r['meta']['chunk_id']}]"
        else:
            # Modo normal: mostra apenas informacoes resumidas
            header = f"[{r['meta']['source_short']}]"
        
        context_parts.append(header + "\n" + r['chunk'])

    # Junta os chunks em uma string
    if context_parts:
        context = "\n\n---\n\n".join(context_parts)
    else:
        context = ""

    history_parts = []
    # Limita o historico
    for u, a in history[-6:]:
        history_parts.append(f"User: {u}\nAssistant: {a}")

    history_text = "\n".join(history_parts)

    # Instrução de citacaoo baseada no modo
    if DEBUG:
        citation_instruction = "Answer concisely and cite the sources in brackets when using the context above"
    else:
        citation_instruction = "Answer concisely and naturally based on the context provided"

    prompt = (
        "You are an assistant that answers using the information from the context below whenever relevant\n\n"
        "CONTEXT:\n"
        f"{context}\n\n"
        "HISTORY:\n"
        f"{history_text}\n\n"
        "USER QUESTION:\n"
        f"{query}\n\n"
        f"{citation_instruction}\n"
    )

    debug_print(f"Built prompt with {len(context)} context chars and {len(history_parts)} history entries")
    if DEBUG:
        debug_print(f"Full prompt:\n{'-'*50}\n{prompt}\n{'-'*50}")

    return prompt

def generate_answer(prompt: str, model=GENERATION_MODEL):
    debug_print(f"Generating answer using model: {model}")
    
    # Usa generate_content para obter texto
    resp = client.models.generate_content(model=model, contents=prompt)

    debug_print(f"Generated response length: {len(resp.text)} chars")
    
    # Retorna a resposta
    return resp.text

def chat_loop():
    print("\nRAG GenAI ready! (type 'quit' or 'exit' to exit)")
    if DEBUG:
        print("DEBUG MODE ENABLED\n")
    
    while True:
        query = input("\nYou: ").strip()
        if not query:
            continue

        if query.lower() in ("quit", "exit", "sair"):
            print("Exiting...")
            break

        debug_print(f"\n{'='*60}\nProcessing query: {query}\n{'='*60}")
        
        retrieved = retrieve_similar(query, top_k=TOP_K)
        prompt = build_prompt(query, retrieved, chat_history)
        answer = generate_answer(prompt)

        # Salva historico simples
        chat_history.append((query, answer))
        debug_print(f"Chat history now has {len(chat_history)} entries")

        # Mostra resposta
        print("\nAgent AI:\n")
        print(answer)
        
        # Mostra no modo DEBUG
        if DEBUG and retrieved:
            print("\n\n[Recovered snippets used (DEBUG MODE):]")
            for r in retrieved:
                print(f"\n- Source: {r['meta']['source_full']}")
                print(f"  Chunk ID: {r['meta']['chunk_id']}")
                print(f"  Preview: {r['meta']['text_preview']}")
                print(f"  Full chunk length: {len(r['chunk'])} chars")

if __name__ == "__main__":
    chat_loop()