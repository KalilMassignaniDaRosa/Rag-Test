# 🧠 RAG com Gemini (Implementação Simples)

> 📚 **Fonte original:** Adaptado do projeto educacional RAG da [Asimov Academy](https://github.com/asimov-academy/rag-in-practice)  
> 🧠 Esta versão foi modificada para usar a **API do Google Gemini** diretamente (sem LangChain)  
> 📄 Licenciado sob MIT License (veja o arquivo [LICENSE](LICENSE))

**Leia em**: [English](README.md) | [Portuguese](README.pt-BR.md)

---

## 🇧🇷 Versão em Português

### 🧩 1. Requisitos

- **Python**: >=3.12
- **Bibliotecas**:

```bash
pip install -U google-genai faiss-cpu python-dotenv numpy
```

---

### 🔐 2. Configuração do Ambiente

Crie um arquivo `.env` na raiz do projeto com sua chave da API do Google:

```env
GOOGLE_API_KEY=sua-chave-aqui
```

Ou use:

```env
GEMINI_API_KEY=sua-chave-aqui
```

🔑 **Obtenha sua chave em**: [Google AI Studio](https://aistudio.google.com/app/apikey)

---

### 📁 3. Estrutura de Documentos

Coloque seus arquivos Markdown (`.md`) dentro da pasta `docs/`. O script irá:
- Carregar todos os arquivos `.md` recursivamente
- Dividi-los em chunks (pedaços)
- Gerar embeddings usando `gemini-embedding-001`
- Indexar com FAISS para busca semântica

---

### ⚙️ 4. Como Executar

**Ativar ambiente virtual (Windows):**

```bash
python -m venv .venv
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\activate
```

**Instalar dependências:**

```bash
pip install -U google-genai faiss-cpu python-dotenv numpy
```

**Executar o script:**

```bash
python rag_simple_genai.py
```

**Interagir com o assistente:**

```
You: O que esse documento explica?
Agent AI: [Resposta baseada no contexto recuperado com citações das fontes]
```

Digite `quit`, `exit` ou `sair` para encerrar.

---

### 💡 5. Como Funciona

1. **Carregamento de Documentos**: Lê todos os arquivos `.md` da pasta `docs/`
2. **Fragmentação**: Divide documentos em chunks com sobreposição (padrão: 1200 caracteres com 200 de overlap)
3. **Geração de Embeddings**: Usa o modelo `gemini-embedding-001` para vetorizar os chunks
4. **Indexação FAISS**: Cria um índice vetorial para busca rápida por similaridade
5. **Processamento de Consultas**: 
   - Gera embedding da pergunta do usuário
   - Recupera os top-K chunks mais similares (padrão: 4)
   - Constrói prompt contextualizado com histórico de conversa
6. **Geração de Resposta**: Usa o modelo `gemini-2.5-flash` para gerar respostas
7. **Atribuição de Fontes**: Mostra quais chunks de documentos foram utilizados

---

### 🔧 6. Configurações

Edite estas constantes em `rag_simple_genai.py`:

```python
DOCS_PATH = "./docs"              # Caminho para os arquivos markdown
CHUNK_SIZE = 1200                 # Caracteres por chunk
CHUNK_OVERLAP = 200               # Sobreposição entre chunks
EMBEDDING_MODEL = "gemini-embedding-001"
GENERATION_MODEL = "gemini-2.5-flash"
TOP_K = 4                         # Número de trechos a recuperar
```

---

### 📖 7. Créditos e Referências

- **Projeto original**: [Asimov Academy - RAG in Practice](https://github.com/asimov-academy/rag-in-practice)
- **Adaptado por**: Uso direto da API do Google Gemini via SDK `google-genai`
- **Tecnologias principais**:
  - `google-genai`: Cliente oficial Python da Google Generative AI
  - `faiss-cpu`: Busca de similaridade vetorial
  - `numpy`: Operações numéricas

---

### 📄 8. Licença

Este projeto está licenciado sob a **MIT License** - veja o arquivo [LICENSE](LICENSE) para detalhes.

Trabalho original Copyright (c) 2025 Asimov Academy  
Modificações e adaptações também estão cobertas pela MIT License.

---

### 🚀 9. Próximos Passos

- Adicione mais documentos à pasta `docs/`
- Ajuste o tamanho dos chunks e sobreposição para seu caso de uso
- Experimente diferentes modelos Gemini
- Implemente armazenamento persistente para embeddings
- Adicione interface web com Streamlit ou Gradio
