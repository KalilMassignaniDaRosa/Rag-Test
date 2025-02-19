# RAG

Este documento explica, passo a passo, como funciona o script que realiza a seguinte tarefa:
- Carrega arquivos Markdown de um diretório.
- Processa esses arquivos para gerar embeddings (vetores numéricos) utilizando a API do OpenAI.
- Indexa esses embeddings com FAISS para realizar buscas semânticas eficientes.
- Configura um agente conversacional (RAG) que utiliza os documentos indexados para responder perguntas do usuário.

Além disso, esta documentação detalha como configurar o ambiente, incluindo a criação do arquivo `.env` na raiz do projeto para armazenar variáveis sensíveis (por exemplo, a chave da API do OpenAI).

---

## 1. Requisitos e Dependências

Para executar este script, você precisará ter instaladas as seguintes bibliotecas (com as versões indicadas):

- **Python:** `^3.12`
- **streamlit:** `^1.42.0`
- **langchain:** `^0.3.18`
- **langchain-openai:** `^0.3.5`
- **python-dotenv:** `^1.0.1`
- **langchain-community:** `0.3.16`
- **numpy:** `^1.23.5`
- **unstructured:** `^0.16.21`
- **faiss-cpu:** `^1.10.0`
- **openai:** `^1.63.2`
- **markdown:** `^3.7`
- **langchain-cli:** `^0.0.35`

### Exemplo de instalação via pip:

```bash
pip install python-dotenv streamlit langchain langchain-openai langchain-community numpy unstructured faiss-cpu openai markdown langchain-cli
```

---

## 2. Configuração do Arquivo de Ambiente (.env)

Para garantir que o script funcione corretamente, é necessário criar um arquivo de ambiente chamado `.env` na raiz do projeto. Esse arquivo armazenará variáveis sensíveis, como a chave de API do OpenAI, sem que elas fiquem diretamente no código-fonte.

### Passos para criar e configurar o arquivo `.env`:

1. **Crie o arquivo:**
   - Na raiz do seu projeto (ou seja, no mesmo diretório onde está o script Python), crie um novo arquivo com o nome:
     ```
     .env
     ```

2. **Defina as variáveis de ambiente:**
   - Abra o arquivo `.env` em um editor de texto e adicione a(s) variável(is) necessárias. Por exemplo, para a chave da API do OpenAI, adicione:
     ```env
     OPENAI_API_KEY=your-api-key-aqui
     ```
   - Substitua `your-api-key-aqui` pela sua chave real da API do OpenAI.

3. **Utilização no Script:**
   - No início do script, a biblioteca `python-dotenv` é utilizada para carregar essas variáveis:
     ```python
     from dotenv import load_dotenv
     load_dotenv()
     ```
   - Assim, a variável `OPENAI_API_KEY` (ou outras que você definir) estará disponível para uso no script sem precisar codificá-la diretamente.

---

## 3. Estrutura do Script

A seguir, uma explicação detalhada de cada parte do script.

### 3.1 Configuração Inicial e Importação das Dependências

**Objetivo:**  
Carregar as variáveis de ambiente e importar as bibliotecas necessárias para:
- Carregar e processar os arquivos Markdown.
- Gerar embeddings via OpenAI.
- Indexar os embeddings com FAISS.
- Configurar o agente conversacional para interação.

**Código:**

```python
from dotenv import load_dotenv  # Carrega variáveis de ambiente do arquivo .env
from langchain_community.document_loaders import DirectoryLoader  # Carrega arquivos de um diretório
from langchain_community.document_loaders import UnstructuredMarkdownLoader  # Processa arquivos Markdown
from langchain_openai.embeddings import OpenAIEmbeddings  # Converte textos em vetores utilizando a API do OpenAI
from langchain_community.vectorstores import FAISS  # Cria um índice vetorial para busca semântica
from langchain.chains.conversational_retrieval.base import ConversationalRetrievalChain  # Combina recuperação de documentos e conversação
from langchain_openai.chat_models import ChatOpenAI  # Modelo de chat baseado na API do OpenAI

# Carrega as variáveis de ambiente definidas no arquivo .env
load_dotenv()
```

### 3.2 Carregamento dos Documentos Markdown

**Objetivo:**  
Localizar e carregar recursivamente arquivos Markdown a partir de um diretório especificado.

**Código:**

```python
# Define o caminho para a pasta que contém os arquivos Markdown
pasta_dos_md = r"/home/samuel/Samuel Sublate/"  # Atualize conforme a localização dos seus arquivos

# Cria um DirectoryLoader para buscar recursivamente por arquivos .md
loader = DirectoryLoader(
    pasta_dos_md,
    glob="**/*.md",  # Padrão que abrange todos os arquivos com extensão .md, inclusive em subdiretórios
    loader_cls=UnstructuredMarkdownLoader  # Processa cada arquivo Markdown
)

# Carrega os documentos encontrados e armazena na variável 'docs'
docs = loader.load()
print(f"{len(docs)} arquivos Markdown carregados.")
```

**Detalhes Importantes:**
- **pasta_dos_md:** Caminho absoluto ou relativo para o diretório onde os arquivos Markdown estão armazenados.
- **DirectoryLoader:** Realiza uma busca recursiva com base no padrão definido (`**/*.md`).
- **UnstructuredMarkdownLoader:** Lida com a formatação dos arquivos Markdown, extraindo o conteúdo textual.

### 3.3 Criação do Índice Vetorial (Embeddings e FAISS)

**Objetivo:**  
Converter o conteúdo dos documentos em embeddings e indexá-los com FAISS para buscas semânticas.

**Código:**

```python
# Cria embeddings dos documentos utilizando a API do OpenAI
embeddings = OpenAIEmbeddings()

# Indexa os documentos com FAISS para realizar buscas eficientes
vector_store = FAISS.from_documents(docs, embeddings)
```

**Detalhes Importantes:**
- **OpenAIEmbeddings:** Responsável por transformar textos em vetores numéricos, permitindo a comparação semântica.
- **FAISS:** Biblioteca de indexação vetorial que facilita a busca dos documentos mais relevantes para uma dada consulta.

### 3.4 Configuração do Agente Conversacional (RAG)

**Objetivo:**  
Integrar o modelo de chat com o mecanismo de recuperação de documentos para responder às perguntas do usuário.

**Código:**

```python
# Inicializa o modelo de chat do OpenAI com temperatura 0 para respostas determinísticas
chat_model = ChatOpenAI(temperature=0)

# Cria uma cadeia conversacional que utiliza o modelo de chat e o índice FAISS para recuperação de documentos
qa_chain = ConversationalRetrievalChain.from_llm(
    llm=chat_model,
    retriever=vector_store.as_retriever(),
)
```

**Detalhes Importantes:**
- **ChatOpenAI:** Configurado com uma temperatura baixa para garantir respostas mais consistentes.
- **ConversationalRetrievalChain:** Combina a capacidade do modelo de linguagem com a busca nos documentos indexados, possibilitando respostas fundamentadas no conteúdo dos Markdown.

### 3.5 Interação com o Usuário

**Objetivo:**  
Permitir a interação via terminal, onde o usuário envia uma consulta e recebe a resposta do agente.

**Código:**

```python
print("Converse com o agente (digite 'sair' para encerrar):")
while True:
    print('\n')
    # Recebe a consulta do usuário via terminal
    query = input("Você:")

    # Encerra o loop se o usuário digitar 'sair', 'exit' ou 'quit'
    if query.lower() in ["sair", "exit", "quit"]:
        break

    # Envia a consulta e um histórico de conversa vazio para a cadeia conversacional
    result = qa_chain({"question": query, "chat_history": []})
    print('\n')
    # Exibe a resposta gerada pelo agente
    print("Agente:", result["answer"])
```

**Detalhes Importantes:**
- **Loop Interativo:** Permite múltiplas interações até que o usuário opte por sair.
- **chat_history:** Inicialmente é uma lista vazia. Em aplicações avançadas, pode ser utilizada para manter o contexto da conversa.

---

## 4. Estrutura do Projeto

Uma estrutura básica do projeto pode ser semelhante a esta:

```
meu_projeto/
├── .env
├── script.py
└── README.md
```

- **.env:** Arquivo de configuração com variáveis de ambiente.
- **script.py:** Contém o código do script explicado nesta documentação.
- **README.md:** Este arquivo de documentação.

---

## 5. Conclusão

Este script integra diversas tecnologias para transformar documentos Markdown em uma base de conhecimento consultável através de um agente conversacional. Através das etapas de:
- Configuração do ambiente (usando um arquivo `.env`),
- Carregamento e processamento dos arquivos Markdown,
- Criação de embeddings e indexação com FAISS,
- Configuração do agente conversacional (RAG) e
- Interação via terminal,

você pode criar uma aplicação robusta que responde perguntas com base no conteúdo dos seus documentos.

Certifique-se de:
1. **Criar e configurar o arquivo `.env`** na raiz do projeto com as variáveis necessárias (por exemplo, `OPENAI_API_KEY`).
2. **Instalar as dependências** com as versões recomendadas.
3. **Ajustar os caminhos** e demais configurações conforme a estrutura do seu ambiente.

Experimente, teste e expanda o script conforme suas necessidades!

---
