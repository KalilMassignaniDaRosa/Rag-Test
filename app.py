# Configuração Inicial e Importação das Dependências
from dotenv import load_dotenv  # Carrega variáveis de ambiente a partir de um arquivo .env
from langchain_community.document_loaders import DirectoryLoader  # Carrega arquivos de um diretório
from langchain_community.document_loaders import UnstructuredMarkdownLoader  # Processa arquivos Markdown
from langchain_openai.embeddings import OpenAIEmbeddings  # Converte textos em vetores (embeddings) usando a API do OpenAI
from langchain_community.vectorstores import FAISS  # Cria um índice vetorial para busca eficiente
from langchain.chains.conversational_retrieval.base import ConversationalRetrievalChain  # Combina recuperação de documentos e conversação
from langchain_openai.chat_models import ChatOpenAI  # Modelo de chat baseado na API do OpenAI

# Carrega as variáveis de ambiente do arquivo .env (ex.: chave da API do OpenAI)
load_dotenv()

# Carregamento dos Documentos Markdown
# Define o caminho para a pasta que contém os arquivos Markdown
pasta_dos_md = r"/home/samuel/Samuel Sublate/"  # Atualize este caminho conforme necessário

# Cria uma instância do DirectoryLoader para buscar recursivamente arquivos .md no diretório
loader = DirectoryLoader(
    pasta_dos_md,
    glob="**/*.md",  # Padrão que busca todos os arquivos com extensão .md, inclusive em subdiretórios
    loader_cls=UnstructuredMarkdownLoader  # Utiliza o UnstructuredMarkdownLoader para processar cada arquivo Markdown
)

# Carrega os documentos encontrados e armazena na variável 'docs'
docs = loader.load()
print(f"{len(docs)} arquivos Markdown carregados.")

# Criação do Índice Vetorial (Embeddings e FAISS)
# Converte o texto dos documentos em vetores numéricos (embeddings) usando a API do OpenAI
embeddings = OpenAIEmbeddings()

# Cria um índice vetorial com FAISS utilizando os documentos e seus embeddings
vector_store = FAISS.from_documents(docs, embeddings)

# Configuração do Agente Conversacional (RAG)
# Instancia o modelo de chat do OpenAI com temperatura 0 para respostas mais determinísticas
chat_model = ChatOpenAI(temperature=0)

# Cria a cadeia conversacional que integra o modelo de chat com o mecanismo de recuperação de documentos (retriever)
# O retriever é gerado a partir do índice FAISS (vector_store)
qa_chain = ConversationalRetrievalChain.from_llm(
    llm=chat_model,
    retriever=vector_store.as_retriever(),
)

# Interação com o Usuário e Exibição das Respostas
print("Converse com o agente (digite 'sair' para encerrar):")
while True:
    print('\n')
    # Recebe a consulta do usuário via terminal
    query = input("Você:")

    # Se o usuário digitar 'sair', 'exit' ou 'quit', encerra o loop e finaliza a aplicação
    if query.lower() in ["sair", "exit", "quit"]:
        break

    # Envia a consulta e um histórico de conversa vazio (lista) para a cadeia conversacional (para a primeira interação)
    result = qa_chain({"question": query, "chat_history": []})
    print('\n')
    # Exibe a resposta gerada pelo agente
    print("Agente:", result["answer"])
