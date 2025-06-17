from src.core.workflow_state import AgentState
from src.core.legal_models import DocumentSnippet, DocumentMetadata, SearchSource
from typing import List
import asyncio # For async sleep placeholder
from src.core.llm_factory import EMBEDDINGS, GOOGLE_API_KEY # Importa os embeddings configurados
from langchain_community.vectorstores import Chroma
import os
import chromadb
from chromadb.utils import embedding_functions

# Configurações do ChromaDB
CHROMA_DB_PATH = "chroma_db_gemini" # Path para o diretório persistente
# Tentar determinar o nome da coleção. Se for complexo, pode precisar ser passado ou configurado.
# Por padrão, muitas implementações usam "langchain" ou um nome específico.
# Vamos assumir um nome padrão ou você pode ajustá-lo.
# Verifique o nome exato dentro do diretório chroma_db_gemini se houver problemas.
# DEFAULT_COLLECTION_NAME = "gemini_collection" # NOME DE EXEMPLO - AJUSTE SE NECESSÁRIO
# Baseado no test.ipynb usando a integração LangChain, o nome padrão é provavelmente "langchain"
DEFAULT_COLLECTION_NAME = "langchain"

# Verifica se os embeddings foram carregados corretamente
if EMBEDDINGS is None:
    raise ImportError("Embeddings Google Gemini não foram inicializados. Verifique a GOOGLE_API_KEY e a configuração em llm_config.py")

# Inicializa o cliente ChromaDB persistente
# Use o embedding_functions do ChromaDB se o seu EMBEDDINGS não for diretamente compatível
# Se EMBEDDINGS for do tipo LangChain (GoogleGenerativeAIEmbeddings), podemos precisar adaptá-lo
# ou usar a função de embedding do Google direto com Chroma.
# ChromaDB >= 0.5 usa uma API diferente.
# Vamos usar a forma padrão de conexão e obter a coleção.
try:
    chroma_client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
    # Tenta obter a coleção. Se não existir, pode dar erro ou precisar ser criada.
    # A criação geralmente ocorre durante a ingestão de dados.
    # Assumimos que a coleção já existe.
    print(f"Tentando obter a coleção ChromaDB: {DEFAULT_COLLECTION_NAME}")
    vectorstore = chroma_client.get_collection(
        name=DEFAULT_COLLECTION_NAME,
        # embedding_function=embedding_functions.GoogleGenerativeAiEmbeddingFunction(api_key=GOOGLE_API_KEY) # Exemplo se necessário
        # Se EMBEDDINGS for compatível com LangChain Chroma, o retriever pode usá-lo diretamente.
    )
    print(f"Coleção ChromaDB '{DEFAULT_COLLECTION_NAME}' obtida com sucesso.")
except Exception as e:
    print(f"ERRO ao conectar ou obter coleção do ChromaDB em '{CHROMA_DB_PATH}' com nome '{DEFAULT_COLLECTION_NAME}': {e}")
    print("Verifique o path, o nome da coleção e se o banco de dados foi inicializado corretamente.")
    # Poderíamos lançar uma exceção mais específica ou configurar um fallback.
    # Por enquanto, vamos deixar o erro impedir a inicialização se a conexão falhar.
    raise e

# --- Placeholder for GraphRAG Interaction ---
# Replace this with actual GraphRAG client/library calls

async def retrieve_documents_graphrag(query: str, jurisdiction: str | None, area_of_law: str | None) -> List[DocumentSnippet]:
    """Placeholder function simulating GraphRAG retrieval."""
    print(f"  Simulating GraphRAG retrieval for: '{query}' (Jurisdiction: {jurisdiction}, Area: {area_of_law})")
    # Simulate network delay or processing time
    await asyncio.sleep(1)
    
    # Simulate finding some documents
    # In a real scenario, this would query the graph database based on query text,
    # extracted entities, jurisdiction, area_of_law, and graph relationships.
    results = [
        DocumentSnippet(
            source_id="CASE-123", 
            text="Relevant precedent text snippet about Article 5 of the relevant code...",
            metadata={"type": "precedent", "date": "2022-01-15", "relevance": 0.9},
            relationship_summary="Cites LAW-XYZ, Interprets STATUTE-ABC"
        ),
        DocumentSnippet(
            source_id="LAW-XYZ", 
            text="Text of Law XYZ regarding the specific issue...",
            metadata={"type": "legislation", "date": "2019-03-10", "relevance": 0.85}
        ),
        DocumentSnippet(
            source_id="ARTICLE-ABC", 
            text="Scholarly article discussing the nuances of LAW-XYZ...",
            metadata={"type": "article", "date": "2023-05-20", "relevance": 0.7}
        )
    ]
    # Simulate filtering based on jurisdiction/area if provided (crude example)
    if jurisdiction:
        results = [doc for doc in results if jurisdiction.lower() in doc.source_id.lower() or not any(j in doc.source_id.lower() for j in ['california', 'brazil'])] # Simplified logic
    
    print(f"  Found {len(results)} simulated documents.")
    return results

# --- LangGraph Agent Node ---

# --- Inicialização do Retriever ---
# Idealmente, o vector store seria carregado uma vez e passado para a função,
# mas para simplificar o exemplo do nó, vamos carregá-lo aqui.
# CUIDADO: Carregar a cada chamada pode ser ineficiente para produção.
def get_retriever():
    if not os.path.exists(CHROMA_DB_PATH):
        raise FileNotFoundError(f"Diretório do ChromaDB não encontrado: {CHROMA_DB_PATH}. Execute o document_processor.py primeiro.")
    
    print(f"  Carregando VectorStore de: {CHROMA_DB_PATH}")
    try:
        vector_store = Chroma(persist_directory=CHROMA_DB_PATH, embedding_function=EMBEDDINGS)
        # Configurar k (número de documentos a retornar)
        retriever = vector_store.as_retriever(search_kwargs={"k": 4})
        print("  Retriever pronto.")
        return retriever
    except Exception as e:
        print(f"  Erro ao carregar ChromaDB ou criar retriever: {e}")
        raise

# Tenta inicializar na importação para falhar cedo se o DB não existir
# Em um app real, isso seria melhor gerenciado (ex: lazy loading)
try:
    retriever_instance = get_retriever()
except Exception as e:
    print(f"ERRO CRÍTICO: Falha ao inicializar o retriever na importação. {e}")
    retriever_instance = None

async def retrieve_documents(state: AgentState) -> dict:
    """Recupera documentos do ChromaDB com base na query atual."""
    print("---NODE: RETRIEVE DOCUMENTS (ChromaDB)---")
    current_query = state.get("current_query")
    if not current_query:
        print("  ERRO: current_query não encontrada no estado.")
        # Retorna um estado indicando falha ou sem documentos
        return {"retrieved_docs": []}

    print(f"  RECUPERADOR (CRAG) - Query para ChromaDB: '{current_query}'")
    retrieved_docs_list = []
    try:
        # Usa o objeto LangChain EMBEDDINGS para gerar o embedding da query
        # Se o ChromaDB não foi configurado com uma função de embedding compatível,
        # precisamos gerar o embedding da query externamente.
        # O objeto EMBEDDINGS (GoogleGenerativeAIEmbeddings) tem um método embed_query.
        query_embedding = EMBEDDINGS.embed_query(current_query)

        # Realiza a busca no ChromaDB
        # Ajuste n_results conforme necessário
        results = vectorstore.query(
            query_embeddings=[query_embedding], # Passa o embedding gerado
            n_results=15, # AUMENTADO PARA DIAGNÓSTICO
            include=['metadatas', 'documents'] # Garantir que documentos e metadados sejam incluídos
        )

        # Processa os resultados
        # A estrutura de 'results' no ChromaDB >= 0.5 é diferente
        # Exemplo: results = {'ids': [['id1', 'id2']], 'distances': [[d1, d2]], 'metadatas': [[m1, m2]], 'documents': [[doc1, doc2]]}
        if results and results.get('documents') and results['documents'][0]:
            num_results = len(results['documents'][0])
            print(f"  ChromaDB retornou {num_results} resultados (solicitados 15).") # LOG ATUALIZADO
            for i in range(num_results):
                doc_text = results['documents'][0][i]
                metadata = results['metadatas'][0][i] if results.get('metadatas') and results['metadatas'][0] else {}
                source_id = results['ids'][0][i] if results.get('ids') and results['ids'][0] else f"chroma_doc_{i}"
                
                # Tenta extrair o source_id real da metadata se disponível (comum em LangChain)
                if 'source' in metadata:
                    source_id = metadata['source']
                
                # LOG MAIS DETALHADO PARA DIAGNÓSTICO DO CRAG
                print(f"    --- CRAG DOC {i+1} (ID: {source_id}) ---") 
                print(f"    METADATA: {metadata}")
                print(f"    TEXTO COMPLETO DO CHUNK:\n{doc_text}")
                print(f"    --- FIM CRAG DOC {i+1} ---")
                # FIM LOG MAIS DETALHADO

                # Criar metadata compatível com novo modelo
                document_metadata = DocumentMetadata(
                    source=SearchSource.VECTORDB,
                    document_type="pdf",  # Assumindo PDF por padrão
                    authority=metadata.get('source', source_id),
                    jurisdiction=None,
                    page_number=metadata.get('page'),
                    confidence_score=0.8,  # Score padrão
                    tags=[],
                    citations=[],
                    created_at=None,
                    last_updated=None
                )
                
                snippet = DocumentSnippet(
                    source_id=source_id,
                    text=doc_text,
                    metadata=document_metadata,
                    relevance_score=0.8,  # Score padrão
                    summary="",  # Vazio por padrão
                    key_points=[],
                    related_concepts=[],
                    legal_areas=[],
                    relationships=[]
                )
                retrieved_docs_list.append(snippet)
        else:
            print("  ChromaDB não retornou resultados.")

    except Exception as e:
        import traceback
        print(f"  ALERTA: Erro durante a busca no ChromaDB: {e}. O fluxo continuará sem documentos recuperados.")
        traceback.print_exc()
        # Permite que o fluxo continue sem documentos CRAG
        return {"retrieved_docs": []} # Retorna lista vazia em caso de erro

    return {"retrieved_docs": retrieved_docs_list}

# Adicione quaisquer outras funções ou lógica necessária para o retriever aqui 