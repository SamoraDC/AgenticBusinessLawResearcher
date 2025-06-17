import os
import shutil # Para limpar o diretório antigo
import time
import fitz # PyMuPDF
from langchain.docstore.document import Document # Para criar documentos LangChain manualmente
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import Chroma
from dotenv import load_dotenv
import sys

load_dotenv()

# --- Configuração ---
DATA_DIR = "data"
CHUNK_SIZE = 1500  # Tamanho máximo do chunk após o RecursiveCharacterTextSplitter
CHUNK_OVERLAP = 300 # Sobreposição para o RecursiveCharacterTextSplitter
PERSIST_DIRECTORY = "chroma_db_gemini"
GEMINI_MODEL_NAME = "models/text-embedding-004"
MIN_CHUNK_LENGTH_CHARS = 100 # Comprimento mínimo para um chunk ser considerado útil após extração PyMuPDF
# --------------------

def is_useful_chunk_heuristic(text: str, page_num: int, total_pages: int) -> bool:
    """
    Filtra heuristicamente chunks que parecem ser lixo ou metadados não relevantes.
    Esta função pode ser bastante expandida.
    """
    text_lower = text.lower()
    
    if len(text_lower.strip()) < MIN_CHUNK_LENGTH_CHARS:
        # print(f"    [Filtro] Chunk muito curto (página {page_num}): '{text_lower[:50]}...'")
        return False

    # Palavras-chave comuns em metadados/lixo (não exaustivo)
    junk_keywords = [
        "copyright", "editora forense", "todos os direitos reservados", "impresso no brasil",
        "sindicato nacional dos editores", "cip – brasil", "catalogação-na-fonte",
        "produção digital:", "capa:", "isbn:", "cdu:", "ficha catalográfica",
        "agradeço a todos", "dedico este livro", "meus queridos pais",
        "sumário", "índice remissivo", "bibliografia" 
        # Cuidado com "sumário", "índice", "bibliografia" se o conteúdo REAL dessas seções for desejado.
        # Esta heurística é mais para descartar páginas que SÃO APENAS essas coisas.
    ]
    # Se o chunk for pequeno e contiver muitas dessas palavras, provavelmente é lixo.
    if len(text_lower) < 300: # Aplicar heurística de palavras-chave mais em chunks menores
        for keyword in junk_keywords:
            if keyword in text_lower:
                # print(f"    [Filtro] Chunk com keyword de lixo (página {page_num}): '{text_lower[:50]}...'")
                return False
    
    # Heurística para citações filosóficas (baseado no seu log anterior)
    # Pode ser muito específico, ajuste ou remova se necessário
    philosopher_names = ["adam smith", "milton friedman", "ludwig von mises", "ayn rand"]
    if any(philosopher in text_lower for philosopher in philosopher_names) and len(text_lower) < 500:
        if "riqueza das nações" in text_lower or "capitalismo e liberdade" in text_lower or "virtue of selfishness" in text_lower:
            # print(f"    [Filtro] Chunk de citação filosófica (página {page_num}): '{text_lower[:50]}...'")
            return False
            
    # TODO: Adicionar mais heurísticas se necessário (ex: detectar listas de figuras, tabelas de conteúdo muito esparsas)
    return True

def load_and_chunk_pdfs_pymupdf(data_directory: str, chunk_size: int, chunk_overlap: int):
    """
    Carrega todos os PDFs de um diretório usando PyMuPDF para extrair blocos de texto,
    depois aplica RecursiveCharacterTextSplitter se os blocos forem muito grandes.
    Filtra chunks heuristicamente.
    """
    script_dir = os.path.dirname(__file__)
    project_root = os.path.dirname(script_dir)
    full_data_path = os.path.join(project_root, data_directory)

    if not os.path.isdir(full_data_path):
        print(f"Erro: Diretório de dados não encontrado: {full_data_path}")
        return []

    all_final_chunks = []
    pdf_files = [f for f in os.listdir(full_data_path) if f.lower().endswith(".pdf")]

    if not pdf_files:
        print(f"Nenhum arquivo PDF encontrado em: {full_data_path}")
        return []

    print(f"Carregando e processando {len(pdf_files)} PDF(s) de '{full_data_path}' usando PyMuPDF...")

    for pdf_file in pdf_files:
        file_path = os.path.join(full_data_path, pdf_file)
        print(f"  - Processando {pdf_file}...")
        try:
            doc = fitz.open(file_path)
            pdf_initial_blocks = []
            for page_num in range(len(doc)):
                page_obj = doc.load_page(page_num)
                # Extrai blocos de texto, preservando a ordem de leitura قدر الإمكان
                blocks = page_obj.get_text("blocks", sort=True) 
                for block_idx, b in enumerate(blocks):
                    text_content = b[4] # O texto do bloco
                    # Limpeza básica
                    text_content = text_content.replace("\r\n", "\n").strip() 
                    
                    # Filtro heurístico primário
                    if is_useful_chunk_heuristic(text_content, page_num + 1, len(doc)):
                        metadata = {
                            "source": os.path.basename(file_path),
                            "page": page_num + 1,
                            "total_pages": len(doc),
                            "block_index_on_page": block_idx,
                            # Coordenadas podem ser úteis para debug ou interfaces visuais
                            # "block_coords": (b[0], b[1], b[2], b[3]) 
                        }
                        pdf_initial_blocks.append(Document(page_content=text_content, metadata=metadata))
            
            print(f"    > {len(doc)} páginas lidas, {len(pdf_initial_blocks)} blocos de texto úteis extraídos inicialmente.")
            doc.close()

            if not pdf_initial_blocks:
                print(f"    > Nenhum bloco de texto útil encontrado em {pdf_file} após filtro heurístico inicial.")
                continue

            # Agora, se os blocos extraídos ainda forem muito grandes, aplicamos o RecursiveCharacterTextSplitter
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                length_function=len,
                add_start_index=True, 
                separators=["\n\n", "\n", ". ", " ", ""]
            )
            
            print(f"    > Dividindo {len(pdf_initial_blocks)} blocos em chunks menores (size={chunk_size}, overlap={chunk_overlap})...")
            further_split_chunks = text_splitter.split_documents(pdf_initial_blocks)
            
            # Filtro final para remover chunks que se tornaram vazios ou muito curtos após o split
            final_pdf_chunks = [chunk for chunk in further_split_chunks if chunk.page_content.strip() and len(chunk.page_content.strip()) >= MIN_CHUNK_LENGTH_CHARS]
            print(f"    > {len(final_pdf_chunks)} chunks finais criados para {pdf_file} (após split e filtro final).")
            all_final_chunks.extend(final_pdf_chunks)

        except Exception as e:
            print(f"Erro ao processar {pdf_file} com PyMuPDF: {e}")

    if not all_final_chunks:
        print("Nenhum chunk foi criado de nenhum PDF.")
        return []
    
    print(f"\nTotal de {len(all_final_chunks)} chunks finais criados de todos os PDFs.")
    return all_final_chunks

# --- Bloco Principal ---
if __name__ == "__main__":
    start_time = time.time()

    # 1. Limpar diretório ChromaDB antigo
    if os.path.exists(PERSIST_DIRECTORY):
        print(f"Limpando diretório ChromaDB existente: {PERSIST_DIRECTORY}...")
        try:
            shutil.rmtree(PERSIST_DIRECTORY)
            print(f"Diretório {PERSIST_DIRECTORY} limpo com sucesso.")
        except Exception as e:
            print(f"Erro ao limpar diretório {PERSIST_DIRECTORY}: {e}. Por favor, delete manualmente e tente novamente.")
            sys.exit(1) # Sai do script se não conseguir limpar
    else:
        print(f"Diretório {PERSIST_DIRECTORY} não encontrado, não precisa limpar.")

    # 2. Carregar e Chunk PDFs usando a nova função
    # Ajusta o DATA_DIR para ser relativo à localização do script
    script_dir = os.path.dirname(__file__)
    project_root = os.path.dirname(script_dir)
    data_dir_path = os.path.join(project_root, DATA_DIR)
    
    chunks = load_and_chunk_pdfs_pymupdf(data_dir_path, CHUNK_SIZE, CHUNK_OVERLAP)

    if chunks:
        print("\n--- Configurando Embeddings e Vector Store ---")
        try:
            print(f"Inicializando Google Embeddings com modelo: {GEMINI_MODEL_NAME}...")
            google_api_key = os.getenv('GOOGLE_API_KEY')
            if not google_api_key:
                 raise ValueError("Variável de ambiente GOOGLE_API_KEY não encontrada.")

            embeddings = GoogleGenerativeAIEmbeddings(
                model=GEMINI_MODEL_NAME,
                google_api_key=google_api_key
            )
            print("Embeddings Google inicializados.")

            print(f"Configurando ChromaDB em: {PERSIST_DIRECTORY} e adicionando chunks...")
            
            # --- REVERTENDO PARA ADIÇÃO MANUAL COM DELAY ---
            # Cria a instância Chroma vazia (o diretório já foi limpo)
            print("Criando novo VectorStore (inicialmente vazio)...")
            vector_store = Chroma(
                persist_directory=PERSIST_DIRECTORY,
                embedding_function=embeddings
            )

            # Define o tamanho do lote e o delay entre os lotes
            BATCH_SIZE = 100  # Número de chunks a processar por lote (ajuste conforme necessário)

            print(f"Iniciando adição de {len(chunks)} chunks ao VectorStore em lotes de {BATCH_SIZE}...")

            added_count = 0
            for i in range(0, len(chunks), BATCH_SIZE):
                batch_chunks = chunks[i:i + BATCH_SIZE]
                if not batch_chunks: # Segurança caso o último lote seja vazio
                    continue
                try:
                    print(f"  Processando lote começando do índice {i} (tamanho do lote: {len(batch_chunks)})...")
                    vector_store.add_documents(batch_chunks) # Adiciona o lote de chunks
                    added_count += len(batch_chunks)
                    print(f"    ... Lote processado. Total de chunks adicionados nesta sessão: {added_count}/{len(chunks)}.")

                except Exception as e:
                    # Melhor tratamento de erro para o loop
                    print(f"\n!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
                    print(f"!! ERRO AO ADICIONAR LOTE DE CHUNKS (começando do índice: {i}, {len(batch_chunks)} chunks no lote) !!")
                    print(f"!! Total adicionado antes do erro: {added_count}")
                    print(f"!! ERRO: {e}")
                    print(f"!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
                    
                    print("Abortando processo de adição devido ao erro.")
                    # Nota: O ChromaDB pode ficar em um estado parcialmente populado com os lotes anteriores bem-sucedidos.
                    # Como o script deleta o DB no início, na próxima execução começará do zero.
                    raise # Re-lança a exceção para parar o script
            
            if added_count == len(chunks) and added_count > 0:
                print(f"\nAdição de todos os {added_count} chunks concluída com sucesso!")
            elif added_count > 0:
                print(f"\nAdição parcial de {added_count}/{len(chunks)} chunks concluída. Ocorreu um erro ou interrupção.")
            elif not chunks:
                 print(f"\nNenhum chunk foi fornecido para processamento.")
            else:
                print(f"\nNenhum chunk foi adicionado. Verifique os logs para erros.")

            if added_count > 0 :
                print(f"VectorStore em '{PERSIST_DIRECTORY}' atualizado com os chunks processados nesta sessão.")
            else:
                print(f"VectorStore em '{PERSIST_DIRECTORY}' não foi modificado ou criado nesta sessão pois nenhum chunk foi adicionado.")
            # --- FIM DA REVERSÃO PARA ADIÇÃO MANUAL ---

            # Comentado o método original que causava o erro
            # print(f"Criando novo VectorStore em: {PERSIST_DIRECTORY} com {len(chunks)} chunks...")
            # vector_store = Chroma.from_documents(
            #     documents=chunks,
            #     embedding=embeddings,
            #     persist_directory=PERSIST_DIRECTORY
            #     # ... (comentários sobre batching removidos) ...
            # )
            # print("VectorStore criado e persistido com sucesso.")

            # Teste rápido (opcional, mas recomendado)
            print("\n--- Testando busca no Vector Store ---")
            try:
                test_queries = [
                    "exclusão de sócio minoritário", 
                    "direito societário", 
                    "dissolução parcial de sociedade" # Termo visto nos seus logs como presente
                ]
                for query in test_queries:
                    print(f"  Buscando por '{query}' (k=3):")
                    results = vector_store.similarity_search(query, k=3)
                    if results:
                        for doc in results:
                            print(f"    - Fonte: {doc.metadata.get('source', 'N/A')}, Página: {doc.metadata.get('page', 'N/A')}, Bloco: {doc.metadata.get('block_index_on_page', 'N/A')}")
                            print(f"      Conteúdo: {doc.page_content[:200]}...") # Preview maior
                    else:
                        print("    Nenhum resultado encontrado.")
            except Exception as e:
                print(f"Erro durante busca teste: {e}")

            end_time = time.time()
            print(f"\nProcessamento completo em {end_time - start_time:.2f} segundos.")
            print(f"Vector Store em '{PERSIST_DIRECTORY}' está pronto para uso.")

        except ImportError as e:
            print(f"\nErro de Importação: {e}. Verifique as bibliotecas (PyMuPDF, langchain, etc.).")
        except ValueError as e:
             print(f"\nErro de Configuração: {e}")
        except Exception as e:
            print(f"\nOcorreu um erro inesperado durante embeddings/vector store: {e}")
            import traceback
            traceback.print_exc()
    else:
        print("Nenhum chunk útil foi criado, Vector Store não foi gerado.") 