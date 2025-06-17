from langgraph.graph import StateGraph, END
from src.core.workflow_state import AgentState
import os # para visualização
from typing import Literal
from pydantic import BaseModel

# Importações dos nós (incluindo os novos)
from src.agents.document_retriever import retrieve_documents
from src.agents.document_grader import grade_documents
from src.agents.query_transformer import transform_query
# Usando agentes unificados do novo sistema
from src.agents.search_coordinator import (
    evaluate_search_necessity, 
    search_jurisprudencia,
    search_web_conditional
)
from src.agents.streaming.response_synthesizer import synthesize_response
from src.core.legal_models import FinalResponse, LegalQuery # Para o Error Handler e Necessário para o estado inicial

# Nomes dos Nós
NODE_RETRIEVE = "retrieve_documents"
NODE_GRADE = "grade_documents"
NODE_TRANSFORM = "transform_query"
NODE_LEXML = "lexml_search"  # Sempre executado após CRAG relevante
NODE_EVALUATE = "evaluate_search_necessity"  # Novo: avalia se precisa busca web
NODE_WEB_SEARCH = "conditional_web_search"  # Novo: busca web condicional
NODE_SYNTHESIZE = "synthesize_response"
NODE_ERROR = "error_handler"

# Função para tratar erro (mantida, mas retorna FinalResponse)
def handle_error(state: AgentState) -> dict:
    """Nó simples para lidar com erros e terminar o fluxo."""
    print("---NODE: ERROR HANDLER---")
    error_message = state.get('error', 'Unknown error')
    print(f"Erro encontrado: {error_message}")
    # Retorna um FinalResponse de erro
    error_response = FinalResponse(
            overall_summary=f"Desculpe, ocorreu um erro: {error_message}",
            retrieved_documents=[],
            detailed_analysis=[]
        )
    # Note que isso sobrescreve qualquer 'final_response' anterior
    return {"final_response": error_response.model_dump()}

# Funções de Roteamento Condicional
def route_after_grading(state: AgentState) -> str:
    """Decide o caminho após avaliar a relevância dos documentos CRAG."""
    print("---DECISION: Route after Grading---")
    grade = state.get("grade")
    print(f"  Grade: {grade}")

    if grade == "relevant":
        # Docs relevantes, sempre buscar jurisprudência primeiro
        print("  Roteando para Busca LexML (sempre após CRAG relevante).")
        return NODE_LEXML
    elif grade == "irrelevant":
        # Docs irrelevantes, transformar a query
        print("  Roteando para Transformar Query.")
        return NODE_TRANSFORM
    else: # Fallback em caso de erro ou grau inesperado
        print(f"  Aviso: Grau inesperado ('{grade}') após avaliação. Indo para LexML.")
        return NODE_LEXML # Tenta prosseguir com jurisprudência

def route_after_transform(state: AgentState) -> str:
    """Decide o que fazer após transformar a query."""
    print("---DECISION: Route after Transform---")
    # Após transformar, sempre buscar jurisprudência primeiro
    print("  Roteando para Busca LexML.")
    return NODE_LEXML

# Funções de roteamento para o novo fluxo
def route_after_evaluation(state: AgentState) -> str:
    """Roteamento após avaliação de necessidade de busca web."""
    needs_web = state.get("needs_web_search", False)
    if needs_web:
        print("  Busca web necessária. Roteando para busca web condicional.")
        return NODE_WEB_SEARCH
    else:
        print("  Busca web não necessária. Roteando para síntese.")
        return NODE_SYNTHESIZE

# --- Construção do Grafo ---
def build_graph():
    """Constrói o StateGraph para o fluxo CRAG unificado."""
    workflow = StateGraph(AgentState)

    # Adicionar Nós
    workflow.add_node(NODE_RETRIEVE, retrieve_documents)
    workflow.add_node(NODE_GRADE, grade_documents)
    workflow.add_node(NODE_TRANSFORM, transform_query)
    workflow.add_node(NODE_LEXML, search_jurisprudencia)  # Sempre executado
    workflow.add_node(NODE_EVALUATE, evaluate_search_necessity)  # Avalia necessidade de web
    workflow.add_node(NODE_WEB_SEARCH, search_web_conditional)  # Busca web condicional
    workflow.add_node(NODE_SYNTHESIZE, synthesize_response)
    workflow.add_node(NODE_ERROR, handle_error)

    # Ponto de Entrada
    workflow.set_entry_point(NODE_RETRIEVE)

    # Fluxo Principal
    workflow.add_edge(NODE_RETRIEVE, NODE_GRADE)

    # Roteamento após Grade
    workflow.add_conditional_edges(
        NODE_GRADE,
        route_after_grading,
        {
            NODE_LEXML: NODE_LEXML,
            NODE_TRANSFORM: NODE_TRANSFORM,
        }
    )

    # Roteamento após Transform
    workflow.add_conditional_edges(
         NODE_TRANSFORM,
         route_after_transform,
         { NODE_LEXML: NODE_LEXML }
    )

    # Após LexML, sempre avaliar necessidade de busca web
    workflow.add_edge(NODE_LEXML, NODE_EVALUATE)

    # Roteamento após avaliação
    workflow.add_conditional_edges(
        NODE_EVALUATE,
        route_after_evaluation,
        {
            NODE_WEB_SEARCH: NODE_WEB_SEARCH,
            NODE_SYNTHESIZE: NODE_SYNTHESIZE
        }
    )

    # Após busca web, sempre ir para síntese
    workflow.add_edge(NODE_WEB_SEARCH, NODE_SYNTHESIZE)

    # Finais
    workflow.add_edge(NODE_SYNTHESIZE, END)
    workflow.add_edge(NODE_ERROR, END)

    # Compila o grafo
    app = workflow.compile()
    print("Grafo CRAG compilado com sucesso.")

    # Opcional: Visualizar o grafo (requer 'pip install pygraphviz')
    try:
        # Tenta salvar a visualização
        output_path = "crag_graph.png"
        # Verifica se o diretório existe, se não, usa o diretório atual
        output_dir = os.path.dirname(output_path)
        if output_dir and not os.path.exists(output_dir):
             output_path = os.path.basename(output_path) # Salva no diretório atual se o path não existir

        img_data = app.get_graph().draw_mermaid_png()
        with open(output_path, "wb") as f:
            f.write(img_data)
        print(f"Visualização do grafo salva em {os.path.abspath(output_path)}")
    except ImportError:
         print("AVISO: pygraphviz não instalado. Não foi possível gerar a visualização do grafo. Instale com: pip install pygraphviz")
    except Exception as e:
        # Captura outros erros potenciais (ex: Graphviz não instalado no sistema)
        print(f"Não foi possível visualizar o grafo (verifique dependências como Graphviz): {e}")


    return app

# Exemplo de como usar (mantido para referência, mas o estado inicial precisará ser ajustado)
if __name__ == '__main__':
    import asyncio
    from src.models import LegalQuery # Necessário para o estado inicial
    from dotenv import load_dotenv
    import pprint

    load_dotenv()
    pp = pprint.PrettyPrinter(indent=2)

    print("Construindo o grafo CRAG...")
    graph_app = build_graph()

    async def run_test():
        # Estado inicial para um novo chat/query
        # A query original vai em 'query', a 'current_query' começa igual
        # mas pode ser modificada pelo transformer.
        initial_state = AgentState(
            query=LegalQuery(text="Quais são as regras para demissão por justa causa segundo a CLT e a jurisprudência recente?"),
            current_query="Quais são as regras para demissão por justa causa segundo a CLT e a jurisprudência recente?",
            retrieved_docs=[], # Será preenchido pelo retriever
            tavily_results=None,
            lexml_results=None,
            # Flags de controle CRAG/Busca: não precisamos mais setar 'needs' inicialmente,
            # o fluxo determinará isso. 'needs_jurisprudencia' será usado pela decisão final.
            # Definimos True inicialmente para que a decisão tente buscá-la se o fluxo chegar lá.
            needs_jurisprudencia=True,
            # 'grade' e 'transformed_query' serão preenchidos pelos nós correspondentes
            grade=None,
            transformed_query=None,
            # Outros campos do estado
            should_synthesize=True,  # CORRIGIDO: Permite síntese no fluxo
            history=[],
            final_response=None,
            error=None,
            next_node=None # Será preenchido pelo nó de decisão
        )

        print(f"\nExecutando grafo com a query: '{initial_state['query'].text}'")
        config = {"recursion_limit": 15} # Aumentar limite para fluxos mais complexos
        final_state = None
        async for event in graph_app.astream(initial_state, config=config):
            print("\n--- Event --- ")
            # Imprime o nome do nó que acabou de rodar e seu resultado (o estado atualizado por ele)
            for node_name, state_update in event.items():
                 print(f"Node '{node_name}' finished.")
                 # Opcional: Imprimir todo o estado atualizado (pode ser verboso)
                 # pp.pprint(state_update)
                 # Guarda o último estado completo visto
                 final_state = state_update
            print("-------------")

        print("\n--- Resultado Final do Grafo --- ")
        if final_state and final_state.get("final_response"):
            pp.pprint(final_state.get("final_response"))
        elif final_state and final_state.get("error"):
             print(f"Erro final: {final_state.get('error')}")
        else:
            print("Execução não produziu resposta final ou erro explícito.")
            print("Estado final completo:")
            pp.pprint(final_state)


    asyncio.run(run_test()) 