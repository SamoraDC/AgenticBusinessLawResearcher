from src.core.workflow_state import AgentState
from src.interfaces.external_search_client import unified_mcp, LexMLSearchRequest, TavilySearchRequest
from src.core.llm_factory import get_pydantic_ai_llm, MODEL_DECISION
from pydantic_ai import Agent
from pydantic import BaseModel, Field
from typing import Literal
import traceback

# Modelo para decisão de busca
class SearchDecision(BaseModel):
    needs_web_search: bool = Field(description="Se é necessário buscar na web após análise dos resultados atuais")
    reasoning: str = Field(description="Justificativa detalhada para a decisão")
    web_search_query: str = Field(None, description="Query otimizada para busca web, se necessário")

# System prompt para o agente de decisão
SEARCH_DECISION_PROMPT = (
    "Você é um especialista em análise de informações jurídicas. Sua tarefa é avaliar se as informações "
    "disponíveis (documentos CRAG e jurisprudência LexML) são suficientes para responder à pergunta do usuário, "
    "ou se é necessário buscar informações complementares na web.\n\n"
    "Critérios para decidir buscar na web:\n"
    "1. As informações do CRAG e LexML são ambíguas ou contraditórias\n"
    "2. Faltam informações práticas ou procedimentais importantes\n"
    "3. A pergunta envolve aspectos muito específicos não cobertos pelos documentos\n"
    "4. É necessário verificar atualizações ou mudanças recentes na legislação\n"
    "5. A pergunta envolve aspectos práticos do dia a dia empresarial\n\n"
    "NÃO busque na web se:\n"
    "1. As informações disponíveis já respondem adequadamente à pergunta\n"
    "2. A pergunta é puramente conceitual e já bem explicada nos documentos\n"
    "3. Os documentos fornecem base legal suficiente\n\n"
    "Analise cuidadosamente e seja conservador - apenas recomende busca web quando realmente necessário."
)

# Agente para decisão de busca web
search_decision_agent = Agent(
    model=get_pydantic_ai_llm(model_name=MODEL_DECISION),
    output_type=SearchDecision,
    system_prompt=SEARCH_DECISION_PROMPT,
    model_settings={"temperature": 0.1}
)

async def search_jurisprudencia(state: AgentState) -> dict:
    """Executa a busca de jurisprudência usando o MCP unificado."""
    print("---NODE: LEXML JURISPRUDENCIA SEARCH (MCP Unificado)---")

    current_query = state["current_query"]
    print(f"  Buscando no LexML por: '{current_query}'")

    try:
        response = await unified_mcp.buscar_jurisprudencia(
            termo=current_query,
            tipo_documento="jurisprudencia",
            max_results=5,
            start_record=1,
            query_original=current_query
        )
        print(f"  LexML MCP retornou {len(response.documentos)} documentos ({response.total_encontrado} total).")
        return {"lexml_results": response.documentos, "needs_jurisprudencia": False}

    except Exception as e:
        print(f"  ERRO: Falha na busca LexML: {e}")
        traceback.print_exc()
        return {"lexml_results": [], "needs_jurisprudencia": False}

async def evaluate_search_necessity(state: AgentState) -> dict:
    """Avalia se é necessário buscar na web baseado nos resultados atuais."""
    print("---NODE: EVALUATE SEARCH NECESSITY---")
    
    # Coleta informações do estado atual
    query_object = state.get("query")
    if not query_object:
        print("  ERRO: Query não encontrada no estado")
        return {"needs_web_search": False, "evaluation_complete": True}
    
    query_text = query_object.text
    retrieved_docs = state.get("retrieved_docs", [])
    lexml_results = state.get("lexml_results", [])
    grade = state.get("grade", "unknown")
    
    # Formato resumo das informações para análise
    crag_summary = f"CRAG Grade: {grade}, Documentos: {len(retrieved_docs)}"
    if retrieved_docs:
        crag_content = "\n".join([f"- {doc.text[:200]}..." for doc in retrieved_docs[:3]])
        crag_summary += f"\nConteúdo CRAG:\n{crag_content}"
    
    lexml_summary = f"LexML: {len(lexml_results)} documentos encontrados"
    if lexml_results:
        lexml_content = "\n".join([f"- {doc.titulo}: {doc.ementa[:200] if doc.ementa else 'Sem ementa'}..." for doc in lexml_results[:3]])
        lexml_summary += f"\nConteúdo LexML:\n{lexml_content}"
    
    analysis_input = (
        f"Pergunta do Usuário: {query_text}\n\n"
        f"Informações Disponíveis:\n"
        f"1. {crag_summary}\n\n"
        f"2. {lexml_summary}\n\n"
        f"Com base nas informações acima, avalie se é necessário buscar informações "
        f"complementares na web para responder adequadamente à pergunta do usuário."
    )
    
    try:
        print("  Analisando necessidade de busca web...")
        result = await search_decision_agent.run(analysis_input)
        decision: SearchDecision = result.output
        
        print(f"  Decisão: {'Buscar na web' if decision.needs_web_search else 'Não buscar na web'}")
        print(f"  Justificativa: {decision.reasoning}")
        
        return {
            "needs_web_search": decision.needs_web_search,
            "web_search_reasoning": decision.reasoning,
            "web_search_query": decision.web_search_query if decision.needs_web_search else None,
            "evaluation_complete": True
        }
        
    except Exception as e:
        print(f"  ERRO na avaliação: {e}. Usando fallback conservador.")
        # Fallback: não buscar na web se houver erro
        return {
            "needs_web_search": False,
            "web_search_reasoning": f"Erro na avaliação: {e}. Prosseguindo sem busca web.",
            "evaluation_complete": True
        }

async def search_web_conditional(state: AgentState) -> dict:
    """Executa busca web apenas se determinado necessário pela avaliação."""
    print("---NODE: CONDITIONAL WEB SEARCH (MCP Unificado)---")
    
    needs_web = state.get("needs_web_search", False)
    if not needs_web:
        print("  Busca web não necessária. Pulando.")
        return {"tavily_results": None, "web_search_skipped": True}
    
    # Usar query otimizada se disponível, senão usar a query atual
    search_query = state.get("web_search_query") or state["current_query"]
    print(f"  Executando busca web para: '{search_query}'")
    
    if not unified_mcp.tavily_client:
        print("  ERRO: Tavily não configurado (sem API Key)")
        return {
            "tavily_results": None,
            "web_search_error": "Tavily API Key não configurada"
        }
    
    try:
        request = TavilySearchRequest(
            query=search_query,
            max_results=5,
            search_depth="basic"
        )
        response = await unified_mcp.buscar_web(request)
        print(f"  Tavily retornou {len(response.results)} resultados")
        
        return {
            "tavily_results": response.results,
            "web_search_performed": True
        }
        
    except Exception as e:
        print(f"  ERRO na busca web: {e}")
        traceback.print_exc()
        return {
            "tavily_results": None,
            "web_search_error": str(e)
        }

# Função para decidir próximo passo após busca jurisprudencial
async def decide_after_jurisprudencia(state: AgentState) -> dict:
    """Decide se deve avaliar necessidade de busca web ou ir direto para síntese."""
    print("---NODE: DECIDE AFTER JURISPRUDENCIA---")
    
    # Sempre avalia necessidade de busca web após jurisprudência
    print("  Roteando para avaliação de necessidade de busca web")
    return {"next_node": "evaluate_search_necessity"}

# Função para decidir próximo passo após avaliação
async def decide_after_evaluation(state: AgentState) -> dict:
    """Decide se deve buscar na web ou ir para síntese."""
    print("---NODE: DECIDE AFTER EVALUATION---")
    
    needs_web = state.get("needs_web_search", False)
    if needs_web:
        print("  Busca web necessária. Roteando para busca web.")
        return {"next_node": "search_web_conditional"}
    else:
        print("  Busca web não necessária. Roteando para síntese.")
        return {"next_node": "synthesize_response"}

# Função para decidir próximo passo após busca web
async def decide_after_web_search(state: AgentState) -> dict:
    """Decide próximo passo após busca web (sempre síntese)."""
    print("---NODE: DECIDE AFTER WEB SEARCH---")
    print("  Roteando para síntese final.")
    return {"next_node": "synthesize_response"} 