# src/agents/grader.py
from typing import List, Literal

from src.core.workflow_state import AgentState, Grade
# from src.llm_config import LLM # Não mais usado diretamente aqui
from src.core.llm_factory import get_pydantic_ai_llm, MODEL_GRADER # Importa a factory OpenRouter e o modelo OpenRouter
from src.core.legal_models import DocumentSnippet # << DESCOMENTADO/ADICIONADO

# Adicionado: Import PydanticAI Agent
from pydantic_ai import Agent
# from langchain_core.pydantic_v1 import BaseModel, Field # Mudar para Pydantic V2
from pydantic import BaseModel, Field # Alterado para Pydantic V2

# Define a estrutura de saída esperada (já era um modelo Pydantic V1)
class GradeDocuments(BaseModel):
    """Define a decisão binária sobre a relevância dos documentos para a pergunta."""
    binary_score: Grade = Field(description="'relevant' se algum documento é relevante, 'irrelevant' se nenhum é relevante, 'needs_web_search' se a informação parece faltar ou desatualizada.")
    reasoning: str = Field(description="Breve explicação do porquê os documentos são (ou não são) relevantes.")

# System Prompt Otimizado para o Agente PydanticAI (Grader - Deepseek Chat)
GRADER_SYSTEM_PROMPT = (
    "Você é um avaliador especialista. Sua tarefa é determinar se os 'Documentos Recuperados' fornecidos são relevantes para responder à 'Pergunta do Usuário'.\n"
    "Analise cada documento em relação à pergunta.\n"
    "Responda com base nas seguintes opções:\n"
    "- 'relevant': Se pelo menos um dos documentos contém informações que ajudam a responder diretamente à pergunta.\n"
    "- 'irrelevant': Se NENHUM dos documentos contém informações úteis para responder à pergunta.\n"
    "- 'needs_web_search': Se os documentos parecem relevantes, mas a informação pode estar desatualizada ou incompleta, sugerindo a necessidade de uma busca na web para complementar (use com moderação).\n"
    "Forneça sua avaliação no formato JSON especificado ('GradeDocuments'), incluindo um breve raciocínio ('reasoning')."
)

# Instancia o Agente PydanticAI para o Grader com o modelo específico
try:
    grader_agent = Agent(
        model=get_pydantic_ai_llm(model_name=MODEL_GRADER),
        output_type=str,
        system_prompt="""
        Você é um avaliador especialista de documentos jurídicos.
        
        Determine se os documentos recuperados são relevantes para a pergunta do usuário.
        
        RESPONDA APENAS COM ESTE FORMATO EXATO:
        
        RELEVANCIA: [relevant/irrelevant/needs_web_search]
        JUSTIFICATIVA: [Breve explicação da sua decisão]
        
        Critérios:
        - 'relevant': Pelo menos um documento responde diretamente à pergunta
        - 'irrelevant': Nenhum documento é útil para a pergunta
        - 'needs_web_search': Documentos relevantes mas podem estar desatualizados
        """,
        model_settings={"temperature": 0} # Avaliação precisa, sem criatividade
    )
except Exception as e:
    print(f"ERRO ao inicializar Grader Agent: {e}")
    grader_agent = None

def format_docs_for_grading(docs: List[DocumentSnippet]) -> str:
    """Formata documentos para o prompt do grader."""
    if not docs:
        return "Nenhum documento recuperado."
    formatted = []
    for i, doc in enumerate(docs):
        # DocumentSnippet é um objeto Pydantic, acessamos os atributos diretamente
        source = getattr(doc, 'source_id', f'Doc {i+1}') # Usar getattr para segurança
        
        # Metadata agora é um objeto DocumentMetadata, não um dict
        metadata = getattr(doc, 'metadata', None)
        if metadata and hasattr(metadata, 'authority'):
            source = metadata.authority or source
        
        text_content = getattr(doc, 'text', '')[:500] # Limita o tamanho do preview
        formatted.append(f"Documento {i+1} (Fonte: {source}):\n{text_content}...")
    return "\n---\n".join(formatted)

async def grade_documents(state: AgentState) -> dict:
    """Avalia a relevância dos documentos recuperados para a query."""
    print("---NODE: GRADE DOCUMENTS---")
    if grader_agent is None:
        print("  ERRO: Grader Agent não inicializado. Pulando a avaliação.")
        # Decide seguir como se fossem relevantes para não parar o fluxo
        # Ou poderia rotear para erro, mas queremos resiliência.
        return {"grade": "relevant"}
        
    query = state.get("query")
    retrieved_docs = state.get("retrieved_docs")

    if not query or not retrieved_docs:
        print("  INFO: Query ou documentos não encontrados. Pulando avaliação.")
        return {"grade": "irrelevant"} # Sem docs, são irrelevantes
    
    query_text = query.text
    # A lista retrieved_docs já contém objetos DocumentSnippet
    formatted_docs = format_docs_for_grading(retrieved_docs)
    
    agent_input_content = (
        f"Pergunta do Usuário: {query_text}\n\n"
        f"Documentos Recuperados:\n{formatted_docs}\n\n"
        f"Avalie a relevância destes documentos para a pergunta e responda no formato JSON solicitado."
    )
    
    print(f"  Avaliando {len(retrieved_docs)} documentos..." )
    try:
        # Executar agent e fazer parsing da resposta de texto
        result = await grader_agent.run(agent_input_content)
        grader_text: str = result.output

        # Fazer parsing manual da resposta
        def parse_grader_response(text: str) -> str:
            """Parse manual da resposta do grader."""
            try:
                import re
                
                # Buscar relevância
                relevancia_match = re.search(r'RELEVANCIA:\s*(relevant|irrelevant|needs_web_search)', text, re.IGNORECASE)
                if relevancia_match:
                    grade = relevancia_match.group(1).lower()
                    return grade
                
                # Fallback: buscar palavras-chave no texto
                text_lower = text.lower()
                if 'irrelevant' in text_lower:
                    return 'irrelevant'
                elif 'needs_web_search' in text_lower:
                    return 'needs_web_search'
                else:
                    return 'relevant'  # Padrão conservador
                    
            except Exception as e:
                print(f"Erro no parsing do grader: {e}")
                return 'relevant'  # Fallback seguro

        grade = parse_grader_response(grader_text)
        print(f"  Resultado da Avaliação: {grade}")
        return {"grade": grade}

    except Exception as e:
        import traceback
        print(f"  ALERTA: Erro durante a avaliação de documentos: {e}. Assumindo 'relevant' para continuar.")
        traceback.print_exc()
        # Fallback para relevante para tentar usar os docs mesmo assim
        return {"grade": "relevant"} 