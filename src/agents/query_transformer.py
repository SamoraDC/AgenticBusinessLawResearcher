# src/agents/transformer.py
# Removido: from langchain_core.prompts import ChatPromptTemplate

from src.core.workflow_state import AgentState
# from src.llm_config import LLM # Não mais usado diretamente aqui
from src.core.llm_factory import get_pydantic_ai_llm, MODEL_TRANSFORMER # Usando OpenRouter

# Adicionado: Import PydanticAI Agent
from pydantic_ai import Agent
from pydantic import BaseModel, Field
# import asyncio # Não será mais necessário se await agent.run() funcionar diretamente

# --- Modelo para a Saída da Transformação ---
class TransformedQuery(BaseModel):
    """Representa a query transformada para melhor recuperação."""
    transformed_query: str = Field(description="A versão reescrita da pergunta do usuário, otimizada para busca em base de conhecimento ou web.")

# --- Prompt para o Agente Transformer ---
TRANSFORMER_SYSTEM_PROMPT = (
    "Você é um especialista em otimização de consultas. Sua tarefa é reescrever a 'Pergunta Original do Usuário' para torná-la mais eficaz para buscar informações relevantes em uma base de conhecimento ou na web.\n"
    "Instruções:\n"
    "1. Analise a pergunta original.\n"
    "2. Identifique os conceitos chave e a intenção principal.\n"
    "3. Reescreva a pergunta de forma mais clara, concisa e focada em palavras-chave, adequada para um sistema de busca.\n"
    "4. Mantenha o significado original da pergunta.\n"
    "5. Gere sua resposta no formato JSON especificado ('TransformedQuery'), contendo apenas a query reescrita no campo 'transformed_query'."
)

# --- Instância do Agente Transformer PydanticAI ---
try:
    transformer_agent = Agent(
        model=get_pydantic_ai_llm(model_name=MODEL_TRANSFORMER), # Usando OpenRouter
        output_type=TransformedQuery,
        system_prompt=TRANSFORMER_SYSTEM_PROMPT,
        model_settings={"temperature": 0} # Transformação focada
    )
except Exception as e:
    print(f"ERRO ao inicializar Transformer Agent: {e}")
    transformer_agent = None

async def transform_query(state: AgentState) -> dict:
    """Transforma a query original para melhorar a recuperação."""
    print("---NODE: TRANSFORM QUERY---")
    if transformer_agent is None:
        print("  ERRO: Transformer Agent não inicializado. Usando query original.")
        # Mantém a query atual sem transformação
        return {}

    original_query = state.get("query")
    if not original_query:
        print("  ERRO: Query original não encontrada no estado.")
        return {}

    original_query_text = original_query.text
    print(f"  Transformando query: '{original_query_text}'")

    agent_input_content = f"Pergunta Original do Usuário: {original_query_text}\n\nReescreva esta pergunta para otimizar a busca."

    try:
        # result = await asyncio.to_thread(transformer_agent.run, agent_input_content)
        result = await transformer_agent.run(agent_input_content) # Alterado para await direto
        transformed_output: TransformedQuery = result.output
        new_query = transformed_output.transformed_query
        print(f"  Query Transformada: '{new_query}'")
        # Atualiza a 'current_query' para ser usada pelas buscas subsequentes
        # Também guarda em 'transformed_query' para referência, se necessário
        return {"current_query": new_query, "transformed_query": new_query}

    except Exception as e:
        import traceback
        print(f"  ALERTA: Erro durante a transformação da query: {e}. Usando query original.")
        traceback.print_exc()
        # Mantém a query original em caso de erro
        return {} 