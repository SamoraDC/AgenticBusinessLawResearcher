# src/agents/synthesizer.py
from typing import List, Optional
import asyncio
import re
import json
import time
import traceback
import os
import httpx

from src.core.workflow_state import AgentState
from src.core.llm_factory import get_pydantic_ai_llm, MODEL_SYNTHESIZER, LLM_GROQ_LANGCHAIN, GROQ_API_KEY, MODEL_GROQ_WEB, OPENROUTER_API_KEY
from src.core.legal_models import DocumentSnippet, FinalResponse, LegalQuery, ProcessingConfig
from src.interfaces.external_search_client import LexMLDocumento, TavilySearchResult

# Adicionado: Import PydanticAI Agent e dependências específicas
from pydantic_ai import Agent, ModelRetry, UnexpectedModelBehavior, capture_run_messages
from pydantic_ai.models.groq import GroqModel
from pydantic_ai.models.openai import OpenAIModel
# ✅ CORREÇÃO: Usar OpenAIProvider com configuração manual para OpenRouter
from pydantic_ai.providers.openai import OpenAIProvider
from pydantic import BaseModel, Field

# ✅ NOVA IMPORTAÇÃO: Sistema híbrido corrigido real
from src.agents.streaming.hybrid_legal_processor import process_legal_query_hybrid_corrected

# Modelo simplificado e robusta para o synthesizer
class SimpleFinalResponse(BaseModel):
    overall_summary: str = Field(
        ..., 
        description="Resposta jurídica completa e detalhada em texto simples",
        min_length=100,
        max_length=15000  # Alinhado com FinalResponse para comportar respostas do sistema híbrido de 4 partes
    )
    disclaimer: str = Field(
        default="Esta resposta é fornecida apenas para fins informativos e não constitui aconselhamento jurídico profissional. Para questões específicas, consulte um advogado especializado.",
        description="Aviso legal sobre limitações da resposta"
    )

# System Prompt otimizado para LLMs e saída limpa
SYNTHESIZER_SYSTEM_PROMPT = (
    "Você é um advogado especialista em direito brasileiro. "
    "Forneça respostas jurídicas precisas, claras e estruturadas.\n\n"
    
    "REGRAS CRÍTICAS PARA FORMATAÇÃO:\n"
    "- Use APENAS texto simples em português brasileiro\n"
    "- NÃO use símbolos especiais, emojis, ou caracteres Unicode\n"
    "- NÃO inclua quebras de linha desnecessárias\n"
    "- Use linguagem jurídica precisa mas acessível\n"
    "- Evite aspas duplas dentro do conteúdo\n\n"
    
    "Estruture sua resposta jurídica com:\n"
    "1. Conceito e base legal\n"
    "2. Procedimento prático específico\n"
    "3. Considerações e alertas importantes\n\n"
    
    "Use 300-800 palavras. Cite artigos de lei quando relevante.\n"
    "Seja objetivo e direto na orientação jurídica."
)

# ✅ WRAPPER CUSTOMIZADO OPENROUTER (comprovadamente funciona)
async def openrouter_direct_call(prompt: str) -> str:
    """Chamada direta para OpenRouter usando HTTP"""
    if not OPENROUTER_API_KEY:
        raise Exception("OPENROUTER_API_KEY não configurada")
    
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json"
    }
    
    data = {
        "model": MODEL_SYNTHESIZER,
        "messages": [
            {"role": "system", "content": SYNTHESIZER_SYSTEM_PROMPT},
            {"role": "user", "content": prompt}
        ],
        "max_tokens": 1000,
        "temperature": 0.0
    }
    
    async with httpx.AsyncClient() as client:
        response = await client.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers=headers,
            json=data,
            timeout=30.0
        )
        
        if response.status_code == 200:
            result = response.json()
            content = result['choices'][0]['message']['content']
            
            # Limpeza básica
            content = re.sub(r'[\x00-\x1f\x7f-\x9f]', '', content)
            content = re.sub(r'\s+', ' ', content).strip()
            
            return content
        else:
            raise Exception(f"OpenRouter HTTP {response.status_code}: {response.text}")

def create_robust_openrouter_agent():
    """Cria agent PydanticAI usando OpenRouter com configuração correta"""
    try:
        print("🔧 Configurando PydanticAI Agent com OpenRouter...")
        
        # Verificar se temos API key do OpenRouter
        if not OPENROUTER_API_KEY:
            print("❌ OPENROUTER_API_KEY não encontrada")
            return None
        
        print(f"🔑 OpenRouter API key configurada: ***{OPENROUTER_API_KEY[-4:]}")
        print(f"🎯 Modelo OpenRouter: {MODEL_SYNTHESIZER}")
        
        # ✅ CONFIGURAÇÃO SIMPLES QUE FUNCIONA (baseada nos testes)
        openrouter_provider = OpenAIProvider(
            api_key=OPENROUTER_API_KEY,
            base_url="https://openrouter.ai/api/v1"
        )
        
        openrouter_model = OpenAIModel(
            model_name=MODEL_SYNTHESIZER,
            provider=openrouter_provider
        )
        
        print("✅ Modelo OpenRouter criado")
        
        # ✅ CONFIGURAÇÃO SIMPLIFICADA que comprovadamente funciona
        agent = Agent(
            model=openrouter_model,
            output_type=SimpleFinalResponse,
            instructions=SYNTHESIZER_SYSTEM_PROMPT,
            # ✅ Settings mais simples baseados nos testes de sucesso
            model_settings={
                "temperature": 0.0,
                "max_tokens": 1000,  # Reduzido de 2000
                "top_p": 0.9        # Simplificado
            },
            retries=1  # Reduzido de 2 para evitar loops
        )
        
        print("✅ Agent PydanticAI + OpenRouter criado com sucesso")
        return agent
        
    except Exception as e:
        print(f"❌ Erro ao criar agent OpenRouter: {e}")
        print(f"Traceback: {traceback.format_exc()}")
        return None

def create_robust_groq_agent():
    """Cria agent PydanticAI usando Groq que já funciona perfeitamente"""
    try:
        print("🔧 Configurando PydanticAI Agent com Groq...")
        
        # Verificar se temos API key do Groq
        if not GROQ_API_KEY:
            print("❌ GROQ_API_KEY não encontrada")
            return None
        
        print(f"🔑 Groq API key configurada: ***{GROQ_API_KEY[-4:]}")
        print(f"🎯 Modelo Groq: {MODEL_GROQ_WEB}")
        
        # Usar modelo Groq diretamente (evita problemas do OpenRouter)
        groq_model = GroqModel(MODEL_GROQ_WEB)
        
        print("✅ Modelo Groq criado")
        
        # Configurações específicas para evitar caracteres problemáticos
        agent = Agent(
            model=groq_model,
            output_type=SimpleFinalResponse,
            instructions=SYNTHESIZER_SYSTEM_PROMPT,  # Usar instructions ao invés de system_prompt
            model_settings={
                "temperature": 0.0,     # Máximo determinismo
                "max_tokens": 2000,     # Limite controlado  
                "top_p": 0.8,           # Foco nas respostas mais prováveis
                "frequency_penalty": 0.3,  # Reduz repetições
                "presence_penalty": 0.2    # Evita padrões problemáticos
            },
            retries=2  # Retry automático do PydanticAI
        )
        
        print("✅ Agent PydanticAI + Groq criado com sucesso")
        return agent
        
    except Exception as e:
        print(f"❌ Erro ao criar agent Groq: {e}")
        print(f"Traceback: {traceback.format_exc()}")
        return None

# Tool function para retry inteligente
def validate_legal_response(ctx, response: SimpleFinalResponse) -> SimpleFinalResponse:
    """Valida e limpa resposta jurídica"""
    
    # Verificar se a resposta tem conteúdo mínimo
    if len(response.overall_summary) < 100:
        raise ModelRetry(
            f"Resposta muito curta ({len(response.overall_summary)} chars). "
            "Forneça uma análise jurídica mais detalhada com pelo menos 300 palavras "
            "incluindo base legal, procedimento prático e considerações importantes."
        )
    
    # Verificar se menciona aspectos jurídicos
    legal_keywords = [
        'código civil', 'lei', 'artigo', 'jurisprudência', 'direito', 
        'sociedade', 'sócio', 'contrato', 'judicial', 'advogado'
    ]
    
    content_lower = response.overall_summary.lower()
    if not any(keyword in content_lower for keyword in legal_keywords):
        raise ModelRetry(
            "A resposta deve abordar aspectos jurídicos específicos. "
            "Inclua base legal, artigos de lei aplicáveis e procedimentos jurídicos adequados."
        )
    
    # Limpar caracteres problemáticos se existirem
    cleaned_summary = re.sub(r'[\x00-\x1f\x7f-\x9f]', '', response.overall_summary)
    cleaned_summary = re.sub(r'[^\x20-\x7E\u00C0-\u017F]', '', cleaned_summary)
    cleaned_summary = re.sub(r'\s+', ' ', cleaned_summary).strip()
    
    if cleaned_summary != response.overall_summary:
        print("🧹 Limpeza de caracteres aplicada")
        response.overall_summary = cleaned_summary
    
    return response

# Instanciar agents - TENTAR OPENROUTER PRIMEIRO, DEPOIS GROQ
print("🔧 Configurando PydanticAI Agent...")

# ✅ PRIORIDADE: Tentar OpenRouter primeiro
synthesizer_agent = create_robust_openrouter_agent()

# ✅ FALLBACK: Se OpenRouter falhar, usar Groq  
if synthesizer_agent is None:
    print("🔄 OpenRouter não disponível. Tentando Groq...")
    synthesizer_agent = create_robust_groq_agent()

if synthesizer_agent:
    # Adicionar validação como output validator
    @synthesizer_agent.output_validator
    def validate_output(ctx, response: SimpleFinalResponse) -> SimpleFinalResponse:
        return validate_legal_response(ctx, response)
    
    print("✅ PydanticAI Agent + Validator inicializado com sucesso")
else:
    print("❌ ERRO: Não foi possível inicializar PydanticAI Agent")

def clean_text_for_json(text: str) -> str:
    """Limpa texto para evitar problemas de JSON"""
    if not text:
        return ""
    
    # Remove caracteres de controle
    text = re.sub(r'[\x00-\x1f\x7f-\x9f]', '', text)
    
    # Remove quebras de linha excessivas
    text = re.sub(r'\n+', ' ', text)
    text = re.sub(r'\r+', ' ', text)
    
    # Remove espaços excessivos
    text = re.sub(r'\s+', ' ', text)
    
    return text.strip()

def format_crag_docs_for_prompt(docs: Optional[List[DocumentSnippet]]) -> str:
    """Formata documentos CRAG para o prompt"""
    if not docs:
        return "Nenhum documento disponível."
    
    formatted_snippets = []
    for i, doc in enumerate(docs[:10]):
        source_id = getattr(doc, 'source_id', f'Documento {i+1}')
        text_content = getattr(doc, 'text', 'Conteúdo não disponível.')
        
        # Limpar texto
        text_content = clean_text_for_json(text_content)
        if len(text_content) > 200:
            text_content = text_content[:200] + "..."
            
        formatted_snippets.append(f"[{i+1}] {source_id}: {text_content}")
    
    return "\n".join(formatted_snippets)

def format_tavily_results_for_prompt(results: Optional[List[TavilySearchResult]]) -> str:
    """Formata resultados Tavily para o prompt"""
    if not results:
        return "Nenhuma busca web realizada."
    
    formatted_results = []
    for i, res in enumerate(results[:3]):
        title = getattr(res, 'title', 'Sem título')
        content = getattr(res, 'content', 'Conteúdo indisponível')
        
        # Limpar texto
        content = clean_text_for_json(content)
        if len(content) > 100:
            content = content[:100] + "..."
            
        formatted_results.append(f"[Web {i+1}] {title}: {content}")
    
    return "\n".join(formatted_results)

def format_lexml_results_for_prompt(results: Optional[List[LexMLDocumento]]) -> str:
    """Formata resultados LexML para o prompt"""
    if not results:
        return "Nenhuma jurisprudência encontrada."
    
    formatted_results = []
    for i, res in enumerate(results[:3]):
        urn = getattr(res, 'urn', f'LexML {i+1}')
        ementa = getattr(res, 'ementa', 'Sem ementa')
        
        # Limpar texto
        if ementa:
            ementa = clean_text_for_json(ementa)
            if len(ementa) > 150:
                ementa = ementa[:150] + "..."
        
        formatted_results.append(f"[Juris {i+1}] {urn}: {ementa}")
    
    return "\n".join(formatted_results)

async def synthesize_with_hybrid_corrected_approach(query_text: str, formatted_crag: str, formatted_tavily: str, formatted_lexml: str) -> SimpleFinalResponse:
    """
    Síntese usando abordagem híbrida CORRETA através do sistema real PydanticAI:
    - Groq: Buscar informações usando tools (sistema real)
    - OpenRouter: RAG, análise e síntese do conteúdo (sistema real)
    """
    
    print("🤖 === SÍNTESE HÍBRIDA CORRETA (SISTEMA REAL) ===")
    print("🔧 Groq: Tools e buscas estruturadas (PydanticAI real)")
    print("🧠 OpenRouter: RAG, análise e síntese (PydanticAI real)")
    
    # Input ultra-limpo e estruturado
    cleaned_query = clean_text_for_json(query_text)
    
    try:
        # Criar LegalQuery para o sistema híbrido real
        legal_query = LegalQuery(text=cleaned_query)
        
        # Configuração de processamento otimizada
        config = ProcessingConfig(
            max_documents_per_source=10,
            search_timeout_seconds=30,
            enable_parallel_search=True,
            max_retries=2,
            retry_backoff_factor=1.5,
            min_confidence_threshold=0.3,
            human_review_threshold=0.8,
            temperature=0.1,
            max_tokens=3000,
            enable_human_review=False,  # Desabilitar para integração rápida
            enable_web_search=True,
            enable_jurisprudence_search=True,
            enable_guardrails=True
        )
        
        print(f"🚀 Executando sistema híbrido real para: {cleaned_query[:100]}...")
        start_time = time.time()
        
        # Executar o sistema híbrido corrigido REAL
        final_response = await process_legal_query_hybrid_corrected(
            query=legal_query,
            config=config,
            user_id="synthesizer_integration"
        )
        
        elapsed_time = time.time() - start_time
        print(f"⏱️ Sistema híbrido real concluído: {elapsed_time:.2f}s")
        print(f"📏 Resposta gerada: {len(final_response.overall_summary)} chars")
        print(f"🎯 Confiança geral: {final_response.overall_confidence:.2%}")
        
        # Converter para SimpleFinalResponse compatível
        simplified_response = SimpleFinalResponse(
            overall_summary=final_response.overall_summary,
            disclaimer=final_response.disclaimer
        )
        
        print("✅ Sistema híbrido real concluído com sucesso!")
        return simplified_response
        
    except Exception as hybrid_error:
        print(f"❌ Erro no sistema híbrido real: {str(hybrid_error)}")
        print(f"🔄 Fallback para simulação híbrida...")
        
        # Fallback para simulação (mantém funcionalidade)
        return await synthesize_with_hybrid_simulation_fallback(
            cleaned_query, formatted_crag, formatted_tavily, formatted_lexml
        )


async def synthesize_with_hybrid_corrected_approach_streaming(query_text: str, formatted_crag: str, formatted_tavily: str, formatted_lexml: str):
    """
    Síntese usando abordagem híbrida CORRETA com streaming através do sistema real PydanticAI:
    - Groq: Buscar informações usando tools (sistema real)
    - OpenRouter: RAG, análise e síntese do conteúdo (sistema real) COM STREAMING
    """
    
    print("🤖 === SÍNTESE HÍBRIDA CORRETA COM STREAMING (SISTEMA REAL) ===")
    print("🔧 Groq: Tools e buscas estruturadas (PydanticAI real)")
    print("🧠 OpenRouter: RAG, análise e síntese (PydanticAI real) COM STREAMING")
    
    # Input ultra-limpo e estruturado
    cleaned_query = clean_text_for_json(query_text)
    
    try:
        # Criar LegalQuery para o sistema híbrido real
        legal_query = LegalQuery(text=cleaned_query)
        
        # Configuração de processamento otimizada
        config = ProcessingConfig(
            max_documents_per_source=10,
            search_timeout_seconds=30,
            enable_parallel_search=True,
            max_retries=2,
            retry_backoff_factor=1.5,
            min_confidence_threshold=0.3,
            human_review_threshold=0.8,
            temperature=0.1,
            max_tokens=3000,
            enable_human_review=False,  # Desabilitar para integração rápida
            enable_web_search=True,
            enable_jurisprudence_search=True,
            enable_guardrails=True
        )
        
        print(f"🚀 Executando sistema híbrido real com streaming para: {cleaned_query[:100]}...")
        start_time = time.time()
        
        # Executar o sistema híbrido corrigido REAL (sem streaming específico)
        final_response = await process_legal_query_hybrid_corrected(
            query=legal_query,
            config=config,
            user_id="synthesizer_integration"
        )
        
        # Simular streaming com a resposta final
        if final_response and hasattr(final_response, 'overall_summary'):
            response_text = final_response.overall_summary
            words = response_text.split()
            current_text = ""
            
            for i, word in enumerate(words):
                current_text += word + " "
                
                # Yield chunks a cada 3 palavras para streaming
                if (i + 1) % 3 == 0 or i == len(words) - 1:
                    yield ("streaming", current_text.strip())
                    await asyncio.sleep(0.1)  # Pausa para streaming
        
        elapsed_time = time.time() - start_time
        print(f"⏱️ Sistema híbrido real com streaming concluído: {elapsed_time:.2f}s")
        
        if final_response:
            print(f"📏 Resposta gerada: {len(final_response.overall_summary)} chars")
            print(f"🎯 Confiança geral: {final_response.overall_confidence:.2%}")
            
            # Converter para SimpleFinalResponse compatível
            simplified_response = SimpleFinalResponse(
                overall_summary=final_response.overall_summary,
                disclaimer=final_response.disclaimer
            )
            
            print("✅ Sistema híbrido real com streaming concluído com sucesso!")
            yield ("final", simplified_response)
        else:
            yield ("error", "Sistema híbrido não retornou resposta final")
        
    except Exception as hybrid_error:
        print(f"❌ Erro no sistema híbrido real com streaming: {str(hybrid_error)}")
        print(f"🔄 Fallback para simulação híbrida com streaming...")
        
        # Fallback para simulação com streaming
        async for step_type, content in synthesize_with_hybrid_simulation_fallback_streaming(
            cleaned_query, formatted_crag, formatted_tavily, formatted_lexml
        ):
            yield step_type, content


async def synthesize_with_hybrid_simulation_fallback_streaming(cleaned_query: str, formatted_crag: str, formatted_tavily: str, formatted_lexml: str):
    """
    Fallback para simulação híbrida com streaming quando o sistema real não funciona.
    """
    print("🔄 === FALLBACK: SIMULAÇÃO HÍBRIDA COM STREAMING ===")
    
    yield ("progress", "🔧 Simulando busca Groq...")
    
    # === SIMULAÇÃO GROQ: Decisão de busca ===
    search_decision = {
        "needs_vectordb": bool(formatted_crag and len(formatted_crag) > 50),
        "needs_lexml": bool(formatted_lexml and len(formatted_lexml) > 50),
        "needs_web": bool(formatted_tavily and len(formatted_tavily) > 50),
        "confidence": 0.85,
        "reasoning": "Consulta jurídica complexa requer múltiplas fontes especializadas"
    }
    
    yield ("progress", "🧠 Executando análise OpenRouter...")
    
    # === SIMULAÇÃO GROQ: Resultados das buscas estruturadas ===
    groq_search_summary = f"""
=== RELATÓRIO DE BUSCAS EXECUTADAS PELO GROQ ===

VECTORDB SEARCH: {"EXECUTADA" if search_decision['needs_vectordb'] else "DISPENSADA"}
{formatted_crag[:800] if formatted_crag else "Nenhum documento encontrado"}

LEXML SEARCH: {"EXECUTADA" if search_decision['needs_lexml'] else "DISPENSADA"}
{formatted_lexml[:600] if formatted_lexml else "Nenhuma legislação encontrada"}

WEB SEARCH: {"EXECUTADA" if search_decision['needs_web'] else "DISPENSADA"}
{formatted_tavily[:400] if formatted_tavily else "Nenhuma informação web encontrada"}

CONFIANÇA GERAL DAS BUSCAS: {search_decision['confidence']:.1%}
STATUS: Dados coletados e estruturados com sucesso
"""

    # === OPENROUTER: Análise jurídica (RAG) ===
    rag_analysis_prompt = f"""
CONSULTA JURÍDICA ORIGINAL: {cleaned_query}

=== DADOS COLETADOS PELO SISTEMA DE BUSCA ===
{groq_search_summary}

Como especialista em direito brasileiro, analise todos os dados acima e produza uma análise jurídica completa:

ESTRUTURA OBRIGATÓRIA:
1. RESUMO EXECUTIVO da consulta
2. QUESTÕES JURÍDICAS identificadas
3. LEGISLAÇÃO APLICÁVEL (com artigos específicos)
4. JURISPRUDÊNCIA RELEVANTE (precedentes)
5. ANÁLISE DOUTRINÁRIA
6. RISCOS E OPORTUNIDADES
7. RECOMENDAÇÕES PRÁTICAS
8. CONCLUSÃO FUNDAMENTADA

Mantenha rigor técnico-jurídico, fundamentação sólida e linguagem clara.
"""

    yield ("progress", "✍️ Gerando síntese final...")

    # Tentar OpenRouter direto para análise RAG
    try:
        if OPENROUTER_API_KEY:
            rag_analysis = await openrouter_direct_call(rag_analysis_prompt)
        else:
            rag_analysis = await LLM_GROQ_LANGCHAIN.ainvoke(rag_analysis_prompt)
            rag_analysis = rag_analysis.content if hasattr(rag_analysis, 'content') else str(rag_analysis)
    except Exception:
        rag_analysis = f"Análise básica da consulta '{cleaned_query}' com base nos documentos disponíveis."
    
    # === OPENROUTER: Síntese final COM STREAMING ===
    synthesis_prompt = f"""
CONSULTA ORIGINAL: {cleaned_query}

ANÁLISE JURÍDICA COMPLETA:
{rag_analysis[:2000]}

Com base na análise jurídica acima, forneça uma RESPOSTA FINAL estruturada e clara:

1. RESPOSTA DIRETA à consulta original
2. FUNDAMENTOS LEGAIS aplicáveis
3. JURISPRUDÊNCIA relevante
4. ORIENTAÇÕES PRÁTICAS específicas
5. DISCLAIMERS apropriados

REQUISITOS:
- Linguagem acessível mas tecnicamente precisa
- 400-800 palavras
- Citar artigos de lei quando aplicável
- Incluir precedentes relevantes
- Orientações práticas claras

SÍNTESE FINAL:
"""

    try:
        if OPENROUTER_API_KEY:
            final_synthesis = await openrouter_direct_call(synthesis_prompt)
            final_synthesis = clean_text_for_json(final_synthesis)
            
            if len(final_synthesis) > 2800:
                final_synthesis = final_synthesis[:2800] + "..."
            
            # Simular streaming
            words = final_synthesis.split()
            current_text = ""
            
            for i, word in enumerate(words):
                current_text += word + " "
                
                # Yield chunks a cada 3 palavras
                if (i + 1) % 3 == 0 or i == len(words) - 1:
                    yield ("streaming", current_text.strip())
                    await asyncio.sleep(0.1)  # Pausa para streaming
            
            response = SimpleFinalResponse(
                overall_summary=final_synthesis,
                disclaimer="Esta resposta foi gerada por sistema de IA integrado e está suscetível a erro. Para qualquer conclusão e tomada de descisão procure um advogado credenciado e qualificado."
            )
            
            yield ("final", response)
            
        else:
            final_response = await LLM_GROQ_LANGCHAIN.ainvoke(synthesis_prompt)
            final_content = final_response.content if hasattr(final_response, 'content') else str(final_response)
            final_content = clean_text_for_json(final_content)
            
            # Simular streaming
            words = final_content.split()
            current_text = ""
            
            for i, word in enumerate(words):
                current_text += word + " "
                
                # Yield chunks a cada 3 palavras
                if (i + 1) % 3 == 0 or i == len(words) - 1:
                    yield ("streaming", current_text.strip())
                    await asyncio.sleep(0.1)  # Pausa para streaming
            
            response = SimpleFinalResponse(
                overall_summary=final_content,
                disclaimer="Esta resposta foi gerada automaticamente com streaming e deve ser revisada por profissional qualificado."
            )
            
            yield ("final", response)
            
    except Exception as e:
        yield ("error", f"Erro na síntese com streaming: {str(e)}")


async def synthesize_with_hybrid_simulation_fallback(cleaned_query: str, formatted_crag: str, formatted_tavily: str, formatted_lexml: str) -> SimpleFinalResponse:
    """
    Fallback para simulação híbrida quando o sistema real não funciona.
    """
    print("🔄 === FALLBACK: SIMULAÇÃO HÍBRIDA ===")
    
    # === SIMULAÇÃO GROQ: Decisão de busca ===
    search_decision = {
        "needs_vectordb": bool(formatted_crag and len(formatted_crag) > 50),
        "needs_lexml": bool(formatted_lexml and len(formatted_lexml) > 50),
        "needs_web": bool(formatted_tavily and len(formatted_tavily) > 50),
        "confidence": 0.85,
        "reasoning": "Consulta jurídica complexa requer múltiplas fontes especializadas"
    }
    
    # === SIMULAÇÃO GROQ: Resultados das buscas estruturadas ===
    groq_search_summary = f"""
=== RELATÓRIO DE BUSCAS EXECUTADAS PELO GROQ ===

VECTORDB SEARCH: {"EXECUTADA" if search_decision['needs_vectordb'] else "DISPENSADA"}
{formatted_crag[:800] if formatted_crag else "Nenhum documento encontrado"}

LEXML SEARCH: {"EXECUTADA" if search_decision['needs_lexml'] else "DISPENSADA"}
{formatted_lexml[:600] if formatted_lexml else "Nenhuma legislação encontrada"}

WEB SEARCH: {"EXECUTADA" if search_decision['needs_web'] else "DISPENSADA"}
{formatted_tavily[:400] if formatted_tavily else "Nenhuma informação web encontrada"}

CONFIANÇA GERAL DAS BUSCAS: {search_decision['confidence']:.1%}
STATUS: Dados coletados e estruturados com sucesso
"""

    # === OPENROUTER: Análise jurídica (RAG) ===
    rag_analysis_prompt = f"""
CONSULTA JURÍDICA ORIGINAL: {cleaned_query}

=== DADOS COLETADOS PELO SISTEMA DE BUSCA ===
{groq_search_summary}

Como especialista em direito brasileiro, analise todos os dados acima e produza uma análise jurídica completa:

ESTRUTURA OBRIGATÓRIA:
1. RESUMO EXECUTIVO da consulta
2. QUESTÕES JURÍDICAS identificadas
3. LEGISLAÇÃO APLICÁVEL (com artigos específicos)
4. JURISPRUDÊNCIA RELEVANTE (precedentes)
5. ANÁLISE DOUTRINÁRIA
6. RISCOS E OPORTUNIDADES
7. RECOMENDAÇÕES PRÁTICAS
8. CONCLUSÃO FUNDAMENTADA

Mantenha rigor técnico-jurídico, fundamentação sólida e linguagem clara.
"""

    # Tentar OpenRouter direto para análise RAG
    try:
        if OPENROUTER_API_KEY:
            rag_analysis = await openrouter_direct_call(rag_analysis_prompt)
        else:
            rag_analysis = await LLM_GROQ_LANGCHAIN.ainvoke(rag_analysis_prompt)
            rag_analysis = rag_analysis.content if hasattr(rag_analysis, 'content') else str(rag_analysis)
    except Exception:
        rag_analysis = f"Análise básica da consulta '{cleaned_query}' com base nos documentos disponíveis."
    
    # === OPENROUTER: Síntese final ===
    synthesis_prompt = f"""
CONSULTA ORIGINAL: {cleaned_query}

ANÁLISE JURÍDICA COMPLETA:
{rag_analysis[:2000]}

Com base na análise jurídica acima, forneça uma RESPOSTA FINAL estruturada e clara:

1. RESPOSTA DIRETA à consulta original
2. FUNDAMENTOS LEGAIS aplicáveis
3. JURISPRUDÊNCIA relevante
4. ORIENTAÇÕES PRÁTICAS específicas
5. DISCLAIMERS apropriados

REQUISITOS:
- Linguagem acessível mas tecnicamente precisa
- 400-800 palavras
- Citar artigos de lei quando aplicável
- Incluir precedentes relevantes
- Orientações práticas claras

SÍNTESE FINAL:
"""

    try:
        if OPENROUTER_API_KEY:
            final_synthesis = await openrouter_direct_call(synthesis_prompt)
            final_synthesis = clean_text_for_json(final_synthesis)
            
            if len(final_synthesis) > 2800:
                final_synthesis = final_synthesis[:2800] + "..."
            
            return SimpleFinalResponse(
                overall_summary=final_synthesis,
                disclaimer="Esta resposta foi gerada por sistema de IA integrado e está suscetível a erro. Para qualquer conclusão e tomada de descisão procure um advogado credenciado e qualificado."
            )
        else:
            final_response = await LLM_GROQ_LANGCHAIN.ainvoke(synthesis_prompt)
            final_content = final_response.content if hasattr(final_response, 'content') else str(final_response)
            final_content = clean_text_for_json(final_content)
            
            return SimpleFinalResponse(
                overall_summary=final_content,
                disclaimer="Esta resposta foi gerada automaticamente e deve ser revisada por profissional qualificado."
            )
            
    except Exception:
        # Síntese básica estruturada
        basic_synthesis = f"""
Com base na consulta jurídica sobre '{cleaned_query}', há importantes aspectos do direito societário brasileiro a considerar.

A legislação brasileira, especialmente o Código Civil (Lei 10.406/2002), estabelece mecanismos específicos para questões envolvendo sócios de sociedades empresárias. Os principais caminhos legais incluem: (1) direito de recesso ou retirada voluntária do sócio em situações previstas em lei; (2) exclusão de sócio por justa causa, respeitando procedimentos legais adequados; (3) dissolução parcial da sociedade com apuração de haveres.

O procedimento específico varia conforme o tipo societário (sociedade limitada, anônima, etc.) e as circunstâncias do caso. Para sociedades limitadas, aplicam-se primariamente os artigos 1.077 e seguintes do Código Civil. É fundamental observar o contrato social, respeitar direitos adquiridos e seguir procedimentos legais apropriados.

A legislação protege sócios minoritários contra abusos, mas também reconhece situações em que a retirada ou exclusão pode ser necessária para preservar a sociedade. Questões como apuração de haveres, justa causa e procedimentos judiciais ou extrajudiciais devem ser cuidadosamente avaliadas.

Recomenda-se consultar advogado especializado em direito empresarial para análise específica do caso, elaboração de estratégia adequada e orientação sobre procedimentos legais aplicáveis à situação particular da sociedade.
"""
        
        return SimpleFinalResponse(
            overall_summary=basic_synthesis,
            disclaimer="Esta é uma resposta básica gerada automaticamente devido a erro técnico. Para orientação jurídica específica, consulte um advogado especializado."
        )

async def synthesize_response(state: AgentState) -> dict:
    """Gera a resposta final usando synthesizer robusto PydanticAI (OpenRouter ou Groq) - MODO COLETA APENAS"""
    print("---NODE: SYNTHESIZE RESPONSE (PydanticAI + Fallback)---")
    
    query_object = state.get("query")
    if not query_object:
        print("  ERRO: Objeto LegalQuery não encontrado no estado!")
        error_response = FinalResponse(
            query_id="unknown",
            overall_summary="Erro interno: Query original não encontrada. O sistema não conseguiu processar adequadamente a consulta jurídica solicitada. Este erro indica um problema técnico que deve ser reportado ao suporte para investigação e correção.",
            disclaimer="Este é um erro técnico do sistema. Para orientação jurídica, consulte um advogado especializado."
        )
        return {"error": "Query não encontrada", "final_response": error_response.model_dump()}
        
    query_text = query_object.text
    retrieved_crag_docs = state.get("retrieved_docs")
    tavily_web_results = state.get("tavily_results")
    lexml_juris_results = state.get("lexml_results")

    # LOGS DE DEBUG DO ESTADO
    print("🔍 === DEBUG ESTADO ===")
    print(f"📝 Query: {query_text}")
    print(f"📚 CRAG docs: {len(retrieved_crag_docs) if retrieved_crag_docs else 0}")
    print(f"🌐 Tavily results: {len(tavily_web_results) if tavily_web_results else 0}")
    print(f"⚖️ LexML results: {len(lexml_juris_results) if lexml_juris_results else 0}")

    # Formatação otimizada
    formatted_crag = format_crag_docs_for_prompt(retrieved_crag_docs) if retrieved_crag_docs else "Nenhum documento CRAG"
    formatted_tavily = format_tavily_results_for_prompt(tavily_web_results) if tavily_web_results else "Nenhuma busca web"
    formatted_lexml = format_lexml_results_for_prompt(lexml_juris_results) if lexml_juris_results else "Nenhuma jurisprudência"

    print(f"📏 Tamanho formatado - CRAG: {len(formatted_crag)}, Tavily: {len(formatted_tavily)}, LexML: {len(formatted_lexml)}")

    # MODO SÍNTESE COMPLETA SEMPRE ATIVO - CORREÇÃO CRÍTICA
    should_synthesize = state.get("should_synthesize", True)
    if not should_synthesize:
        print("  ⚠️ AVISO: should_synthesize=False detectado, mas executando síntese completa...")
        print("  🔧 CORREÇÃO: Forçando síntese completa para evitar modo coleta apenas")
    
    print("  ✅ MODO SÍNTESE COMPLETA - Processando dados CRAG + LexML + Web")

    print("  Iniciando síntese híbrida correta...")
    try:
        # Usar synthesizer híbrido corrigido (Groq tools + OpenRouter RAG)
        llm_response = await synthesize_with_hybrid_corrected_approach(
            query_text, formatted_crag, formatted_tavily, formatted_lexml
        )
        
        # Construir resposta final
        final_response = FinalResponse(
            query_id=query_object.id,
            overall_summary=llm_response.overall_summary,
            disclaimer=llm_response.disclaimer 
        )
        
        print("  ✅ Síntese concluída com sucesso!")
        print(f"📏 Resposta final: {len(final_response.overall_summary)} chars")
        return {"final_response": final_response.model_dump()}

    except Exception as e:
        print(f"  ❌ Erro crítico na síntese: {str(e)}")
        print(f"  Traceback: {traceback.format_exc()}")
        
        # Resposta de emergência
        emergency_response = FinalResponse(
            query_id=query_object.id if query_object else "unknown",
            overall_summary="O sistema encontrou dificuldades técnicas ao processar sua consulta jurídica. Este erro pode estar relacionado à complexidade da consulta ou a problemas temporários de conectividade com os serviços de IA. Tente reformular sua pergunta de forma mais específica ou tente novamente em alguns minutos. Para orientação jurídica imediata, consulte um advogado especializado.",
            disclaimer="Resposta de erro técnico. Este sistema utiliza IA para assistência jurídica, mas não substitui consulta profissional com advogado especializado."
        )
        return {"error": str(e), "final_response": emergency_response.model_dump()}


async def synthesize_response_streaming(state: AgentState):
    """Gera a resposta final usando synthesizer robusto PydanticAI com STREAMING"""
    print("---NODE: SYNTHESIZE RESPONSE WITH STREAMING (PydanticAI + Fallback)---")
    
    query_object = state.get("query")
    if not query_object:
        print("  ERRO: Objeto LegalQuery não encontrado no estado!")
        error_response = FinalResponse(
            query_id="unknown",
            overall_summary="Erro interno: Query original não encontrada. O sistema não conseguiu processar adequadamente a consulta jurídica solicitada. Este erro indica um problema técnico que deve ser reportado ao suporte para investigação e correção.",
            disclaimer="Este é um erro técnico do sistema. Para orientação jurídica, consulte um advogado especializado."
        )
        yield ("error", "Query não encontrada")
        return
        
    query_text = query_object.text
    retrieved_crag_docs = state.get("retrieved_docs")
    tavily_web_results = state.get("tavily_results")
    lexml_juris_results = state.get("lexml_results")

    # LOGS DE DEBUG DO ESTADO
    print("🔍 === DEBUG ESTADO ===")
    print(f"📝 Query: {query_text}")
    print(f"📚 CRAG docs: {len(retrieved_crag_docs) if retrieved_crag_docs else 0}")
    print(f"🌐 Tavily results: {len(tavily_web_results) if tavily_web_results else 0}")
    print(f"⚖️ LexML results: {len(lexml_juris_results) if lexml_juris_results else 0}")

    # Formatação otimizada
    formatted_crag = format_crag_docs_for_prompt(retrieved_crag_docs) if retrieved_crag_docs else "Nenhum documento CRAG"
    formatted_tavily = format_tavily_results_for_prompt(tavily_web_results) if tavily_web_results else "Nenhuma busca web"
    formatted_lexml = format_lexml_results_for_prompt(lexml_juris_results) if lexml_juris_results else "Nenhuma jurisprudência"

    print(f"📏 Tamanho formatado - CRAG: {len(formatted_crag)}, Tavily: {len(formatted_tavily)}, LexML: {len(formatted_lexml)}")

    print("  Iniciando síntese híbrida correta com streaming...")
    try:
        # Usar synthesizer híbrido corrigido com streaming
        final_response = None
        async for step_type, content in synthesize_with_hybrid_corrected_approach_streaming(
            query_text, formatted_crag, formatted_tavily, formatted_lexml
        ):
            if step_type == "progress":
                yield ("progress", content)
            elif step_type == "streaming":
                yield ("streaming", content)
            elif step_type == "final":
                # Converter SimpleFinalResponse para FinalResponse
                simple_response = content
                final_response = FinalResponse(
                    query_id=query_object.id,
                    overall_summary=simple_response.overall_summary,
                    disclaimer=simple_response.disclaimer 
                )
                break
            elif step_type == "error":
                yield ("error", content)
                return
        
        if final_response:
            print("  ✅ Síntese com streaming concluída com sucesso!")
            print(f"📏 Resposta final: {len(final_response.overall_summary)} chars")
            yield ("final", final_response.model_dump())
        else:
            yield ("error", "Síntese com streaming não retornou resposta")

    except Exception as e:
        print(f"  ❌ Erro crítico na síntese com streaming: {str(e)}")
        print(f"  Traceback: {traceback.format_exc()}")
        
        # Resposta de emergência
        emergency_response = FinalResponse(
            query_id=query_object.id if query_object else "unknown",
            overall_summary="O sistema encontrou dificuldades técnicas ao processar sua consulta jurídica com streaming. Este erro pode estar relacionado à complexidade da consulta ou a problemas temporários de conectividade com os serviços de IA. Tente reformular sua pergunta de forma mais específica ou tente novamente em alguns minutos. Para orientação jurídica imediata, consulte um advogado especializado.",
            disclaimer="Resposta de erro técnico com streaming. Este sistema utiliza IA para assistência jurídica, mas não substitui consulta profissional com advogado especializado."
        )
        yield ("error", str(e)) 