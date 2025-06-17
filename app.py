import streamlit as st
import asyncio
import sys
import traceback
from datetime import datetime
from typing import Optional, AsyncGenerator, Dict, List, Tuple, Union
import json
import time

# Adicionar src ao path para importações
sys.path.append('src')

from dotenv import load_dotenv

from src.core.legal_models import LegalQuery, Priority, ValidationLevel
from src.core.workflow_builder import build_graph
from src.agents.streaming.response_synthesizer import synthesize_response_streaming

# Importar sistema de observabilidade COMPLETO
from src.core.observability import (
    trace_legal_query, 
    trace_crag_execution, 
    trace_hybrid_processing,
    track_crag_retrieval,
    track_data_integration,
    log_detailed_state,
    log_data_flow_checkpoint,
    log_performance_metrics,
    log_state_transition,
    serialize_for_langfuse,
    extract_metadata
)

# Carregar variáveis de ambiente
load_dotenv()

# Dicionário de traduções
TRANSLATIONS = {
    "pt": {
        "page_title": "🔥 Sistema Jurídico AI Avançado",
        "page_subtitle": "PydanticAI + LangGraph + ChromaDB + LexML + Tavily",
        "page_description": "Sistema de IA jurídica avançado para consultas especializadas ⚡",
        "language_selector": "Idioma / Language",
        "legal_consultation": "💬 Consulta Jurídica",
        "chat_placeholder": "Digite sua consulta jurídica...",
        "processing": "🤖 **Processando sua consulta jurídica...**",
        "processing_complete": "✅ Processamento concluído!",
        "legal_response": "📋 Resposta Jurídica",
        "disclaimer": "⚠️ Disclaimer:",
        "error_processing": "Erro no processamento:",
        "invalid_response_format": "Formato de resposta inválido",
        "unexpected_response_type": "Tipo de resposta inesperado:",
        "processing_failed": "Falha ao processar consulta",
        "unexpected_error": "Erro inesperado no sistema:",
        "technical_details": "Ver detalhes técnicos",
        "footer_title": "Sistema Jurídico AI Avançado",
        "footer_subtitle": "PydanticAI + LangGraph + ChromaDB",
        "footer_apis": "🏗️ Todas as APIs: OpenRouter + Groq + Tavily + LexML + Google",
        "footer_disclaimer": "⚖️ Para consultas específicas, sempre consulte um advogado especializado",
        "disclaimer_text": "Esta resposta foi gerada por sistema de IA integrado e está suscetível a erro. Para qualquer conclusão e tomada de descisão procure um advogado credenciado e qualificado.",
        "how_it_works": "🧠 Como o Sistema Funciona",
        "system_description": "Este é um sistema de IA jurídica avançado que combina múltiplas tecnologias para fornecer respostas jurídicas precisas e bem fundamentadas.",
        "step1_title": "1. 📚 Coleta de Informações",
        "step1_desc": "O sistema busca informações em múltiplas fontes: banco de dados vetorial ChromaDB com documentos jurídicos, jurisprudências do LexML, e informações atualizadas da web via Tavily.",
        "step2_title": "2. 🔍 Análise Inteligente",
        "step2_desc": "Usando LangGraph e PydanticAI, o sistema analisa sua consulta, seleciona os documentos mais relevantes e processa as informações de forma estruturada.",
        "step3_title": "3. ⚖️ Síntese Jurídica",
        "step3_desc": "O sistema sintetiza todas as informações coletadas em uma resposta jurídica completa, com análise doutrinária, jurisprudencial e considerações práticas.",
        "technologies_title": "🛠️ Tecnologias Utilizadas",
        "tech_langgraph": "**LangGraph**: Orquestração de fluxos de trabalho complexos",
        "tech_pydantic": "**PydanticAI**: Validação e estruturação de dados",
        "tech_chromadb": "**ChromaDB**: Banco de dados vetorial para busca semântica",
        "tech_lexml": "**LexML**: Acesso a jurisprudências e legislação",
        "tech_tavily": "**Tavily**: Busca web para informações atualizadas",
        "tech_openrouter": "**OpenRouter**: Modelos de IA avançados para análise",
        "tech_groq": "**Groq**: Processamento rápido de consultas"
    },
    "en": {
        "page_title": "🔥 Advanced Legal AI System",
        "page_subtitle": "PydanticAI + LangGraph + ChromaDB + LexML + Tavily",
        "page_description": "Advanced legal AI system for specialized consultations ⚡",
        "language_selector": "Language / Idioma",
        "legal_consultation": "💬 Legal Consultation",
        "chat_placeholder": "Enter your legal query...",
        "processing": "🤖 **Processing your legal query...**",
        "processing_complete": "✅ Processing completed!",
        "legal_response": "📋 Legal Response",
        "disclaimer": "⚠️ Disclaimer:",
        "error_processing": "Processing error:",
        "invalid_response_format": "Invalid response format",
        "unexpected_response_type": "Unexpected response type:",
        "processing_failed": "Failed to process query",
        "unexpected_error": "Unexpected system error:",
        "technical_details": "View technical details",
        "footer_title": "Advanced Legal AI System",
        "footer_subtitle": "PydanticAI + LangGraph + ChromaDB",
        "footer_apis": "🏗️ All APIs: OpenRouter + Groq + Tavily + LexML + Google",
        "footer_disclaimer": "⚖️ For specific queries, always consult a specialized lawyer",
        "disclaimer_text": "This response was generated by an integrated AI system and is subject to error. For any conclusion and decision making, seek a licensed and qualified lawyer.",
        "how_it_works": "🧠 How the System Works",
        "system_description": "This is an advanced legal AI system that combines multiple technologies to provide accurate and well-founded legal responses.",
        "step1_title": "1. 📚 Information Collection",
        "step1_desc": "The system searches information from multiple sources: ChromaDB vector database with legal documents, LexML jurisprudence, and updated web information via Tavily.",
        "step2_title": "2. 🔍 Intelligent Analysis",
        "step2_desc": "Using LangGraph and PydanticAI, the system analyzes your query, selects the most relevant documents and processes information in a structured way.",
        "step3_title": "3. ⚖️ Legal Synthesis",
        "step3_desc": "The system synthesizes all collected information into a complete legal response, with doctrinal analysis, jurisprudence and practical considerations.",
        "technologies_title": "🛠️ Technologies Used",
        "tech_langgraph": "**LangGraph**: Orchestration of complex workflows",
        "tech_pydantic": "**PydanticAI**: Data validation and structuring",
        "tech_chromadb": "**ChromaDB**: Vector database for semantic search",
        "tech_lexml": "**LexML**: Access to jurisprudence and legislation",
        "tech_tavily": "**Tavily**: Web search for updated information",
        "tech_openrouter": "**OpenRouter**: Advanced AI models for analysis",
        "tech_groq": "**Groq**: Fast query processing"
    }
}

# Configuração da página
st.set_page_config(
    page_title="🔥 Legal AI System - Sistema Jurídico AI",
    page_icon="⚖️",
    layout="wide"
)

# Seletor de idioma no topo
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    language = st.selectbox(
        "🌐 Language / Idioma",
        options=["pt", "en"],
        format_func=lambda x: "🇧🇷 Português" if x == "pt" else "🇺🇸 English",
        index=0
    )

# Obter traduções para o idioma selecionado
t = TRANSLATIONS[language]

# CSS customizado
st.markdown("""
<style>
.main-header {
    background: linear-gradient(90deg, #1e3c72 0%, #2a5298 100%);
    padding: 2rem;
    border-radius: 10px;
    color: white;
    margin-bottom: 2rem;
    text-align: center;
}

.error-box {
    background: #ffebee;
    padding: 1rem;
    border-radius: 8px;
    border-left: 4px solid #f44336;
    margin: 1rem 0;
}

.success-box {
    background: #f1f8e9;
    padding: 1rem;
    border-radius: 8px;
    border-left: 4px solid #4CAF50;
    margin: 1rem 0;
}


</style>
""", unsafe_allow_html=True)

# Header
st.markdown(f"""
<div class="main-header">
    <h1>{t["page_title"]}</h1>
    <h3>{t["page_subtitle"]}</h3>
    <p>{t["page_description"]}</p>
</div>
""", unsafe_allow_html=True)

# Seção opcional "Como Funciona" com expander
with st.expander(f"🧠 {t['how_it_works']}", expanded=False):
    
    # Descrição principal
    st.markdown(f"<p style='font-size: 1.1em; margin-bottom: 1rem; text-align: center; color: #333333; line-height: 1.5;'>{t['system_description']}</p>", unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Passos do sistema usando containers nativos do Streamlit
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown(f"""
        <div style="background: white; padding: 1.5rem; border-radius: 8px; margin: 0.5rem 0; box-shadow: 0 2px 4px rgba(0,0,0,0.1); border-left: 4px solid #4CAF50;">
            <h4 style="color: #2a5298; margin-top: 0;">{t["step1_title"]}</h4>
            <p style="margin-bottom: 0; color: #333333; font-size: 14px; line-height: 1.4;">{t["step1_desc"]}</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div style="background: white; padding: 1.5rem; border-radius: 8px; margin: 0.5rem 0; box-shadow: 0 2px 4px rgba(0,0,0,0.1); border-left: 4px solid #2196F3;">
            <h4 style="color: #2a5298; margin-top: 0;">{t["step2_title"]}</h4>
            <p style="margin-bottom: 0; color: #333333; font-size: 14px; line-height: 1.4;">{t["step2_desc"]}</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div style="background: white; padding: 1.5rem; border-radius: 8px; margin: 0.5rem 0; box-shadow: 0 2px 4px rgba(0,0,0,0.1); border-left: 4px solid #FF9800;">
            <h4 style="color: #2a5298; margin-top: 0;">{t["step3_title"]}</h4>
            <p style="margin-bottom: 0; color: #333333; font-size: 14px; line-height: 1.4;">{t["step3_desc"]}</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Tecnologias
    st.markdown(f"<h3 style='text-align: center; color: #2a5298; margin-top: 1rem;'>{t['technologies_title']}</h3>", unsafe_allow_html=True)
    
    # Grid de tecnologias usando colunas do Streamlit
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown(f"""
        <div style="background: white; padding: 1rem; border-radius: 8px; margin: 0.5rem 0; border-left: 4px solid #2a5298; box-shadow: 0 1px 3px rgba(0,0,0,0.1);">
            <p style="margin: 0; font-weight: bold; color: #2a5298; font-size: 14px;">{t["tech_langgraph"]}</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown(f"""
        <div style="background: white; padding: 1rem; border-radius: 8px; margin: 0.5rem 0; border-left: 4px solid #2a5298; box-shadow: 0 1px 3px rgba(0,0,0,0.1);">
            <p style="margin: 0; font-weight: bold; color: #2a5298; font-size: 14px;">{t["tech_chromadb"]}</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown(f"""
        <div style="background: white; padding: 1rem; border-radius: 8px; margin: 0.5rem 0; border-left: 4px solid #2a5298; box-shadow: 0 1px 3px rgba(0,0,0,0.1);">
            <p style="margin: 0; font-weight: bold; color: #2a5298; font-size: 14px;">{t["tech_tavily"]}</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown(f"""
        <div style="background: white; padding: 1rem; border-radius: 8px; margin: 0.5rem 0; border-left: 4px solid #2a5298; box-shadow: 0 1px 3px rgba(0,0,0,0.1);">
            <p style="margin: 0; font-weight: bold; color: #2a5298; font-size: 14px;">{t["tech_groq"]}</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div style="background: white; padding: 1rem; border-radius: 8px; margin: 0.5rem 0; border-left: 4px solid #2a5298; box-shadow: 0 1px 3px rgba(0,0,0,0.1);">
            <p style="margin: 0; font-weight: bold; color: #2a5298; font-size: 14px;">{t["tech_pydantic"]}</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown(f"""
        <div style="background: white; padding: 1rem; border-radius: 8px; margin: 0.5rem 0; border-left: 4px solid #2a5298; box-shadow: 0 1px 3px rgba(0,0,0,0.1);">
            <p style="margin: 0; font-weight: bold; color: #2a5298; font-size: 14px;">{t["tech_lexml"]}</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown(f"""
        <div style="background: white; padding: 1rem; border-radius: 8px; margin: 0.5rem 0; border-left: 4px solid #2a5298; box-shadow: 0 1px 3px rgba(0,0,0,0.1);">
            <p style="margin: 0; font-weight: bold; color: #2a5298; font-size: 14px;">{t["tech_openrouter"]}</p>
        </div>
        """, unsafe_allow_html=True)

# Função para importar e carregar o sistema
@st.cache_resource
def load_system():
    """Carrega o sistema uma vez e mantém em cache"""
    try:
        from src.core.legal_models import LegalQuery, Priority, ValidationLevel
        from src.core.workflow_builder import build_graph
        
        # Construir o grafo
        app = build_graph()
        
        return {
            "app": app,
            "LegalQuery": LegalQuery,
            "Priority": Priority,
            "ValidationLevel": ValidationLevel,
            "status": "success",
            "message": "Sistema carregado com sucesso!"
        }
    except Exception as e:
        return {
            "app": None,
            "status": "error",
            "message": f"Erro ao carregar sistema: {str(e)}",
            "error": traceback.format_exc()
        }

# Carregar sistema
system = load_system()

# Verificar status do sistema (sem mostrar detalhes)
if system["status"] != "success":
    st.markdown(f"""
    <div class="error-box">
        <h4>❌ System Error / Erro no Sistema</h4>
        <p>{system["message"]}</p>
        <details>
            <summary>Ver detalhes do erro / View error details</summary>
            <pre>{system.get("error", "Sem detalhes")}</pre>
        </details>
    </div>
    """, unsafe_allow_html=True)
    st.stop()

# Interface principal
st.markdown(f"## {t['legal_consultation']}")

# Função para processar query
async def process_legal_query(query_text: str, priority, validation_level, use_hybrid: bool):
    """
    Processa consulta jurídica com observabilidade COMPLETA.
    Rastreia cada etapa e transferência de dados com Langfuse.
    """
    
    start_time = time.time()
    language = "pt"  # Português brasileiro
    
    try:
        # Criar objeto LegalQuery
        query = LegalQuery(
            id=f"query_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}",
            text=query_text,
            user_id="streamlit_user",
            priority=priority,
            validation_level=validation_level,
            language=language
        )
        
        # INICIAR RASTREAMENTO LANGFUSE COMPLETO
        with trace_legal_query(query.id, query_text) as trace:
            
            log_data_flow_checkpoint("query_creation", {
                "query_id": query.id,
                "query_text": query_text[:100],
                "priority": str(priority),
                "validation_level": str(validation_level),
                "use_hybrid": use_hybrid
            })
            
            # Carregar sistema uma vez
            system = load_system()
            
            if use_hybrid:
                yield ("progress", "🚀 Sistema híbrido integrado com observabilidade..." if language == "pt" else "🚀 Integrated hybrid system with observability...")
                
                # ===================================================================
                # ETAPA 1: COLETA DE DADOS CRAG (COM RASTREAMENTO COMPLETO)
                # ===================================================================
                
                with trace_crag_execution(query_text) as crag_span:
                    yield ("progress", "📚 Coletando dados do CRAG (rastreado)..." if language == "pt" else "📚 Collecting CRAG data (tracked)...")
                    
                    # Estado inicial para o grafo CRAG (apenas para busca de dados)
                    initial_crag_state = {
                        "query": query,
                        "current_query": query.text,
                        "retrieved_docs": [],
                        "tavily_results": None,
                        "lexml_results": None,
                        "needs_jurisprudencia": True,
                        "grade": None,
                        "transformed_query": None,
                        "should_synthesize": True,   # CORRIGIDO: Permite transferência de dados para híbrido
                        "history": [],
                        "final_response": None,
                        "error": None,
                        "next_node": None
                    }
                    
                    # LOG DETALHADO: Estado inicial
                    log_detailed_state(initial_crag_state, "crag_initial_state")
                    
                    # Configuração do grafo
                    config = {
                        "recursion_limit": 50,
                        "configurable": {
                            "thread_id": f"crag-{query.id}"
                        }
                    }
                    
                    # Executar CRAG apenas para coleta de dados (parar antes da síntese) - RASTREADO
                    crag_data = {}
                    event_count = 0
                    final_crag_state = None
                    
                    crag_start_time = time.time()
                    
                    async for event in system["app"].astream(initial_crag_state, config=config):
                        event_count += 1
                        
                        for node_name, state_update in event.items():
                            yield ("progress", f"CRAG: {node_name.replace('_', ' ').title()}")
                            
                            # LOG DETALHADO: Cada nó do CRAG
                            if state_update:
                                detailed_state = log_detailed_state(state_update, f"crag_node_{node_name}")
                                
                                # CHECKPOINT: Estado após cada nó
                                log_data_flow_checkpoint(f"crag_after_{node_name}", detailed_state)
                                
                                # Log transição de estado
                                log_state_transition(
                                    from_state=final_crag_state.get("next_node", "unknown") if final_crag_state else "initial",
                                    to_state=node_name,
                                    reason=f"CRAG workflow step: {node_name}"
                                )
                            
                            final_crag_state = state_update
                            
                            # CORREÇÃO: Acumular dados críticos preservando coletas anteriores
                            if state_update:
                                # Preservar dados coletados nos nós anteriores
                                for key in ["retrieved_docs", "tavily_results", "lexml_results"]:
                                    if state_update.get(key) and key not in crag_data:
                                        crag_data[key] = state_update[key]
                                    elif state_update.get(key) and crag_data.get(key):
                                        # Combinar dados se ambos existem
                                        if isinstance(state_update[key], list) and isinstance(crag_data[key], list):
                                            crag_data[key].extend(state_update[key])
                                        else:
                                            crag_data[key] = state_update[key]
                                
                                # Atualizar outros campos normalmente
                                for key, value in state_update.items():
                                    if key not in ["retrieved_docs", "tavily_results", "lexml_results"]:
                                        crag_data[key] = value
                            
                            # NÃO parar - deixar CRAG completar para preservar dados
                            # CORREÇÃO: Remover quebra prematura que perde dados
                            pass
                        
                        # Limitar eventos para evitar loops
                        if event_count >= 15:
                            break
                    
                    crag_duration = time.time() - crag_start_time
                    log_performance_metrics("crag_data_collection", crag_duration * 1000, events_processed=event_count)
                    
                    # CORREÇÃO: Usar dados acumulados em vez do estado final que pode estar limpo
                    if final_crag_state:
                        # Manter dados já coletados se final_state estiver vazio
                        for key in ["retrieved_docs", "tavily_results", "lexml_results"]:
                            if final_crag_state.get(key) and not crag_data.get(key):
                                crag_data[key] = final_crag_state[key]
                            elif not final_crag_state.get(key) and crag_data.get(key):
                                # Preservar dados acumulados se final_state perdeu os dados
                                pass
                        
                        # LOG DETALHADO: Estado final do CRAG
                        final_state_summary = log_detailed_state(crag_data, "crag_final_state_corrected")
                        log_data_flow_checkpoint("crag_completion", final_state_summary)
                    
                    # ===================================================================
                    # EXTRAÇÃO E PROCESSAMENTO DE DADOS CRAG (RASTREAMENTO DETALHADO)
                    # ===================================================================
                    
                    # Debug: verificar dados coletados do CRAG - RASTREAMENTO COMPLETO
                    retrieved_docs = crag_data.get("retrieved_docs", [])
                    tavily_results = crag_data.get("tavily_results", [])
                    lexml_results = crag_data.get("lexml_results", [])
                    
                    # LOG CRÍTICO: Dados brutos extraídos
                    raw_data_summary = {
                        "retrieved_docs_type": type(retrieved_docs).__name__,
                        "retrieved_docs_count": len(retrieved_docs) if retrieved_docs else 0,
                        "tavily_results_type": type(tavily_results).__name__,
                        "tavily_results_count": len(tavily_results) if isinstance(tavily_results, list) else (1 if tavily_results else 0),
                        "lexml_results_type": type(lexml_results).__name__,
                        "lexml_results_count": len(lexml_results) if isinstance(lexml_results, list) else (1 if lexml_results else 0)
                    }
                    
                    log_data_flow_checkpoint("raw_data_extraction", raw_data_summary)
                    
                    # RASTREAMENTO: Documentos CRAG
                    if retrieved_docs:
                        track_crag_retrieval(query_text, retrieved_docs)
                    
                    # CORREÇÃO CRÍTICA: Processar estrutura real dos documentos - RASTREADO
                    processed_docs = []
                    if retrieved_docs and isinstance(retrieved_docs, list):
                        print(f"🔍 DEBUG: Processando {len(retrieved_docs)} documentos CRAG...")
                        for i, doc in enumerate(retrieved_docs):
                            print(f"  📄 Doc {i+1}: Tipo = {type(doc).__name__}")
                            
                            doc_metadata = extract_metadata(doc)
                            
                            # ESTRATÉGIAS OTIMIZADAS DE EXTRAÇÃO (hierárquica)
                            content = None
                            extraction_method = ""
                            
                            # Estratégia 1: DocumentSnippet.text (Sistema personalizado - OTIMIZADO)
                            if hasattr(doc, 'text') and doc.text:
                                content = doc.text
                                extraction_method = "document_snippet_text"
                                print(f"    ✅ Extraído via DocumentSnippet.text: {len(content)} chars")
                            
                            # Estratégia 2: Atributo page_content (LangChain padrão)
                            elif hasattr(doc, 'page_content') and doc.page_content:
                                content = doc.page_content
                                extraction_method = "langchain_page_content"
                                print(f"    ✅ Extraído via page_content: {len(content)} chars")
                            
                            # Estratégia 3: Dict com 'content'
                            elif isinstance(doc, dict) and 'content' in doc:
                                content = doc['content']
                                extraction_method = "dict_content"
                                print(f"    ✅ Extraído via dict['content']: {len(content)} chars")
                            
                            # Estratégia 4: Dict com 'page_content'
                            elif isinstance(doc, dict) and 'page_content' in doc:
                                content = doc['page_content']
                                extraction_method = "dict_page_content"
                                print(f"    ✅ Extraído via dict['page_content']: {len(content)} chars")
                            
                            # Estratégia 5: String direta
                            elif isinstance(doc, str):
                                content = doc
                                extraction_method = "direct_string"
                                print(f"    ✅ Extraído via string direta: {len(content)} chars")
                            
                            # Estratégia 6: Fallback - Serializar objeto completo
                            else:
                                content = str(doc)
                                extraction_method = "fallback_str"
                                print(f"    ⚠️ Fallback para str(doc): {len(content)} chars")
                            
                            if content and len(content.strip()) > 10:
                                processed_docs.append({
                                    'content': content,
                                    'metadata': getattr(doc, 'metadata', {}) if hasattr(doc, 'metadata') else {},
                                    'source': 'crag_vectordb',
                                    'index': i,
                                    'original_type': type(doc).__name__,
                                    'extraction_method': extraction_method,  # OTIMIZAÇÃO: Track método usado
                                    'content_preview': content[:100] + "..." if len(content) > 100 else content,
                                    'content_length': len(content)  # OTIMIZAÇÃO: Track tamanho
                                })
                                print(f"    ✅ Doc {i+1} processado e adicionado via {extraction_method}!")
                            else:
                                print(f"    ❌ Doc {i+1} descartado - conteúdo insuficiente ({len(content.strip()) if content else 0} chars)")
                        
                        # MÉTRICAS DE OTIMIZAÇÃO: Contabilizar métodos de extração
                        extraction_stats = {}
                        total_content_length = 0
                        for doc in processed_docs:
                            method = doc.get('extraction_method', 'unknown')
                            extraction_stats[method] = extraction_stats.get(method, 0) + 1
                            total_content_length += doc.get('content_length', 0)
                        
                        print(f"📊 RESULTADO: {len(processed_docs)}/{len(retrieved_docs)} docs CRAG processados com sucesso")
                        print(f"📈 MÉTRICAS DE EXTRAÇÃO:")
                        for method, count in extraction_stats.items():
                            emoji = "🚀" if method == "document_snippet_text" else "⚠️" if method == "fallback_str" else "✅"
                            print(f"    {emoji} {method}: {count} docs")
                        print(f"📏 CONTEÚDO TOTAL: {total_content_length:,} chars ({total_content_length/len(processed_docs):.0f} chars/doc)")
                        
                        # Log das métricas para observabilidade
                        log_data_flow_checkpoint("extraction_optimization_metrics", {
                            "extraction_methods": extraction_stats,
                            "total_docs": len(processed_docs),
                            "total_content_length": total_content_length,
                            "avg_content_per_doc": total_content_length / len(processed_docs) if processed_docs else 0,
                            "optimization_success": extraction_stats.get("document_snippet_text", 0) > extraction_stats.get("fallback_str", 0)
                        })
                    
                    # CORREÇÃO CRÍTICA: Processar resultados Tavily - RASTREADO
                    processed_tavily = []
                    if tavily_results:
                        if isinstance(tavily_results, list):
                            for i, result in enumerate(tavily_results):
                                processed_tavily.append({
                                    **serialize_for_langfuse(result),
                                    'source': 'tavily_web',
                                    'index': i
                                })
                        elif tavily_results is not None:
                            processed_tavily = [{
                                **serialize_for_langfuse(tavily_results),
                                'source': 'tavily_web',
                                'index': 0
                            }]
                    
                    # CORREÇÃO CRÍTICA: Processar resultados LexML - RASTREADO
                    processed_lexml = []
                    if lexml_results:
                        if isinstance(lexml_results, list):
                            for i, result in enumerate(lexml_results):
                                processed_lexml.append({
                                    **serialize_for_langfuse(result),
                                    'source': 'lexml_jurisprudence',
                                    'index': i
                                })
                        elif lexml_results is not None:
                            processed_lexml = [{
                                **serialize_for_langfuse(lexml_results),
                                'source': 'lexml_jurisprudence', 
                                'index': 0
                            }]
                    
                    # LOG CRÍTICO: Dados processados
                    processed_data_summary = {
                        "processed_docs_count": len(processed_docs),
                        "processed_tavily_count": len(processed_tavily),
                        "processed_lexml_count": len(processed_lexml),
                        "total_processed_sources": len(processed_docs) + len(processed_tavily) + len(processed_lexml)
                    }
                    
                    log_data_flow_checkpoint("processed_data_summary", processed_data_summary)
                    
                    # RASTREAMENTO: Integração de dados
                    if processed_docs or processed_tavily or processed_lexml:
                        integration_data = track_data_integration(processed_docs, processed_lexml, processed_tavily)
                        crag_span.update(output=integration_data)
                    
                    yield ("progress", f"📊 CRAG PROCESSADO E RASTREADO: {len(processed_docs)} docs, {len(processed_tavily)} tavily, {len(processed_lexml)} lexml")
                
                # ===================================================================
                # ETAPA 2: PROCESSAMENTO HÍBRIDO INTEGRADO (COM RASTREAMENTO)
                # ===================================================================
                
                hybrid_data_summary = {
                    "total_sources": len(processed_docs) + len(processed_tavily) + len(processed_lexml),
                    "crag_docs": len(processed_docs),
                    "tavily_results": len(processed_tavily),
                    "lexml_results": len(processed_lexml)
                }
                
                with trace_hybrid_processing(query_text, hybrid_data_summary) as hybrid_span:
                    yield ("progress", "🔧 Processamento híbrido integrado e rastreado..." if language == "pt" else "🔧 Integrated and tracked hybrid processing...")
                    
                    # LOG CRÍTICO: Dados sendo passados para o sistema híbrido
                    log_data_flow_checkpoint("hybrid_input_data", {
                        "processed_docs_sample": processed_docs[:2] if processed_docs else [],
                        "processed_tavily_sample": processed_tavily[:1] if processed_tavily else [],
                        "processed_lexml_sample": processed_lexml[:1] if processed_lexml else [],
                        "total_sources_passed": hybrid_data_summary["total_sources"]
                    })
                    
                    # Importar função híbrida específica que aceita dados externos
                    from src.agents.streaming.hybrid_legal_processor import process_legal_query_hybrid_with_crag_data
                    
                    hybrid_start_time = time.time()
                    final_result = None
                    
                    async for step_type, content in process_legal_query_hybrid_with_crag_data(
                        query, 
                        crag_retrieved_docs=processed_docs,  # Usar dados processados e rastreados
                        crag_tavily_results=processed_tavily,  # Usar dados processados e rastreados
                        crag_lexml_results=processed_lexml  # Usar dados processados e rastreados
                    ):
                        if step_type == "progress":
                            yield ("progress", content)
                        elif step_type == "final":
                            final_result = content
                            break
                        elif step_type == "error":
                            yield ("error", content)
                            return
                    
                    hybrid_duration = time.time() - hybrid_start_time
                    log_performance_metrics("hybrid_processing", hybrid_duration * 1000, 
                                           sources_used=hybrid_data_summary["total_sources"])
                    
                    if final_result:
                        # LOG FINAL: Resultado híbrido
                        final_result_metadata = extract_metadata(final_result)
                        log_data_flow_checkpoint("hybrid_final_result", final_result_metadata)
                        
                        hybrid_span.update(output={
                            "result_metadata": final_result_metadata,
                            "processing_successful": True
                        })
                        
                        yield ("final", final_result)
                    else:
                        error_msg = "Sistema híbrido integrado não retornou resultado" if language == "pt" else "Integrated hybrid system did not return result"
                        log_data_flow_checkpoint("hybrid_error", {"error": error_msg})
                        yield ("error", error_msg)
            
            else:
                # Sistema original CRAG sem híbrido - TAMBÉM RASTREADO
                yield ("progress", "🤖 Processando com sistema CRAG padrão rastreado..." if language == "pt" else "🤖 Processing with tracked standard CRAG system...")
                
                # Estado inicial para o grafo
                initial_state = {
                    "query": query,
                    "current_query": query.text,
                    "retrieved_docs": [],
                    "tavily_results": None,
                    "lexml_results": None,
                    "needs_jurisprudencia": True,
                    "grade": None,
                    "transformed_query": None,
                    "should_synthesize": True,   # CORRIGIDO: Permite síntese normal
                    "history": [],
                    "final_response": None,
                    "error": None,
                    "next_node": None
                }
                
                # Configuração do grafo
                config = {
                    "recursion_limit": 50,
                    "configurable": {
                        "thread_id": f"streamlit-{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                    }
                }
                
                # Executar o grafo
                final_state = None
                event_count = 0
                
                async for event in system["app"].astream(initial_state, config=config):
                    event_count += 1
                    
                    for node_name, state_update in event.items():
                        yield ("progress", f"🔄 {node_name.replace('_', ' ').title()}")
                        final_state = state_update
                    
                    if event_count >= 20:
                        break
                
                if final_state and final_state.get("final_response"):
                    yield ("final", final_state["final_response"])
                else:
                    yield ("error", "Sistema CRAG não retornou resposta final" if language == "pt" else "CRAG system did not return final response")
            
            total_duration = time.time() - start_time
            log_performance_metrics("complete_legal_query", total_duration * 1000,
                                   query_length=len(query_text),
                                   use_hybrid=use_hybrid)
        
    except Exception as e:
        error_msg = f"Erro ao processar consulta: {str(e)}" if language == "pt" else f"Error processing query: {str(e)}"
        
        # LOG ERRO COMPLETO
        log_data_flow_checkpoint("fatal_error", {
            "error_message": str(e),
            "error_type": type(e).__name__,
            "query_text": query_text[:100]
        })
        
        yield ("error", error_msg)

# Interface de chat
if "messages" not in st.session_state:
    st.session_state.messages = []

# Mostrar mensagens anteriores
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Input do usuário
if prompt := st.chat_input(t["chat_placeholder"]):
    
    # Adicionar mensagem do usuário
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Processar resposta
    with st.chat_message("assistant"):
        
        # Container para resposta
        response_container = st.container()
        
        with response_container:
            st.markdown(t["processing"])
            
            # Elementos de interface
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            try:
                final_result = None
                
                # Função para executar processamento
                def run_processing():
                    async def process_coroutine():
                        progress_count = 0
                        async for step_type, content in process_legal_query(
                            prompt, 
                            system["Priority"].MEDIUM,  # Prioridade padrão
                            system["ValidationLevel"].MODERATE,  # Validação padrão
                            True  # Usar híbrido
                        ):
                            
                            if step_type == "progress":
                                progress_count += 1
                                progress = min(progress_count / 10.0, 0.9)
                                progress_bar.progress(progress)
                                status_text.text(content)
                            
                            elif step_type == "final":
                                return content
                            
                            elif step_type == "error":
                                st.error(f"{t['error_processing']} {content}")
                                return None
                        
                        return None
                    
                    return asyncio.run(process_coroutine())
                
                # Executar processamento
                final_result = run_processing()
                
                # Limpar progress quando terminar
                progress_bar.progress(1.0)
                status_text.text(t["processing_complete"])
                
                if final_result:
                    # Limpar elementos de progresso
                    time.sleep(1)
                    progress_bar.empty()
                    status_text.empty()
                    
                    # Processar resultado
                    if isinstance(final_result, dict):
                        if "overall_summary" in final_result:
                            summary = final_result["overall_summary"]
                            disclaimer = final_result.get("disclaimer", "")
                            
                            # Exibir resposta final formatada
                            st.markdown(f"### {t['legal_response']}")
                            st.markdown(summary)
                            
                            if disclaimer:
                                st.markdown("---")
                                st.markdown(f"**{t['disclaimer']}**")
                                st.markdown(disclaimer)
                            
                            # Adicionar à sessão
                            full_response = f"### {t['legal_response']}\n\n{summary}"
                            if disclaimer:
                                full_response += f"\n\n**{t['disclaimer']}** {disclaimer}"
                            
                            st.session_state.messages.append({
                                "role": "assistant", 
                                "content": full_response
                            })
                            
                        else:
                            error_msg = t["invalid_response_format"]
                            st.error(error_msg)
                            st.session_state.messages.append({
                                "role": "assistant", 
                                "content": f"❌ {error_msg}"
                            })
                    
                    else:
                        error_msg = f"{t['unexpected_response_type']} {type(final_result)}"
                        st.error(error_msg)
                        st.session_state.messages.append({
                            "role": "assistant",
                            "content": f"❌ {error_msg}"
                        })
                
                else:
                    error_msg = t["processing_failed"]
                    st.error(error_msg)
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": f"❌ {error_msg}"
                    })
                    
            except Exception as e:
                # Limpar progress em caso de erro
                progress_bar.empty()
                status_text.empty()
                
                error_msg = f"{t['unexpected_error']} {str(e)}"
                st.error(error_msg)
                
                # Mostrar traceback em expander
                with st.expander(t["technical_details"]):
                    st.code(traceback.format_exc())
                
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": f"❌ {error_msg}"
                })

# Footer
st.markdown("---")
st.markdown(f"""
<div style="text-align: center; color: #666; margin-top: 2rem;">
    <p><strong>{t["footer_title"]}</strong> | {t["footer_subtitle"]}</p>
    <p>{t["footer_apis"]}</p>
    <p>{t["footer_disclaimer"]}</p>
</div>
""", unsafe_allow_html=True) 