"""
Módulo de observabilidade completa usando Langfuse para rastreamento de sistema híbrido jurídico.
Implementa tracking de todas as operações, dados CRAG, análises e sínteses.
"""

import os
import json
from datetime import datetime
from typing import Any, Dict, List, Optional, Union
from contextlib import contextmanager
import structlog

# CRÍTICO: Carregar variáveis de ambiente do .env
from dotenv import load_dotenv
load_dotenv()  # Carrega o arquivo .env

try:
    from langfuse import Langfuse
    LANGFUSE_AVAILABLE = True
except ImportError:
    LANGFUSE_AVAILABLE = False

# Decorador simples sem dependência dos decoradores do Langfuse
def observe(func=None, *, name: str = None, **kwargs):
    """Decorador simples para observabilidade."""
    def decorator(f):
        return f
    return decorator(func) if func else decorator

class DummyContext:
    def update_current_trace(self, **kwargs): pass
    def update_current_observation(self, **kwargs): pass
    def create_trace(self, **kwargs): return self
    def span(self, **kwargs): return self
    def generation(self, **kwargs): return self
    def update(self, **kwargs): pass
    def __enter__(self): return self
    def __exit__(self, *args): pass

# Usar DummyContext sempre, já que os decoradores do Langfuse não estão funcionando
langfuse_context = DummyContext()

logger = structlog.get_logger(__name__)

# ===============================
# CONFIGURAÇÃO LANGFUSE
# ===============================

def initialize_langfuse() -> Optional[Langfuse]:
    """Inicializa cliente Langfuse com configurações do ambiente."""
    
    if not LANGFUSE_AVAILABLE:
        logger.warning("Langfuse não disponível - observabilidade limitada")
        return None
    
    try:
        # Configurações do ambiente
        public_key = os.getenv("LANGFUSE_PUBLIC_KEY")
        secret_key = os.getenv("LANGFUSE_SECRET_KEY")
        host = os.getenv("LANGFUSE_HOST", "https://cloud.langfuse.com")
        
        if not public_key or not secret_key:
            logger.warning("Chaves Langfuse não configuradas - usando observabilidade local")
            return None
        
        # Inicializar cliente
        langfuse = Langfuse(
            public_key=public_key,
            secret_key=secret_key,
            host=host
        )
        
        # Testar conectividade
        try:
            auth_result = langfuse.auth_check()
            if auth_result:
                logger.info("Langfuse inicializado com sucesso", host=host, auth=auth_result)
                return langfuse
            else:
                logger.warning("Falha na autenticação Langfuse")
                return None
        except Exception as e:
            logger.warning("Erro ao verificar autenticação Langfuse", error=str(e))
            return None
        
    except Exception as e:
        logger.error("Erro ao inicializar Langfuse", error=str(e))
        return None

# Cliente global
_langfuse_client = initialize_langfuse()

# ===============================
# UTILITÁRIOS DE OBSERVABILIDADE
# ===============================

def serialize_for_langfuse(obj: Any) -> Any:
    """Serializa objetos complexos para Langfuse."""
    
    if isinstance(obj, dict):
        return {k: serialize_for_langfuse(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [serialize_for_langfuse(item) for item in obj]
    elif hasattr(obj, 'model_dump'):
        # Objetos Pydantic
        return obj.model_dump()
    elif hasattr(obj, '__dict__'):
        # Objetos com atributos
        return {k: serialize_for_langfuse(v) for k, v in obj.__dict__.items() 
                if not k.startswith('_')}
    elif isinstance(obj, datetime):
        return obj.isoformat()
    else:
        try:
            json.dumps(obj)  # Teste de serialização
            return obj
        except (TypeError, ValueError):
            return str(obj)

def extract_metadata(obj: Any) -> Dict[str, Any]:
    """Extrai metadados relevantes de objetos para tracking."""
    
    metadata = {}
    
    if hasattr(obj, 'id'):
        metadata['id'] = str(obj.id)
    if hasattr(obj, '__class__'):
        metadata['type'] = obj.__class__.__name__
    if hasattr(obj, 'created_at'):
        metadata['created_at'] = obj.created_at.isoformat() if obj.created_at else None
    if hasattr(obj, 'status'):
        metadata['status'] = str(obj.status)
    
    # Para documentos
    if hasattr(obj, 'page_content'):
        metadata['content_length'] = len(str(obj.page_content))
        if hasattr(obj, 'metadata'):
            metadata['doc_metadata'] = serialize_for_langfuse(obj.metadata)
    
    # Para listas
    if isinstance(obj, list):
        metadata['list_length'] = len(obj)
        if obj and hasattr(obj[0], '__class__'):
            metadata['item_type'] = obj[0].__class__.__name__
    
    return metadata

# ===============================
# DECORADORES DE OBSERVABILIDADE
# ===============================

@observe(name="crag_document_retrieval")
def track_crag_retrieval(query: str, docs_retrieved: List[Any]) -> Dict[str, Any]:
    """Rastreia recuperação de documentos CRAG."""
    
    tracking_data = {
        "query": query,
        "documents_count": len(docs_retrieved),
        "documents_metadata": []
    }
    
    # Extrair metadados de cada documento
    for i, doc in enumerate(docs_retrieved[:5]):  # Primeiros 5 para não sobrecarregar
        doc_meta = extract_metadata(doc)
        doc_meta['index'] = i
        tracking_data["documents_metadata"].append(doc_meta)
    
    # Atualizar contexto Langfuse
    try:
        if LANGFUSE_AVAILABLE and _langfuse_client:
            _langfuse_client.update_current_span(
                input={"query": query},
                output=tracking_data,
                metadata={
                    "step": "crag_retrieval",
                    "documents_retrieved": len(docs_retrieved)
                }
            )
    except Exception as e:
        logger.warning("Erro ao atualizar span CRAG", error=str(e))
    
    logger.info("CRAG retrieval tracked", 
               query=query[:50], 
               docs_count=len(docs_retrieved))
    
    return tracking_data

@observe(name="lexml_search")
def track_lexml_search(query: str, results: List[Any]) -> Dict[str, Any]:
    """Rastreia busca LexML."""
    
    tracking_data = {
        "query": query,
        "results_count": len(results),
        "results_metadata": []
    }
    
    # Extrair metadados dos resultados
    for i, result in enumerate(results[:3]):  # Primeiros 3
        result_meta = extract_metadata(result)
        result_meta['index'] = i
        tracking_data["results_metadata"].append(result_meta)
    
    try:
        if LANGFUSE_AVAILABLE and _langfuse_client:
            _langfuse_client.update_current_span(
                input={"query": query},
                output=tracking_data,
                metadata={
                    "step": "lexml_search",
                    "results_found": len(results)
                }
            )
    except Exception as e:
        logger.warning("Erro ao atualizar span LexML", error=str(e))
    
    logger.info("LexML search tracked", 
               query=query[:50], 
               results_count=len(results))
    
    return tracking_data

@observe(name="web_search")
def track_web_search(query: str, results: List[Any]) -> Dict[str, Any]:
    """Rastreia busca web (Tavily)."""
    
    tracking_data = {
        "query": query,
        "results_count": len(results),
        "results_metadata": []
    }
    
    # Extrair metadados dos resultados
    for i, result in enumerate(results[:3]):  # Primeiros 3
        result_meta = extract_metadata(result)
        result_meta['index'] = i
        tracking_data["results_metadata"].append(result_meta)
    
    langfuse_context.update_current_observation(
        input={"query": query},
        output=tracking_data,
        metadata={
            "step": "web_search",
            "results_found": len(results)
        }
    )
    
    logger.info("Web search tracked", 
               query=query[:50], 
               results_count=len(results))
    
    return tracking_data

@observe(name="data_integration")
def track_data_integration(crag_docs: List[Any], lexml_results: List[Any], 
                          web_results: List[Any]) -> Dict[str, Any]:
    """Rastreia integração de dados de múltiplas fontes."""
    
    integration_data = {
        "crag_documents": {
            "count": len(crag_docs),
            "metadata": [extract_metadata(doc) for doc in crag_docs[:3]]
        },
        "lexml_results": {
            "count": len(lexml_results),
            "metadata": [extract_metadata(result) for result in lexml_results[:3]]
        },
        "web_results": {
            "count": len(web_results),
            "metadata": [extract_metadata(result) for result in web_results[:3]]
        },
        "total_sources": len(crag_docs) + len(lexml_results) + len(web_results)
    }
    
    langfuse_context.update_current_observation(
        input={
            "crag_count": len(crag_docs),
            "lexml_count": len(lexml_results),
            "web_count": len(web_results)
        },
        output=integration_data,
        metadata={
            "step": "data_integration",
            "total_sources": integration_data["total_sources"]
        }
    )
    
    logger.info("Data integration tracked",
               crag_count=len(crag_docs),
               lexml_count=len(lexml_results),
               web_count=len(web_results),
               total_sources=integration_data["total_sources"])
    
    return integration_data

@observe(name="openrouter_analysis")
def track_openrouter_analysis(query: str, context_data: str, analysis_result: str) -> Dict[str, Any]:
    """Rastreia análise jurídica OpenRouter."""
    
    analysis_data = {
        "query": query,
        "context_length": len(context_data),
        "analysis_length": len(analysis_result),
        "analysis_word_count": len(analysis_result.split()),
        "context_preview": context_data[:200] + "..." if len(context_data) > 200 else context_data
    }
    
    langfuse_context.update_current_observation(
        input={
            "query": query,
            "context": context_data[:500]  # Primeiros 500 chars
        },
        output={
            "analysis": analysis_result[:1000],  # Primeiros 1000 chars
            "analysis_stats": {
                "length": len(analysis_result),
                "word_count": len(analysis_result.split())
            }
        },
        metadata={
            "step": "openrouter_analysis",
            "context_length": len(context_data),
            "analysis_length": len(analysis_result)
        }
    )
    
    logger.info("OpenRouter analysis tracked",
               query=query[:50],
               context_length=len(context_data),
               analysis_length=len(analysis_result))
    
    return analysis_data

@observe(name="groq_searches")
def track_groq_searches(query: str, web_result: str, lexml_result: str) -> Dict[str, Any]:
    """Rastreia buscas Groq (WEB + LexML)."""
    
    groq_data = {
        "query": query,
        "web_result_length": len(web_result),
        "lexml_result_length": len(lexml_result),
        "total_content_length": len(web_result) + len(lexml_result)
    }
    
    langfuse_context.update_current_observation(
        input={"query": query},
        output={
            "web_result": web_result[:500],  # Primeiros 500 chars
            "lexml_result": lexml_result[:500],
            "stats": groq_data
        },
        metadata={
            "step": "groq_searches",
            "web_length": len(web_result),
            "lexml_length": len(lexml_result)
        }
    )
    
    logger.info("Groq searches tracked",
               query=query[:50],
               web_length=len(web_result),
               lexml_length=len(lexml_result))
    
    return groq_data

@observe(name="synthesis_streaming")
def track_synthesis_streaming(query: str, analysis_text: str, response_parts: List[str]) -> Dict[str, Any]:
    """Rastreia síntese final com streaming."""
    
    synthesis_data = {
        "query": query,
        "analysis_length": len(analysis_text),
        "response_parts": len(response_parts),
        "total_response_length": sum(len(part) for part in response_parts),
        "response_word_count": sum(len(part.split()) for part in response_parts)
    }
    
    langfuse_context.update_current_observation(
        input={
            "query": query,
            "analysis": analysis_text[:500]
        },
        output={
            "response_parts": [part[:200] for part in response_parts],  # Primeiros 200 chars de cada parte
            "stats": synthesis_data
        },
        metadata={
            "step": "synthesis_streaming",
            "parts_count": len(response_parts),
            "total_words": synthesis_data["response_word_count"]
        }
    )
    
    logger.info("Synthesis streaming tracked",
               query=query[:50],
               parts_count=len(response_parts),
               total_words=synthesis_data["response_word_count"])
    
    return synthesis_data

# ===============================
# CONTEXT MANAGERS
# ===============================

@contextmanager
def trace_legal_query(query_id: str, query_text: str):
    """Context manager para rastrear consulta jurídica completa."""
    
    if LANGFUSE_AVAILABLE and _langfuse_client:
        try:
            # Usar API correta do Langfuse v3.0.2
            trace_id = query_id if query_id else f"trace_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"
            trace = _langfuse_client.start_span(
                name="legal_query_processing",
                input={"query": query_text[:100]},  # Limitar tamanho
                metadata={
                    "system": "hybrid_legal_ai",
                    "version": "1.0.0",
                    "trace_id": trace_id,
                    "start_time": datetime.now().isoformat()
                }
            )
        except Exception as e:
            logger.warning("Erro ao criar trace Langfuse", error=str(e))
            trace = DummyContext()
    else:
        trace = DummyContext()
    
    try:
        logger.info("Starting legal query trace", query_id=query_id, query=query_text[:50])
        yield trace
        
        if trace and hasattr(trace, 'update'):
            trace.update(
                output={"status": "completed"},
                metadata={"end_time": datetime.now().isoformat()}
            )
            
    except Exception as e:
        logger.error("Error in legal query trace", query_id=query_id, error=str(e))
        if trace and hasattr(trace, 'update'):
            trace.update(
                output={"status": "error", "error": str(e)},
                metadata={"end_time": datetime.now().isoformat()}
            )
        raise
    finally:
        logger.info("Legal query trace completed", query_id=query_id)

@contextmanager 
def trace_crag_execution(query_text: str):
    """Context manager para rastrear execução CRAG."""
    
    if LANGFUSE_AVAILABLE and _langfuse_client:
        try:
            span = _langfuse_client.start_span(
                name="crag_execution",
                input={"query": query_text[:100]},  # Limitar tamanho
                metadata={"step": "crag_data_collection"}
            )
        except Exception as e:
            logger.warning("Erro ao criar span CRAG", error=str(e))
            span = DummyContext()
    else:
        span = DummyContext()
        
    try:
        logger.info("Starting CRAG execution trace", query=query_text[:50])
        yield span
        
        if span and hasattr(span, 'update'):
            span.update(output={"status": "completed"})
            
    except Exception as e:
        logger.error("Error in CRAG execution", error=str(e))
        if span and hasattr(span, 'update'):
            span.update(output={"error": str(e)})
        raise

@contextmanager
def trace_hybrid_processing(query_text: str, crag_data_summary: Dict[str, Any]):
    """Context manager para rastrear processamento híbrido."""
    
    if LANGFUSE_AVAILABLE and _langfuse_client:
        try:
            span = _langfuse_client.start_span(
                name="hybrid_processing",
                input={"query": query_text[:100], "crag_summary": crag_data_summary},  # Limitar tamanho
                metadata={"step": "hybrid_analysis_synthesis"}
            )
        except Exception as e:
            logger.warning("Erro ao criar span híbrido", error=str(e))
            span = DummyContext()
    else:
        span = DummyContext()
        
    try:
        logger.info("Starting hybrid processing trace", 
                   query=query_text[:50],
                   crag_sources=crag_data_summary.get("total_sources", 0))
        yield span
        
        if span and hasattr(span, 'update'):
            span.update(output={"status": "completed"})
            
    except Exception as e:
        logger.error("Error in hybrid processing", error=str(e))
        if span and hasattr(span, 'update'):
            span.update(output={"error": str(e)})
        raise

# ===============================
# MÉTRICAS E ANALYTICS
# ===============================

def log_state_transition(from_state: str, to_state: str, reason: str = ""):
    """Log transições de estado para analytics."""
    
    try:
        if LANGFUSE_AVAILABLE and _langfuse_client:
            langfuse_context.update_current_observation(
                metadata={
                    "state_transition": {
                        "from": from_state,
                        "to": to_state,
                        "reason": reason,
                        "timestamp": datetime.now().isoformat()
                    }
                }
            )
    except Exception as e:
        # Silenciosamente ignorar erros de contexto Langfuse
        pass
    
    logger.info("State transition logged",
               from_state=from_state,
               to_state=to_state,
               reason=reason)

def log_performance_metrics(operation: str, duration_ms: float, **kwargs):
    """Log métricas de performance."""
    
    metrics = {
        "operation": operation,
        "duration_ms": duration_ms,
        **kwargs
    }
    
    try:
        if LANGFUSE_AVAILABLE and _langfuse_client:
            langfuse_context.update_current_observation(
                metadata={"performance_metrics": metrics}
            )
    except Exception as e:
        # Silenciosamente ignorar erros de contexto Langfuse
        pass
    
    # CORRIGIDO: Evitar conflitos de parâmetros com structlog
    logger.info("Performance metrics logged",
               operation=operation,
               duration_ms=duration_ms,
               additional_metrics=kwargs)

def log_data_flow_checkpoint(checkpoint_name: str, data_summary: Dict[str, Any]):
    """Log checkpoints do fluxo de dados para debugging."""
    
    try:
        if LANGFUSE_AVAILABLE and _langfuse_client:
            langfuse_context.update_current_observation(
                metadata={
                    "data_checkpoint": {
                        "name": checkpoint_name,
                        "summary": data_summary,
                        "timestamp": datetime.now().isoformat()
                    }
                }
            )
    except Exception as e:
        # Silenciosamente ignorar erros de contexto Langfuse
        pass
    
    # CORRIGIDO: Evitar conflitos de parâmetros com structlog
    logger.info("Data flow checkpoint logged",
               checkpoint_name=checkpoint_name,
               data_summary=data_summary)

# ===============================
# UTILITÁRIOS DE DEBUG
# ===============================

def log_detailed_state(state: Dict[str, Any], step_name: str):
    """Log detalhado do estado para debugging."""
    
    # Extrair informações relevantes do estado
    state_summary = {
        "retrieved_docs_count": len(state.get("retrieved_docs", [])),
        "tavily_results_count": len(state.get("tavily_results", [])) if state.get("tavily_results") else 0,
        "lexml_results_count": len(state.get("lexml_results", [])) if state.get("lexml_results") else 0,
        "has_query": bool(state.get("query")),
        "current_query": state.get("current_query", ""),
        "should_synthesize": state.get("should_synthesize", None),
        "final_response": bool(state.get("final_response"))
    }
    
    # Log detalhado
    try:
        if LANGFUSE_AVAILABLE and _langfuse_client:
            langfuse_context.update_current_observation(
                metadata={
                    "detailed_state": state_summary,
                    "step": step_name
                }
            )
    except Exception as e:
        # Silenciosamente ignorar erros de contexto Langfuse
        pass
    
    logger.info("Detailed state logged",
               step_name=step_name,  # CORRIGIDO: usar step_name em vez de step
               **state_summary)
    
    return state_summary

# ===============================
# INICIALIZAÇÃO
# ===============================

def setup_observability():
    """Configura observabilidade completa do sistema."""
    
    if LANGFUSE_AVAILABLE and _langfuse_client:
        logger.info("✅ Observabilidade Langfuse configurada com sucesso")
    else:
        logger.warning("⚠️ Observabilidade limitada - Langfuse não disponível")
    
    # Configurar logging estruturado adicional
    structlog.configure(
        processors=[
            structlog.processors.TimeStamper(fmt="ISO"),
            structlog.processors.add_log_level,
            structlog.processors.JSONRenderer()
        ],
        wrapper_class=structlog.make_filtering_bound_logger(20),  # INFO level
        logger_factory=structlog.PrintLoggerFactory(),
        cache_logger_on_first_use=True,
    )
    
    logger.info("Sistema de observabilidade inicializado")

# Inicializar automaticamente ao importar
setup_observability() 