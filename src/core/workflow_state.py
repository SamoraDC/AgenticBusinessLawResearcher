"""
Estado do grafo para sistema jurídico AI com PydanticAI e LangGraph.
Implementa tipagem forte, retry, human-in-the-loop e paralelismo.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Literal, Optional, TypedDict
from typing_extensions import NotRequired

from src.core.legal_models import (
    LegalQuery,
    DocumentSnippet, 
    SearchResult,
    AnalysisResult,
    FinalResponse,
    ProcessingConfig,
    RetryableError,
    HumanReview,
    SearchSource,
    Status,
    Priority
)

# Tipos para compatibilidade com sistema legado
Grade = Literal["relevant", "irrelevant", "needs_web"]


class ParallelSearchState(TypedDict):
    """Estado para buscas paralelas."""
    vectordb_result: NotRequired[SearchResult]
    lexml_result: NotRequired[SearchResult] 
    web_result: NotRequired[SearchResult]
    jurisprudence_result: NotRequired[SearchResult]
    
    # Status das buscas
    vectordb_started: NotRequired[bool]
    lexml_started: NotRequired[bool]
    web_started: NotRequired[bool]
    jurisprudence_started: NotRequired[bool]
    
    # Controle de tempo
    search_start_time: NotRequired[datetime]
    search_timeout_seconds: NotRequired[int]
    
    # Erros de busca
    search_errors: NotRequired[List[RetryableError]]


class RetryState(TypedDict):
    """Estado para controle de retry."""
    retry_count: NotRequired[int]
    max_retries: NotRequired[int]
    backoff_factor: NotRequired[float]
    last_error: NotRequired[RetryableError]
    retry_history: NotRequired[List[Dict[str, Any]]]
    
    # Configurações específicas por operação
    retrieval_retry: NotRequired[RetryableError]
    search_retry: NotRequired[RetryableError]
    analysis_retry: NotRequired[RetryableError]
    synthesis_retry: NotRequired[RetryableError]


class HumanInLoopState(TypedDict):
    """Estado para human-in-the-loop."""
    requires_review: NotRequired[bool]
    review_reason: NotRequired[str]
    review_request: NotRequired[HumanReview]
    review_completed: NotRequired[bool]
    review_approved: NotRequired[bool]
    reviewer_feedback: NotRequired[str]
    
    # Configurações
    confidence_threshold: NotRequired[float]
    auto_approve_high_confidence: NotRequired[bool]
    
    # Histórico de revisões
    review_history: NotRequired[List[HumanReview]]


class QualityState(TypedDict):
    """Estado para controle de qualidade."""
    overall_confidence: NotRequired[float]
    completeness_score: NotRequired[float]
    validation_passed: NotRequired[bool]
    validation_errors: NotRequired[List[str]]
    
    # Guardrails
    guardrails_passed: NotRequired[bool]
    guardrail_violations: NotRequired[List[str]]
    
    # Métricas de qualidade
    response_length_adequate: NotRequired[bool]
    citation_quality_score: NotRequired[float]
    legal_accuracy_score: NotRequired[float]


class ProcessingState(TypedDict):
    """Estado de processamento e performance."""
    start_time: NotRequired[datetime]
    end_time: NotRequired[datetime]
    processing_duration_ms: NotRequired[float]
    
    # Status dos nós
    nodes_completed: NotRequired[List[str]]
    current_node: NotRequired[str]
    next_node: NotRequired[str]
    
    # Métricas de performance
    total_tokens_used: NotRequired[int]
    api_calls_made: NotRequired[int]
    search_operations: NotRequired[int]
    
    # Configuração
    config: NotRequired[ProcessingConfig]


class AgentState(TypedDict):
    """
    Estado principal do agente com tipagem forte e recursos avançados.
    Implementa paralelismo, retry, human-in-the-loop e validação robusta.
    """
    
    # ===============================
    # ENTRADA E IDENTIFICAÇÃO
    # ===============================
    
    # Consulta original
    query: LegalQuery
    session_id: NotRequired[str]  # ID da sessão para tracking
    user_id: NotRequired[str]     # ID do usuário
    
    # ===============================
    # DOCUMENTOS E BUSCA
    # ===============================
    
    # Documentos do vetor store (CRAG tradicional)
    retrieved_docs: NotRequired[List[DocumentSnippet]]
    
    # Resultados de busca consolidados por fonte
    search_results: NotRequired[Dict[SearchSource, SearchResult]]
    
    # Estado de buscas paralelas
    parallel_search: NotRequired[ParallelSearchState]
    
    # ===============================
    # ANÁLISE E SÍNTESE
    # ===============================
    
    # Análises por aspecto/área
    analysis_results: NotRequired[List[AnalysisResult]]
    
    # Resposta final
    final_response: NotRequired[FinalResponse]
    
    # ===============================
    # CONTROLE DE FLUXO
    # ===============================
    
    # Status geral
    status: NotRequired[Status]
    priority: NotRequired[Priority]
    
    # Decisões de roteamento
    needs_vectordb_search: NotRequired[bool]
    needs_lexml_search: NotRequired[bool] 
    needs_web_search: NotRequired[bool]
    needs_jurisprudence_search: NotRequired[bool]
    needs_analysis: NotRequired[bool]
    needs_synthesis: NotRequired[bool]
    
    # Roteamento condicional
    route_reason: NotRequired[str]  # Motivo da decisão de roteamento
    routing_confidence: NotRequired[float]  # Confiança na decisão
    
    # ===============================
    # RETRY E RECUPERAÇÃO
    # ===============================
    
    # Estado de retry
    retry_state: NotRequired[RetryState]
    
    # Erros encontrados
    errors: NotRequired[List[RetryableError]]
    last_error: NotRequired[str]
    
    # ===============================
    # HUMAN-IN-THE-LOOP
    # ===============================
    
    # Estado de revisão humana
    human_loop: NotRequired[HumanInLoopState]
    
    # ===============================
    # QUALIDADE E VALIDAÇÃO
    # ===============================
    
    # Estado de qualidade
    quality: NotRequired[QualityState]
    
    # Validações realizadas
    validations_passed: NotRequired[List[str]]
    validations_failed: NotRequired[List[str]]
    
    # ===============================
    # PROCESSAMENTO E MÉTRICAS
    # ===============================
    
    # Estado de processamento
    processing: NotRequired[ProcessingState]
    
    # Histórico de conversação
    conversation_history: NotRequired[List[Dict[str, Any]]]
    
    # Metadados extras
    metadata: NotRequired[Dict[str, Any]]
    
    # ===============================
    # COMPATIBILIDADE (mantido temporariamente)
    # ===============================
    
    # Campos legados para compatibilidade - serão removidos gradualmente
    current_query: NotRequired[str]
    grade: NotRequired[str] 
    transformed_query: NotRequired[str]
    tavily_results: NotRequired[List[Any]]
    lexml_results: NotRequired[List[Any]]
    needs_web_search: NotRequired[bool]  # Duplicado, usar needs_web_search
    needs_jurisprudencia: NotRequired[bool]  # Renomeado para needs_jurisprudence_search
    should_synthesize: NotRequired[bool]  # Usar needs_synthesis
    history: NotRequired[List[tuple[str, str]]]  # Usar conversation_history
    next_node: NotRequired[str]  # Usar processing.next_node
    web_search_reasoning: NotRequired[str]
    web_search_query: NotRequired[str]
    evaluation_complete: NotRequired[bool]
    web_search_performed: NotRequired[bool]
    web_search_skipped: NotRequired[bool]
    web_search_error: NotRequired[str]
    error: NotRequired[str]  # Usar errors list


# ===============================
# FUNÇÕES AUXILIARES DE ESTADO
# ===============================

def create_initial_state(
    query: LegalQuery,
    config: Optional[ProcessingConfig] = None,
    session_id: Optional[str] = None,
    user_id: Optional[str] = None
) -> AgentState:
    """Cria estado inicial do agente."""
    import uuid
    
    if config is None:
        config = ProcessingConfig()
    
    return AgentState(
        query=query,
        session_id=session_id or str(uuid.uuid4()),
        user_id=user_id,
        status=Status.PENDING,
        priority=query.priority,
        
        # Inicializar estruturas
        search_results={},
        analysis_results=[],
        errors=[],
        validations_passed=[],
        validations_failed=[],
        conversation_history=[],
        metadata={},
        
        # Estado inicial de buscas
        needs_vectordb_search=True,
        needs_lexml_search=True,
        needs_web_search=False,  # Será determinado dinamicamente
        needs_jurisprudence_search=True,
        needs_analysis=True,
        needs_synthesis=True,
        
        # Estado de processamento
        processing=ProcessingState(
            start_time=datetime.now(),
            nodes_completed=[],
            config=config,
            total_tokens_used=0,
            api_calls_made=0,
            search_operations=0
        ),
        
        # Estado de qualidade
        quality=QualityState(
            overall_confidence=0.0,
            completeness_score=0.0,
            validation_passed=False,
            guardrails_passed=False,
            validation_errors=[],
            guardrail_violations=[]
        ),
        
        # Estado de retry
        retry_state=RetryState(
            retry_count=0,
            max_retries=config.max_retries,
            backoff_factor=config.retry_backoff_factor,
            retry_history=[]
        ),
        
        # Human-in-the-loop
        human_loop=HumanInLoopState(
            requires_review=False,
            confidence_threshold=config.human_review_threshold,
            auto_approve_high_confidence=True,
            review_history=[]
        )
    )


def update_processing_metrics(state: AgentState, node_name: str) -> None:
    """Atualiza métricas de processamento."""
    if "processing" not in state:
        state["processing"] = ProcessingState(
            start_time=datetime.now(),
            nodes_completed=[],
            total_tokens_used=0,
            api_calls_made=0,
            search_operations=0
        )
    
    processing = state["processing"]
    if "nodes_completed" not in processing:
        processing["nodes_completed"] = []
    
    processing["nodes_completed"].append(node_name)
    processing["current_node"] = node_name


def should_trigger_human_review(state: AgentState) -> bool:
    """Verifica se deve acionar revisão humana."""
    if not state.get("human_loop", {}).get("confidence_threshold"):
        return False
    
    quality = state.get("quality", {})
    confidence = quality.get("overall_confidence", 0.0)
    threshold = state["human_loop"]["confidence_threshold"]
    
    return confidence < threshold


def is_retry_needed(state: AgentState, error: RetryableError) -> bool:
    """Verifica se retry é necessário e possível."""
    retry_state = state.get("retry_state", {})
    current_retries = retry_state.get("retry_count", 0)
    max_retries = retry_state.get("max_retries", 3)
    
    return error.can_retry and current_retries < max_retries


def calculate_overall_confidence(state: AgentState) -> float:
    """Calcula confiança geral baseada nas análises."""
    analyses = state.get("analysis_results", [])
    if not analyses:
        return 0.0
    
    total_confidence = sum(analysis.confidence_score for analysis in analyses)
    return total_confidence / len(analyses) 