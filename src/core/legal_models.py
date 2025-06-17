"""
Modelos de dados para o sistema jurídico AI com validação robusta e tipagem forte.
"""

from __future__ import annotations

import uuid
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Literal, Optional, Union

from pydantic import BaseModel, Field, validator, ConfigDict
from typing_extensions import Annotated


class ValidationLevel(str, Enum):
    """Níveis de validação para entrada de dados."""
    STRICT = "strict"
    MODERATE = "moderate" 
    LENIENT = "lenient"


class Priority(str, Enum):
    """Prioridade da consulta jurídica."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    URGENT = "urgent"


class Status(str, Enum):
    """Status do processamento da consulta."""
    PENDING = "pending"
    PROCESSING = "processing"
    REVIEWING = "reviewing"  # Human-in-the-loop
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class JurisdictionType(str, Enum):
    """Tipos de jurisdição."""
    FEDERAL = "federal"
    STATE = "state"
    MUNICIPAL = "municipal"
    INTERNATIONAL = "international"
    UNKNOWN = "unknown"


class LegalAreaType(str, Enum):
    """Áreas do direito."""
    CIVIL = "civil"
    CRIMINAL = "criminal"
    LABOR = "labor"
    COMMERCIAL = "commercial"
    CONSTITUTIONAL = "constitutional"
    ADMINISTRATIVE = "administrative"
    TAX = "tax"
    ENVIRONMENTAL = "environmental"
    FAMILY = "family"
    CONSUMER = "consumer"
    OTHER = "other"


class SearchSource(str, Enum):
    """Fontes de busca de informações."""
    VECTORDB = "vectordb"
    LEXML = "lexml"
    WEB = "web"
    JURISPRUDENCE = "jurisprudence"
    LEGISLATION = "legislation"


class RetryableError(BaseModel):
    """Erro que pode ser reprocessado."""
    model_config = ConfigDict(frozen=True)
    
    error_type: str = Field(description="Tipo do erro")
    message: str = Field(description="Mensagem de erro")
    retry_count: int = Field(0, ge=0, description="Número de tentativas")
    max_retries: int = Field(3, ge=1, description="Máximo de tentativas")
    backoff_factor: float = Field(1.0, ge=0, description="Fator de backoff")
    
    @property
    def can_retry(self) -> bool:
        """Verifica se ainda pode tentar novamente."""
        return self.retry_count < self.max_retries


class HumanReview(BaseModel):
    """Solicitação de revisão humana."""
    model_config = ConfigDict(frozen=True)
    
    review_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    reason: str = Field(description="Motivo da revisão")
    confidence_threshold: float = Field(0.7, ge=0, le=1, description="Limite de confiança")
    reviewed_at: Optional[datetime] = Field(None, description="Quando foi revisado")
    reviewer_id: Optional[str] = Field(None, description="ID do revisor")
    approved: Optional[bool] = Field(None, description="Se foi aprovado")
    feedback: Optional[str] = Field(None, description="Feedback do revisor")


class LegalQuery(BaseModel):
    """Consulta jurídica com validação robusta."""
    model_config = ConfigDict(validate_assignment=True)
    
    # Identificação única
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    
    # Conteúdo da consulta
    text: Annotated[str, Field(
        min_length=10,
        max_length=5000,
        description="Texto da consulta jurídica"
    )]
    
    # Metadados inferidos ou fornecidos
    jurisdiction: Optional[JurisdictionType] = Field(
        None, 
        description="Jurisdição legal determinada"
    )
    area_of_law: Optional[LegalAreaType] = Field(
        None,
        description="Área do direito identificada"
    )
    
    # Configurações de processamento
    priority: Priority = Field(Priority.MEDIUM, description="Prioridade da consulta")
    validation_level: ValidationLevel = Field(
        ValidationLevel.MODERATE,
        description="Nível de validação"
    )
    
    # Timestamps
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)
    
    # Status e controle
    status: Status = Field(Status.PENDING, description="Status atual")
    retry_config: Optional[RetryableError] = Field(None, description="Configuração de retry")
    human_review: Optional[HumanReview] = Field(None, description="Revisão humana")
    
    @validator('text')
    def validate_text_content(cls, v: str) -> str:
        """Valida o conteúdo do texto."""
        # Remove espaços extras
        v = v.strip()
        
        # Verifica se não é apenas espaços ou caracteres especiais
        if not v or len(v.strip(".,?!;:")) < 5:
            raise ValueError("Consulta deve conter texto significativo")
            
        return v


class DocumentMetadata(BaseModel):
    """Metadados enriquecidos de documentos."""
    model_config = ConfigDict(frozen=True)
    
    document_type: str = Field(description="Tipo do documento")
    publication_date: Optional[datetime] = Field(None, description="Data de publicação")
    authority: Optional[str] = Field(None, description="Autoridade emissora")
    jurisdiction: Optional[JurisdictionType] = Field(None, description="Jurisdição")
    confidence_score: float = Field(0.0, ge=0, le=1, description="Score de confiança")
    language: str = Field("pt-BR", description="Idioma do documento")
    source: SearchSource = Field(description="Fonte da busca")
    access_date: datetime = Field(default_factory=datetime.now)
    
    # Informações adicionais
    tags: List[str] = Field(default_factory=list, description="Tags do documento")
    citations: List[str] = Field(default_factory=list, description="Citações")
    related_laws: List[str] = Field(default_factory=list, description="Leis relacionadas")


class DocumentSnippet(BaseModel):
    """Trecho de documento com metadados completos."""
    model_config = ConfigDict(validate_assignment=True)
    
    # Identificação
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    source_id: str = Field(description="ID da fonte do documento")
    
    # Conteúdo
    text: Annotated[str, Field(
        min_length=1,
        max_length=10000,
        description="Texto do trecho"
    )]
    
    # Metadados
    metadata: DocumentMetadata = Field(description="Metadados do documento")
    
    # Relacionamentos
    relationship_summary: Optional[str] = Field(
        None, 
        description="Como este trecho se relaciona com outros"
    )
    parent_document_id: Optional[str] = Field(None, description="ID do documento pai")
    
    # Relevância
    relevance_score: float = Field(0.0, ge=0, le=1, description="Score de relevância")
    extraction_method: str = Field("vector_search", description="Método de extração")
    
    # Timestamps
    extracted_at: datetime = Field(default_factory=datetime.now)


class SearchResult(BaseModel):
    """Resultado de busca unificado."""
    model_config = ConfigDict(frozen=True)
    
    # Identificação
    result_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    query: str = Field(description="Query de busca")
    source: SearchSource = Field(description="Fonte da busca")
    
    # Resultados
    documents: List[DocumentSnippet] = Field(
        default_factory=list,
        description="Documentos encontrados"
    )
    
    # Estatísticas
    total_results: int = Field(0, ge=0, description="Total de resultados")
    search_time_ms: float = Field(0.0, ge=0, description="Tempo de busca em ms")
    
    # Metadados de busca
    search_metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Metadados específicos da fonte"
    )
    
    # Controle de qualidade
    success: bool = Field(True, description="Se a busca foi bem-sucedida")
    error_message: Optional[str] = Field(None, description="Mensagem de erro")
    
    # Timestamps
    searched_at: datetime = Field(default_factory=datetime.now)


class AnalysisResult(BaseModel):
    """Resultado de análise jurídica."""
    model_config = ConfigDict(validate_assignment=True)
    
    # Identificação
    analysis_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    
    # Conteúdo da análise
    summary: Annotated[str, Field(
        min_length=50,
        max_length=5000,
        description="Resumo da análise jurídica"
    )]
    
    # Qualidade e confiabilidade
    confidence_score: float = Field(0.0, ge=0, le=1, description="Score de confiança")
    legal_certainty: float = Field(0.0, ge=0, le=1, description="Certeza jurídica")
    
    # Documentos de apoio
    supporting_documents: List[DocumentSnippet] = Field(
        default_factory=list,
        description="Documentos que apoiam a análise"
    )
    
    # Classificação
    legal_area: Optional[LegalAreaType] = Field(None, description="Área do direito")
    jurisdiction: Optional[JurisdictionType] = Field(None, description="Jurisdição")
    
    # Revisão e validação
    needs_human_review: bool = Field(False, description="Precisa revisão humana")
    review_reason: Optional[str] = Field(None, description="Motivo da revisão")
    human_review: Optional[HumanReview] = Field(None, description="Dados da revisão")
    
    # Timestamps
    analyzed_at: datetime = Field(default_factory=datetime.now)
    validated_at: Optional[datetime] = Field(None, description="Quando foi validado")


class FinalResponse(BaseModel):
    """Resposta final com validação completa."""
    model_config = ConfigDict(validate_assignment=True)
    
    # Identificação
    response_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    query_id: str = Field(description="ID da consulta original")
    
    # Resposta principal
    overall_summary: Annotated[str, Field(
        min_length=100,
        max_length=15000,
        description="Resposta final completa"
    )]
    
    # Análises detalhadas
    detailed_analyses: List[AnalysisResult] = Field(
        default_factory=list,
        description="Análises detalhadas por aspecto"
    )
    
    # Qualidade geral
    overall_confidence: float = Field(0.0, ge=0, le=1, description="Confiança geral")
    completeness_score: float = Field(0.0, ge=0, le=1, description="Score de completude")
    
    # Fontes utilizadas
    search_results: List[SearchResult] = Field(
        default_factory=list,
        description="Resultados de busca utilizados"
    )
    
    # Disclaimers e avisos
    disclaimer: str = Field(
        default="Esta informação é gerada por IA e não substitui assessoria jurídica profissional. Busque sempre orientação de advogado especializado para sua situação específica.",
        description="Disclaimer jurídico"
    )
    
    warnings: List[str] = Field(
        default_factory=list,
        description="Avisos importantes"
    )
    
    # Status e controle
    status: Status = Field(Status.COMPLETED, description="Status da resposta")
    validated: bool = Field(False, description="Se foi validada")
    human_reviewed: bool = Field(False, description="Se foi revisada por humano")
    
    # Timestamps
    generated_at: datetime = Field(default_factory=datetime.now)
    validated_at: Optional[datetime] = Field(None, description="Quando foi validada")
    
    @validator('overall_summary')
    def validate_summary_quality(cls, v: str) -> str:
        """Valida a qualidade do resumo."""
        # Verifica se contém estrutura mínima (mais inteligente)
        sentences = [s.strip() for s in v.split('.') if s.strip()]
        if len(sentences) < 2:  # Reduzido de 3 para 2 frases
            raise ValueError("Resposta deve conter pelo menos 2 frases completas")
        
        # Verifica comprimento mínimo total
        if len(v.strip()) < 50:  # Mínimo absoluto muito baixo
            raise ValueError("Resposta muito curta - mínimo 50 caracteres")
            
        # Verifica se não é muito repetitiva (mais tolerante)
        words = v.lower().split()
        if len(words) < 10:  # Se muito curta, não aplicar regra de repetição
            return v
            
        unique_words = set(words)
        if len(unique_words) / len(words) < 0.25:  # Reduzido de 0.3 para 0.25
            raise ValueError("Resposta muito repetitiva")
            
        return v


class ProcessingConfig(BaseModel):
    """Configuração de processamento."""
    model_config = ConfigDict(frozen=True)
    
    # Configurações de busca
    max_documents_per_source: int = Field(10, ge=1, le=50)
    search_timeout_seconds: int = Field(30, ge=5, le=300)
    enable_parallel_search: bool = Field(True)
    
    # Configurações de retry
    max_retries: int = Field(3, ge=1, le=10)
    retry_backoff_factor: float = Field(1.5, ge=1.0, le=5.0)
    
    # Configurações de qualidade
    min_confidence_threshold: float = Field(0.3, ge=0, le=1)
    human_review_threshold: float = Field(0.7, ge=0, le=1)
    
    # Configurações de modelo
    temperature: float = Field(0.1, ge=0, le=2)
    max_tokens: int = Field(4000, ge=100, le=8000)
    
    # Features habilitadas
    enable_human_review: bool = Field(True)
    enable_web_search: bool = Field(True)
    enable_jurisprudence_search: bool = Field(True)
    enable_guardrails: bool = Field(True)


# Unions para diferentes tipos de saída
SearchSourceUnion = Union[SearchResult, RetryableError]
AnalysisResultUnion = Union[AnalysisResult, HumanReview, RetryableError]
FinalResponseUnion = Union[FinalResponse, HumanReview, RetryableError] 