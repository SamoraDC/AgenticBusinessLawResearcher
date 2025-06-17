"""
Módulo Agents - Agentes especializados para processamento legal
Contém agentes para recuperação, classificação, transformação e busca
"""

from .document_retriever import retrieve_documents
from .document_grader import grade_documents
from .query_transformer import transform_query
from .search_coordinator import (
    evaluate_search_necessity,
    search_jurisprudencia,
    search_web_conditional
)

__all__ = [
    "retrieve_documents",
    "grade_documents", 
    "transform_query",
    "evaluate_search_necessity",
    "search_jurisprudencia",
    "search_web_conditional"
] 