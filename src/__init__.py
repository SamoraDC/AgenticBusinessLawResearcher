"""
Sistema de Consulta Jurídica com IA
Módulo principal com componentes de processamento legal
"""

from .core.legal_models import LegalQuery, Priority, ValidationLevel, FinalResponse
from .core.workflow_builder import build_graph
from .agents.streaming.response_synthesizer import synthesize_response_streaming

__version__ = "1.0.0"
__author__ = "Sistema Jurídico AI"

__all__ = [
    "LegalQuery",
    "Priority", 
    "ValidationLevel",
    "FinalResponse",
    "build_graph",
    "synthesize_response_streaming"
] 