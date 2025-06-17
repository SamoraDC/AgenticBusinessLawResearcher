"""
Módulo Streaming - Componentes de streaming e processamento híbrido
Contém sintetizadores de resposta e processadores legais com streaming
"""

from .response_synthesizer import synthesize_response_streaming, synthesize_response
from .hybrid_legal_processor import process_legal_query_hybrid_corrected

__all__ = [
    "synthesize_response_streaming",
    "synthesize_response", 
    "process_legal_query_hybrid_corrected"
] 