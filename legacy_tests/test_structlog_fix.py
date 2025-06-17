#!/usr/bin/env python3
"""
Teste para verificar se o conflito do structlog foi resolvido.
"""

import sys
sys.path.append('src')

try:
    from src.core.observability import log_data_flow_checkpoint, log_performance_metrics, log_detailed_state
    
    def test_logging_functions():
        """Testa se as funções de logging não têm mais conflitos."""
        
        print("🧪 Testando funções de logging corrigidas...")
        
        # Testar log_data_flow_checkpoint
        try:
            log_data_flow_checkpoint("test_checkpoint", {
                "step": "test_step",  # Este era um dos parâmetros conflitantes
                "query_id": "test_123",
                "count": 5
            })
            print("✅ log_data_flow_checkpoint - OK")
        except Exception as e:
            print(f"❌ log_data_flow_checkpoint - ERRO: {e}")
        
        # Testar log_performance_metrics  
        try:
            log_performance_metrics("test_operation", 100.5, 
                                   step="performance_step",  # Potencial conflito
                                   events_processed=10)
            print("✅ log_performance_metrics - OK")
        except Exception as e:
            print(f"❌ log_performance_metrics - ERRO: {e}")
        
        # Testar log_detailed_state
        try:
            test_state = {
                "retrieved_docs": [],
                "tavily_results": [],
                "lexml_results": [],
                "query": "test query",
                "current_query": "test",
                "should_synthesize": False,
                "final_response": None
            }
            log_detailed_state(test_state, "test_step_name")
            print("✅ log_detailed_state - OK")
        except Exception as e:
            print(f"❌ log_detailed_state - ERRO: {e}")
        
        print("\n🎯 TESTE DO PROBLEMA ESPECÍFICO:")
        try:
            # Simular exatamente as chamadas que estavam causando erro
            log_data_flow_checkpoint("query_creation", {
                "query_id": "test_id",
                "query_text": "test query",
                "priority": "Priority.MEDIUM",
                "validation_level": "ValidationLevel.MODERATE",
                "use_hybrid_streaming": True
            })
            print("✅ Erro de structlog conflito RESOLVIDO!")
        except Exception as e:
            print(f"❌ Erro ainda persiste: {e}")

    if __name__ == "__main__":
        test_logging_functions()
        print("\n🎉 Teste de conflitos structlog completo!")

except ImportError as e:
    print(f"❌ Erro de importação: {e}")
    print("Verifique se todas as dependências estão instaladas.") 