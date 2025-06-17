#!/usr/bin/env python3
"""
Teste simples para verificar se a correção DummyContext funciona.
"""

import sys
sys.path.append('src')

from src.core.observability import langfuse_context, trace_legal_query, trace_crag_execution, trace_hybrid_processing

def test_dummy_context():
    """Testa se todos os métodos DummyContext funcionam."""
    
    print("🧪 Testando DummyContext métodos...")
    
    # Testar métodos básicos
    try:
        langfuse_context.update_current_trace(test="value")
        print("✅ update_current_trace - OK")
    except Exception as e:
        print(f"❌ update_current_trace - ERRO: {e}")
    
    try:
        langfuse_context.update_current_observation(test="value")
        print("✅ update_current_observation - OK")
    except Exception as e:
        print(f"❌ update_current_observation - ERRO: {e}")
    
    try:
        span = langfuse_context.span(name="test")
        print("✅ span - OK")
    except Exception as e:
        print(f"❌ span - ERRO: {e}")
    
    try:
        span.update(test="value")  # MÉTODO QUE ESTAVA FALTANDO
        print("✅ span.update - OK")
    except Exception as e:
        print(f"❌ span.update - ERRO: {e}")
    
    # Testar context managers
    try:
        with trace_legal_query("test_id", "test query") as trace:
            print("✅ trace_legal_query context manager - OK")
    except Exception as e:
        print(f"❌ trace_legal_query - ERRO: {e}")
    
    try:
        with trace_crag_execution("test query") as span:
            print("✅ trace_crag_execution context manager - OK")
    except Exception as e:
        print(f"❌ trace_crag_execution - ERRO: {e}")
    
    try:
        with trace_hybrid_processing("test query", {"total_sources": 0}) as span:
            print("✅ trace_hybrid_processing context manager - OK")
    except Exception as e:
        print(f"❌ trace_hybrid_processing - ERRO: {e}")

if __name__ == "__main__":
    test_dummy_context()
    print("\n🎉 Teste completo! Se não houve erros, a correção funcionou.") 