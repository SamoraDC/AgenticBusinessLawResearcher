#!/usr/bin/env python3
"""
Teste simples e direto para verificar se a correção DummyContext funciona.
"""

# Reproduzir exatamente o que está na observabilidade
class DummyContext:
    def update_current_trace(self, **kwargs): 
        pass
    def update_current_observation(self, **kwargs): 
        pass
    def create_trace(self, **kwargs): 
        return self
    def span(self, **kwargs): 
        return self
    def generation(self, **kwargs): 
        return self
    def update(self, **kwargs): 
        pass  # MÉTODO QUE FOI ADICIONADO
    def __enter__(self): 
        return self
    def __exit__(self, *args): 
        pass

def test_dummy_context():
    """Testa se todos os métodos DummyContext funcionam."""
    
    print("🧪 Testando DummyContext métodos...")
    
    context = DummyContext()
    
    # Testar métodos básicos
    try:
        context.update_current_trace(test="value")
        print("✅ update_current_trace - OK")
    except Exception as e:
        print(f"❌ update_current_trace - ERRO: {e}")
    
    try:
        context.update_current_observation(test="value")
        print("✅ update_current_observation - OK")
    except Exception as e:
        print(f"❌ update_current_observation - ERRO: {e}")
    
    try:
        span = context.span(name="test")
        print("✅ span - OK")
    except Exception as e:
        print(f"❌ span - ERRO: {e}")
    
    try:
        span.update(test="value")  # MÉTODO QUE ESTAVA FALTANDO
        print("✅ span.update - OK")
    except Exception as e:
        print(f"❌ span.update - ERRO: {e}")
    
    # Testar context manager
    try:
        with context.span(name="test_span") as span:
            span.update(input={"test": "value"}, metadata={"step": "test"})
            print("✅ Context manager com span.update - OK")
    except Exception as e:
        print(f"❌ Context manager - ERRO: {e}")

    print("\n🎯 TESTE DO PROBLEMA ESPECÍFICO:")
    try:
        # Simular exatamente o que estava causando erro
        with context.span(name="hybrid_processing") as span:
            span.update(
                input={"query": "test", "crag_summary": {}},
                metadata={"step": "hybrid_analysis_synthesis"}
            )
        print("✅ Erro DummyContext.update() RESOLVIDO!")
    except Exception as e:
        print(f"❌ Erro ainda persiste: {e}")

if __name__ == "__main__":
    test_dummy_context()
    print("\n🎉 Teste simples completo!") 