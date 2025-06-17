#!/usr/bin/env python3
"""
Teste simples para Langfuse v3.0.2
"""

import sys
import os
sys.path.append('src')

def test_langfuse_simple():
    print("🔍 === TESTE SIMPLES LANGFUSE v3.0.2 ===")
    
    from dotenv import load_dotenv
    load_dotenv()
    
    from langfuse import Langfuse
    
    public_key = os.getenv("LANGFUSE_PUBLIC_KEY")
    secret_key = os.getenv("LANGFUSE_SECRET_KEY")
    
    if not public_key or not secret_key:
        print("❌ Chaves não encontradas")
        return False
    
    try:
        # Inicializar cliente
        langfuse = Langfuse(
            public_key=public_key,
            secret_key=secret_key,
            host="https://cloud.langfuse.com"
        )
        
        print("✅ Cliente criado")
        
        # Verificar autenticação
        auth_result = langfuse.auth_check()
        print(f"✅ Autenticação: {auth_result}")
        
        # Criar trace ID
        trace_id = langfuse.create_trace_id()
        print(f"✅ Trace ID: {trace_id}")
        
        # Método correto para criar evento no v3.0.2
        event = langfuse.create_event(
            name="test_event",
            input={"query": "teste"},
            output={"response": "ok"}
        )
        print(f"✅ Evento criado: {event}")
        
        # Testar span
        span = langfuse.start_span(
            name="test_span",
            input={"data": "teste"}
        )
        print(f"✅ Span criado: {span}")
        
        # Flush para enviar dados
        langfuse.flush()
        print("✅ Dados enviados")
        
        return True
        
    except Exception as e:
        print(f"❌ Erro: {e}")
        return False

if __name__ == "__main__":
    success = test_langfuse_simple()
    
    if success:
        print("🎉 Langfuse funcionando!")
    else:
        print("❌ Langfuse com problemas") 