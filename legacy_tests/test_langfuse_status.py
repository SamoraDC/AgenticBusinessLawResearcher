#!/usr/bin/env python3
"""
Teste específico para verificar status do Langfuse
"""

import sys
import os
sys.path.append('src')

def test_langfuse():
    print("🔍 === TESTE STATUS LANGFUSE ===")
    
    # 1. Verificar variáveis de ambiente
    print("\n1️⃣ Verificando variáveis de ambiente...")
    
    from dotenv import load_dotenv
    load_dotenv()
    
    public_key = os.getenv("LANGFUSE_PUBLIC_KEY")
    secret_key = os.getenv("LANGFUSE_SECRET_KEY")
    
    print(f"LANGFUSE_PUBLIC_KEY: {'✅ Presente' if public_key else '❌ Ausente'}")
    print(f"LANGFUSE_SECRET_KEY: {'✅ Presente' if secret_key else '❌ Ausente'}")
    
    if public_key:
        print(f"Public Key: {public_key[:10]}...")
    if secret_key:
        print(f"Secret Key: {secret_key[:10]}...")
    
    # 2. Verificar importação do Langfuse
    print("\n2️⃣ Verificando importação do Langfuse...")
    
    try:
        from langfuse import Langfuse
        print("✅ Langfuse importado com sucesso")
        langfuse_available = True
    except ImportError as e:
        print(f"❌ Erro ao importar Langfuse: {e}")
        langfuse_available = False
    
    # 3. Verificar inicialização manual
    if langfuse_available and public_key and secret_key:
        print("\n3️⃣ Testando inicialização manual...")
        
        try:
            langfuse = Langfuse(
                public_key=public_key,
                secret_key=secret_key,
                host="https://cloud.langfuse.com"
            )
            print("✅ Cliente Langfuse criado com sucesso")
            
            # Teste básico - API correta para v3.0.2
            try:
                # Método correto no Langfuse v3
                trace_id = langfuse.create_trace_id()
                print(f"✅ Trace ID criado: {trace_id}")
                
                # Criar trace manualmente
                trace = langfuse.trace(id=trace_id, name="test_trace")
                print("✅ Trace criado com sucesso")
                
                # Testar auth
                auth_result = langfuse.auth_check()
                print(f"✅ Autenticação: {auth_result}")
                
                return True
                
            except Exception as e:
                print(f"⚠️ Erro nos testes: {e}")
                return False
                
        except Exception as e:
            print(f"❌ Erro ao criar cliente: {e}")
            return False
    
    # 4. Verificar sistema de observabilidade
    print("\n4️⃣ Verificando sistema de observabilidade...")
    
    try:
        from src.core.observability import _langfuse_client, LANGFUSE_AVAILABLE, initialize_langfuse
        
        print(f"LANGFUSE_AVAILABLE: {LANGFUSE_AVAILABLE}")
        print(f"Cliente global: {'✅ Presente' if _langfuse_client else '❌ None'}")
        
        if _langfuse_client:
            print(f"Tipo do cliente: {type(_langfuse_client)}")
        else:
            print("🔄 Tentando reinicializar...")
            new_client = initialize_langfuse()
            print(f"Nova tentativa: {'✅ Sucesso' if new_client else '❌ Falhou'}")
            
    except Exception as e:
        print(f"❌ Erro no sistema de observabilidade: {e}")
    
    print("\n📊 === RESUMO ===")
    return False

if __name__ == "__main__":
    result = test_langfuse()
    
    if result == True:
        print("🎉 Langfuse totalmente funcional!")
    else:
        print("⚠️ Verificando métodos disponíveis...")
        
        try:
            from langfuse import Langfuse
            from dotenv import load_dotenv
            load_dotenv()
            
            client = Langfuse(
                public_key=os.getenv("LANGFUSE_PUBLIC_KEY"),
                secret_key=os.getenv("LANGFUSE_SECRET_KEY"),
                host="https://cloud.langfuse.com"
            )
            
            methods = [m for m in dir(client) if not m.startswith('_')]
            print(f"Métodos disponíveis: {methods}")
            
        except Exception as e:
            print(f"Erro: {e}") 