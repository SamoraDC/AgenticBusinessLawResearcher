#!/usr/bin/env python3
"""
Teste carregamento variáveis de ambiente
"""

import os
from dotenv import load_dotenv

load_dotenv()

print("🔍 === TESTE VARIÁVEIS DE AMBIENTE ===")

public_key = os.getenv("LANGFUSE_PUBLIC_KEY")
secret_key = os.getenv("LANGFUSE_SECRET_KEY")

print(f"PUBLIC_KEY: {'✅ Presente' if public_key else '❌ Ausente'}")
print(f"SECRET_KEY: {'✅ Presente' if secret_key else '❌ Ausente'}")

if public_key:
    print(f"Public Key: {public_key[:10]}...")
if secret_key:
    print(f"Secret Key: {secret_key[:10]}...")

# Testar importação do Langfuse
try:
    from langfuse import Langfuse
    print("✅ Langfuse importável")
    
    if public_key and secret_key:
        try:
            client = Langfuse(
                public_key=public_key,
                secret_key=secret_key,
                host="https://cloud.langfuse.com"
            )
            
            auth_result = client.auth_check()
            print(f"✅ Autenticação: {auth_result}")
            
        except Exception as e:
            print(f"❌ Erro ao criar cliente: {e}")
    
except ImportError as e:
    print(f"❌ Erro ao importar Langfuse: {e}")

print("\nTestando sistema de observabilidade...")

try:
    import sys
    sys.path.append('src')
    
    from src.core.observability import initialize_langfuse, LANGFUSE_AVAILABLE
    
    print(f"LANGFUSE_AVAILABLE: {LANGFUSE_AVAILABLE}")
    
    client = initialize_langfuse()
    print(f"Cliente criado: {client is not None}")
    
    if client:
        print(f"Tipo: {type(client)}")
    
except Exception as e:
    print(f"❌ Erro no sistema: {e}")
    import traceback
    traceback.print_exc() 