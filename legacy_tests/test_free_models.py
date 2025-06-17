#!/usr/bin/env python3
"""
Script rápido para testar modelos gratuitos da OpenRouter
Execute: python test_free_models.py
"""

import sys
import os
import asyncio

# Adicionar src ao path
sys.path.append('src')
sys.path.append('tests')

# Importar o teste completo
from tests.test_openrouter_free_models import main

if __name__ == "__main__":
    print("🚀 Executando teste dos modelos gratuitos da OpenRouter...")
    print("=" * 60)
    
    # Verificar se a API key está configurada
    if not os.getenv("OPENROUTER_API_KEY"):
        print("❌ ERRO: OPENROUTER_API_KEY não configurada!")
        print("💡 Configure a variável de ambiente antes de executar:")
        print("   export OPENROUTER_API_KEY='sua_chave_aqui'")
        print("   ou adicione no arquivo .env")
        sys.exit(1)
    
    # Executar testes
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n⚠️ Teste interrompido pelo usuário")
    except Exception as e:
        print(f"\n❌ Erro durante execução: {str(e)}")
        sys.exit(1) 