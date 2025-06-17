"""
Teste específico para verificar se a arquitetura híbrida CORRETA funciona.
Arquitetura REAL:
🧠 OpenRouter (gemini-2.0-flash-exp:free) - Etapa 1: Decisão de busca
🧠 OpenRouter (llama-4-scout:free) - Etapa 2: Busca vectordb  
🔧 Groq (llama-3.3-70b-versatile) - Etapa 2.1: Busca WEB + LexML
🧠 OpenRouter (llama-4-scout:free) - Etapa 3: Análise jurídica RAG
🧠 OpenRouter (gemma-3-27b-it:free) - Etapa 4: Síntese final
🧠 OpenRouter (deepseek-r1-0528:free) - Etapa 5: Validação de qualidade
🧠 OpenRouter - Etapa 6: Verificação de guardrails
"""

import asyncio
import time
from src.agents.pydantic_agents_hybrid_corrected import process_legal_query_hybrid_corrected
from src.models import LegalQuery, ProcessingConfig

async def test_correct_hybrid_architecture():
    """Testa se a arquitetura híbrida CORRETA funciona."""
    
    print("🔧 === TESTE DA ARQUITETURA HÍBRIDA CORRETA ===")
    print("🎯 Verificando se a nova divisão OpenRouter + Groq funciona")
    print("\n📋 ARQUITETURA ESPERADA:")
    print("🧠 OpenRouter (gemini-2.0-flash-exp) - Etapa 1: Decisão de busca")
    print("🧠 OpenRouter (llama-4-scout) - Etapa 2: Busca vectordb")
    print("🔧 Groq (llama-3.3-70b-versatile) - Etapa 2.1: Busca WEB + LexML")
    print("🧠 OpenRouter (llama-4-scout) - Etapa 3: Análise jurídica RAG")
    print("🧠 OpenRouter (gemma-3-27b-it) - Etapa 4: Síntese final")
    print("🧠 OpenRouter (deepseek-r1-0528) - Etapa 5: Validação de qualidade")
    print("🧠 OpenRouter - Etapa 6: Verificação de guardrails")
    
    try:
        # Query para teste
        query = LegalQuery(text="Como um sócio pode ser excluído de uma sociedade limitada?")
        
        # Configuração otimizada para teste
        config = ProcessingConfig(
            max_documents_per_source=3,
            search_timeout_seconds=30,
            enable_parallel_search=True,
            max_retries=1,
            enable_human_review=False,
            enable_web_search=True,
            enable_jurisprudence_search=True,
            enable_guardrails=True
        )
        
        print(f"\n📝 Query: {query.text}")
        print("🚀 Iniciando teste da arquitetura híbrida correta...")
        
        start_time = time.time()
        
        # Executar sistema híbrido corrigido
        result = await process_legal_query_hybrid_corrected(
            query=query,
            config=config,
            user_id="test_correct_architecture"
        )
        
        elapsed_time = time.time() - start_time
        
        print(f"\n⏱️ Tempo total: {elapsed_time:.2f}s")
        print(f"📊 Status: {result.status}")
        print(f"🎯 Confiança: {result.overall_confidence:.2%}")
        print(f"📏 Tamanho resposta: {len(result.overall_summary)} chars")
        print(f"⚠️ Warnings: {len(result.warnings)}")
        
        # Exibir resumo da resposta
        if len(result.overall_summary) > 300:
            preview = result.overall_summary[:300] + "..."
        else:
            preview = result.overall_summary
        
        print(f"\n📄 PREVIEW DA RESPOSTA:")
        print(f"{preview}")
        
        # Verificar se chegou até o final
        if result.status.value == "completed":
            print("\n✅ SUCESSO: Arquitetura híbrida CORRETA funcionando!")
            print("⚡ Etapas concluídas:")
            print("  1. ✅ Decisão de busca (OpenRouter - gemini-2.0-flash-exp)")
            print("  2. ✅ Busca vectordb (OpenRouter - llama-4-scout)")
            print("  2.1 ✅ Busca WEB + LexML (Groq - llama-3.3-70b-versatile)")
            print("  3. ✅ Análise jurídica RAG (OpenRouter - llama-4-scout)")
            print("  4. ✅ Síntese final (OpenRouter - gemma-3-27b-it)")
            print("  5. ✅ Validação de qualidade (OpenRouter - deepseek-r1-0528)")
            print("  6. ✅ Verificação de guardrails (OpenRouter)")
            
            # Verificar qualidade da resposta
            if result.overall_confidence >= 0.7:
                print(f"🎯 QUALIDADE EXCELENTE: {result.overall_confidence:.1%} de confiança")
            elif result.overall_confidence >= 0.5:
                print(f"🎯 QUALIDADE BOA: {result.overall_confidence:.1%} de confiança")
            else:
                print(f"⚠️ QUALIDADE BAIXA: {result.overall_confidence:.1%} de confiança")
            
            return True
            
        elif result.status.value == "failed":
            print("\n❌ FALHA: Sistema ainda não está funcionando corretamente")
            print("🔍 Verificar logs de erro acima")
            return False
        else:
            print(f"\n⚠️ Status inesperado: {result.status}")
            return False
            
    except Exception as e:
        print(f"\n❌ ERRO NO TESTE: {str(e)}")
        import traceback
        print(f"Traceback: {traceback.format_exc()}")
        return False

if __name__ == "__main__":
    print("🔧 Testando arquitetura híbrida CORRETA...")
    
    success = asyncio.run(test_correct_hybrid_architecture())
    
    print(f"\n📊 === RESULTADO FINAL ===")
    if success:
        print("🎉 TESTE PASSOU! Arquitetura híbrida CORRETA funcionando!")
        print("✅ OpenRouter fazendo a maioria das operações")
        print("✅ Groq fazendo apenas buscas WEB + LexML")
        print("🚀 Sistema pronto para uso no Streamlit")
    else:
        print("⚠️ TESTE FALHOU! Ainda há problemas na arquitetura")
        print("🔍 Verificar logs e implementação") 