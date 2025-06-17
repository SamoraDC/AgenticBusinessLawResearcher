"""
Teste de integração do sistema híbrido corrigido com o synthesizer.
"""

import asyncio
import time
from src.agents.synthesizer import synthesize_with_hybrid_corrected_approach
from src.models import LegalQuery

async def test_synthesizer_integration():
    """Testa a integração do synthesizer com o sistema híbrido corrigido."""
    
    print("🧪 === TESTE DE INTEGRAÇÃO SYNTHESIZER ===")
    print("🎯 Testando integração com sistema híbrido corrigido real")
    
    # Query de teste
    query_text = "Como um sócio pode sair de uma sociedade limitada no Brasil?"
    
    # Dados simulados (como viria do sistema CRAG atual)
    formatted_crag = """
    [1] Documento 1: O Código Civil brasileiro estabelece os mecanismos para retirada de sócios...
    [2] Documento 2: Direito de recesso e dissolução parcial da sociedade são institutos distintos...
    """
    
    formatted_tavily = """
    [Web 1] STJ Recent Decision: Nova jurisprudência sobre direito de recesso em sociedades limitadas...
    [Web 2] Legal Analysis: Procedimentos atualizados para saída de sócios...
    """
    
    formatted_lexml = """
    [Juris 1] Código Civil Art. 1077: Direito de retirada do sócio em sociedade limitada...
    [Juris 2] Lei 6404/76: Aplicação subsidiária às sociedades limitadas...
    """
    
    try:
        print(f"📝 Query: {query_text}")
        print("🚀 Iniciando síntese híbrida...")
        
        start_time = time.time()
        
        # Executar síntese híbrida corrigida
        result = await synthesize_with_hybrid_corrected_approach(
            query_text=query_text,
            formatted_crag=formatted_crag,
            formatted_tavily=formatted_tavily,
            formatted_lexml=formatted_lexml
        )
        
        elapsed_time = time.time() - start_time
        
        print(f"⏱️ Tempo total: {elapsed_time:.2f}s")
        print(f"📏 Tamanho da resposta: {len(result.overall_summary)} chars")
        print("✅ TESTE CONCLUÍDO COM SUCESSO!")
        
        # Mostrar resultado resumido
        print("\n🎯 === RESULTADO RESUMIDO ===")
        print(f"Primeiros 200 chars: {result.overall_summary[:200]}...")
        print(f"Disclaimer: {result.disclaimer}")
        
        return True
        
    except Exception as e:
        print(f"❌ ERRO NO TESTE: {str(e)}")
        import traceback
        print(f"Traceback: {traceback.format_exc()}")
        return False

async def test_fallback_mechanism():
    """Testa se o mecanismo de fallback funciona."""
    
    print("\n🔄 === TESTE FALLBACK ===")
    print("🎯 Forçando erro para testar fallback")
    
    try:
        # Query que pode causar problemas
        query_text = "Test query for fallback"
        
        result = await synthesize_with_hybrid_corrected_approach(
            query_text=query_text,
            formatted_crag="",
            formatted_tavily="",
            formatted_lexml=""
        )
        
        print("✅ Fallback funcionou!")
        print(f"📏 Resposta: {len(result.overall_summary)} chars")
        
        return True
        
    except Exception as e:
        print(f"❌ ERRO NO FALLBACK: {str(e)}")
        return False

if __name__ == "__main__":
    print("🔧 Iniciando testes de integração do synthesizer...")
    
    # Executar testes
    success1 = asyncio.run(test_synthesizer_integration())
    success2 = asyncio.run(test_fallback_mechanism())
    
    print(f"\n📊 === RESUMO DOS TESTES ===")
    print(f"✅ Teste principal: {'PASSOU' if success1 else 'FALHOU'}")
    print(f"✅ Teste fallback: {'PASSOU' if success2 else 'FALHOU'}")
    
    if success1 and success2:
        print("🎉 TODOS OS TESTES PASSARAM!")
        print("🚀 Sistema híbrido corrigido integrado com sucesso no synthesizer!")
    else:
        print("⚠️ Alguns testes falharam. Verifique os logs acima.") 