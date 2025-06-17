#!/usr/bin/env python3
"""
Teste da implementação híbrida CORRETA:
- Groq: Tools e operações estruturadas 
- OpenRouter: RAG, análise e síntese
"""

import asyncio
import os
import time
from src.agents.synthesizer import synthesize_with_hybrid_corrected_approach
from src.graph_state import AgentState
from src.models import LegalQuery

# Mock de dados para teste
MOCK_CRAG_DOCS = """
[1] Código Civil - Artigos sobre sociedades: Art. 1.085 do Código Civil estabelece que a exclusão de sócio pode ser feita por deliberação dos demais sócios, respeitado o contraditório e ampla defesa. O procedimento deve observar as regras do contrato social.

[2] Lei 6.404/76 - Sociedades Anônimas: A Lei das S.A. prevê mecanismos específicos para dissolução parcial e retirada de acionistas, com regras próprias para apuração de haveres conforme artigos 206 a 214.

[3] Jurisprudência STJ - Exclusão de sócios: REsp 1.234.567/SP estabeleceu que a exclusão deve ser sempre fundamentada em justa causa e precedida de oportunidade para exercício do contraditório.
"""

MOCK_LEXML_RESULTS = """
[Juris 1] Lei 10.406/2002: Art. 1.085 - A exclusão do sócio pode ser feita por deliberação dos demais sócios, mediante alteração do contrato social, desde que respeitado o contraditório e ampla defesa.

[Juris 2] Lei 6.404/1976: Arts. 206-214 - Regras específicas para dissolução de sociedades anônimas e apuração de haveres de acionistas dissidentes.

[Juris 3] Súmula 413 STJ: Em sociedade limitada, é necessário procedimento específico para exclusão de sócio, respeitando direitos fundamentais do contraditório.
"""

MOCK_TAVILY_RESULTS = """
[Web 1] STJ Decisões 2024: "Exclusão de sócio deve respeitar contraditório e ser fundamentada em justa causa específica para preservar direitos do sócio minoritário."

[Web 2] TJSP Jurisprudência: "Procedimento extrajudicial para exclusão é válido desde que previamente estabelecido no contrato social e respeitados direitos processuais."

[Web 3] Doutrina Especializada: "Apuração de haveres deve considerar valor real da empresa no momento da exclusão, não apenas valor contábil."
"""

async def test_hybrid_corrected_approach():
    """Teste da abordagem híbrida correta"""
    
    print("🧪 === TESTE DA IMPLEMENTAÇÃO HÍBRIDA CORRETA ===")
    print("🔧 Groq: Tools e buscas estruturadas (simulado)")
    print("🧠 OpenRouter: RAG, análise e síntese")
    print()
    
    # Query de teste
    test_query = "Como proceder com a exclusão de um sócio em sociedade limitada que não está cumprindo suas obrigações sociais?"
    
    print(f"📝 Query de teste: {test_query}")
    print()
    
    # Executar síntese híbrida
    try:
        print("🚀 Iniciando teste da síntese híbrida...")
        start_time = time.time()
        
        result = await synthesize_with_hybrid_corrected_approach(
            query_text=test_query,
            formatted_crag=MOCK_CRAG_DOCS,
            formatted_tavily=MOCK_TAVILY_RESULTS,
            formatted_lexml=MOCK_LEXML_RESULTS
        )
        
        elapsed_time = time.time() - start_time
        
        print()
        print(f"⏱️ Tempo total: {elapsed_time:.2f}s")
        print(f"📏 Tamanho da resposta: {len(result.overall_summary)} caracteres")
        print(f"💬 Disclaimer: {result.disclaimer}")
        print()
        
        print("📄 === RESPOSTA FINAL ===")
        print(result.overall_summary)
        print()
        
        # Validações básicas
        success_indicators = [
            (len(result.overall_summary) >= 300, "Resposta com tamanho adequado"),
            ("direito" in result.overall_summary.lower(), "Menciona aspecto jurídico"),
            ("código civil" in result.overall_summary.lower() or "lei" in result.overall_summary.lower(), "Cita legislação"),
            ("exclusão" in result.overall_summary.lower(), "Aborda o tema da consulta"),
            ("sócio" in result.overall_summary.lower(), "Menciona sócios"),
            (result.disclaimer is not None, "Inclui disclaimer"),
        ]
        
        print("✅ === VALIDAÇÕES ===")
        for condition, description in success_indicators:
            status = "✅ PASS" if condition else "❌ FAIL"
            print(f"  {status}: {description}")
        
        passed = sum(1 for condition, _ in success_indicators if condition)
        total = len(success_indicators)
        success_rate = (passed / total) * 100
        
        print()
        print(f"📊 Taxa de sucesso: {passed}/{total} ({success_rate:.1f}%)")
        
        if success_rate >= 80:
            print("🎉 TESTE APROVADO! Implementação híbrida funcionando corretamente.")
        elif success_rate >= 60:
            print("⚠️ TESTE PARCIAL: Funcionamento básico OK, mas precisa melhorias.")
        else:
            print("❌ TESTE FALHADO: Implementação precisa de correções.")
            
        return result
        
    except Exception as e:
        print(f"❌ ERRO NO TESTE: {str(e)}")
        import traceback
        print(f"📋 Traceback: {traceback.format_exc()}")
        return None


async def test_integration_with_synthesizer():
    """Teste de integração com o synthesizer do sistema"""
    
    print("\n🔗 === TESTE DE INTEGRAÇÃO COM SYNTHESIZER ===")
    
    # Criar estado mock
    from src.models import LegalQuery
    
    mock_query = LegalQuery(
        text="Como excluir sócio de sociedade limitada por descumprimento de obrigações?",
        area=None,
        jurisdiction=None
    )
    
    # Simular estado do agent
    state = {
        "query": mock_query,
        "retrieved_docs": [
            type('MockDoc', (), {
                'source_id': 'Código Civil',
                'text': 'Art. 1.085 - Exclusão de sócio pode ser feita por deliberação...'
            })()
        ],
        "tavily_results": [
            type('MockTavily', (), {
                'title': 'STJ Jurisprudência',
                'content': 'Exclusão deve respeitar contraditório...'
            })()
        ],
        "lexml_results": [
            type('MockLexML', (), {
                'urn': 'Lei 10.406/2002',
                'ementa': 'Código Civil - Artigos sobre sociedades...'
            })()
        ]
    }
    
    try:
        print("🚀 Executando synthesize_response...")
        
        # Importar e executar função do synthesizer
        from src.agents.synthesizer import synthesize_response
        
        result = await synthesize_response(state)
        
        print(f"✅ Integração bem-sucedida!")
        print(f"📊 Tipo do resultado: {type(result)}")
        
        if 'final_response' in result:
            final_response = result['final_response']
            print(f"📝 Resposta final presente: {len(str(final_response))} chars")
            
            if isinstance(final_response, dict) and 'overall_summary' in final_response:
                print(f"📄 Summary: {final_response['overall_summary'][:200]}...")
        
        return result
        
    except Exception as e:
        print(f"❌ ERRO NA INTEGRAÇÃO: {str(e)}")
        import traceback
        print(f"📋 Traceback: {traceback.format_exc()}")
        return None


async def main():
    """Função principal de teste"""
    
    print("🏁 Iniciando testes da implementação híbrida correta...")
    print()
    
    # Verificar variáveis de ambiente
    groq_key = os.getenv("GROQ_API_KEY")
    openrouter_key = os.getenv("OPENROUTER_API_KEY")
    
    print("🔑 Verificação das APIs:")
    print(f"  GROQ_API_KEY: {'✅ Configurada' if groq_key else '❌ Ausente'}")
    print(f"  OPENROUTER_API_KEY: {'✅ Configurada' if openrouter_key else '❌ Ausente'}")
    
    if not groq_key:
        print("⚠️ GROQ_API_KEY ausente - sistema usará fallbacks")
    if not openrouter_key:
        print("⚠️ OPENROUTER_API_KEY ausente - síntese será feita com Groq")
    
    print()
    
    # Teste 1: Abordagem híbrida isolada
    print("📋 TESTE 1: Síntese híbrida isolada")
    result1 = await test_hybrid_corrected_approach()
    
    print("\n" + "="*50)
    
    # Teste 2: Integração com synthesizer
    print("📋 TESTE 2: Integração com synthesizer do sistema")
    result2 = await test_integration_with_synthesizer()
    
    print("\n" + "="*50)
    print("🏆 RESUMO DOS TESTES:")
    print(f"  Teste híbrido isolado: {'✅ OK' if result1 else '❌ FALHOU'}")
    print(f"  Teste de integração: {'✅ OK' if result2 else '❌ FALHOU'}")
    
    if result1 and result2:
        print("\n🎉 TODOS OS TESTES PASSARAM! Sistema híbrido corrigido está funcional.")
    elif result1 or result2:
        print("\n⚠️ TESTES PARCIAIS: Alguns componentes funcionam, outros precisam ajustes.")
    else:
        print("\n❌ FALHA GERAL: Sistema precisa de correções antes de usar em produção.")


if __name__ == "__main__":
    asyncio.run(main()) 