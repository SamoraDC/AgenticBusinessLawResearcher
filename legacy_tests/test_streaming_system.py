import asyncio
import sys
import traceback
from dotenv import load_dotenv

# Carregar env
load_dotenv()

# Adicionar src ao path
sys.path.append('src')

async def test_streaming_system():
    """Teste simples do sistema híbrido com streaming"""
    
    try:
        print("🔥 Testando Sistema Híbrido com STREAMING...")
        print("=" * 60)
        
        # Importar módulos
        from src.models import LegalQuery, ProcessingConfig
        from src.agents.pydantic_agents_hybrid_corrected import process_legal_query_hybrid_corrected_streaming
        
        print("✅ Módulos carregados com sucesso")
        
        # Criar consulta de teste
        query = LegalQuery(text="O que são direitos do consumidor no e-commerce?")
        
        # Configuração básica
        config = ProcessingConfig(
            max_documents_per_source=3,
            search_timeout_seconds=20,
            enable_web_search=True,
            enable_jurisprudence_search=True,
            enable_guardrails=True,
            max_retries=1
        )
        
        print(f"📝 Consulta: {query.text}")
        print(f"⚙️ Config: docs={config.max_documents_per_source}, timeout={config.search_timeout_seconds}s")
        print()
        
        # Processar com streaming
        print("🚀 Iniciando processamento híbrido com streaming...")
        print("-" * 60)
        
        final_result = None
        streaming_chunks = []
        
        async for step_type, content in process_legal_query_hybrid_corrected_streaming(
            query=query,
            config=config,
            user_id="test_user"
        ):
            
            if step_type == "progress":
                print(f"📊 PROGRESSO: {content}")
            
            elif step_type == "streaming":
                # Simular streaming visual
                print(f"✍️ STREAMING: {len(content)} chars...")
                streaming_chunks.append(content)
                
                # Mostrar apenas as últimas palavras para não poluir
                words = content.split()
                if len(words) > 10:
                    preview = " ".join(words[-10:])
                    print(f"   📝 Preview: ...{preview}")
                else:
                    print(f"   📝 Texto: {content}")
            
            elif step_type == "final":
                final_result = content
                print(f"🎯 FINAL: Resultado obtido!")
                break
            
            elif step_type == "error":
                print(f"❌ ERRO: {content}")
                break
        
        print()
        print("=" * 60)
        
        if final_result:
            print("✅ SISTEMA HÍBRIDO COM STREAMING FUNCIONOU!")
            print(f"📊 Status: {final_result.status}")
            print(f"🎯 Confiança: {final_result.overall_confidence:.1%}")
            print(f"📝 Completude: {final_result.completeness_score:.1%}")
            print(f"⚠️ Avisos: {len(final_result.warnings)}")
            print(f"📏 Texto final: {len(final_result.overall_summary)} chars")
            
            if streaming_chunks:
                print(f"⚡ Chunks de streaming: {len(streaming_chunks)}")
                print(f"📈 Crescimento do texto: {[len(chunk) for chunk in streaming_chunks[:5]]}...")
            
            print()
            print("📋 RESPOSTA HÍBRIDA:")
            print("-" * 40)
            print(final_result.overall_summary[:500] + "..." if len(final_result.overall_summary) > 500 else final_result.overall_summary)
            
            if final_result.warnings:
                print("\n⚠️ AVISOS:")
                for warning in final_result.warnings[:3]:
                    print(f"   • {warning}")
        
        else:
            print("❌ FALHA: Sistema não retornou resultado final")
        
        print()
        print("🔥 Teste de Streaming Concluído!")
        
    except Exception as e:
        print(f"💥 ERRO CRÍTICO: {str(e)}")
        print(f"📍 Traceback:")
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_streaming_system()) 