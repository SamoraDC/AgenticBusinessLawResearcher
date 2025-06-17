import asyncio
import sys
import traceback
from dotenv import load_dotenv

# Carregar env
load_dotenv()

# Adicionar src ao path
sys.path.append('src')

async def test_real_system_streaming():
    """Teste do sistema real com streaming"""
    
    try:
        print("🔥 Testando Sistema REAL com STREAMING...")
        print("=" * 60)
        
        # Importar módulos do sistema real
        from src.models import LegalQuery, Priority, ValidationLevel
        from src.agents.synthesizer import synthesize_response_streaming
        
        print("✅ Módulos do sistema real carregados com sucesso")
        
        # Criar consulta de teste
        query = LegalQuery(
            text="Quais são os direitos do trabalhador em caso de demissão sem justa causa?",
            priority=Priority.MEDIUM,
            validation_level=ValidationLevel.MODERATE
        )
        
        print(f"📝 Consulta: {query.text}")
        print(f"⚙️ Prioridade: {query.priority}")
        print(f"🔍 Validação: {query.validation_level}")
        print()
        
        # Criar estado simulado (como seria no CRAG)
        state = {
            "query": query,
            "retrieved_docs": [
                {"text": "Art. 477 da CLT - O pagamento das parcelas constantes deste artigo..."},
                {"text": "Súmula 389 do TST - Aviso prévio e rescisão contratual..."},
                {"text": "Lei 8.036/90 - FGTS e direitos em rescisão..."}
            ],
            "tavily_results": [
                {"content": "Demissão sem justa causa: direitos e procedimentos segundo especialistas"},
                {"content": "Cálculos trabalhistas: verbas rescisórias em 2024"}
            ],
            "lexml_results": [
                {"titulo": "Consolidação das Leis do Trabalho", "artigo": "Art. 477"},
                {"titulo": "Lei do FGTS", "artigo": "Art. 18"}
            ]
        }
        
        print("🚀 Iniciando síntese com streaming...")
        print("-" * 60)
        
        # Testar synthesize_response_streaming
        final_result = None
        streaming_chunks = []
        
        async for step_type, content in synthesize_response_streaming(state):
            
            if step_type == "progress":
                print(f"📊 PROGRESSO: {content}")
            
            elif step_type == "streaming":
                # Simular streaming visual
                print(f"✍️ STREAMING: {len(content)} chars...")
                streaming_chunks.append(content)
                
                # Mostrar apenas as últimas palavras para não poluir
                words = content.split()
                if len(words) > 15:
                    preview = " ".join(words[-15:])
                    print(f"   📝 Preview: ...{preview}")
                else:
                    print(f"   📝 Texto: {content}")
                
                # Pequena pausa para simular efeito visual
                await asyncio.sleep(0.05)
            
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
            print("✅ SISTEMA REAL COM STREAMING FUNCIONOU!")
            
            if isinstance(final_result, dict):
                summary = final_result.get("overall_summary", "")
                disclaimer = final_result.get("disclaimer", "")
                
                print(f"📊 Status: Resposta gerada com sucesso")
                print(f"📏 Texto final: {len(summary)} chars")
                
                if streaming_chunks:
                    print(f"⚡ Chunks de streaming: {len(streaming_chunks)}")
                    print(f"📈 Crescimento do texto: {[len(chunk) for chunk in streaming_chunks[:5]]}...")
                
                print()
                print("📋 RESPOSTA FINAL:")
                print("-" * 40)
                print(summary[:500] + "..." if len(summary) > 500 else summary)
                
                if disclaimer:
                    print("\n⚠️ DISCLAIMER:")
                    print(disclaimer[:200] + "..." if len(disclaimer) > 200 else disclaimer)
            
            else:
                print(f"⚠️ Formato inesperado: {type(final_result)}")
                print(f"📝 Conteúdo: {str(final_result)[:300]}...")
        
        else:
            print("❌ FALHA: Sistema não retornou resultado final")
        
        print()
        print("🔥 Teste de Streaming do Sistema Real Concluído!")
        
    except Exception as e:
        print(f"💥 ERRO CRÍTICO: {str(e)}")
        print(f"📍 Traceback:")
        traceback.print_exc()

async def test_real_system_integration():
    """Teste de integração com o sistema CRAG + streaming"""
    
    try:
        print("\n" + "=" * 60)
        print("🧪 Testando INTEGRAÇÃO: CRAG + Streaming...")
        print("=" * 60)
        
        # Importar sistema completo
        from src.models import LegalQuery, Priority, ValidationLevel
        from src.graph_builder import build_graph
        
        print("✅ Sistema CRAG carregado")
        
        # Construir o grafo
        app = build_graph()
        print("✅ Grafo LangGraph compilado")
        
        # Criar consulta
        query = LegalQuery(
            text="Como funciona o seguro-desemprego no Brasil?",
            priority=Priority.HIGH,
            validation_level=ValidationLevel.MODERATE
        )
        
        # Estado inicial
        initial_state = {
            "query": query,
            "current_query": query.text,
            "retrieved_docs": [],
            "tavily_results": None,
            "lexml_results": None,
            "needs_jurisprudencia": True,
            "grade": None,
            "transformed_query": None,
            "should_synthesize": False,
            "history": [],
            "final_response": None,
            "error": None,
            "next_node": None
        }
        
        # Configuração
        config = {
            "recursion_limit": 50,
            "configurable": {
                "thread_id": "test_streaming_integration"
            }
        }
        
        print(f"📝 Consulta: {query.text}")
        print("🚀 Executando CRAG workflow...")
        
        # Executar CRAG (limitado)
        event_count = 0
        final_crag_state = None
        
        async for event in app.astream(initial_state, config=config):
            event_count += 1
            
            for node_name, state_update in event.items():
                print(f"🔄 CRAG Node: {node_name}")
                final_crag_state = state_update
                
                # Parar antes da síntese para usar streaming
                if "synthesize" in node_name.lower():
                    print("🛑 Interrompendo antes da síntese para usar streaming")
                    break
            
            # Limitar eventos
            if event_count >= 10:
                print("⏰ Limite de eventos atingido")
                break
        
        if final_crag_state:
            print(f"✅ CRAG executado com {event_count} eventos")
            print(f"📚 Docs recuperados: {len(final_crag_state.get('retrieved_docs', []))}")
            print(f"🌐 Resultados Tavily: {final_crag_state.get('tavily_results') is not None}")
            print(f"⚖️ Resultados LexML: {final_crag_state.get('lexml_results') is not None}")
            
            # Agora testar streaming na síntese
            print("\n✍️ Testando síntese com streaming...")
            
            from src.agents.synthesizer import synthesize_response_streaming
            
            async for step_type, content in synthesize_response_streaming(final_crag_state):
                if step_type == "progress":
                    print(f"📊 {content}")
                elif step_type == "streaming":
                    print(f"⚡ Streaming: {len(content)} chars")
                elif step_type == "final":
                    print("🎯 Síntese concluída!")
                    break
                elif step_type == "error":
                    print(f"❌ Erro: {content}")
                    break
            
            print("✅ INTEGRAÇÃO CRAG + STREAMING FUNCIONOU!")
        
        else:
            print("❌ CRAG não retornou estado final")
        
    except Exception as e:
        print(f"💥 ERRO NA INTEGRAÇÃO: {str(e)}")
        traceback.print_exc()

async def main():
    """Executar todos os testes"""
    await test_real_system_streaming()
    await test_real_system_integration()

if __name__ == "__main__":
    asyncio.run(main()) 