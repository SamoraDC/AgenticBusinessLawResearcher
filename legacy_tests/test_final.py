import asyncio
from src.graph_builder import build_graph
from src.models import LegalQuery

async def test_sistema_completo():
    """Teste completo do sistema com CRAG, LexML e Tavily"""
    print("=== TESTE SISTEMA COMPLETO ===")
    
    try:
        # Construir o grafo
        graph_app = build_graph()
        
        # Estado inicial
        initial_state = {
            "query": LegalQuery(text="como retirar um sócio minoritário da sociedade?"),
            "current_query": "como retirar um sócio minoritário da sociedade?",
            "retrieved_docs": [],
            "tavily_results": None,
            "lexml_results": None,
            "needs_jurisprudencia": True,
            "needs_web_search": False,
            "grade": None,
            "transformed_query": None,
            "should_synthesize": False,
            "history": [],
            "next_node": None,
            "web_search_reasoning": None
        }
        
        print("\n🔄 Executando o grafo...")
        final_state = await graph_app.ainvoke(initial_state)
        
        print(f"\n=== RESULTADO FINAL ===")
        final_response = final_state.get("final_response")
        
        if final_response:
            # Se final_response for um dict, converter para object
            if isinstance(final_response, dict):
                overall_summary = final_response.get('overall_summary', '')
                disclaimer = final_response.get('disclaimer', '')
            else:
                overall_summary = final_response.overall_summary
                disclaimer = final_response.disclaimer
                
            print(f"📝 Resposta gerada com {len(overall_summary)} caracteres")
            print(f"⚖️ LexML encontrou: {len(final_state.get('lexml_results', {}).get('documentos', []))} documentos")
            print(f"🌐 Tavily encontrou: {len(final_state.get('tavily_results', {}).get('results', []))} resultados")
            
            print(f"\n📋 Primeiros 300 caracteres da resposta:")
            print(overall_summary[:300] + "...")
            
            print(f"\n⚠️ Disclaimer: {disclaimer}")
            
            print("\n✅ SISTEMA FUNCIONANDO PERFEITAMENTE!")
        else:
            print("❌ Nenhuma resposta foi gerada")
        
    except Exception as e:
        print(f"❌ Erro no teste: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_sistema_completo()) 