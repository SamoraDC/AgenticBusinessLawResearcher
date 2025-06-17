#!/usr/bin/env python3
"""
Script de teste para validar o sistema jurídico real com PydanticAI + LangGraph
"""

import sys
import asyncio
import traceback
from datetime import datetime

# Adicionar src ao path
sys.path.append('src')

async def test_real_system():
    """Testa o sistema jurídico real passo a passo"""
    
    print("🔄 Iniciando teste do sistema real...")
    
    try:
        # 1. Testar importações básicas
        print("\n1️⃣ Testando importações básicas...")
        
        from src.models import LegalQuery, ProcessingConfig, Priority, ValidationLevel
        print("✅ Modelos importados com sucesso")
        
        from src.llm_config import get_pydantic_ai_llm, MODEL_TRANSFORMER
        print("✅ Configuração LLM importada com sucesso")
        
        # 2. Testar criação de query
        print("\n2️⃣ Testando criação de LegalQuery...")
        
        query = LegalQuery(
            text="Como remover um sócio minoritário de uma sociedade limitada?",
            priority=Priority.MEDIUM,
            validation_level=ValidationLevel.MODERATE
        )
        print(f"✅ LegalQuery criada: {query.id}")
        
        # 3. Testar LangGraph
        print("\n3️⃣ Testando construção do grafo LangGraph...")
        
        from src.graph_builder import build_graph
        app = build_graph()
        print("✅ Grafo LangGraph construído com sucesso")
        
        # 4. Testar estado inicial
        print("\n4️⃣ Preparando estado inicial...")
        
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
        print("✅ Estado inicial preparado")
        
        # 5. Testar execução do grafo (com timeout)
        print("\n5️⃣ Executando grafo LangGraph...")
        
        config = {
            "recursion_limit": 50,
            "configurable": {
                "thread_id": f"test-{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            }
        }
        
        # Executar com timeout
        final_state = None
        event_count = 0
        
        try:
            async for event in app.astream(initial_state, config=config):
                event_count += 1
                print(f"📝 Evento {event_count}: {list(event.keys())}")
                
                for node_name, state_update in event.items():
                    final_state = state_update
                    
                # Limitar eventos para teste
                if event_count >= 10:
                    print("⏰ Limitando execução para teste...")
                    break
                    
        except Exception as e:
            print(f"❌ Erro durante execução: {e}")
            traceback.print_exc()
            return False
        
        # 6. Verificar resultado
        print(f"\n6️⃣ Verificando resultado final...")
        
        if final_state:
            print(f"✅ Estado final obtido com {event_count} eventos")
            
            if final_state.get("final_response"):
                response = final_state["final_response"]
                if isinstance(response, dict):
                    summary = response.get("overall_summary", "")[:200]
                    print(f"✅ Resposta gerada: {summary}...")
                else:
                    print(f"✅ Resposta final: {str(response)[:200]}...")
            else:
                print("⚠️ Nenhuma resposta final gerada ainda")
                print(f"📊 Estado atual: {list(final_state.keys())}")
        
        else:
            print("❌ Nenhum estado final obtido")
            return False
        
        print("\n🎉 Teste do sistema real concluído com sucesso!")
        return True
        
    except Exception as e:
        print(f"\n❌ Erro durante teste: {e}")
        traceback.print_exc()
        return False

async def test_individual_components():
    """Testa componentes individuais"""
    
    print("\n🔧 Testando componentes individuais...")
    
    try:
        # Testar PydanticAI Agent
        print("\n🤖 Testando PydanticAI Agent...")
        
        from src.agents.transformer import transformer_agent
        if transformer_agent:
            print("✅ Transformer Agent inicializado")
        else:
            print("❌ Transformer Agent não inicializado")
            
        # Testar MCP Unificado
        print("\n🔗 Testando MCP Unificado...")
        
        from src.mcp.unified_mcp import unified_mcp
        print("✅ Unified MCP importado")
        
        # Testar retriever
        print("\n📚 Testando Retriever...")
        
        from src.agents.retriever import retriever_instance
        if retriever_instance:
            print("✅ Retriever inicializado")
        else:
            print("⚠️ Retriever não inicializado (normal se ChromaDB não estiver pronto)")
        
        return True
        
    except Exception as e:
        print(f"❌ Erro testando componentes: {e}")
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("🚀 Iniciando testes do sistema jurídico real...")
    
    # Teste dos componentes
    components_ok = asyncio.run(test_individual_components())
    
    if components_ok:
        # Teste do sistema completo
        system_ok = asyncio.run(test_real_system())
        
        if system_ok:
            print("\n✅ TODOS OS TESTES PASSARAM!")
            print("✅ Sistema jurídico real PydanticAI + LangGraph está funcionando!")
        else:
            print("\n❌ FALHA NOS TESTES DO SISTEMA")
    else:
        print("\n❌ FALHA NOS TESTES DOS COMPONENTES") 