#!/usr/bin/env python3
"""
Script de debug para investigar a perda de dados CRAG na transferência.
"""

import sys
sys.path.append('src')

import asyncio
from src.core.workflow_state import AgentState
from src.core.llm_factory import setup_llm_factory, MODEL_GROQ_WEB
from src.core.legal_models import LegalQuery
from src.core.workflow_builder import create_legal_workflow_system

async def debug_crag_data_flow():
    """Debug completo do fluxo de dados CRAG."""
    
    print("🔍 === DEBUG CRAG DATA FLOW ===")
    
    # Configurar sistema
    system = create_legal_workflow_system()
    
    # Query de teste (mesma do log)
    query_text = "o que É patenteável como invenção?"
    query = LegalQuery(text=query_text)
    
    print(f"📝 Query: {query_text}")
    
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
    
    print("🚀 Estado inicial criado")
    
    # Configuração
    config = {
        "recursion_limit": 50,
        "configurable": {
            "thread_id": "debug_session"
        }
    }
    
    # Executar até síntese
    final_state = None
    event_count = 0
    
    print("🔧 Executando grafo CRAG...")
    
    async for event in system["app"].astream(initial_state, config=config):
        event_count += 1
        
        for node_name, state_update in event.items():
            print(f"\n--- NODE: {node_name.upper()} ---")
            
            if state_update:
                # Debug: verificar dados no estado
                retrieved_docs = state_update.get("retrieved_docs", [])
                tavily_results = state_update.get("tavily_results", [])
                lexml_results = state_update.get("lexml_results", [])
                
                print(f"📚 Retrieved docs: {len(retrieved_docs) if retrieved_docs else 0}")
                print(f"🌐 Tavily results: {len(tavily_results) if isinstance(tavily_results, list) else (1 if tavily_results else 0)}")
                print(f"⚖️ LexML results: {len(lexml_results) if isinstance(lexml_results, list) else (1 if lexml_results else 0)}")
                
                # Debug: tipos de dados
                print(f"📚 Retrieved docs type: {type(retrieved_docs)}")
                print(f"🌐 Tavily results type: {type(tavily_results)}")
                print(f"⚖️ LexML results type: {type(lexml_results)}")
                
                # Debug: preview de conteúdo
                if retrieved_docs and len(retrieved_docs) > 0:
                    doc = retrieved_docs[0]
                    print(f"📄 First doc type: {type(doc)}")
                    if hasattr(doc, 'page_content'):
                        print(f"📄 First doc content preview: {doc.page_content[:100]}...")
                    elif isinstance(doc, dict):
                        print(f"📄 First doc keys: {list(doc.keys())}")
                    else:
                        print(f"📄 First doc: {str(doc)[:100]}...")
                
                final_state = state_update
            
            # Parar quando chegar na síntese
            if node_name == "synthesize" or "synthesize" in node_name.lower():
                print("🛑 Parando na síntese")
                break
        
        # Limitar eventos
        if event_count >= 15:
            print("🛑 Limite de eventos atingido")
            break
    
    print(f"\n🏁 === RESULTADO FINAL ===")
    
    if final_state:
        # Testar extração como no app.py
        retrieved_docs = final_state.get("retrieved_docs", [])
        tavily_results = final_state.get("tavily_results", [])
        lexml_results = final_state.get("lexml_results", [])
        
        print(f"📚 Final retrieved docs: {len(retrieved_docs) if retrieved_docs else 0}")
        print(f"🌐 Final tavily results: {len(tavily_results) if isinstance(tavily_results, list) else (1 if tavily_results else 0)}")
        print(f"⚖️ Final lexml results: {len(lexml_results) if isinstance(lexml_results, list) else (1 if lexml_results else 0)}")
        
        # Simular processamento como no app.py
        processed_docs = []
        if retrieved_docs and isinstance(retrieved_docs, list):
            print(f"🔄 Processando {len(retrieved_docs)} documentos...")
            for i, doc in enumerate(retrieved_docs):
                if hasattr(doc, 'page_content'):
                    processed_docs.append({
                        'content': doc.page_content,
                        'metadata': getattr(doc, 'metadata', {}),
                        'source': 'crag_vectordb',
                        'index': i,
                        'original_type': type(doc).__name__
                    })
                    print(f"✅ Doc {i+1} processado: {len(doc.page_content)} chars")
                elif isinstance(doc, dict):
                    processed_docs.append({**doc, 'source': 'crag_vectordb', 'index': i})
                    print(f"✅ Doc {i+1} processado: dict")
                elif isinstance(doc, str):
                    processed_docs.append({
                        'content': doc, 
                        'metadata': {}, 
                        'source': 'crag_vectordb', 
                        'index': i,
                        'original_type': 'string'
                    })
                    print(f"✅ Doc {i+1} processado: string")
        
        print(f"🎯 RESULTADO: {len(processed_docs)} documentos processados com sucesso")
        
        if len(processed_docs) == 0:
            print("❌ PROBLEMA: Nenhum documento foi processado!")
            print("🔍 Debug estado final completo:")
            for key, value in final_state.items():
                print(f"  {key}: {type(value)} = {str(value)[:200]}...")
        
    else:
        print("❌ ERRO: Nenhum estado final recebido")

if __name__ == "__main__":
    asyncio.run(debug_crag_data_flow()) 