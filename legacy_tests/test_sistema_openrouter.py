#!/usr/bin/env python3
"""
Teste do Sistema Completo com OpenRouter Funcionando
Verifica se OpenRouter é usado como prioridade no sistema
"""

import asyncio
import sys
import os

# Adicionar src ao path
sys.path.append("src")

async def test_synthesizer_integration():
    """Testa se o synthesizer está usando OpenRouter como prioridade"""
    
    print("🧪 === TESTE SISTEMA COMPLETO COM OPENROUTER ===")
    
    try:
        # Importar o módulo do synthesizer
        from src.agents.synthesizer import synthesizer_agent, create_robust_openrouter_agent, create_robust_groq_agent
        
        print("🔍 Verificando qual agent foi inicializado...")
        
        if synthesizer_agent is None:
            print("❌ Synthesizer agent não foi inicializado!")
            return False
            
        # Verificar se é OpenRouter ou Groq
        agent_model = synthesizer_agent.model
        model_info = "Desconhecido"
        provider_info = "Desconhecido"
        
        if hasattr(agent_model, 'model_name'):
            model_info = agent_model.model_name
            
        if hasattr(agent_model, 'provider'):
            provider = agent_model.provider
            if hasattr(provider, 'base_url'):
                base_url = provider.base_url
                if "openrouter.ai" in str(base_url):
                    provider_info = "OpenRouter"
                elif "groq.com" in str(base_url):
                    provider_info = "Groq (API)"
                else:
                    provider_info = f"Outro ({base_url})"
            elif "GroqModel" in str(type(agent_model)):
                provider_info = "Groq (nativo)"
            else:
                # Verificar pelo tipo do provider
                provider_type = str(type(provider))
                if "OpenAI" in provider_type:
                    # Se é OpenAIProvider, verificar se tem deepseek no modelo (indicativo de OpenRouter)
                    if "deepseek" in str(model_info).lower():
                        provider_info = "OpenRouter (via OpenAIProvider)"
                    else:
                        provider_info = "OpenAI"
                else:
                    provider_info = provider_type
        
        print(f"🎯 Modelo: {model_info}")
        print(f"🔧 Provider: {provider_info}")
        
        # Teste prático - criar query simples
        from src.models import LegalQuery
        from src.graph_state import AgentState
        
        test_query = LegalQuery(
            text="O que é sociedade limitada?",
            id="test_openrouter_001"
        )
        
        # Simular estado mínimo
        test_state = AgentState(
            query=test_query,
            retrieved_docs=None,
            tavily_results=None,
            lexml_results=None
        )
        
        print("🚀 Testando síntese com query jurídica...")
        
        # Importar função de síntese
        from src.agents.synthesizer import synthesize_response
        
        result = await synthesize_response(test_state)
        
        if "final_response" in result:
            final_resp = result["final_response"]
            summary = final_resp.get("overall_summary", "")
            
            print(f"✅ Síntese funcionou! Tamanho: {len(summary)} chars")
            print(f"📝 Resposta: {summary[:150]}...")
            
            if provider_info == "OpenRouter":
                print("🎉 SUCESSO: Sistema está usando OpenRouter como prioridade!")
                return True
            elif provider_info == "OpenRouter (via OpenAIProvider)":
                print("🎉 SUCESSO: Sistema está usando OpenRouter via OpenAIProvider!")
                return True
            elif provider_info == "Groq (nativo)":
                print("⚠️ Sistema está usando Groq. OpenRouter não inicializou.")
                return False
            else:
                print(f"⚠️ Sistema está usando provider desconhecido: {provider_info}")
                return False
        else:
            print("❌ Síntese falhou")
            return False
            
    except Exception as e:
        print(f"❌ Erro no teste: {type(e).__name__}: {str(e)}")
        import traceback
        print(f"Traceback: {traceback.format_exc()}")
        return False

async def test_comparison_agents():
    """Testa ambos os agents diretamente"""
    print("\n🔄 === TESTE DIRETO DOS AGENTS ===")
    
    try:
        from src.agents.synthesizer import create_robust_openrouter_agent, create_robust_groq_agent
        
        # Teste OpenRouter
        print("1️⃣ Testando criação do agent OpenRouter...")
        openrouter_agent = create_robust_openrouter_agent()
        
        # Teste Groq
        print("\n2️⃣ Testando criação do agent Groq...")
        groq_agent = create_robust_groq_agent()
        
        print(f"\n📊 Resultado:")
        print(f"OpenRouter Agent: {'✅ Criado' if openrouter_agent else '❌ Falhou'}")
        print(f"Groq Agent: {'✅ Criado' if groq_agent else '❌ Falhou'}")
        
        if openrouter_agent and groq_agent:
            print("⚡ Ambos os agents foram criados com sucesso!")
            return True
        elif openrouter_agent:
            print("✅ OpenRouter funcionando, Groq com problema")
            return True
        elif groq_agent:
            print("⚠️ Apenas Groq funcionando, OpenRouter com problema")
            return False
        else:
            print("❌ Ambos os agents falharam!")
            return False
            
    except Exception as e:
        print(f"❌ Erro: {str(e)}")
        return False

async def main():
    """Executa todos os testes"""
    print("🚀 INICIANDO TESTES DO SISTEMA OPENROUTER")
    
    # Teste 1: Agents diretos
    agents_ok = await test_comparison_agents()
    
    # Teste 2: Sistema integrado
    system_ok = await test_synthesizer_integration()
    
    print(f"\n🏁 === RESULTADO FINAL ===")
    print(f"Criação de Agents: {'✅' if agents_ok else '❌'}")
    print(f"Sistema Integrado: {'✅' if system_ok else '❌'}")
    
    if system_ok:
        print("🎉 SUCESSO TOTAL: OpenRouter funcionando no sistema!")
        print("🌟 O problema original foi resolvido com sucesso!")
    elif agents_ok:
        print("⚠️ SUCESSO PARCIAL: OpenRouter funciona, mas sistema está usando Groq")
        print("💡 Sistema tem fallback confiável")
    else:
        print("❌ PROBLEMA: Verificar configuração das API keys")

if __name__ == "__main__":
    asyncio.run(main()) 