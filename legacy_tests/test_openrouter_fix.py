#!/usr/bin/env python3
"""
Solução específica para problema PydanticAI + OpenRouter
Teste diferentes abordagens para fazer funcionar
"""

import os
import asyncio
import json
from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai.providers.openai import OpenAIProvider
from pydantic import BaseModel, Field, ValidationError

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
MODEL_SYNTHESIZER = "deepseek/deepseek-chat-v3-0324:free"

class FlexibleResponse(BaseModel):
    content: str = Field(..., description="Resposta")

class LegalResponse(BaseModel):
    summary: str = Field(..., description="Resumo jurídico")

async def test_approach_1_minimal():
    """Abordagem 1: Configuração mínima"""
    print("\n🧪 === ABORDAGEM 1: CONFIGURAÇÃO MÍNIMA ===")
    
    try:
        provider = OpenAIProvider(
            api_key=OPENROUTER_API_KEY,
            base_url="https://openrouter.ai/api/v1"
        )
        
        model = OpenAIModel(model_name=MODEL_SYNTHESIZER, provider=provider)
        
        # Configuração ULTRA mínima
        agent = Agent(model, output_type=str)  # Sem instruções, sem settings
        
        result = await agent.run("O que é sociedade? Responda em 1 palavra.")
        print(f"✅ SUCESSO! Resultado: {result.output}")
        return True
        
    except Exception as e:
        print(f"❌ Falhou: {e}")
        return False

async def test_approach_2_simple_model():
    """Abordagem 2: Modelo simples com settings básicos"""
    print("\n🧪 === ABORDAGEM 2: MODELO SIMPLES ===")
    
    try:
        provider = OpenAIProvider(
            api_key=OPENROUTER_API_KEY,
            base_url="https://openrouter.ai/api/v1"
        )
        
        model = OpenAIModel(model_name=MODEL_SYNTHESIZER, provider=provider)
        
        agent = Agent(
            model,
            output_type=FlexibleResponse,
            model_settings={"temperature": 0.0, "max_tokens": 100}
        )
        
        result = await agent.run("Explique sociedade limitada em 1 frase.")
        print(f"✅ SUCESSO! Resultado: {result.output.content}")
        return True
        
    except Exception as e:
        print(f"❌ Falhou: {e}")
        return False

async def test_approach_3_different_model():
    """Abordagem 3: Testar modelo diferente do OpenRouter"""
    print("\n🧪 === ABORDAGEM 3: MODELO DIFERENTE ===")
    
    # Testar com modelo gratuito diferente
    alternative_model = "microsoft/phi-3-mini-128k-instruct:free"
    
    try:
        provider = OpenAIProvider(
            api_key=OPENROUTER_API_KEY,
            base_url="https://openrouter.ai/api/v1"
        )
        
        model = OpenAIModel(model_name=alternative_model, provider=provider)
        
        agent = Agent(
            model,
            output_type=FlexibleResponse,
            model_settings={"temperature": 0.0, "max_tokens": 100}
        )
        
        result = await agent.run("O que é uma empresa? Responda em 1 frase.")
        print(f"✅ SUCESSO com {alternative_model}! Resultado: {result.output.content}")
        return True
        
    except Exception as e:
        print(f"❌ Falhou com {alternative_model}: {e}")
        return False

async def test_approach_4_direct_http():
    """Abordagem 4: Wrapper customizado usando HTTP direto"""
    print("\n🧪 === ABORDAGEM 4: WRAPPER CUSTOMIZADO ===")
    
    try:
        import httpx
        
        async def openrouter_call(prompt: str) -> str:
            """Chamada direta para OpenRouter"""
            headers = {
                "Authorization": f"Bearer {OPENROUTER_API_KEY}",
                "Content-Type": "application/json"
            }
            
            data = {
                "model": MODEL_SYNTHESIZER,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": 500,
                "temperature": 0.0
            }
            
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    "https://openrouter.ai/api/v1/chat/completions",
                    headers=headers,
                    json=data,
                    timeout=30.0
                )
                
                if response.status_code == 200:
                    result = response.json()
                    return result['choices'][0]['message']['content']
                else:
                    raise Exception(f"HTTP {response.status_code}: {response.text}")
        
        # Testar chamada direta
        prompt = "Explique sociedade limitada no Brasil em 2 frases."
        response = await openrouter_call(prompt)
        
        # Validar com Pydantic separadamente
        legal_resp = LegalResponse(summary=response)
        
        print(f"✅ SUCESSO com wrapper! Resultado: {legal_resp.summary}")
        return True
        
    except Exception as e:
        print(f"❌ Falhou wrapper: {e}")
        return False

async def main():
    """Testa todas as abordagens"""
    print("🔧 === TESTE DE SOLUÇÕES PARA PYDANTICAI + OPENROUTER ===")
    
    approaches = [
        ("Configuração Mínima", test_approach_1_minimal),
        ("Modelo Simples", test_approach_2_simple_model),
        ("Modelo Diferente", test_approach_3_different_model),
        ("Wrapper Customizado", test_approach_4_direct_http)
    ]
    
    results = {}
    
    for name, test_func in approaches:
        print(f"\n{'='*60}")
        print(f"Testando: {name}")
        print('='*60)
        
        try:
            success = await test_func()
            results[name] = success
        except Exception as e:
            print(f"❌ Erro geral: {e}")
            results[name] = False
    
    # Resumo final
    print(f"\n🏁 === RESUMO DAS SOLUÇÕES ===")
    for name, success in results.items():
        status = "✅ FUNCIONOU" if success else "❌ FALHOU"
        print(f"{name}: {status}")
    
    working_solutions = [name for name, success in results.items() if success]
    
    if working_solutions:
        print(f"\n🎉 SOLUÇÕES QUE FUNCIONAM: {working_solutions}")
        print("💡 Use uma dessas abordagens no sistema principal!")
    else:
        print("\n❌ Nenhuma solução funcionou. Problema mais profundo no PydanticAI.")

if __name__ == "__main__":
    asyncio.run(main()) 