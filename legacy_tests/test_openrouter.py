#!/usr/bin/env python3
"""
Script de teste específico para OpenRouter com PydanticAI
Testa a nova configuração corrigida do OpenRouter
"""

import os
import asyncio
import time
from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai.providers.openai import OpenAIProvider
from pydantic import BaseModel, Field

# Configurações
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
MODEL_SYNTHESIZER = "deepseek/deepseek-chat-v3-0324:free"

class TestResponse(BaseModel):
    content: str = Field(..., description="Resposta de teste")

async def test_openrouter():
    """Testa OpenRouter com configuração corrigida"""
    
    print("🧪 === TESTE OPENROUTER CORRIGIDO ===")
    print(f"🔑 API Key disponível: {bool(OPENROUTER_API_KEY)}")
    
    if not OPENROUTER_API_KEY:
        print("❌ OPENROUTER_API_KEY não encontrada!")
        return False
    
    print(f"🔑 API Key: ***{OPENROUTER_API_KEY[-4:]}")
    print(f"🎯 Modelo: {MODEL_SYNTHESIZER}")
    
    try:
        # ✅ CONFIGURAÇÃO CORRIGIDA: OpenAIProvider com configuração manual
        print("🔧 Criando provider OpenRouter (compatível OpenAI API)...")
        
        openrouter_provider = OpenAIProvider(
            api_key=OPENROUTER_API_KEY,
            base_url="https://openrouter.ai/api/v1"
        )
        
        openrouter_model = OpenAIModel(
            model_name=MODEL_SYNTHESIZER,
            provider=openrouter_provider
        )
        
        print("✅ Provider e modelo criados")
        
        # Criar agent
        test_agent = Agent(
            model=openrouter_model,
            output_type=TestResponse,
            instructions="Você é um assistente de teste. Responda de forma clara e simples.",
            model_settings={
                "temperature": 0.0,
                "max_tokens": 500,
                "top_p": 0.8
            },
            retries=1
        )
        
        print("✅ Agent criado")
        
        # Teste simples
        test_prompt = "Explique em uma frase o que é direito empresarial."
        
        print(f"🚀 Testando com prompt: '{test_prompt}'")
        start_time = time.time()
        
        result = await test_agent.run(test_prompt)
        
        elapsed_time = time.time() - start_time
        print(f"⏱️ Tempo de resposta: {elapsed_time:.2f}s")
        
        # ✅ CORREÇÃO: Usar .output ao invés de .data (deprecated)
        if hasattr(result, 'output') and result.output:
            response = result.output.content
            print(f"✅ SUCESSO! Resposta recebida: {len(response)} chars")
            print(f"📝 Conteúdo: {response[:100]}...")
            return True
        elif hasattr(result, 'data') and result.data:
            response = result.data.content
            print(f"✅ SUCESSO! Resposta recebida: {len(response)} chars")
            print(f"📝 Conteúdo: {response[:100]}...")
            return True
        else:
            print(f"❌ Resultado inválido: {type(result)}")
            return False
            
    except Exception as e:
        print(f"❌ ERRO: {type(e).__name__}: {str(e)}")
        import traceback
        print(f"Traceback: {traceback.format_exc()}")
        return False

async def test_comparison():
    """Compara OpenRouter vs Groq"""
    print("\n🔄 === COMPARAÇÃO OPENROUTER vs GROQ ===")
    
    # Teste OpenRouter
    print("1️⃣ Testando OpenRouter...")
    openrouter_success = await test_openrouter()
    
    # Teste Groq se disponível
    groq_success = False
    GROQ_API_KEY = os.getenv("GROQ_API_KEY")
    
    if GROQ_API_KEY:
        print("\n2️⃣ Testando Groq...")
        try:
            from pydantic_ai.models.groq import GroqModel
            
            groq_model = GroqModel("llama-3.3-70b-versatile")
            groq_agent = Agent(
                model=groq_model,
                output_type=TestResponse,
                instructions="Você é um assistente de teste. Responda de forma clara e simples.",
                model_settings={"temperature": 0.0, "max_tokens": 500},
                retries=1
            )
            
            start_time = time.time()
            result = await groq_agent.run("Explique em uma frase o que é direito empresarial.")
            elapsed_time = time.time() - start_time
            
            # ✅ CORREÇÃO: Usar .output ao invés de .data (deprecated)
            if hasattr(result, 'output') and result.output:
                print(f"✅ Groq funcionou! Tempo: {elapsed_time:.2f}s")
                print(f"📝 Conteúdo: {result.output.content[:100]}...")
                groq_success = True
            elif hasattr(result, 'data') and result.data:
                print(f"✅ Groq funcionou! Tempo: {elapsed_time:.2f}s")
                print(f"📝 Conteúdo: {result.data.content[:100]}...")
                groq_success = True
            else:
                print("❌ Groq falhou")
                
        except Exception as e:
            print(f"❌ Erro no Groq: {str(e)}")
    else:
        print("⚠️ GROQ_API_KEY não disponível para comparação")
    
    # Resultado final
    print(f"\n📊 === RESULTADO FINAL ===")
    print(f"OpenRouter: {'✅ FUNCIONOU' if openrouter_success else '❌ FALHOU'}")
    print(f"Groq: {'✅ FUNCIONOU' if groq_success else '❌ FALHOU'}")
    
    if openrouter_success:
        print("🎉 OpenRouter está funcionando com a nova configuração!")
        if groq_success:
            print("⚡ Ambos os providers funcionam! Sistema totalmente robusto!")
    else:
        print("⚠️ OpenRouter ainda tem problemas. Verificar logs acima.")
        if groq_success:
            print("✅ Pelo menos Groq funciona como fallback confiável.")

if __name__ == "__main__":
    asyncio.run(test_comparison()) 