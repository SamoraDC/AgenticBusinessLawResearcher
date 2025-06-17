#!/usr/bin/env python3
"""
Debug OpenRouter - Ver exatamente o que está sendo retornado
"""

import os
import asyncio
import json
from pydantic_ai import Agent, capture_run_messages
from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai.providers.openai import OpenAIProvider
from pydantic import BaseModel, Field

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
MODEL_SYNTHESIZER = "deepseek/deepseek-chat-v3-0324:free"

class SimpleResponse(BaseModel):
    content: str = Field(..., description="Qualquer resposta")

async def debug_openrouter():
    """Debug detalhado do OpenRouter"""
    
    print("🔍 === DEBUG OPENROUTER - VER O QUE ESTÁ SENDO RETORNADO ===")
    
    try:
        # Configuração OpenRouter
        provider = OpenAIProvider(
            api_key=OPENROUTER_API_KEY,
            base_url="https://openrouter.ai/api/v1"
        )
        
        model = OpenAIModel(model_name=MODEL_SYNTHESIZER, provider=provider)
        
        # Agent simples SEM validação restritiva
        agent = Agent(
            model=model,
            output_type=SimpleResponse,
            instructions="Responda de forma simples e direta.",
            model_settings={
                "temperature": 0.0,
                "max_tokens": 500
            },
            retries=0  # SEM retries para ver o erro bruto
        )
        
        prompt = "O que é sociedade limitada? Responda em 2 frases."
        
        print(f"📤 Enviando: {prompt}")
        
        # Capturar todas as mensagens para debug
        with capture_run_messages() as messages:
            try:
                result = await agent.run(prompt)
                
                print("✅ SUCESSO!")
                print(f"📥 Resposta: {result.output.content}")
                
                return True
                
            except Exception as e:
                print(f"❌ ERRO: {e}")
                
                # Analisar mensagens capturadas
                print(f"\n📚 Mensagens capturadas ({len(messages)}):")
                for i, msg in enumerate(messages):
                    print(f"\n{i+1}. {type(msg).__name__}")
                    
                    # Se for ModelResponse, ver o conteúdo
                    if hasattr(msg, 'content'):
                        print(f"   Content: {repr(msg.content)}")
                    if hasattr(msg, 'text'):
                        print(f"   Text: {repr(msg.text)}")
                    if hasattr(msg, 'response'):
                        print(f"   Response: {msg.response}")
                
                return False
                
    except Exception as e:
        print(f"❌ Erro na configuração: {e}")
        return False

async def test_raw_openrouter():
    """Teste RAW do OpenRouter sem PydanticAI"""
    print("\n🔧 === TESTE RAW OPENROUTER (sem PydanticAI) ===")
    
    try:
        import httpx
        
        headers = {
            "Authorization": f"Bearer {OPENROUTER_API_KEY}",
            "Content-Type": "application/json"
        }
        
        data = {
            "model": MODEL_SYNTHESIZER,
            "messages": [
                {"role": "user", "content": "O que é sociedade limitada? Responda em 1 frase."}
            ],
            "max_tokens": 200,
            "temperature": 0.0
        }
        
        async with httpx.AsyncClient() as client:
            response = await client.post(
                "https://openrouter.ai/api/v1/chat/completions",
                headers=headers,
                json=data,
                timeout=30.0
            )
            
            print(f"Status: {response.status_code}")
            
            if response.status_code == 200:
                result = response.json()
                content = result['choices'][0]['message']['content']
                print(f"✅ RAW Response: {repr(content)}")
                
                # Verificar caracteres problemáticos
                problematic_chars = []
                for i, char in enumerate(content):
                    if ord(char) < 32 and char not in ['\n', '\t']:
                        problematic_chars.append((i, char, ord(char)))
                
                if problematic_chars:
                    print(f"⚠️ Caracteres problemáticos encontrados: {problematic_chars}")
                else:
                    print("✅ Nenhum caractere problemático encontrado")
                
                return True
            else:
                print(f"❌ Erro HTTP: {response.text}")
                return False
                
    except Exception as e:
        print(f"❌ Erro no teste RAW: {e}")
        return False

async def main():
    print("🧪 DEBUG COMPLETO OPENROUTER")
    
    # Teste 1: PydanticAI com debug
    print("1️⃣ Testando com PydanticAI (debug)...")
    pydantic_ok = await debug_openrouter()
    
    # Teste 2: Raw HTTP
    print("\n2️⃣ Testando Raw HTTP...")
    raw_ok = await test_raw_openrouter()
    
    print(f"\n🏁 === DIAGNÓSTICO ===")
    print(f"PydanticAI: {'✅' if pydantic_ok else '❌'}")
    print(f"Raw HTTP: {'✅' if raw_ok else '❌'}")
    
    if raw_ok and not pydantic_ok:
        print("💡 OpenRouter funciona, mas PydanticAI tem problema de parsing/validação")
    elif pydantic_ok:
        print("🎉 Tudo funcionando!")
    else:
        print("❌ Problema na API do OpenRouter")

if __name__ == "__main__":
    asyncio.run(main()) 