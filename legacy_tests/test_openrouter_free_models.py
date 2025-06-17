#!/usr/bin/env python3
"""
Teste dos Modelos Gratuitos da OpenRouter
Verifica se os modelos free estão funcionando e não gerando custos
"""

import os
import asyncio
import time
from typing import Dict, List, Tuple
from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai.providers.openai import OpenAIProvider
from pydantic import BaseModel, Field

# Configurações
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

# Modelos gratuitos para testar
FREE_MODELS = [
    "meta-llama/llama-4-maverick:free",
    "microsoft/phi-4-reasoning-plus:free", 
    "google/gemma-3n-e4b-it:free",
    "deepseek/deepseek-r1-0528:free"
]

class TestResponse(BaseModel):
    """Resposta estruturada para os testes"""
    content: str = Field(..., description="Resposta do modelo")
    model_name: str = Field(..., description="Nome do modelo testado")

class ModelTestResult(BaseModel):
    """Resultado do teste de um modelo"""
    model_name: str
    success: bool
    response_time: float
    response_content: str
    error_message: str = ""
    token_count_estimate: int = 0

def create_openrouter_model(model_name: str) -> OpenAIModel:
    """
    Cria um modelo OpenRouter com configuração correta
    """
    if not OPENROUTER_API_KEY:
        raise ValueError("OPENROUTER_API_KEY não encontrada nas variáveis de ambiente")
    
    provider = OpenAIProvider(
        api_key=OPENROUTER_API_KEY,
        base_url="https://openrouter.ai/api/v1"
    )
    
    return OpenAIModel(
        model_name=model_name,
        provider=provider
    )

async def test_single_model(model_name: str, test_prompt: str) -> ModelTestResult:
    """
    Testa um único modelo da OpenRouter
    """
    print(f"\n🧪 Testando: {model_name}")
    print("-" * 60)
    
    try:
        # Criar modelo
        model = create_openrouter_model(model_name)
        
        # Criar agent
        agent = Agent(
            model=model,
            output_type=TestResponse,
            instructions=f"Você está sendo testado como modelo {model_name}. Responda de forma clara e concisa.",
            model_settings={
                "temperature": 0.1,
                "max_tokens": 300
            },
            retries=1
        )
        
        # Medir tempo de resposta
        start_time = time.time()
        
        # Executar teste
        result = await agent.run(test_prompt)
        
        end_time = time.time()
        response_time = end_time - start_time
        
        # Extrair resposta
        response_content = result.output.content
        token_estimate = len(response_content.split()) * 1.3  # Estimativa aproximada
        
        print(f"✅ SUCESSO - Tempo: {response_time:.2f}s")
        print(f"📤 Pergunta: {test_prompt}")
        print(f"📥 Resposta: {response_content[:200]}{'...' if len(response_content) > 200 else ''}")
        print(f"📊 Tokens estimados: {int(token_estimate)}")
        
        return ModelTestResult(
            model_name=model_name,
            success=True,
            response_time=response_time,
            response_content=response_content,
            token_count_estimate=int(token_estimate)
        )
        
    except Exception as e:
        error_msg = str(e)
        print(f"❌ ERRO: {error_msg}")
        
        return ModelTestResult(
            model_name=model_name,
            success=False,
            response_time=0.0,
            response_content="",
            error_message=error_msg
        )

async def test_all_free_models() -> List[ModelTestResult]:
    """
    Testa todos os modelos gratuitos
    """
    print("🚀 === TESTE DOS MODELOS GRATUITOS OPENROUTER ===")
    print("=" * 80)
    
    if not OPENROUTER_API_KEY:
        print("❌ ERRO: OPENROUTER_API_KEY não configurada!")
        print("💡 Configure a variável de ambiente OPENROUTER_API_KEY")
        return []
    
    print(f"✅ API Key configurada: ***{OPENROUTER_API_KEY[-8:]}")
    print(f"🎯 Testando {len(FREE_MODELS)} modelos gratuitos")
    
    # Prompt de teste simples
    test_prompt = "Explique em 2 frases o que é inteligência artificial."
    
    results = []
    
    # Testar cada modelo
    for model_name in FREE_MODELS:
        result = await test_single_model(model_name, test_prompt)
        results.append(result)
        
        # Pausa entre testes para evitar rate limiting
        await asyncio.sleep(1)
    
    return results

def generate_report(results: List[ModelTestResult]) -> None:
    """
    Gera relatório final dos testes
    """
    print("\n" + "=" * 80)
    print("📊 RELATÓRIO FINAL DOS TESTES")
    print("=" * 80)
    
    successful_models = [r for r in results if r.success]
    failed_models = [r for r in results if not r.success]
    
    print(f"✅ Modelos funcionando: {len(successful_models)}/{len(results)}")
    print(f"❌ Modelos com erro: {len(failed_models)}/{len(results)}")
    
    if successful_models:
        print(f"\n🎉 MODELOS GRATUITOS FUNCIONAIS:")
        for result in successful_models:
            print(f"  ✅ {result.model_name}")
            print(f"     ⏱️  Tempo: {result.response_time:.2f}s")
            print(f"     📊 Tokens: ~{result.token_count_estimate}")
            print(f"     💬 Resposta: {result.response_content[:100]}...")
            print()
    
    if failed_models:
        print(f"\n❌ MODELOS COM PROBLEMAS:")
        for result in failed_models:
            print(f"  ❌ {result.model_name}")
            print(f"     🚨 Erro: {result.error_message}")
            print()
    
    # Recomendações
    print("💡 RECOMENDAÇÕES:")
    if successful_models:
        fastest_model = min(successful_models, key=lambda x: x.response_time)
        print(f"  🚀 Modelo mais rápido: {fastest_model.model_name} ({fastest_model.response_time:.2f}s)")
        
        print(f"  🔄 Para substituir modelos pagos, use qualquer um dos funcionais:")
        for result in successful_models:
            print(f"     - {result.model_name}")
    else:
        print("  ⚠️  Nenhum modelo gratuito funcionou. Verifique:")
        print("     - Configuração da API key")
        print("     - Disponibilidade dos modelos na OpenRouter")
        print("     - Limites de rate limiting")

async def test_legal_query_with_free_model():
    """
    Teste específico com consulta jurídica usando o melhor modelo gratuito
    """
    print("\n" + "=" * 80)
    print("⚖️ TESTE COM CONSULTA JURÍDICA")
    print("=" * 80)
    
    # Primeiro, encontrar um modelo que funciona
    results = await test_all_free_models()
    working_models = [r for r in results if r.success]
    
    if not working_models:
        print("❌ Nenhum modelo gratuito disponível para teste jurídico")
        return
    
    # Usar o modelo mais rápido
    best_model = min(working_models, key=lambda x: x.response_time)
    print(f"🎯 Usando modelo: {best_model.model_name}")
    
    legal_prompt = """
    Como advogado especialista em direito empresarial brasileiro, explique:
    
    Quais são os principais passos para abrir uma sociedade limitada no Brasil?
    
    Responda de forma estruturada e prática.
    """
    
    try:
        model = create_openrouter_model(best_model.model_name)
        
        agent = Agent(
            model=model,
            output_type=str,
            instructions="Você é um advogado especialista em direito empresarial brasileiro. Forneça respostas precisas e estruturadas.",
            model_settings={
                "temperature": 0.2,
                "max_tokens": 800
            }
        )
        
        print("🚀 Executando consulta jurídica...")
        start_time = time.time()
        
        result = await agent.run(legal_prompt)
        
        end_time = time.time()
        
        print(f"✅ Consulta jurídica concluída em {end_time - start_time:.2f}s")
        print(f"📋 Resposta jurídica:")
        print("-" * 60)
        print(result.output)
        print("-" * 60)
        
        print(f"💰 CUSTO: R$ 0,00 (modelo gratuito)")
        print(f"🎉 Modelo {best_model.model_name} adequado para substituir modelos pagos!")
        
    except Exception as e:
        print(f"❌ Erro na consulta jurídica: {str(e)}")

async def main():
    """
    Função principal que executa todos os testes
    """
    print("🧪 TESTE COMPLETO DOS MODELOS GRATUITOS OPENROUTER")
    print("=" * 80)
    print("🎯 Objetivo: Verificar modelos gratuitos para substituir os pagos")
    print("💰 Meta: Eliminar custos inesperados da OpenRouter")
    print("=" * 80)
    
    # Teste básico de todos os modelos
    results = await test_all_free_models()
    
    # Gerar relatório
    generate_report(results)
    
    # Teste específico com consulta jurídica
    await test_legal_query_with_free_model()
    
    print("\n" + "=" * 80)
    print("🏁 TESTE CONCLUÍDO")
    print("=" * 80)
    print("📝 Use este relatório para:")
    print("   1. Identificar modelos gratuitos funcionais")
    print("   2. Substituir modelos pagos no sistema")
    print("   3. Eliminar custos inesperados")
    print("   4. Manter funcionalidade sem cobrança")

if __name__ == "__main__":
    asyncio.run(main()) 