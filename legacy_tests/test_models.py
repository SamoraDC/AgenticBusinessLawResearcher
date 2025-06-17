#!/usr/bin/env python3
"""
Script para testar todos os modelos LLM configurados no sistema
- Modelos OpenRouter (via langchain_openai)
- Modelo Groq (via langchain_groq)
"""

import os
import time
from datetime import datetime
from dotenv import load_dotenv

# Carrega variáveis de ambiente
load_dotenv()

# Imports para LangChain
from langchain_openai import ChatOpenAI
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage

# --- Configuração dos Modelos ---
# OpenRouter Models (atualizada conforme llm_config.py)
MODEL_TRANSFORMER = "deepseek/deepseek-chat-v3-0324:free"  # ✅ FUNCIONANDO - Excelente qualidade
MODEL_GRADER = "deepseek/deepseek-chat-v3-0324:free"       # 🔄 MUDANÇA: Era deepseek-r1-zero (resposta vazia)
MODEL_SYNTHESIZER = "meta-llama/llama-4-scout:free"        # ✅ FUNCIONANDO - Rápido e eficiente  
MODEL_DECISION = "deepseek/deepseek-chat-v3-0324:free"     # 🔄 MUDANÇA: Era gemini (rate limit)

# Groq Model 
MODEL_GROQ_WEB = "llama-3.3-70b-versatile"

# --- Mensagens de Teste ---
TEST_MESSAGES = {
    "transformer": "Transforme esta pergunta jurídica em uma consulta mais técnica: 'Como retirar sócio da empresa?'",
    "grader": "Avalie se este documento é relevante para 'retirada de sócio': 'Art. 1085 do Código Civil trata da exclusão de sócio por justa causa.' Responda apenas SIM ou NÃO.",
    "synthesizer": "Resuma em português: A exclusão de sócio pode ocorrer por deliberação da maioria quando há justa causa.",
    "decision": "Preciso buscar mais informações na web sobre este tópico jurídico? Responda apenas SIM ou NÃO: 'retirada de sócio minoritário'",
    "groq_web": "Explique brevemente o que é direito societário brasileiro."
}

def test_openrouter_model(model_name: str, test_name: str, message: str):
    """Testa um modelo do OpenRouter"""
    print(f"\n🔍 Testando {test_name.upper()} ({model_name})")
    print("-" * 60)
    
    try:
        # Configuração do modelo OpenRouter
        llm = ChatOpenAI(
            model=model_name,
            openai_api_base="https://openrouter.ai/api/v1",
            openai_api_key=os.getenv("OPENROUTER_API_KEY"),
            temperature=0.1,
            max_tokens=500
        )
        
        # Mede tempo de resposta
        start_time = time.time()
        
        # Envia mensagem
        response = llm.invoke([HumanMessage(content=message)])
        
        end_time = time.time()
        response_time = round(end_time - start_time, 2)
        
        # Resultados
        print(f"✅ SUCESSO - Tempo: {response_time}s")
        print(f"📤 Pergunta: {message}")
        print(f"📥 Resposta: {response.content}")
        print(f"📊 Tokens estimados: {len(response.content)}")
        
        return True, response_time, len(response.content)
        
    except Exception as e:
        print(f"❌ ERRO: {str(e)}")
        return False, 0, 0

def test_groq_model(model_name: str, message: str):
    """Testa o modelo do Groq"""
    print(f"\n🔍 Testando GROQ WEB ({model_name})")
    print("-" * 60)
    
    try:
        # Configuração do modelo Groq
        llm = ChatGroq(
            model=model_name,
            groq_api_key=os.getenv("GROQ_API_KEY"),
            temperature=0.1,
            max_tokens=500
        )
        
        # Mede tempo de resposta
        start_time = time.time()
        
        # Envia mensagem
        response = llm.invoke([HumanMessage(content=message)])
        
        end_time = time.time()
        response_time = round(end_time - start_time, 2)
        
        # Resultados
        print(f"✅ SUCESSO - Tempo: {response_time}s")
        print(f"📤 Pergunta: {message}")
        print(f"📥 Resposta: {response.content}")
        print(f"📊 Tokens estimados: {len(response.content)}")
        
        return True, response_time, len(response.content)
        
    except Exception as e:
        print(f"❌ ERRO: {str(e)}")
        return False, 0, 0

def main():
    """Função principal para testar todos os modelos"""
    print("🚀 INICIANDO TESTE DE MODELOS LLM")
    print("=" * 70)
    print(f"⏰ Data/Hora: {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}")
    print()
    
    # Verifica variáveis de ambiente
    print("🔑 Verificando Chaves de API...")
    openrouter_key = os.getenv("OPENROUTER_API_KEY")
    groq_key = os.getenv("GROQ_API_KEY")
    
    if not openrouter_key:
        print("❌ OPENROUTER_API_KEY não encontrada!")
        return
    else:
        print(f"✅ OPENROUTER_API_KEY: ...{openrouter_key[-10:]}")
    
    if not groq_key:
        print("❌ GROQ_API_KEY não encontrada!")
        return
    else:
        print(f"✅ GROQ_API_KEY: ...{groq_key[-10:]}")
    
    # Resultados dos testes
    results = {}
    
    # Testa modelos OpenRouter
    openrouter_models = [
        (MODEL_TRANSFORMER, "transformer", TEST_MESSAGES["transformer"]),
        (MODEL_GRADER, "grader", TEST_MESSAGES["grader"]),
        (MODEL_SYNTHESIZER, "synthesizer", TEST_MESSAGES["synthesizer"]),
        (MODEL_DECISION, "decision", TEST_MESSAGES["decision"])
    ]
    
    for model, name, message in openrouter_models:
        success, time_taken, tokens = test_openrouter_model(model, name, message)
        results[name] = {
            "model": model,
            "success": success,
            "time": time_taken,
            "tokens": tokens,
            "provider": "OpenRouter"
        }
    
    # Testa modelo Groq
    success, time_taken, tokens = test_groq_model(MODEL_GROQ_WEB, TEST_MESSAGES["groq_web"])
    results["groq_web"] = {
        "model": MODEL_GROQ_WEB,
        "success": success,
        "time": time_taken,
        "tokens": tokens,
        "provider": "Groq"
    }
    
    # Relatório Final
    print("\n" + "=" * 70)
    print("📊 RELATÓRIO FINAL DOS TESTES")
    print("=" * 70)
    
    successful_tests = 0
    total_tests = len(results)
    
    for test_name, result in results.items():
        status = "✅ FUNCIONANDO" if result["success"] else "❌ FALHOU"
        provider = result["provider"]
        model = result["model"]
        time_str = f"{result['time']}s" if result["success"] else "N/A"
        tokens_str = str(result["tokens"]) if result["success"] else "N/A"
        
        print(f"{status} | {test_name.upper():<12} | {provider:<10} | {time_str:<8} | {tokens_str:<6} tokens")
        print(f"           Modelo: {model}")
        print()
        
        if result["success"]:
            successful_tests += 1
    
    # Resumo
    success_rate = (successful_tests / total_tests) * 100
    print("-" * 70)
    print(f"📈 TAXA DE SUCESSO: {successful_tests}/{total_tests} ({success_rate:.1f}%)")
    
    if success_rate == 100:
        print("🎉 TODOS OS MODELOS ESTÃO FUNCIONANDO PERFEITAMENTE!")
    elif success_rate >= 80:
        print("⚠️  A maioria dos modelos está funcionando. Verifique os que falharam.")
    else:
        print("🚨 VÁRIOS MODELOS COM PROBLEMAS. Verifique configurações e chaves de API.")
    
    print(f"⏰ Teste concluído em: {datetime.now().strftime('%H:%M:%S')}")

if __name__ == "__main__":
    main() 