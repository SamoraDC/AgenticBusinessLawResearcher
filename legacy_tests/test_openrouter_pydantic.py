"""
Teste do OpenRouter com PydanticAI - Implementação oficial sem wrappers.
Baseado na documentação oficial do PydanticAI para OpenRouter.
"""

import asyncio
import os
from typing import List, Optional

from dotenv import load_dotenv
from pydantic import BaseModel, Field
from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai.providers.openrouter import OpenRouterProvider

# Carregar variáveis de ambiente
load_dotenv()


# ===============================
# MODELOS PYDANTIC
# ===============================

class AnaliseJuridica(BaseModel):
    """Análise jurídica estruturada."""
    
    resumo_principal: str = Field(description="Resumo principal da análise")
    fundamentos_legais: List[str] = Field(description="Principais fundamentos legais")
    jurisprudencia_relevante: List[str] = Field(description="Jurisprudência relevante")
    conclusao: str = Field(description="Conclusão da análise")
    nivel_confianca: float = Field(ge=0, le=1, description="Nível de confiança da análise")
    recomendacoes: List[str] = Field(description="Recomendações práticas")


# ===============================
# CONFIGURAÇÃO DOS MODELOS
# ===============================

def create_openrouter_model(model_name: str) -> OpenAIModel:
    """
    Cria um modelo OpenRouter seguindo a documentação oficial do PydanticAI.
    
    Documentação: https://ai.pydantic.dev/models/openai/#openrouter
    """
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise ValueError("OPENROUTER_API_KEY não encontrada nas variáveis de ambiente")
    
    # Usar OpenRouterProvider conforme documentação oficial
    return OpenAIModel(
        model_name,
        provider=OpenRouterProvider(api_key=api_key)
    )


# ===============================
# AGENTES ESPECIALIZADOS
# ===============================

# Agente com DeepSeek R1
deepseek_agent = Agent[None, AnaliseJuridica](
    model=create_openrouter_model('deepseek/deepseek-r1-0528:free'),
    output_type=AnaliseJuridica,
    system_prompt="""
    Você é um especialista em direito brasileiro com foco em análises jurídicas precisas.
    
    Sua tarefa é fornecer análises jurídicas estruturadas e fundamentadas, incluindo:
    - Resumo claro da questão jurídica
    - Fundamentos legais aplicáveis
    - Jurisprudência relevante quando disponível
    - Conclusão objetiva
    - Recomendações práticas
    
    Sempre base suas respostas em:
    - Constituição Federal
    - Códigos (Civil, Penal, Processo Civil, etc.)
    - Leis específicas
    - Jurisprudência consolidada dos tribunais superiores
    
    Seja preciso, objetivo e evite especulações sem fundamento legal.
    """
)

# Agente com Gemini 2.0 Flash
gemini_agent = Agent[None, AnaliseJuridica](
    model=create_openrouter_model('google/gemini-2.0-flash-exp:free'),
    output_type=AnaliseJuridica,
    system_prompt="""
    Você é um especialista em direito brasileiro com experiência em análises jurídicas detalhadas.
    
    Forneça análises jurídicas completas e bem estruturadas, considerando:
    - Aspectos doutrinários
    - Precedentes jurisprudenciais
    - Impactos práticos
    - Diferentes interpretações possíveis
    
    Mantenha sempre:
    - Rigor técnico-jurídico
    - Linguagem clara e acessível
    - Fundamentação sólida
    - Conclusões objetivas
    
    Base suas análises na legislação brasileira vigente e jurisprudência consolidada.
    """
)


# ===============================
# FUNÇÕES DE TESTE
# ===============================

async def test_deepseek_model():
    """Testa o modelo DeepSeek R1 via OpenRouter."""
    
    print("🧠 Testando DeepSeek R1 via OpenRouter...")
    print("-" * 60)
    
    consulta = """
    Um funcionário foi demitido por justa causa alegando insubordinação, 
    mas ele afirma que apenas questionou uma ordem que poderia ser considerada 
    assédio moral. Quais os direitos do trabalhador nesta situação e como 
    deve proceder segundo a CLT?
    """
    
    try:
        resultado = await deepseek_agent.run(consulta)
        analise = resultado.output
        
        print(f"📊 Análise DeepSeek R1 (Confiança: {analise.nivel_confianca:.1%})")
        print(f"💭 Resumo: {analise.resumo_principal}")
        print(f"\n📚 Fundamentos Legais:")
        for i, fundamento in enumerate(analise.fundamentos_legais, 1):
            print(f"  {i}. {fundamento}")
        
        print(f"\n⚖️ Jurisprudência:")
        for i, jurisprudencia in enumerate(analise.jurisprudencia_relevante, 1):
            print(f"  {i}. {jurisprudencia}")
        
        print(f"\n✅ Conclusão: {analise.conclusao}")
        
        print(f"\n💡 Recomendações:")
        for i, recomendacao in enumerate(analise.recomendacoes, 1):
            print(f"  {i}. {recomendacao}")
        
        print(f"\n📈 Tokens utilizados: {resultado.usage()}")
        
        return True
        
    except Exception as e:
        print(f"❌ Erro no DeepSeek: {e}")
        return False


async def test_gemini_model():
    """Testa o modelo Gemini 2.0 Flash via OpenRouter."""
    
    print("\n🤖 Testando Gemini 2.0 Flash via OpenRouter...")
    print("-" * 60)
    
    consulta = """
    Uma empresa deseja implementar trabalho remoto permanente para alguns 
    funcionários. Quais as obrigações legais da empresa em relação a 
    equipamentos, ergonomia, controle de jornada e segurança do trabalho 
    segundo a legislação brasileira?
    """
    
    try:
        resultado = await gemini_agent.run(consulta)
        analise = resultado.output
        
        print(f"📊 Análise Gemini 2.0 (Confiança: {analise.nivel_confianca:.1%})")
        print(f"💭 Resumo: {analise.resumo_principal}")
        print(f"\n📚 Fundamentos Legais:")
        for i, fundamento in enumerate(analise.fundamentos_legais, 1):
            print(f"  {i}. {fundamento}")
        
        print(f"\n⚖️ Jurisprudência:")
        for i, jurisprudencia in enumerate(analise.jurisprudencia_relevante, 1):
            print(f"  {i}. {jurisprudencia}")
        
        print(f"\n✅ Conclusão: {analise.conclusao}")
        
        print(f"\n💡 Recomendações:")
        for i, recomendacao in enumerate(analise.recomendacoes, 1):
            print(f"  {i}. {recomendacao}")
        
        print(f"\n📈 Tokens utilizados: {resultado.usage()}")
        
        return True
        
    except Exception as e:
        print(f"❌ Erro no Gemini: {e}")
        return False


async def test_comparative_analysis():
    """Executa análise comparativa entre os dois modelos."""
    
    print("\n🔄 Executando análise comparativa...")
    print("=" * 80)
    
    consulta_teste = """
    Um contrato de prestação de serviços contém cláusula de exclusividade, 
    mas o prestador argumenta que se caracteriza vínculo empregatício devido 
    à subordinação e habitualidade. Como o tribunal deve decidir esta questão?
    """
    
    print(f"🔍 Consulta de teste:")
    print(f"📝 {consulta_teste}")
    print("\n" + "=" * 80)
    
    # Executar em paralelo para comparar performance
    try:
        resultados = await asyncio.gather(
            deepseek_agent.run(consulta_teste),
            gemini_agent.run(consulta_teste),
            return_exceptions=True
        )
        
        deepseek_result, gemini_result = resultados
        
        print("\n📊 COMPARAÇÃO DE RESULTADOS:")
        print("-" * 50)
        
        if not isinstance(deepseek_result, Exception):
            print(f"🧠 DeepSeek - Confiança: {deepseek_result.output.nivel_confianca:.1%}")
            print(f"   Resumo: {deepseek_result.output.resumo_principal[:100]}...")
            print(f"   Tokens: {deepseek_result.usage()}")
        else:
            print(f"🧠 DeepSeek - ERRO: {deepseek_result}")
        
        if not isinstance(gemini_result, Exception):
            print(f"\n🤖 Gemini - Confiança: {gemini_result.output.nivel_confianca:.1%}")
            print(f"   Resumo: {gemini_result.output.resumo_principal[:100]}...")
            print(f"   Tokens: {gemini_result.usage()}")
        else:
            print(f"\n🤖 Gemini - ERRO: {gemini_result}")
        
        return True
        
    except Exception as e:
        print(f"❌ Erro na análise comparativa: {e}")
        return False


async def verify_openrouter_connection():
    """Verifica se a conexão com OpenRouter está funcionando."""
    
    print("🔗 Verificando conexão com OpenRouter...")
    print("-" * 50)
    
    # Teste simples com o DeepSeek
    simple_agent = Agent[None, str](
        model=create_openrouter_model('deepseek/deepseek-r1-0528:free'),
        system_prompt="Responda de forma breve e direta."
    )
    
    try:
        resultado = await simple_agent.run("Teste de conexão: responda apenas 'OK'")
        print(f"✅ Conexão estabelecida: {resultado.output}")
        print(f"📊 Usage: {resultado.usage()}")
        return True
        
    except Exception as e:
        print(f"❌ Falha na conexão: {e}")
        return False


# ===============================
# FUNÇÃO PRINCIPAL
# ===============================

async def main():
    """Função principal que executa todos os testes."""
    
    print("🚀 TESTE DO OPENROUTER COM PYDANTIC AI")
    print("=" * 80)
    print("📖 Implementação baseada na documentação oficial:")
    print("   https://ai.pydantic.dev/models/openai/#openrouter")
    print("=" * 80)
    
    # Verificar se a API key está configurada
    if not os.getenv("OPENROUTER_API_KEY"):
        print("❌ ERRO: OPENROUTER_API_KEY não configurada!")
        print("💡 Configure a variável de ambiente OPENROUTER_API_KEY")
        print("   Obtenha sua chave em: https://openrouter.ai/keys")
        return
    
    print(f"✅ API Key configurada: {os.getenv('OPENROUTER_API_KEY')[:20]}...")
    
    # Executar testes
    tests_passed = 0
    total_tests = 4
    
    # 1. Verificar conexão
    if await verify_openrouter_connection():
        tests_passed += 1
    
    # 2. Testar DeepSeek
    if await test_deepseek_model():
        tests_passed += 1
    
    # 3. Testar Gemini
    if await test_gemini_model():
        tests_passed += 1
    
    # 4. Análise comparativa
    if await test_comparative_analysis():
        tests_passed += 1
    
    # Resultado final
    print("\n" + "=" * 80)
    print("📊 RESULTADO DOS TESTES")
    print("=" * 80)
    print(f"✅ Testes aprovados: {tests_passed}/{total_tests}")
    
    if tests_passed == total_tests:
        print("🎉 Todos os testes passaram! OpenRouter está funcionando perfeitamente com PydanticAI.")
        print("\n💡 PRÓXIMOS PASSOS:")
        print("  1. Integre esta configuração no seu sistema principal")
        print("  2. Substitua a configuração do modelo nos agentes existentes")
        print("  3. Teste com consultas mais complexas")
        print("  4. Monitore o uso de tokens e custos")
    else:
        print("⚠️  Alguns testes falharam. Verifique:")
        print("  - Conexão com internet")
        print("  - Validade da API key")
        print("  - Disponibilidade dos modelos no OpenRouter")


if __name__ == "__main__":
    asyncio.run(main())