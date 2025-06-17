"""
Teste do OpenRouter com PydanticAI usando OpenAIProvider base.
Implementação compatível com versões anteriores do PydanticAI.
"""

import asyncio
import os
from typing import List, Optional

from dotenv import load_dotenv
from pydantic import BaseModel, Field
from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai.providers.openai import OpenAIProvider

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
    Cria um modelo OpenRouter usando OpenAIProvider base com base_url configurada.
    
    Método compatível com versões anteriores do PydanticAI.
    """
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise ValueError("OPENROUTER_API_KEY não encontrada nas variáveis de ambiente")
    
    # Usar OpenAIProvider com base_url do OpenRouter
    return OpenAIModel(
        model_name,
        provider=OpenAIProvider(
            base_url='https://openrouter.ai/api/v1',
            api_key=api_key
        )
    )


# ===============================
# AGENTES ESPECIALIZADOS
# ===============================

# Agente com DeepSeek R1 via OpenRouter
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

# Agente com Gemini 2.0 Flash via OpenRouter
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
        import traceback
        traceback.print_exc()
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
        import traceback
        traceback.print_exc()
        return False


async def test_simple_connection():
    """Teste simples de conexão."""
    
    print("🔗 Teste simples de conexão...")
    print("-" * 50)
    
    # Agente simples para teste de conectividade
    simple_agent = Agent[None, str](
        model=create_openrouter_model('deepseek/deepseek-r1-0528:free'),
        system_prompt="Responda de forma breve e direta."
    )
    
    try:
        resultado = await simple_agent.run("Responda apenas 'Sistema funcionando' se conseguir me entender.")
        print(f"✅ Resposta: {resultado.output}")
        print(f"📊 Usage: {resultado.usage()}")
        return True
        
    except Exception as e:
        print(f"❌ Falha na conexão: {e}")
        import traceback
        traceback.print_exc()
        return False


async def verify_model_integration():
    """Verifica integração dos modelos no sistema existente."""
    
    print("\n🔧 Verificando integração com sistema existente...")
    print("-" * 60)
    
    try:
        # Importar AgentDependencies existente
        from src.agents.pydantic_agents import AgentDependencies
        from src.models import ProcessingConfig
        
        print("✅ Classes do sistema existente importadas com sucesso")
        
        # Criar dependencies básicas
        config = ProcessingConfig()
        deps = AgentDependencies(
            config=config,
            session_id="test-session"
        )
        
        print("✅ Dependências configuradas")
        
        # Teste de integração usando modelo OpenRouter
        integration_agent = Agent[AgentDependencies, str](
            model=create_openrouter_model('deepseek/deepseek-r1-0528:free'),
            system_prompt="Você é um assistente jurídico especializado."
        )
        
        resultado = await integration_agent.run(
            "Teste de integração: cite apenas um artigo da Constituição Federal.",
            deps=deps
        )
        
        print(f"✅ Integração bem-sucedida: {resultado.output[:100]}...")
        return True
        
    except Exception as e:
        print(f"❌ Erro na integração: {e}")
        import traceback
        traceback.print_exc()
        return False


# ===============================
# FUNÇÃO PRINCIPAL
# ===============================

async def main():
    """Função principal que executa todos os testes."""
    
    print("🚀 TESTE DO OPENROUTER COM PYDANTIC AI (VERSÃO BASE)")
    print("=" * 80)
    print("📖 Usando OpenAIProvider base com configuração manual do OpenRouter")
    print("🔗 Base URL: https://openrouter.ai/api/v1")
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
    
    # 1. Teste simples de conexão
    if await test_simple_connection():
        tests_passed += 1
    
    # 2. Testar DeepSeek
    if await test_deepseek_model():
        tests_passed += 1
    
    # 3. Testar Gemini
    if await test_gemini_model():
        tests_passed += 1
    
    # 4. Verificar integração
    if await verify_model_integration():
        tests_passed += 1
    
    # Resultado final
    print("\n" + "=" * 80)
    print("📊 RESULTADO DOS TESTES")
    print("=" * 80)
    print(f"✅ Testes aprovados: {tests_passed}/{total_tests}")
    
    if tests_passed == total_tests:
        print("🎉 Todos os testes passaram! OpenRouter está funcionando perfeitamente com PydanticAI.")
        print("\n💡 PRÓXIMOS PASSOS:")
        print("  1. Atualize src/agents/pydantic_agents.py para usar esta configuração")
        print("  2. Substitua create_openrouter_model() nos agentes existentes")
        print("  3. Teste com o sistema completo")
        print("  4. Configure monitoramento de uso e custos")
        print("\n🔧 CÓDIGO PARA INTEGRAÇÃO:")
        print("```python")
        print("# Em src/agents/pydantic_agents.py, substitua OpenAIModel('gpt-4o') por:")
        print("model = OpenAIModel(")
        print("    'deepseek/deepseek-r1-0528:free',")
        print("    provider=OpenAIProvider(")
        print("        base_url='https://openrouter.ai/api/v1',")
        print("        api_key=os.getenv('OPENROUTER_API_KEY')")
        print("    )")
        print(")")
        print("```")
    else:
        print("⚠️  Alguns testes falharam. Verifique:")
        print("  - Conexão com internet")
        print("  - Validade da API key do OpenRouter")
        print("  - Disponibilidade dos modelos especificados")
        print("  - Configuração correta do base_url")


if __name__ == "__main__":
    asyncio.run(main())