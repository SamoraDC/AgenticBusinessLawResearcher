"""
Teste do sistema híbrido: OpenRouter (RAG/síntese) + Groq (tools estruturados).
Demonstra a funcionalidade completa do sistema jurídico.
"""

import asyncio
import os
from typing import List, Optional

from dotenv import load_dotenv
from pydantic import BaseModel, Field

# Carregar variáveis de ambiente
load_dotenv()


async def test_hybrid_workflow():
    """Testa o workflow completo do sistema híbrido."""
    
    print("🔄 TESTE DO SISTEMA HÍBRIDO OPENROUTER + GROQ")
    print("=" * 80)
    print("🤖 OpenRouter: RAG, análise e síntese")
    print("⚡ Groq: Decisões estruturadas e validação")
    print("=" * 80)
    
    # Verificar configurações necessárias
    required_keys = ["OPENROUTER_API_KEY", "GROQ_API_KEY"]
    missing_keys = [key for key in required_keys if not os.getenv(key)]
    
    if missing_keys:
        print(f"❌ ERRO: Variáveis de ambiente faltando: {', '.join(missing_keys)}")
        print("💡 Configure as chaves de API:")
        print("   - OPENROUTER_API_KEY: https://openrouter.ai/keys")
        print("   - GROQ_API_KEY: https://console.groq.com/keys")
        return False
    
    print("✅ Todas as API keys configuradas")
    
    try:
        # Importar dependências do sistema
        from src.agents.pydantic_agents_hybrid import (
            AgentDependencies,
            SearchDecision,
            search_decision_agent,
            vectordb_search_agent,
            lexml_search_agent,
            web_search_agent,
            legal_analyzer_agent,
            final_synthesizer_agent,
            quality_assessor_agent,
            guardrail_agent,
            run_parallel_searches,
            check_human_review_needed,
            validate_with_guardrails
        )
        from src.models import ProcessingConfig
        
        print("✅ Módulos do sistema importados com sucesso")
        
        # Configurar dependências
        config = ProcessingConfig()
        deps = AgentDependencies(
            config=config,
            session_id="test-hybrid-session"
        )
        
        # Consulta de teste
        consulta_juridica = """
        Uma empresa multinacional deseja implementar um programa de home office 
        permanente para 60% dos seus funcionários no Brasil. Quais são as 
        obrigações legais da empresa em relação ao controle de jornada, 
        fornecimento de equipamentos, ergonomia e segurança do trabalho? 
        Há jurisprudência recente do TST sobre essa questão?
        """
        
        print(f"\n🔍 CONSULTA DE TESTE:")
        print(f"📝 {consulta_juridica}")
        print("\n" + "=" * 80)
        
        # === ETAPA 1: DECISÃO DE BUSCA (GROQ) ===
        print("\n⚡ ETAPA 1: Decisão de busca com Groq...")
        
        decision_result = await search_decision_agent.run(
            f"Analise esta consulta jurídica e decida quais buscas realizar: {consulta_juridica}",
            deps=deps
        )
        
        decision: SearchDecision = decision_result.output
        
        print(f"📊 Decisão tomada:")
        print(f"   VectorDB: {'✓' if decision.needs_vectordb else '✗'}")
        print(f"   LexML: {'✓' if decision.needs_lexml else '✗'}")
        print(f"   Web: {'✓' if decision.needs_web else '✗'}")
        print(f"   Jurisprudência: {'✓' if decision.needs_jurisprudence else '✗'}")
        print(f"   Confiança: {decision.confidence:.1%}")
        print(f"   Justificativa: {decision.reasoning}")
        
        # === ETAPA 2: BUSCAS PARALELAS (OPENROUTER) ===
        print("\n🤖 ETAPA 2: Executando buscas paralelas com OpenRouter...")
        
        search_results = await run_parallel_searches(deps, consulta_juridica, decision)
        
        print(f"📊 Buscas concluídas: {len(search_results)} fontes")
        for source, result in search_results.items():
            print(f"   {source}: {len(result)} caracteres")
        
        # === ETAPA 3: ANÁLISE JURÍDICA (OPENROUTER) ===
        print("\n🤖 ETAPA 3: Análise jurídica com OpenRouter...")
        
        # Combinar resultados das buscas
        combined_results = "\n\n".join([
            f"=== {source} ===\n{result}" 
            for source, result in search_results.items()
        ])
        
        analysis_prompt = f"""
        Analise esta consulta jurídica: {consulta_juridica}
        
        Baseado nos seguintes resultados de busca:
        {combined_results[:2000]}...
        
        Forneça uma análise jurídica detalhada seguindo a estrutura solicitada.
        """
        
        analysis_result = await legal_analyzer_agent.run(analysis_prompt, deps=deps)
        analysis_text = analysis_result.output
        
        print(f"📊 Análise produzida: {len(analysis_text)} caracteres")
        print(f"📄 Prévia: {analysis_text[:200]}...")
        
        # === ETAPA 4: SÍNTESE FINAL (OPENROUTER) ===
        print("\n🤖 ETAPA 4: Síntese final com OpenRouter...")
        
        synthesis_prompt = f"""
        Sintetize uma resposta final para esta consulta: {consulta_juridica}
        
        Baseado nesta análise jurídica:
        {analysis_text[:1500]}...
        
        E nos resultados de busca de {len(search_results)} fontes.
        
        Produza uma resposta final clara e completa.
        """
        
        synthesis_result = await final_synthesizer_agent.run(synthesis_prompt, deps=deps)
        final_response = synthesis_result.output
        
        print(f"📊 Resposta final produzida: {len(final_response)} caracteres")
        
        # === ETAPA 5: AVALIAÇÃO DE QUALIDADE (GROQ) ===
        print("\n⚡ ETAPA 5: Avaliação de qualidade com Groq...")
        
        quality_result = await quality_assessor_agent.run(
            f"Avalie esta resposta jurídica: {final_response}",
            deps=deps
        )
        
        quality_assessment = quality_result.output
        
        print(f"📊 Avaliação de qualidade:")
        print(f"   Score geral: {quality_assessment.overall_score:.1%}")
        print(f"   Completude: {quality_assessment.completeness:.1%}")
        print(f"   Precisão: {quality_assessment.accuracy:.1%}")
        print(f"   Clareza: {quality_assessment.clarity:.1%}")
        print(f"   Revisão humana: {'✓' if quality_assessment.needs_human_review else '✗'}")
        
        # === ETAPA 6: VERIFICAÇÃO DE GUARDRAILS (GROQ) ===
        print("\n⚡ ETAPA 6: Verificação de guardrails com Groq...")
        
        guardrail_result = await guardrail_agent.run(
            f"Verifique esta resposta jurídica: {final_response}",
            deps=deps
        )
        
        guardrail_check = guardrail_result.output
        
        print(f"📊 Verificação de guardrails:")
        print(f"   Passou: {'✓' if guardrail_check.passed else '✗'}")
        print(f"   Violações: {len(guardrail_check.violations)}")
        print(f"   Nível de risco: {guardrail_check.overall_risk_level}")
        
        # === RESULTADO FINAL ===
        print("\n" + "=" * 80)
        print("📋 RESPOSTA JURÍDICA FINAL")
        print("=" * 80)
        print(final_response)
        
        if quality_assessment.needs_human_review:
            print(f"\n⚠️  REVISÃO HUMANA RECOMENDADA:")
            print(f"   Motivo: {quality_assessment.review_reason}")
        
        if not guardrail_check.passed:
            print(f"\n🚨 VIOLAÇÕES DE GUARDRAILS:")
            for violation in guardrail_check.violations:
                print(f"   - {violation.violation_type}: {violation.description}")
        
        print("\n" + "=" * 80)
        print("📊 MÉTRICAS DO TESTE")
        print("=" * 80)
        print(f"✅ Workflow híbrido executado com sucesso")
        print(f"🤖 OpenRouter: Buscas, análise e síntese")
        print(f"⚡ Groq: Decisões estruturadas e validação")
        print(f"📈 Qualidade geral: {quality_assessment.overall_score:.1%}")
        print(f"🔒 Guardrails: {'✓ Aprovado' if guardrail_check.passed else '✗ Violações detectadas'}")
        
        return True
        
    except Exception as e:
        print(f"❌ Erro no teste híbrido: {e}")
        import traceback
        traceback.print_exc()
        return False


async def test_individual_components():
    """Testa componentes individuais do sistema híbrido."""
    
    print("\n🧩 TESTE DE COMPONENTES INDIVIDUAIS")
    print("=" * 50)
    
    try:
        from src.agents.pydantic_agents_hybrid import (
            search_decision_agent,
            quality_assessor_agent,
            AgentDependencies
        )
        from src.models import ProcessingConfig
        
        # Configurar dependências básicas
        config = ProcessingConfig()
        deps = AgentDependencies(config=config, session_id="test-components")
        
        # Teste 1: Decisão de busca (Groq)
        print("\n⚡ Testando decisão de busca (Groq)...")
        decision_result = await search_decision_agent.run(
            "Preciso saber sobre direitos trabalhistas em home office",
            deps=deps
        )
        print(f"✅ Decisão estruturada gerada: {decision_result.output.confidence:.1%} confiança")
        
        # Teste 2: Avaliação de qualidade (Groq)
        print("\n⚡ Testando avaliação de qualidade (Groq)...")
        quality_result = await quality_assessor_agent.run(
            "Esta é uma resposta jurídica de teste sobre direitos trabalhistas.",
            deps=deps
        )
        print(f"✅ Avaliação estruturada gerada: {quality_result.output.overall_score:.1%} qualidade")
        
        return True
        
    except Exception as e:
        print(f"❌ Erro nos componentes: {e}")
        return False


async def main():
    """Função principal do teste."""
    
    # Executar testes
    component_test = await test_individual_components()
    workflow_test = await test_hybrid_workflow()
    
    # Resultado final
    print("\n" + "=" * 80)
    print("🏁 RESULTADO FINAL DOS TESTES")
    print("=" * 80)
    
    if component_test and workflow_test:
        print("🎉 TODOS OS TESTES PASSARAM!")
        print("\n💡 SISTEMA HÍBRIDO FUNCIONANDO PERFEITAMENTE:")
        print("  ✅ OpenRouter: Executa RAG, análise e síntese sem limitações de tools")
        print("  ✅ Groq: Manuseia saídas estruturadas e validação com tools")
        print("  ✅ Integração: Workflow completo executado com sucesso")
        print("\n📋 PRÓXIMOS PASSOS:")
        print("  1. Integrar esta configuração no sistema principal")
        print("  2. Atualizar src/pydantic_graph_builder.py para usar os novos agentes")
        print("  3. Testar com consultas jurídicas reais")
        print("  4. Configurar monitoramento de custos para ambas as APIs")
    else:
        print("⚠️  ALGUNS TESTES FALHARAM")
        print(f"  Componentes: {'✅' if component_test else '❌'}")
        print(f"  Workflow: {'✅' if workflow_test else '❌'}")


if __name__ == "__main__":
    asyncio.run(main()) 