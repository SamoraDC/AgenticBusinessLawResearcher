"""
Teste rápido de integração do synthesizer corrigido.
"""

import asyncio
import sys
import os

# Adicionar o diretório src ao path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

async def quick_test():
    """Teste rápido do sistema corrigido."""
    print("🔧 Teste rápido de integração...")
    
    try:
        # Testar imports do sistema híbrido corrigido
        from src.agents.streaming.hybrid_legal_processor import synthesize_with_openrouter_streaming
        from src.core.legal_models import LegalQuery, Priority, ValidationLevel
        print("✅ Imports OK")
        
        # Testar criação de objetos básicos
        query = LegalQuery(
            text="retirada de sócio minoritário",
            priority=Priority.MEDIUM,
            validation_level=ValidationLevel.MODERATE,
            use_hybrid_streaming=True
        )
        print("✅ Modelos OK")
        
        # Teste básico de funcionamento (sem execução real)
        print("✅ Sistema corrigido carregado com sucesso!")
        
        # Verificar se as funções problemáticas foram corrigidas
        print("🔍 Verificando correções...")
        
        # Ler o código corrigido e verificar se as palavras problemáticas foram removidas
        with open('src/agents/streaming/hybrid_legal_processor.py', 'r', encoding='utf-8') as f:
            content = f.read()
            
        # Verificar se os prompts problemáticos foram removidos
        problematic_phrases = [
            'CONCEITO PRINCIPAL',
            'FUNDAMENTAÇÃO LEGAL', 
            'APLICAÇÃO PRÁTICA',
            'EXATAMENTE 400-500 palavras',
            'OBRIGATÓRIO: Esta seção deve conter NO MÍNIMO'
        ]
        
        issues_found = []
        for phrase in problematic_phrases:
            if phrase in content:
                issues_found.append(phrase)
        
        if issues_found:
            print(f"⚠️ Ainda há frases problemáticas: {issues_found}")
            return False
        else:
            print("✅ Frases problemáticas removidas!")
        
        print("🎉 Sistema corrigido validado!")
        return True
        
    except Exception as e:
        print(f"❌ Erro: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = asyncio.run(quick_test())
    if success:
        print("✅ TESTE RÁPIDO: PASSOU")
    else:
        print("❌ TESTE RÁPIDO: FALHOU") 