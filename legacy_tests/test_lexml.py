import asyncio
from src.mcp.unified_mcp import unified_mcp

async def test_lexml():
    """Teste específico do LexML com debugging"""
    print("=== TESTE LEXML COM DEBUGGING ===")
    
    try:
        print("\n1. Testando busca simples...")
        response = await unified_mcp.buscar_jurisprudencia(
            termo="sociedade",
            max_results=2
        )
        print(f"Resultado: {response.total_encontrado} documentos encontrados")
        
        if response.documentos:
            print("\nDocumentos encontrados:")
            for i, doc in enumerate(response.documentos):
                print(f"  {i+1}. {doc.titulo}")
        
    except Exception as e:
        print(f"Erro no teste: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        await unified_mcp.close()

if __name__ == "__main__":
    asyncio.run(test_lexml()) 