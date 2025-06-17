import httpx
from typing import List, Dict, Any, Optional, Literal
from pydantic import BaseModel, Field
import xml.etree.ElementTree as ET
import asyncio
import os
from tavily import TavilyClient
from dotenv import load_dotenv

# Carrega variáveis de ambiente
load_dotenv()

# ==== MODELOS LEXML ====
class LexMLSearchRequest(BaseModel):
    termo: str
    tribunal: Optional[str] = None
    tipo_documento: Optional[str] = "jurisprudencia"
    max_results: int = 10
    start_record: int = 1

class LexMLDocumento(BaseModel):
    id: str
    urn: Optional[str] = None
    titulo: Optional[str] = None
    ementa: Optional[str] = None
    conteudo_html: Optional[str] = None
    data_publicacao: Optional[str] = None
    url_lexml: Optional[str] = None

class LexMLSearchResponse(BaseModel):
    documentos: List[LexMLDocumento]
    total_encontrado: int
    termo_pesquisa_cql: str
    start_record: int
    max_records: int
    query_original: Optional[str] = None

# ==== MODELOS TAVILY ====
class TavilySearchRequest(BaseModel):
    query: str
    max_results: int = 5
    search_depth: str = "basic"

class TavilySearchResult(BaseModel):
    url: str
    content: str
    score: float = 0.0
    title: Optional[str] = None

class TavilySearchResponse(BaseModel):
    results: List[TavilySearchResult]
    query: str

# ==== MODELO DE RESPOSTA UNIFICADA ====
class SearchNecessity(BaseModel):
    needs_jurisprudencia: bool = Field(description="Se é necessário buscar jurisprudência")
    needs_web_search: bool = Field(description="Se é necessário buscar na web")
    reasoning: str = Field(description="Justificativa para as necessidades identificadas")

class UnifiedMCP:
    """MCP unificado que combina funcionalidades do LexML e Tavily"""
    
    def __init__(self, tavily_api_key: Optional[str] = None):
        # Configuração LexML
        self.lexml_base_url = "https://www.lexml.gov.br/busca/SRU"
        self.lexml_client = httpx.AsyncClient(timeout=60.0)
        self.namespaces = {
            'srw': 'http://www.loc.gov/zing/srw/',
            'dc': 'http://purl.org/dc/elements/1.1/',
            'lexml': 'http://www.lexml.gov.br/srw/lexml-dc-schema.xsd',
            'srw_dc': 'info:srw/schema/1/dc-schema'
        }
        
        # Configuração Tavily
        self.tavily_api_key = tavily_api_key or os.getenv("TAVILY_API_KEY")
        self.tavily_client = None
        if self.tavily_api_key:
            self.tavily_client = TavilyClient(api_key=self.tavily_api_key)
        else:
            print("AVISO: TAVILY_API_KEY não encontrada. Busca web será desabilitada.")
    
    def _build_cql_query(self, termo: str, tipo_documento: Optional[str]) -> str:
        """Constrói uma string de consulta CQL simplificada e mais eficaz"""
        parts = []
        
        if termo:
            # Lista reduzida de stop words mais críticas
            stop_words = set([
                "a", "o", "os", "as", "da", "das", "do", "dos", "de", "e", "em", "para", "com", "por",
                "que", "não", "se", "um", "uma", "na", "no", "como", "mais", "sobre", "pelo", "pela",
                "quero", "gostaria", "posso", "poderia", "preciso", "seria", "fazer", "da", "de",
                "maneira", "forma", "correta", "isso"
            ])
            
            # Extrair palavras-chave importantes de forma mais simples
            palavras_do_termo = termo.lower().replace('?', '').replace('.', '').replace(',', '').split()
            palavras_chave = [
                palavra for palavra in palavras_do_termo
                if palavra not in stop_words and len(palavra) > 3  # Palavras com mais de 3 caracteres
            ]
            
            # Remover duplicatas mantendo ordem
            palavras_chave_unicas = list(dict.fromkeys(palavras_chave))
            
            if palavras_chave_unicas:
                # Estratégia mais simples: sempre usar as primeiras 2 palavras com AND
                if len(palavras_chave_unicas) >= 2:
                    principais = palavras_chave_unicas[:2]
                    termo_cql = f"{principais[0]} AND {principais[1]}"
                    parts.append(f"({termo_cql})")
                else:
                    # Para 1 palavra, usar diretamente
                    parts.append(palavras_chave_unicas[0])
        
        # Filtrar por tipo de documento após a busca, não na query CQL
        # (O campo tipoDocumento parece não funcionar corretamente na query CQL do LexML)
        
        # Se ainda não temos critérios, fazer busca mais geral
        if not parts:
            print("  ALERTA: Nenhum critério específico. Fazendo busca geral por jurisprudência.")
            return 'tipoDocumento exact "jurisprudencia"'

        final_query = " AND ".join(parts)
        print(f"  Query CQL construída: {final_query}")
        return final_query
    
    async def buscar_jurisprudencia(self, 
                                  termo: str,
                                  tipo_documento: Optional[str] = "jurisprudencia",
                                  max_results: int = 5,
                                  start_record: int = 1,
                                  query_original: Optional[str] = None) -> LexMLSearchResponse:
        """Busca jurisprudência no LexML com estratégias de fallback"""
        print(f"--- MCP LexML (SRU Tool): Buscando para termo '{termo}', tipo '{tipo_documento}' ---")
        
        # Primeira tentativa com query específica
        cql_query = self._build_cql_query(termo, tipo_documento)
        documentos, total_encontrado = await self._executar_busca_lexml(cql_query, max_results, start_record)
        
        # Se não encontrou nada, tentar busca mais ampla
        if total_encontrado == 0 and termo:
            print("  Primeira busca sem resultados. Tentando busca mais ampla...")
            
            # Primeiro fallback: usar apenas uma palavra-chave principal
            palavras_do_termo = termo.lower().replace('?', '').replace('.', '').replace(',', '').split()
            palavras_relevantes = [p for p in palavras_do_termo if len(p) > 4]  # Palavras maiores
            
            if palavras_relevantes:
                palavra_principal = palavras_relevantes[0]  # Primeira palavra relevante
                fallback_query = palavra_principal
                if tipo_documento:
                    fallback_query += f' AND tipoDocumento exact "{tipo_documento}"'
                
                print(f"  Tentando busca fallback com palavra-chave: {fallback_query}")
                documentos, total_encontrado = await self._executar_busca_lexml(fallback_query, max_results, start_record)
            
            # Segundo fallback: usar palavras-chave jurídicas comuns
            if total_encontrado == 0:
                palavras_importantes = ["sociedade", "sócio", "direito", "empresarial", "jurisprudencia", "tribunal"]
                termo_clean = termo.lower()
                palavras_encontradas = [p for p in palavras_importantes if p in termo_clean]
                
                if palavras_encontradas:
                    fallback_query = palavras_encontradas[0]  # Usar apenas uma palavra
                    if tipo_documento:
                        fallback_query += f' AND tipoDocumento exact "{tipo_documento}"'
                    
                    print(f"  Tentando busca fallback jurídica: {fallback_query}")
                    documentos, total_encontrado = await self._executar_busca_lexml(fallback_query, max_results, start_record)
        
        # Se ainda não encontrou, busca geral por tipo de documento
        if total_encontrado == 0 and tipo_documento:
            print("  Tentando busca apenas por tipo de documento...")
            fallback_query = f'tipoDocumento exact "{tipo_documento}"'
            documentos, total_encontrado = await self._executar_busca_lexml(fallback_query, max_results, start_record)
        
        # Último recurso: busca muito geral
        if total_encontrado == 0:
            print("  Tentando busca geral por 'direito'...")
            documentos, total_encontrado = await self._executar_busca_lexml("direito", max_results, start_record)
        
        return LexMLSearchResponse(
            documentos=documentos,
            total_encontrado=total_encontrado,
            termo_pesquisa_cql=cql_query,
            start_record=start_record,
            max_records=max_results,
            query_original=query_original
        )
    
    async def _executar_busca_lexml(self, cql_query: str, max_results: int, start_record: int) -> tuple[List[LexMLDocumento], int]:
        """Executa uma busca no LexML e retorna documentos e total encontrado"""
        params = {
            "operation": "searchRetrieve",
            "version": "1.1", 
            "query": cql_query,
            "startRecord": str(start_record),
            "maximumRecords": str(max_results),
        }

        print(f"  URL da API: {self.lexml_base_url}")
        print(f"  Parâmetros SRU: {params}")

        try:
            response = await self.lexml_client.get(self.lexml_base_url, params=params)
            response.raise_for_status()
            xml_content = response.text
            
            # Debug XML se necessário (comentado para produção)
            # print("---- RAW XML RESPONSE START ----")
            # print(xml_content[:1500] + "..." if len(xml_content) > 1500 else xml_content)
            # print("---- RAW XML RESPONSE END ----")
            print("  Resposta XML da API recebida.")

            root = ET.fromstring(xml_content)
            
            documentos = []
            total_encontrado = 0

            # Extrair o número total de resultados com múltiplas tentativas
            total_encontrado = 0
            for xpath in ['.//srw:numberOfRecords', './/numberOfRecords', './/*[local-name()="numberOfRecords"]']:
                number_element = root.find(xpath, self.namespaces)
                if number_element is not None and number_element.text:
                    total_encontrado = int(number_element.text)
                    break
            
            # Contar records manualmente se o número total não foi encontrado
            records_found = root.findall('.//srw:record', self.namespaces)
            if not records_found:
                records_found = root.findall('.//*[local-name()="record"]')  # Sem namespace
            
            for record_element in records_found:
                # Tentar múltiplos caminhos para encontrar os dados do record
                record_data = None
                for xpath in [
                    './/srw:recordData/srw_dc:dc',  # Formato padrão do LexML
                    './/srw:recordData/lexml:lexml/lexml:item',
                    './/srw:recordData/dc:dc', 
                    './/recordData/srw_dc/dc',
                    './/*[local-name()="recordData"]/*'
                ]:
                    record_data = record_element.find(xpath, self.namespaces)
                    if record_data is not None:
                        break
                
                if record_data is not None:
                    # Tentar múltiplas formas de encontrar URN/ID
                    urn_element = None
                    for urn_xpath in ['urn', 'dc:identifier', 'lexml:urn']:
                        urn_element = record_data.find(urn_xpath, self.namespaces)
                        if urn_element is not None:
                            break
                    
                    urn_text = urn_element.text if urn_element is not None else f"URN_DESCONHECIDA_{len(documentos)}"
                    
                    # Título
                    title_element = record_data.find('dc:title', self.namespaces)
                    titulo = title_element.text if title_element is not None else "Título não disponível"
                    
                    # Tipo de documento para filtro
                    tipo_doc_element = record_data.find('tipoDocumento')
                    tipo_doc = tipo_doc_element.text if tipo_doc_element is not None else None
                    
                    # Ementa/descrição (pode estar em dc:subject)
                    ementa_element = record_data.find('dc:description', self.namespaces)
                    if ementa_element is None:
                        ementa_element = record_data.find('dc:subject', self.namespaces)
                    
                    ementa = ementa_element.text if ementa_element is not None else None
                    
                    # Data de publicação
                    data_pub_element = record_data.find('dc:date', self.namespaces)
                    data_pub = data_pub_element.text if data_pub_element is not None else None

                    # URL do LexML
                    url_lexml_final = None
                    if urn_text and "URN_DESCONHECIDA" not in urn_text:
                        url_lexml_final = f"https://www.lexml.gov.br/urn/{urn_text}"

                    documento = LexMLDocumento(
                        id=urn_text,
                        urn=urn_text,
                        titulo=titulo,
                        ementa=ementa,
                        data_publicacao=data_pub,
                        url_lexml=url_lexml_final
                    )
                    
                    # Se necessário, adicionar todos ou filtrar por tipo depois
                    documentos.append(documento)
                    # print(f"  ✓ Documento extraído: {titulo[:50]}... (Tipo: {tipo_doc})")  # Debug

            print(f"  {len(documentos)} documentos processados desta página. Total geral (informado pela API): {total_encontrado}")
            return documentos, total_encontrado

        except Exception as e:
            print(f"  Erro na busca LexML: {e}")
            return [], 0
    
    async def buscar_web(self, request: TavilySearchRequest) -> TavilySearchResponse:
        """Busca informações na web usando Tavily"""
        if not self.tavily_client:
            raise ValueError("Tavily API Key não configurada")
        
        try:
            print(f"--- MCP Tavily: Buscando na web por '{request.query}' ---")
            
            # O cliente Tavily é síncrono, então usamos asyncio.to_thread
            results_raw = await asyncio.to_thread(
                self.tavily_client.search,
                query=request.query,
                max_results=request.max_results,
                search_depth=request.search_depth
            )

            formatted_results = [
                TavilySearchResult(
                    url=result.get("url", ""),
                    content=result.get("content", ""),
                    score=result.get("score", 0.0),
                    title=result.get("title")
                )
                for result in results_raw.get("results", [])
            ]

            print(f"  Tavily retornou {len(formatted_results)} resultados")
            
            return TavilySearchResponse(
                results=formatted_results,
                query=request.query
            )
        except Exception as e:
            print(f"  Erro na busca Tavily: {e}")
            raise ValueError(f"Erro na busca Tavily: {e}")
    
    async def close(self):
        """Fecha as conexões"""
        if self.lexml_client:
            await self.lexml_client.aclose()

# Instância global
unified_mcp = UnifiedMCP()

# Exemplo de uso
if __name__ == '__main__':
    async def main():
        # Teste LexML
        try:
            lexml_response = await unified_mcp.buscar_jurisprudencia(
                termo="direito empresarial",
                max_results=3
            )
            print(f"LexML encontrou {len(lexml_response.documentos)} documentos")
            for doc in lexml_response.documentos:
                print(f"  - {doc.titulo}")
        except Exception as e:
            print(f"Erro no teste LexML: {e}")
        
        # Teste Tavily
        try:
            if unified_mcp.tavily_client:
                tavily_request = TavilySearchRequest(query="direito empresarial Brasil")
                tavily_response = await unified_mcp.buscar_web(tavily_request)
                print(f"Tavily encontrou {len(tavily_response.results)} resultados")
                for result in tavily_response.results:
                    print(f"  - {result.title}: {result.url}")
        except Exception as e:
            print(f"Erro no teste Tavily: {e}")
        
        await unified_mcp.close()

    asyncio.run(main()) 