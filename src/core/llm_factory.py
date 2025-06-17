import os
from langchain_groq import ChatGroq
from langchain_community.tools import TavilySearchResults
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv

# Adicionado: Importações para PydanticAI e LangChain OpenAI (para OpenRouter)
from pydantic_ai.providers.openai import OpenAIProvider
from pydantic_ai.models.openai import OpenAIModel # Usaremos a interface OpenAI para OpenRouter/Groq
# Removido import ChatOpenAI pois PydanticAI usa seu próprio provider

load_dotenv() # Carrega variáveis do .env

# --- Chaves API --- 
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# --- Nomes dos Modelos --- #
# OpenRouter Models (usando modelo gratuito funcionando baseado nos testes)
MODEL_TRANSFORMER = "meta-llama/llama-4-maverick:free"  # ✅ FUNCIONANDO - Único modelo gratuito testado
MODEL_GRADER = "meta-llama/llama-4-maverick:free"       # ✅ FUNCIONANDO - Único modelo gratuito testado
MODEL_SYNTHESIZER = "meta-llama/llama-4-maverick:free"  # ✅ FUNCIONANDO - Único modelo gratuito testado
MODEL_DECISION = "meta-llama/llama-4-maverick:free"     # ✅ FUNCIONANDO - Único modelo gratuito testado

# Groq Model (reservado para web) - FUNCIONANDO PERFEITAMENTE
MODEL_GROQ_WEB = "llama-3.3-70b-versatile"
# Google Gemini Embeddings (consistente com pdf_processor.py)
GEMINI_EMBEDDING_MODEL_NAME = "models/text-embedding-004"

# --- LLM Configuration (Groq - LangChain) ---
# Mantido para uso potencial futuro ou comparação
LLM_GROQ_LANGCHAIN = None
if GROQ_API_KEY:
    LLM_GROQ_LANGCHAIN = ChatGroq(
        model_name=MODEL_GROQ_WEB,
        temperature=0,
        api_key=GROQ_API_KEY
    )
    print(f"LLM LangChain Groq ({MODEL_GROQ_WEB}) inicializado.")
else:
    print("AVISO: GROQ_API_KEY não encontrada. LLM LangChain Groq não está disponível.")

# --- LLM Configuration (PydanticAI via OpenRouter) ---
OPENROUTER_API_BASE = "https://openrouter.ai/api/v1"

def get_pydantic_ai_llm(model_name: str) -> OpenAIModel:
    """Configura e retorna um modelo PydanticAI para usar com OpenRouter."""
    if not OPENROUTER_API_KEY:
        raise ValueError("OPENROUTER_API_KEY não encontrada no ambiente.")

    provider = OpenAIProvider(
        api_key=OPENROUTER_API_KEY,
        base_url=OPENROUTER_API_BASE
        # headers podem ser necessários para OpenRouter se não usar api_key direta
        # Ex: headers={"HTTP-Referer": "YOUR_SITE_URL", "X-Title": "YOUR_PROJECT_TITLE"}
    )

    llm_pydantic = OpenAIModel(
        model_name=model_name, # Usa o nome específico do modelo OpenRouter
        provider=provider
    )
    print(f"Configurando PydanticAI LLM com OpenRouter para modelo: {model_name}")
    return llm_pydantic

# --- LLM Configuration (PydanticAI via Groq) ---
GROQ_API_BASE = "https://api.groq.com/openai/v1" # URL base da API Groq compatível com OpenAI

def get_pydantic_ai_llm_groq(model_name: str) -> OpenAIModel:
    """Configura e retorna um modelo PydanticAI para usar com Groq."""
    if not GROQ_API_KEY:
        raise ValueError("GROQ_API_KEY não encontrada no ambiente.")

    provider = OpenAIProvider(
        api_key=GROQ_API_KEY,
        base_url=GROQ_API_BASE
    )

    llm_pydantic = OpenAIModel(
        model_name=model_name, # Usa o nome específico do modelo Groq
        provider=provider
    )
    print(f"Configurando PydanticAI LLM com Groq para modelo: {model_name}")
    return llm_pydantic

# --- Web Search Tool Configuration ---
WEB_SEARCH_TOOL = None
if TAVILY_API_KEY:
    WEB_SEARCH_TOOL = TavilySearchResults(max_results=3, api_key=TAVILY_API_KEY)
    print("Ferramenta Tavily Web Search inicializada.")
else:
    print("AVISO: TAVILY_API_KEY não encontrada. Busca web não funcionará.")

# --- Embeddings Configuration (Google Gemini) --- # Modificado Bloco

EMBEDDINGS = None
if GOOGLE_API_KEY:
    try:
        EMBEDDINGS = GoogleGenerativeAIEmbeddings(
            model=GEMINI_EMBEDDING_MODEL_NAME,
            google_api_key=GOOGLE_API_KEY
        )
        print(f"Embeddings Google Gemini ({GEMINI_EMBEDDING_MODEL_NAME}) inicializados.")
    except Exception as e:
        print(f"ERRO ao inicializar Google Embeddings: {e}")
        print("  >> Verifique sua GOOGLE_API_KEY e se o modelo '{GEMINI_EMBEDDING_MODEL_NAME}' é válido.")
else:
    print(f"AVISO: GOOGLE_API_KEY não encontrada. Embeddings Google Gemini não estão disponíveis.")
    print("       O ChromaDB ({'chroma_db_gemini'}) não poderá ser carregado/consultado corretamente sem eles.")


print("--- Configuração de LLMs e Ferramentas concluída ---")

# --- Deprecated / Old Config --- #
# ... (removido) 