# 🏢⚖️ Business Law AI System / Sistema de IA para Direito Empresarial

[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![LangGraph](https://img.shields.io/badge/LangGraph-v0.4.1-green.svg)](https://github.com/langchain-ai/langgraph)
[![PydanticAI](https://img.shields.io/badge/PydanticAI-v0.1.8-purple.svg)](https://github.com/pydantic/pydantic-ai)
[![Streamlit](https://img.shields.io/badge/Streamlit-v1.32.0-red.svg)](https://streamlit.io/)

> **Advanced AI-powered legal consultation system specialized in Brazilian Business Law, built with state-of-the-art technologies for reliable, accurate, and comprehensive legal analysis.**

---

## 🌟 English Version

### 📖 Overview

The **Business Law AI System** is a comprehensive artificial intelligence platform specifically designed for Brazilian business law consultations. It combines multiple advanced technologies including **PydanticAI**, **LangGraph**, **ChromaDB**, and **LexML** using a **CRAG (Corrective Retrieval-Augmented Generation)** workflow with **hybrid AI processing** to provide accurate, well-founded, and reliable legal responses for business law matters.

### 🎯 Key Features

- **🏢 Business Law Specialization**: Focused on Brazilian corporate, commercial, and business law
- **🔍 CRAG Workflow**: Corrective Retrieval-Augmented Generation for enhanced accuracy
- **🤖 Hybrid AI Architecture**: OpenRouter + Groq integration for optimal performance
- **⚖️ Multi-Source Research**: ChromaDB vector database + LexML + web search
- **📊 Quality Assurance**: Built-in confidence scoring and guardrails validation
- **🌐 Bilingual Support**: Portuguese and English interfaces
- **📈 Real-time Processing**: Streaming responses with progress tracking
- **📊 Observability**: Complete monitoring with Langfuse integration

### 🏗️ Architecture

#### CRAG Workflow

![CRAG Graph](crag_graph.png)

The system implements a Corrective Retrieval-Augmented Generation (CRAG) workflow that intelligently manages document retrieval, evaluation, and query processing:

1. **Document Retrieval**: Searches the ChromaDB vector database
2. **Document Grading**: AI evaluates relevance and quality
3. **Query Transformation**: Optimizes queries when documents are insufficient
4. **LexML Search**: Accesses Brazilian legal jurisprudence
5. **Web Search Evaluation**: Determines if additional web search is needed
6. **Conditional Web Search**: Performs targeted web searches when necessary
7. **Response Synthesis**: Generates comprehensive legal responses

#### Complete Legal AI Workflow

![Legal AI Workflow](legal_ai_workflow.png)

The complete workflow shows the integration between CRAG processing and the hybrid AI system, including quality validation, guardrails, and observability components.

### 🧠 How It Works

1. **📝 Query Processing**: The system analyzes your business law question using the CRAG workflow
2. **🔍 Document Retrieval**: Searches ChromaDB vector database containing 60MB+ of business law documents
3. **⚖️ Document Grading**: AI evaluates document relevance and decides on additional searches
4. **🔄 Query Transformation**: Optimizes search queries when needed for better results
5. **📚 Multi-Source Search**:
   - **LexML**: Brazilian legal database for jurisprudence and legislation
   - **Web Search**: Current legal information and updates (when needed)
6. **🤖 Hybrid AI Analysis**:
   - **OpenRouter**: Legal analysis, synthesis, and quality validation
   - **Groq**: Structured processing and web/LexML searches
7. **📋 Response Generation**: Comprehensive legal response with confidence scoring

### 📚 Knowledge Base

The system includes specialized business law documentation:

- **Corporate Law** (Direito Societário) - Marlon Tomazette Vol. 1 (16MB)
- **Commercial Law** (Direito Empresarial) - André Luiz Santa Cruz Ramos (11MB)
- **Bankruptcy & Recovery** (Falência e Recuperação) - Marlon Tomazette Vol. 3 (14MB)
- **Credit Instruments** (Títulos de Crédito) - Marlon Tomazette Vol. 2 (11MB)
- **Business Law Fundamentals** - Mônica Gusmão (3.9MB)
- **Systematic Business Law** - Tarcísio Teixeira (3.6MB)

> **📁 Note on Legal Documents**: The original PDF documents are stored in the `data/` folder, which is included in `.gitignore` to prevent sharing copyrighted legal materials. Users should obtain their own copies of Brazilian business law documents for the knowledge base.

### 🚀 Quick Start

#### Prerequisites

- Python 3.12 or higher
- [uv](https://github.com/astral-sh/uv) package manager (recommended)
- API keys for external services (optional for basic functionality)
- Brazilian business law PDF documents for the knowledge base

#### Installation

```bash
# Clone the repository
git clone https://github.com/SamoraDC/AgenticBusinessLawResearcher.git
cd AgenticBusinessLawResearcher

# Install dependencies using uv (recommended)
uv sync

# Or using pip
pip install -r requirements.txt
```

#### Setting Up the Knowledge Base

1. **Create the data folder**:

   ```bash
   mkdir data
   ```
2. **Add your Brazilian business law PDF documents** to the `data/` folder:

   - Corporate law textbooks
   - Commercial law references
   - Jurisprudence collections
   - Legal doctrine materials
3. **Process the documents** to create the vector database:

   ```bash
   uv run python src/utils/document_processor.py
   ```

#### Environment Setup

Create a `.env` file in the root directory:

```env
# Required for enhanced functionality
GOOGLE_API_KEY=your_google_key_here
OPENROUTER_API_KEY=your_openrouter_key_here
GROQ_API_KEY=your_groq_key_here

# Optional for additional features
TAVILY_API_KEY=your_tavily_key_here
LANGFUSE_PUBLIC_KEY=your_langfuse_public_key
LANGFUSE_SECRET_KEY=your_langfuse_secret_key
```

#### Running the Application

```bash
# Using uv (recommended)
uv run streamlit run app.py

# Or using pip
streamlit run app.py
```

Visit `http://localhost:8501` in your browser.

### 💡 Usage Examples

#### Corporate Partnership Questions

```
"I want to remove a minority partner from the company. What are the legal procedures?"
```

#### Business Formation

```
"What are the requirements to establish a limited liability company (LTDA) in Brazil?"
```

#### Commercial Contracts

```
"What are the essential clauses for a commercial distribution agreement?"
```

#### Bankruptcy and Recovery

```
"What are the differences between judicial and extrajudicial business recovery?"
```

### 🛠️ Technologies Used

| Technology           | Purpose                              | Version  |
| -------------------- | ------------------------------------ | -------- |
| **PydanticAI** | AI agents with strong typing         | v0.1.8+  |
| **LangGraph**  | CRAG workflow orchestration          | v0.4.1+  |
| **ChromaDB**   | Vector database for semantic search  | v1.0.7+  |
| **Streamlit**  | Web interface                        | v1.32.0+ |
| **OpenRouter** | AI analysis and synthesis            | -        |
| **Groq**       | Structured processing                | -        |
| **LexML**      | Brazilian legal database integration | -        |
| **Tavily**     | Web search for current information   | v0.7.0+  |
| **Langfuse**   | Observability and monitoring         | v3.0.2+  |

### 📊 System Performance

- **Response Time**: 20-40 seconds for comprehensive analysis
- **Accuracy**: 85%+ confidence scoring on complex queries
- **Coverage**: 60MB+ of specialized business law content
- **Architecture**: CRAG + Hybrid AI for optimal quality
- **Languages**: Full bilingual support (Portuguese/English)

### 🔧 Configuration

The system can be configured through the `ProcessingConfig` model:

```python
config = ProcessingConfig(
    max_documents_per_source=20,        # Documents per search source
    search_timeout_seconds=60,          # Timeout for searches
    enable_parallel_search=True,        # Parallel processing
    min_confidence_threshold=0.4,       # Minimum confidence
    temperature=0.1,                    # AI creativity (low for legal)
    max_tokens=4000,                    # Response length
    enable_guardrails=True,             # Safety validations
    enable_web_search=True,             # Web search capability
    enable_jurisprudence_search=True    # LexML integration
)
```

### 🛡️ Legal Disclaimers and Quality Control

- **Professional Disclaimer**: All responses include clear disclaimers
- **Scope Limitation**: Focuses on general business law information
- **Quality Scoring**: Confidence scores from 0.0 to 1.0 on all responses
- **Guardrails**: Built-in checks for appropriate legal guidance
- **Source Attribution**: All responses cite their sources
- **CRAG Validation**: Corrective mechanisms ensure response accuracy

### 🧪 Development

#### Project Structure

```
├── src/                          # Core application code
│   ├── core/                     # Core models and configuration
│   │   ├── legal_models.py      # Pydantic models
│   │   ├── workflow_builder.py  # LangGraph CRAG workflows
│   │   └── observability.py     # Monitoring and logging
│   ├── agents/                   # AI agents
│   │   ├── document_retriever.py # ChromaDB interface
│   │   ├── document_grader.py   # Document relevance evaluation
│   │   ├── query_transformer.py # Query optimization
│   │   ├── search_coordinator.py # Multi-source search
│   │   └── streaming/           # Streaming processors
│   │       ├── hybrid_legal_processor.py # Hybrid AI system
│   │       └── response_synthesizer.py  # Response generation
│   └── interfaces/              # External integrations
│       └── external_search_client.py # LexML and Tavily
├── data/                        # Legal documents (60MB+) - .gitignore
├── chroma_db_gemini/           # Vector database
├── crag_graph.png              # CRAG workflow diagram
├── legal_ai_workflow.png       # Complete system workflow
├── app.py                      # Main Streamlit application
├── pyproject.toml             # Project configuration
└── README.md                  # This file
```

#### Running Tests

```bash
# Run specific tests
uv run python tests/test_system.py

# Test the complete integration
uv run python tests/test_complete_hybrid_integration.py
```

#### Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

### 📝 API Reference

#### LegalQuery Model

```python
query = LegalQuery(
    text="Your business law question",
    priority=Priority.MEDIUM,
    validation_level=ValidationLevel.MODERATE,
    area_of_law=LegalAreaType.COMMERCIAL
)
```

#### Response Structure

```python
response = FinalResponse(
    overall_summary="Comprehensive legal analysis...",
    overall_confidence=0.85,
    search_results=[...],
    detailed_analyses=[...],
    disclaimer="Professional legal disclaimer..."
)
```

### ⚖️ Legal Notice

This system provides general business law information and is not a substitute for professional legal advice. Always consult with qualified legal professionals for specific legal matters.

---

## 🌟 Versão em Português

### 📖 Visão Geral

O **Sistema de IA para Direito Empresarial** é uma plataforma abrangente de inteligência artificial especificamente projetada para consultas de direito empresarial brasileiro. Combina múltiplas tecnologias avançadas incluindo **PydanticAI**, **LangGraph**, **ChromaDB** e **LexML** usando um workflow **CRAG (Corrective Retrieval-Augmented Generation)** com **processamento híbrido de IA** para fornecer respostas jurídicas precisas, bem fundamentadas e confiáveis para questões de direito empresarial.

### 🎯 Características Principais

- **🏢 Especialização em Direito Empresarial**: Focado no direito societário, comercial e empresarial brasileiro
- **🔍 Workflow CRAG**: Corrective Retrieval-Augmented Generation para maior precisão
- **🤖 Arquitetura Híbrida de IA**: Integração OpenRouter + Groq para performance otimizada
- **⚖️ Pesquisa Multi-Fonte**: Banco vetorial ChromaDB + LexML + busca web
- **📊 Garantia de Qualidade**: Pontuação de confiança integrada e validação de guardrails
- **🌐 Suporte Bilíngue**: Interfaces em português e inglês
- **📈 Processamento em Tempo Real**: Respostas streaming com acompanhamento de progresso
- **📊 Observabilidade**: Monitoramento completo com integração Langfuse

### 🏗️ Arquitetura

#### Workflow CRAG

![Grafo CRAG](crag_graph.png)

O sistema implementa um workflow CRAG (Corrective Retrieval-Augmented Generation) que gerencia inteligentemente a recuperação, avaliação e processamento de documentos:

1. **Recuperação de Documentos**: Busca no banco vetorial ChromaDB
2. **Avaliação de Documentos**: IA avalia relevância e qualidade
3. **Transformação de Query**: Otimiza consultas quando documentos são insuficientes
4. **Busca LexML**: Acessa jurisprudência brasileira
5. **Avaliação de Busca Web**: Determina se busca web adicional é necessária
6. **Busca Web Condicional**: Realiza buscas web direcionadas quando necessário
7. **Síntese de Resposta**: Gera respostas jurídicas abrangentes

#### Workflow Completo de IA Jurídica

![Workflow de IA Jurídica](legal_ai_workflow.png)

O workflow completo mostra a integração entre o processamento CRAG e o sistema híbrido de IA, incluindo validação de qualidade, guardrails e componentes de observabilidade.

### 🧠 Como Funciona

1. **📝 Processamento da Consulta**: O sistema analisa sua pergunta de direito empresarial usando o workflow CRAG
2. **🔍 Recuperação de Documentos**: Busca no banco vetorial ChromaDB contendo mais de 60MB de documentos de direito empresarial
3. **⚖️ Avaliação de Documentos**: IA avalia relevância dos documentos e decide sobre buscas adicionais
4. **🔄 Transformação de Query**: Otimiza consultas de busca quando necessário para melhores resultados
5. **📚 Busca Multi-Fonte**:
   - **LexML**: Base de dados jurídica brasileira para jurisprudência e legislação
   - **Busca Web**: Informações jurídicas atuais e atualizações (quando necessário)
6. **🤖 Análise Híbrida de IA**:
   - **OpenRouter**: Análise jurídica, síntese e validação de qualidade
   - **Groq**: Processamento estruturado e buscas web/LexML
7. **📋 Geração de Resposta**: Resposta jurídica abrangente com pontuação de confiança

### 📚 Base de Conhecimento

O sistema inclui documentação especializada em direito empresarial:

- **Direito Societário** - Marlon Tomazette Vol. 1 (16MB)
- **Direito Empresarial Esquematizado** - André Luiz Santa Cruz Ramos (11MB)
- **Falência e Recuperação de Empresas** - Marlon Tomazette Vol. 3 (14MB)
- **Títulos de Crédito** - Marlon Tomazette Vol. 2 (11MB)
- **Lições de Direito Empresarial** - Mônica Gusmão (3.9MB)
- **Direito Empresarial Sistematizado** - Tarcísio Teixeira (3.6MB)

> **📁 Nota sobre Documentos Jurídicos**: Os PDFs originais são armazenados na pasta `data/`, que está incluída no `.gitignore` para evitar compartilhamento de materiais jurídicos com direitos autorais. Os usuários devem obter suas próprias cópias de documentos de direito empresarial brasileiro para a base de conhecimento.

### 🚀 Início Rápido

#### Pré-requisitos

- Python 3.12 ou superior
- Gerenciador de pacotes [uv](https://github.com/astral-sh/uv) (recomendado)
- Chaves de API para serviços externos (opcional para funcionalidade básica)
- PDFs de direito empresarial brasileiro para a base de conhecimento

#### Instalação

```bash
# Clone o repositório
git clone https://github.com/SamoraDC/AgenticBusinessLawResearcher.git
cd AgenticBusinessLawResearcher

# Instale as dependências usando uv (recomendado)
uv sync

# Ou usando pip
pip install -r requirements.txt
```

#### Configurando a Base de Conhecimento

1. **Crie a pasta data**:

   ```bash
   mkdir data
   ```
2. **Adicione seus PDFs de direito empresarial brasileiro** na pasta `data/`:

   - Livros de direito societário
   - Referências de direito comercial
   - Coleções de jurisprudência
   - Materiais de doutrina jurídica
3. **Processe os documentos** para criar o banco vetorial:

   ```bash
   uv run python src/utils/document_processor.py
   ```

#### Configuração do Ambiente

Crie um arquivo `.env` no diretório raiz:

```env
# Necessário para funcionalidade aprimorada
GOOGLE_API_KEY=sua_chave_google_aqui
OPENROUTER_API_KEY=sua_chave_openrouter_aqui
GROQ_API_KEY=sua_chave_groq_aqui

# Opcional para recursos adicionais
TAVILY_API_KEY=sua_chave_tavily_aqui
LANGFUSE_PUBLIC_KEY=sua_chave_publica_langfuse
LANGFUSE_SECRET_KEY=sua_chave_secreta_langfuse
```

#### Executando a Aplicação

```bash
# Usando uv (recomendado)
uv run streamlit run app.py

# Ou usando pip
streamlit run app.py
```

Visite `http://localhost:8501` no seu navegador.

### 💡 Exemplos de Uso

#### Questões de Sociedade Empresarial

```
"Quero retirar um sócio minoritário da sociedade. Quais são os procedimentos legais?"
```

#### Constituição de Empresas

```
"Quais são os requisitos para constituir uma sociedade limitada (LTDA) no Brasil?"
```

#### Contratos Comerciais

```
"Quais são as cláusulas essenciais para um contrato de distribuição comercial?"
```

#### Falência e Recuperação

```
"Quais são as diferenças entre recuperação judicial e extrajudicial de empresas?"
```

### 🛠️ Tecnologias Utilizadas

| Tecnologia           | Propósito                                 | Versão  |
| -------------------- | ------------------------------------------ | -------- |
| **PydanticAI** | Agentes de IA com tipagem forte            | v0.1.8+  |
| **LangGraph**  | Orquestração de workflow CRAG            | v0.4.1+  |
| **ChromaDB**   | Banco vetorial para busca semântica       | v1.0.7+  |
| **Streamlit**  | Interface web                              | v1.32.0+ |
| **OpenRouter** | Análise e síntese de IA                  | -        |
| **Groq**       | Processamento estruturado                  | -        |
| **LexML**      | Integração com base jurídica brasileira | -        |
| **Tavily**     | Busca web para informações atuais        | v0.7.0+  |
| **Langfuse**   | Observabilidade e monitoramento            | v3.0.2+  |

### 📊 Performance do Sistema

- **Tempo de Resposta**: 20-40 segundos para análise abrangente
- **Precisão**: Pontuação de confiança 85%+ em consultas complexas
- **Cobertura**: Mais de 60MB de conteúdo especializado em direito empresarial
- **Arquitetura**: CRAG + IA Híbrida para qualidade otimizada
- **Idiomas**: Suporte bilíngue completo (Português/Inglês)

### 🔧 Configuração

O sistema pode ser configurado através do modelo `ProcessingConfig`:

```python
config = ProcessingConfig(
    max_documents_per_source=20,        # Documentos por fonte de busca
    search_timeout_seconds=60,          # Timeout para buscas
    enable_parallel_search=True,        # Processamento paralelo
    min_confidence_threshold=0.4,       # Confiança mínima
    temperature=0.1,                    # Criatividade da IA (baixa para jurídico)
    max_tokens=4000,                    # Tamanho da resposta
    enable_guardrails=True,             # Validações de segurança
    enable_web_search=True,             # Capacidade de busca web
    enable_jurisprudence_search=True    # Integração LexML
)
```

### 🛡️ Disclaimers Legais e Controle de Qualidade

- **Disclaimer Profissional**: Todas as respostas incluem disclaimers claros
- **Limitação de Escopo**: Foca em informações gerais de direito empresarial
- **Pontuação de Qualidade**: Scores de confiança de 0.0 a 1.0 em todas as respostas
- **Guardrails**: Verificações integradas para orientação jurídica apropriada
- **Atribuição de Fontes**: Todas as respostas citam suas fontes
- **Validação CRAG**: Mecanismos corretivos garantem precisão das respostas

### 🧪 Desenvolvimento

#### Estrutura do Projeto

```
├── src/                          # Código principal da aplicação
│   ├── core/                     # Modelos e configuração principal
│   │   ├── legal_models.py      # Modelos Pydantic
│   │   ├── workflow_builder.py  # Workflows CRAG LangGraph
│   │   └── observability.py     # Monitoramento e logging
│   ├── agents/                   # Agentes de IA
│   │   ├── document_retriever.py # Interface ChromaDB
│   │   ├── document_grader.py   # Avaliação de relevância
│   │   ├── query_transformer.py # Otimização de consultas
│   │   ├── search_coordinator.py # Busca multi-fonte
│   │   └── streaming/           # Processadores streaming
│   │       ├── hybrid_legal_processor.py # Sistema híbrido de IA
│   │       └── response_synthesizer.py  # Geração de respostas
│   └── interfaces/              # Integrações externas
│       └── external_search_client.py # LexML e Tavily
├── data/                        # Documentos jurídicos (60MB+) - .gitignore
├── chroma_db_gemini/           # Banco de dados vetorial
├── crag_graph.png              # Diagrama do workflow CRAG
├── legal_ai_workflow.png       # Workflow completo do sistema
├── app.py                      # Aplicação Streamlit principal
├── pyproject.toml             # Configuração do projeto
└── README.md                  # Este arquivo
```

#### Executando Testes

```bash
# Execute testes específicos
uv run python tests/test_system.py

# Teste a integração completa
uv run python tests/test_complete_hybrid_integration.py
```

#### Contribuindo

1. Faça um fork do repositório
2. Crie uma branch de feature (`git checkout -b feature/funcionalidade-incrivel`)
3. Commit suas mudanças (`git commit -m 'Adiciona funcionalidade incrível'`)
4. Push para a branch (`git push origin feature/funcionalidade-incrivel`)
5. Abra um Pull Request

### 📝 Referência da API

#### Modelo LegalQuery

```python
query = LegalQuery(
    text="Sua pergunta de direito empresarial",
    priority=Priority.MEDIUM,
    validation_level=ValidationLevel.MODERATE,
    area_of_law=LegalAreaType.COMMERCIAL
)
```

#### Estrutura de Resposta

```python
response = FinalResponse(
    overall_summary="Análise jurídica abrangente...",
    overall_confidence=0.85,
    search_results=[...],
    detailed_analyses=[...],
    disclaimer="Disclaimer jurídico profissional..."
)
```

### ⚖️ Aviso Legal

Este sistema fornece informações gerais de direito empresarial e não substitui assessoria jurídica profissional. Sempre consulte profissionais jurídicos qualificados para questões legais específicas.

---

## 📄 License / Licença

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

Este projeto está licenciado sob a Licença MIT - veja o arquivo [LICENSE](LICENSE) para detalhes.
