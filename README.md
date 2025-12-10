# Sistema de Análise SWOT com RAG Local

Sistema Python para geração de análises SWOT individuais usando RAG (Retrieval-Augmented Generation) local com LangChain, ChromaDB e Ollama.

## 📋 Pré-requisitos

### 1. Ollama (LLM e Embeddings Locais)

Instale o Ollama seguindo as instruções em: https://ollama.com/download

**Comandos para baixar os modelos necessários:**

```bash
# Modelo de embeddings (vetorização de texto)
ollama pull nomic-embed-text

# Modelo LLM para geração de texto
ollama pull llama3
```

Verifique se o Ollama está rodando:
```bash
ollama list
```

### 2. Dependências Python

```bash
# Crie um ambiente virtual (recomendado)
python -m venv venv
source venv/bin/activate  # Linux/Mac
# ou: venv\Scripts\activate  # Windows

# Instale as dependências
pip install langchain langchain-community langchain-core chromadb
```

## 🚀 Execução

```bash
# Navegue até o diretório do projeto
cd /home/lbarbedo/projetos_skynet02/ping_v3

# Execute o script
python swot_analyzer.py
```

## ⚙️ Configuração

Para analisar diferentes perfis, edite a lista `perfis_para_analisar` no arquivo `swot_analyzer.py`:

```python
perfis_para_analisar = [
    "reitor.json",
    "joserodrigues.json",
    "sintuff.json",
    "fabiopassos.json",
    # Adicione mais perfis aqui
]
```

## 📁 Estrutura de Saída

Os relatórios são salvos em `./swot_reports/` no formato:
```
SWOT_{perfil}_{timestamp}.md
```

## 🔧 Parâmetros Ajustáveis

| Parâmetro | Valor | Descrição |
|-----------|-------|-----------|
| `EMBEDDING_MODEL` | nomic-embed-text | Modelo para vetorização |
| `LLM_MODEL` | llama3 | Modelo para geração de texto |
| `k` | 15 | Número de documentos recuperados por consulta |
| `temperature` | 0.3 | Criatividade do LLM (0-1) |
| `num_ctx` | 8192 | Tamanho do contexto do LLM |

## 🛡️ Isolamento de Dados

O sistema garante que cada análise SWOT seja gerada **exclusivamente** com dados do perfil especificado através do filtro de metadados:

```python
retriever = vector_store.as_retriever(
    search_kwargs={
        "filter": {"source": nome_arquivo_alvo}  # ISOLAMENTO
    }
)
```
