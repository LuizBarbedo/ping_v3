#!/usr/bin/env python3
"""
Sistema de Análise SWOT Individual com RAG Local
=================================================
Utiliza LangChain + ChromaDB + Ollama para gerar análises SWOT 
separadas para cada perfil, garantindo isolamento de dados.

Autor: Engenharia de Dados e IA
"""

import os
import json
import glob
from datetime import datetime
from typing import List, Dict, Any

# LangChain Core & Community
from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.llms import Ollama

# Configurações
WORKSPACE_DIR = os.path.dirname(os.path.abspath(__file__))
CHROMA_PERSIST_DIR = os.path.join(WORKSPACE_DIR, "chroma_db")
REPORTS_DIR = os.path.join(WORKSPACE_DIR, "swot_reports")

# Modelos Ollama
EMBEDDING_MODEL = "nomic-embed-text"
LLM_MODEL = "qwen3:30b"  


def carregar_e_processar_jsons(diretorio: str) -> List[Document]:
    """
    ETL: Carrega todos os arquivos .json e cria objetos Document
    com page_content e metadata conforme especificado.
    
    Args:
        diretorio: Caminho do diretório contendo os arquivos .json
        
    Returns:
        Lista de objetos Document prontos para indexação
    """
    documentos = []
    arquivos_json = glob.glob(os.path.join(diretorio, "*.json"))
    
    print(f"\n📂 Encontrados {len(arquivos_json)} arquivos JSON para processar...")
    
    for arquivo_path in arquivos_json:
        nome_arquivo = os.path.basename(arquivo_path)
        print(f"  ├── Processando: {nome_arquivo}")
        
        try:
            with open(arquivo_path, 'r', encoding='utf-8') as f:
                dados = json.load(f)
            
            # Garante que dados seja uma lista
            if isinstance(dados, dict):
                dados = [dados]
            
            posts_processados = 0
            for post in dados:
                # Extrai campos com tratamento de valores None/vazios
                caption = post.get('caption', '') or ''
                alt = post.get('alt', '') or ''
                timestamp = post.get('timestamp', '') or ''
                likes = post.get('likesCount', 0)
                if likes == -1:  # Valor sentinela no JSON
                    likes = 0
                
                # Processa comentários
                comentarios_formatados = []
                latest_comments = post.get('latestComments', []) or []
                for comentario in latest_comments:
                    if isinstance(comentario, dict):
                        user = comentario.get('ownerUsername', 'Anônimo')
                        texto = comentario.get('text', '')
                        if texto:
                            comentarios_formatados.append(f"{user}: {texto}")
                
                comentarios_str = "\n".join(comentarios_formatados) if comentarios_formatados else "Sem comentários"
                
                # Monta o page_content conforme especificação
                page_content = f"""
PERFIL/FONTE: {nome_arquivo}
---
LEGENDA DO POST:
{caption}
---
DESCRIÇÃO DA IMAGEM (ALT):
{alt}
---
COMENTÁRIOS:
{comentarios_str}
""".strip()
                
                # Cria o Document com metadata para filtragem
                doc = Document(
                    page_content=page_content,
                    metadata={
                        "source": nome_arquivo,  # CRÍTICO: usado para filtro
                        "timestamp": timestamp,
                        "likes": likes,
                        "post_id": post.get('id', ''),
                        "type": post.get('type', 'Unknown')
                    }
                )
                documentos.append(doc)
                posts_processados += 1
            
            print(f"  │   └── {posts_processados} posts extraídos")
            
        except json.JSONDecodeError as e:
            print(f"  │   ⚠️  Erro ao decodificar JSON: {e}")
        except Exception as e:
            print(f"  │   ⚠️  Erro inesperado: {e}")
    
    print(f"\n✅ Total de documentos criados: {len(documentos)}")
    return documentos


def criar_vector_store(documentos: List[Document], persist_directory: str) -> Chroma:
    """
    Cria ou carrega o Vector Store ChromaDB com embeddings Ollama.
    
    Args:
        documentos: Lista de Documents para indexar
        persist_directory: Diretório para persistência do ChromaDB
        
    Returns:
        Instância do ChromaDB
    """
    print("\n🔄 Inicializando embeddings com Ollama (nomic-embed-text)...")
    
    embeddings = OllamaEmbeddings(
        model=EMBEDDING_MODEL,
        base_url="http://localhost:11434"
    )
    
    # Remove banco antigo para garantir dados atualizados
    if os.path.exists(persist_directory):
        import shutil
        print("  ├── Removendo banco de vetores antigo...")
        shutil.rmtree(persist_directory)
    
    print("  ├── Criando novo banco de vetores...")
    print(f"  ├── Indexando {len(documentos)} documentos...")
    
    vector_store = Chroma.from_documents(
        documents=documentos,
        embedding=embeddings,
        persist_directory=persist_directory,
        collection_name="perfis_swot"
    )
    
    print("  └── ✅ Vector Store criado com sucesso!")
    return vector_store


def gerar_swot_individual(
    vector_store: Chroma,
    nome_arquivo_alvo: str,
    k: int = 15
) -> str:
    """
    Gera análise SWOT para um perfil específico usando filtro de metadados.
    
    CRITICAL: O filtro {"source": nome_arquivo_alvo} garante que apenas
    documentos do perfil especificado sejam considerados na análise.
    
    Args:
        vector_store: Instância do ChromaDB
        nome_arquivo_alvo: Nome do arquivo (ex: "reitor.json")
        k: Número de documentos a recuperar (default: 15)
        
    Returns:
        Texto da análise SWOT gerada
    """
    print(f"\n🎯 Gerando SWOT para: {nome_arquivo_alvo}")
    print(f"  ├── Configurando filtro de metadados: source = {nome_arquivo_alvo}")
    print(f"  ├── Recuperando top {k} documentos relevantes...")
    
    # Configura o retriever com FILTRO DE METADADOS - CRÍTICO!
    retriever = vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={
            "k": k,
            "filter": {"source": nome_arquivo_alvo}  # ISOLAMENTO DE DADOS
        }
    )
    
    # Verifica se há documentos para o perfil
    docs_teste = retriever.invoke("análise geral do perfil")
    if not docs_teste:
        return f"⚠️ Nenhum documento encontrado para o perfil: {nome_arquivo_alvo}"
    
    print(f"  ├── {len(docs_teste)} documentos recuperados para análise")
    
    # Configura o LLM Ollama
    llm = Ollama(
        model=LLM_MODEL,
        base_url="http://localhost:11434",
        temperature=0.3,  # Menor temperatura para análises mais focadas
        num_ctx=8192,     # Contexto expandido para análises longas
    )
    
    # Prompt especializado para análise SWOT
    prompt_template = """Você é um analista estratégico especializado em análise de perfis públicos e institucionais.

CONTEXTO DOS DOCUMENTOS:
{context}

TAREFA:
Baseado EXCLUSIVAMENTE nos documentos fornecidos acima, crie uma Análise SWOT detalhada e profunda.

INSTRUÇÕES ESPECÍFICAS:
1. **FORÇAS (Strengths)**: Identifique pontos fortes. CITE frases específicas dos comentários ou legendas como evidência.
2. **FRAQUEZAS (Weaknesses)**: Identifique pontos fracos ou áreas de melhoria. CITE frases que demonstrem críticas ou problemas.
3. **OPORTUNIDADES (Opportunities)**: Identifique oportunidades de crescimento baseadas nos padrões das legendas e engajamento.
4. **AMEAÇAS (Threats)**: Identifique potenciais riscos ou ameaças baseadas nas interações e contexto.

FORMATO DE RESPOSTA:
Use o formato abaixo, sendo detalhado e citando evidências textuais:

## 📊 ANÁLISE SWOT

### 💪 FORÇAS (Strengths)
[Liste cada força com citação de evidência]

### ⚠️ FRAQUEZAS (Weaknesses)  
[Liste cada fraqueza com citação de evidência]

### 🚀 OPORTUNIDADES (Opportunities)
[Liste cada oportunidade identificada]

### 🔥 AMEAÇAS (Threats)
[Liste cada ameaça identificada]

### 📈 PADRÕES IDENTIFICADOS NAS LEGENDAS
[Analise temas recorrentes, tom de comunicação, estratégias]

### 💡 RECOMENDAÇÕES ESTRATÉGICAS
[Baseado na análise, sugira 3-5 ações concretas]

Resposta detalhada:"""

    prompt = PromptTemplate(
        template=prompt_template,
        input_variables=["context"]
    )
    
    # Função para formatar documentos
    def format_docs(docs):
        return "\n\n---\n\n".join(doc.page_content for doc in docs)
    
    # Cria a chain usando LCEL (LangChain Expression Language)
    rag_chain = (
        {"context": retriever | format_docs}
        | prompt
        | llm
        | StrOutputParser()
    )
    
    # Executa a análise
    print("  ├── Executando análise com LLM (llama3)...")
    
    query = f"Realize uma análise SWOT completa e aprofundada do perfil '{nome_arquivo_alvo}'."
    
    resultado = rag_chain.invoke(query)
    
    print("  └── ✅ Análise concluída!")
    
    return resultado


def salvar_relatorio(nome_arquivo: str, conteudo: str, diretorio_saida: str) -> str:
    """
    Salva o relatório SWOT em arquivo Markdown.
    
    Args:
        nome_arquivo: Nome do perfil analisado
        conteudo: Conteúdo da análise SWOT
        diretorio_saida: Diretório para salvar os relatórios
        
    Returns:
        Caminho do arquivo salvo
    """
    os.makedirs(diretorio_saida, exist_ok=True)
    
    nome_base = nome_arquivo.replace('.json', '')
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    nome_relatorio = f"SWOT_{nome_base}_{timestamp}.md"
    caminho_completo = os.path.join(diretorio_saida, nome_relatorio)
    
    # Monta o relatório completo
    relatorio = f"""# Relatório de Análise SWOT
## Perfil: {nome_arquivo}
**Data de Geração:** {datetime.now().strftime("%d/%m/%Y às %H:%M:%S")}
**Modelo LLM:** {LLM_MODEL}
**Modelo de Embeddings:** {EMBEDDING_MODEL}

---

{conteudo}

---
*Relatório gerado automaticamente pelo Sistema de Análise SWOT com RAG Local*
"""
    
    with open(caminho_completo, 'w', encoding='utf-8') as f:
        f.write(relatorio)
    
    return caminho_completo


def main():
    """
    Função principal que orquestra todo o pipeline de análise SWOT.
    """
    print("=" * 60)
    print("🔬 SISTEMA DE ANÁLISE SWOT COM RAG LOCAL")
    print("=" * 60)
    print(f"📅 Execução iniciada em: {datetime.now().strftime('%d/%m/%Y às %H:%M:%S')}")
    print(f"📂 Diretório de trabalho: {WORKSPACE_DIR}")
    
    # ========================================
    # 1. ETL - Carregamento e Processamento
    # ========================================
    print("\n" + "=" * 60)
    print("📥 ETAPA 1: CARREGAMENTO E PROCESSAMENTO DE DADOS (ETL)")
    print("=" * 60)
    
    documentos = carregar_e_processar_jsons(WORKSPACE_DIR)
    
    if not documentos:
        print("❌ Nenhum documento foi carregado. Verifique os arquivos JSON.")
        return
    
    # ========================================
    # 2. Indexação no Vector Store
    # ========================================
    print("\n" + "=" * 60)
    print("🗄️ ETAPA 2: INDEXAÇÃO NO VECTOR STORE (CHROMADB)")
    print("=" * 60)
    
    vector_store = criar_vector_store(documentos, CHROMA_PERSIST_DIR)
    
    # ========================================
    # 3. Geração de Análises SWOT
    # ========================================
    print("\n" + "=" * 60)
    print("📊 ETAPA 3: GERAÇÃO DE ANÁLISES SWOT INDIVIDUAIS")
    print("=" * 60)
    
    # Lista de perfis a analisar (CONFIGURÁVEL)
    # Adicione ou remova arquivos conforme necessário
    perfis_para_analisar = [
        "reitor.json",
        "joserodrigues.json",
        # "sintuff.json",
        # "fabiopassos.json",
        # Adicione mais perfis aqui conforme necessário
    ]
    
    # Verifica quais perfis existem
    arquivos_disponiveis = set(os.path.basename(f) for f in glob.glob(os.path.join(WORKSPACE_DIR, "*.json")))
    
    print(f"\n📋 Perfis configurados para análise: {len(perfis_para_analisar)}")
    for perfil in perfis_para_analisar:
        status = "✅" if perfil in arquivos_disponiveis else "❌ (não encontrado)"
        print(f"  ├── {perfil} {status}")
    
    # Processa cada perfil sequencialmente
    relatorios_gerados = []
    
    for i, perfil in enumerate(perfis_para_analisar, 1):
        if perfil not in arquivos_disponiveis:
            print(f"\n⚠️ Pulando {perfil}: arquivo não encontrado")
            continue
            
        print(f"\n{'─' * 50}")
        print(f"📌 Processando perfil {i}/{len(perfis_para_analisar)}: {perfil}")
        print('─' * 50)
        
        try:
            # Gera a análise SWOT com isolamento de dados
            analise = gerar_swot_individual(
                vector_store=vector_store,
                nome_arquivo_alvo=perfil,
                k=15  # Recupera bastante contexto
            )
            
            # Salva o relatório
            caminho_relatorio = salvar_relatorio(
                nome_arquivo=perfil,
                conteudo=analise,
                diretorio_saida=REPORTS_DIR
            )
            
            relatorios_gerados.append(caminho_relatorio)
            print(f"  📄 Relatório salvo: {caminho_relatorio}")
            
        except Exception as e:
            print(f"  ❌ Erro ao processar {perfil}: {e}")
            import traceback
            traceback.print_exc()
    
    # ========================================
    # 4. Resumo Final
    # ========================================
    print("\n" + "=" * 60)
    print("✅ EXECUÇÃO CONCLUÍDA")
    print("=" * 60)
    print(f"📊 Total de relatórios gerados: {len(relatorios_gerados)}")
    print(f"📂 Diretório dos relatórios: {REPORTS_DIR}")
    print("\n📋 Relatórios gerados:")
    for relatorio in relatorios_gerados:
        print(f"  └── {os.path.basename(relatorio)}")
    
    print(f"\n⏱️ Execução finalizada em: {datetime.now().strftime('%d/%m/%Y às %H:%M:%S')}")


if __name__ == "__main__":
    main()
