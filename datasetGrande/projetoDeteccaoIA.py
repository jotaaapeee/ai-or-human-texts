# CRISP-DM: Detector de Texto - IA vs Humano
import pandas as pd
import numpy as np
import pickle
import re

# CRISP-DM: análise e visualização de dados - fase 2/3
import matplotlib.pyplot as plt

# CRISP-DM: engenharia de features de texto - fase 3
from sklearn.feature_extraction.text import TfidfVectorizer

# CRISP-DM: pré-processamento - fase 3
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

# CRISP-DM: modelos - fase 4
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier

# CRISP-DM: avaliação - fase 5
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# -------------------------------------------------------
# CONFIGURAÇÕES GLOBAIS
# -------------------------------------------------------
NOMEARQUIVO      = "model_training_dataset.parquet"  # Parquet com human_text / ai_text
NOMEMODELO       = "detectorIA.pickle"
NOME_VETORIZADOR = "vetorizador.pickle"
NOME_ENCODER     = "labelEncoder.pickle"

# Quantas linhas usar no treino (o Parquet tem 1M — usar tudo é lento e desnecessário).
# 100k já garante generalização muito melhor que os 6k do CSV antigo.
# Aumente se tiver RAM/tempo disponível.
N_TREINO = 100_000


# DIAGNÓSTICO - Descubra o vazamento
def diagnosticar_dataset(nomeArquivo):
    """Verifica se há padrões suspeitos no dataset"""
    df_full = pd.read_parquet(nomeArquivo, engine="pyarrow")
    
    print("="*60)
    print("DIAGNÓSTICO DO DATASET")
    print("="*60)
    
    # 1. Verificar colunas
    print(f"\nColunas disponíveis: {list(df_full.columns)}")
    
    # 2. Verificar se há marcadores nos textos
    print("\n--- Verificando marcadores nos primeiros textos ---")
    for i in range(min(5, len(df_full))):
        print(f"\nExemplo {i}:")
        print(f"  human_text (início): '{df_full.iloc[i]['human_text'][:100]}'")
        print(f"  ai_text (início):    '{df_full.iloc[i]['ai_text'][:100]}'")
        if 'instructions' in df_full.columns:
            print(f"  instructions: '{df_full.iloc[i]['instructions'][:100]}'")
    
    # 3. Verificar se a instrução é a mesma para humanos e IA
    if 'instructions' in df_full.columns:
        print("\n--- Analisando coluna instructions ---")
        unique_instructions = df_full['instructions'].nunique()
        print(f"Instruções únicas: {unique_instructions}")
        
        # Ver se humanos e IA recebem instruções diferentes
        sample = df_full.head(100)
        print(f"Exemplo de instruções nas primeiras 100 linhas:")
        print(sample['instructions'].value_counts().head())
    
    # 4. Verificar padrões de pontuação/capitalização
    print("\n--- Padrões suspeitos ---")
    for col in ['human_text', 'ai_text']:
        textos = df_full[col].astype(str)
        
        # Verificar se começam com marcador
        starts_with_label = textos.str.match(r'^(Human|AI|human|ai):', case=False).sum()
        print(f"{col}: {starts_with_label}/{len(textos)} começam com 'Human:' ou 'AI:'")
        
        # Verificar comprimento médio
        avg_len = textos.str.len().mean()
        print(f"{col}: comprimento médio = {avg_len:.0f} caracteres")
    
    return df_full

# Execute antes do carregamento normal
df_diagnostic = diagnosticar_dataset("model_training_dataset.parquet")


# -------------------------------------------------------
# FASE 2 — ENTENDIMENTO DOS DADOS
# -------------------------------------------------------

def carregarDados(nomeArquivo):
    """
    Lê o Parquet com colunas human_text / ai_text e o transforma
    em um DataFrame com colunas 'text' e 'author', igual ao formato
    esperado pelo restante do pipeline.

    Carrega apenas as colunas necessárias (formato colunar do Parquet)
    e amostra N_TREINO linhas para manter o treino manejável.
    """
    try:
        print(f"[INFO] Lendo Parquet '{nomeArquivo}' (colunas: human_text, ai_text)...")
        df = pd.read_parquet(nomeArquivo,
                             columns=["human_text", "ai_text"],
                             engine="pyarrow")
        print(f"[OK] Parquet carregado: {len(df)} linhas")

        # amostra N_TREINO linhas antes de desempilhar (economiza memória)
        n = min(N_TREINO, len(df))
        df = df.dropna(subset=["human_text", "ai_text"]).sample(n=n, random_state=42)
        print(f"[OK] Amostra para treino: {n} linhas ({n} Human + {n} AI = {n*2} textos)")

        # "desempilha" as duas colunas em linhas com label
        humanos = pd.DataFrame({"text": df["human_text"], "author": "Human"})
        ias     = pd.DataFrame({"text": df["ai_text"],    "author": "AI"})
        dados   = pd.concat([humanos, ias], ignore_index=True)
        dados   = dados.sample(frac=1, random_state=42).reset_index(drop=True)

        print(f"[OK] Dataset final: {len(dados)} textos, {dados.shape[1]} colunas")
        return dados

    except Exception as e:
        print(f"[ERRO] Não foi possível carregar os dados: {e}")
        print("[DICA] Instale o engine: pip install pyarrow")
        return None

# -------------------------------------------------------
# FASE 3 — PREPARAÇÃO DOS DADOS + ENGENHARIA DE FEATURES
# -------------------------------------------------------

def limparTexto(texto):
    """
    Limpeza básica do texto:
    - converte para minúsculas
    - remove caracteres especiais e números isolados
    - remove espaços extras
    """
    if not isinstance(texto, str):
        return ""
    texto = texto.lower()
    texto = re.sub(r"http\S+|www\S+", "", texto)       # remove URLs
    texto = re.sub(r"[^a-záéíóúãõâêîôûàèìòùç\s]", " ", texto)  # só letras
    texto = re.sub(r"\s+", " ", texto).strip()          # espaços extras
    return texto

def prepararDados(dados):
    """Limpeza geral do DataFrame (colunas: text, author)."""
    print("\n--- Informações iniciais ---")
    print(dados.info())
    print(dados.head())

    # remove linhas sem texto ou sem label (segurança)
    dados.dropna(subset=["text", "author"], inplace=True)

    # limpeza do texto
    dados["text_limpo"] = dados["text"].apply(limparTexto)

    # remove textos que ficaram vazios após limpeza
    dados = dados[dados["text_limpo"].str.strip() != ""]

    # LabelEncoder: AI=0, Human=1 (ordem alfabética)
    le = LabelEncoder()
    dados = dados.copy()
    dados["label"] = le.fit_transform(dados["author"])

    print(f"\n[OK] Dados após limpeza: {dados.shape[0]} registros")
    print(f"Distribuição de classes:\n{dados['author'].value_counts()}")

    # IMPORTANTE: confirma o mapeamento real gerado pelo LabelEncoder
    print(f"\n[INFO] Mapeamento do LabelEncoder (ordem alfabética):")
    for i, classe in enumerate(le.classes_):
        print(f"  {classe!r} → {i}")

    return dados, le

def visualizarDados(dados):
    """Gera gráficos exploratórios básicos."""

    # distribuição de classes
    fig, ax = plt.subplots()
    contagem = dados["author"].value_counts()
    ax.bar(contagem.index, contagem.values, color=["tab:blue", "tab:orange"])
    ax.set_title("Distribuição: Humano vs IA")
    ax.set_ylabel("Quantidade de textos")
    plt.tight_layout()
    plt.show()

    # comprimento dos textos por classe
    dados["tamanho_texto"] = dados["text_limpo"].apply(len)
    dados.boxplot(column="tamanho_texto", by="author")
    plt.title("Comprimento dos textos por classe")
    plt.suptitle("")
    plt.ylabel("Caracteres")
    plt.tight_layout()
    plt.show()

def extrairFeatures(dados):
    """
    Engenharia de features via TF-IDF.
    
    TF-IDF (Term Frequency–Inverse Document Frequency) transforma
    cada texto em um vetor numérico com base na relevância das palavras.
    É a ponte entre texto bruto e os modelos de ML.
    
    Parâmetros escolhidos:
    - max_features=10000 : usa as 10k palavras mais relevantes
    - ngram_range=(1,2)  : considera palavras sozinhas E pares de palavras (bigramas)
    - sublinear_tf=True  : suaviza frequências altas (evita dominância de palavras repetitivas)
    - min_df=2           : ignora palavras que aparecem em menos de 2 documentos
    """
    print("\n[INFO] Extraindo features com TF-IDF...")
    vetorizador = TfidfVectorizer(
        max_features=10000,
        ngram_range=(1, 2),
        sublinear_tf=True,
        min_df=2,
        strip_accents="unicode"
    )

    X = vetorizador.fit_transform(dados["text_limpo"])
    Y = dados["label"]

    print(f"[OK] Matriz de features: {X.shape[0]} textos × {X.shape[1]} features")
    return X, Y, vetorizador

# -------------------------------------------------------
# FASE 3 — SEPARAÇÃO TREINO / TESTE
# -------------------------------------------------------

def separarDados(X, Y):
    """Divide em 70% treino e 30% teste com estratificação."""
    print("\n[INFO] Separando dados de treino e teste (70/30)...")

    x_train, x_test, y_train, y_test = train_test_split(
        X, Y,
        test_size=0.3,
        train_size=0.7,
        shuffle=True,
        random_state=42,
        stratify=Y           # garante proporção igual de classes em treino e teste
    )

    print(f"[OK] Treino: {x_train.shape[0]} | Teste: {x_test.shape[0]}")
    return x_train, x_test, y_train, y_test

# -------------------------------------------------------
# FASE 4 — MODELAGEM
# -------------------------------------------------------

def treinarModelo(x_train, y_train):
    """
    Treina os dois modelos pedidos:
    - Regressão Logística: rápida, interpretável, ótima baseline para texto
    - MLP (Rede Neural): camadas densas, captura padrões mais complexos
    """
    listaModelos = []

    listaAlgoritmos = [
        LogisticRegression(
            max_iter=1000,
            C=1.0,
            solver="lbfgs",
            class_weight="balanced",  # penaliza mais os erros na classe minoritária
            random_state=42
        ),
        MLPClassifier(
            hidden_layer_sizes=(256, 128),
            activation="relu",
            solver="adam",
            alpha=1e-4,
            max_iter=300,
            random_state=42,
            early_stopping=True,
            validation_fraction=0.1
            # MLP não tem class_weight — o balanceamento no dataset já resolve
        )
    ]

    nomes = ["Regressão Logística", "Rede Neural (MLP)"]

    for nome, algoritmo in zip(nomes, listaAlgoritmos):
        try:
            print(f"\n[INFO] Treinando: {nome}...")
            algoritmo.fit(x_train, y_train)
            listaModelos.append((nome, algoritmo))
            print(f"[OK] {nome} treinado.")
        except Exception as e:
            print(f"[ERRO] Falha ao treinar {nome}: {e}")

    return listaModelos

# -------------------------------------------------------
# FASE 5 — AVALIAÇÃO
# -------------------------------------------------------

def avaliarListaModelos(listaModelos, x_test, y_test, le):
    """
    Avalia todos os modelos e retorna o de maior F1-score macro.
    F1 macro é mais confiável que acurácia quando há desbalanceamento:
    ela penaliza o modelo que ignora uma classe inteira.
    """
    from sklearn.metrics import f1_score
    listaF1 = []

    for nome, modelo in listaModelos:
        y_pred = modelo.predict(x_test)
        acuracia = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average="macro", zero_division=0)
        listaF1.append(f1)

        print(f"\n{'='*50}")
        print(f"Modelo: {nome}")
        print(f"Acurácia : {acuracia:.4f} ({acuracia*100:.2f}%)")
        print(f"F1 macro : {f1:.4f}  ← critério de seleção")
        print("\nRelatório de Classificação:")
        print(classification_report(y_test, y_pred,
                                    target_names=le.classes_,
                                    zero_division=0))
        cm = confusion_matrix(y_test, y_pred)
        print("Matriz de Confusão:")
        print(pd.DataFrame(cm, index=le.classes_, columns=le.classes_))

    melhorIdx = listaF1.index(max(listaF1))
    nomeMelhor, melhorModelo = listaModelos[melhorIdx]
    print(f"\n[RESULTADO] Melhor modelo: {nomeMelhor} "
          f"— F1 macro: {max(listaF1):.4f}")

    return melhorModelo

# -------------------------------------------------------
# PERSISTÊNCIA DO MODELO
# -------------------------------------------------------

def salvarModelo(nomeModelo, nomeVetorizador, nomeEncoder, modelo, vetorizador, le):
    """Salva modelo, vetorizador TF-IDF e LabelEncoder. Os três são necessários para predição."""
    with open(nomeModelo, "wb") as f:
        pickle.dump(modelo, f)
    with open(nomeVetorizador, "wb") as f:
        pickle.dump(vetorizador, f)
    with open(nomeEncoder, "wb") as f:
        pickle.dump(le, f)
    print(f"\n[OK] Modelo salvo em '{nomeModelo}'")
    print(f"[OK] Vetorizador salvo em '{nomeVetorizador}'")
    print(f"[OK] LabelEncoder salvo em '{nomeEncoder}'")

def carregarModelo(nomeModelo, nomeVetorizador, nomeEncoder):
    """Carrega o modelo, o vetorizador e o LabelEncoder salvos."""
    with open(nomeModelo, "rb") as f:
        modelo = pickle.load(f)
    with open(nomeVetorizador, "rb") as f:
        vetorizador = pickle.load(f)
    with open(nomeEncoder, "rb") as f:
        le = pickle.load(f)
    print(f"[OK] Modelo, vetorizador e LabelEncoder carregados.")
    print(f"[INFO] Classes do modelo: { {c: i for i, c in enumerate(le.classes_)} }")
    return modelo, vetorizador, le

# -------------------------------------------------------
# VALIDAÇÃO / INFERÊNCIA
# -------------------------------------------------------

def validarTexto(modelo, vetorizador, le, textos):
    """
    Recebe uma lista de textos brutos e retorna a predição (Human ou IA).
    
    Exemplo de uso:
        validarTexto(modelo, vetorizador, le, ["Este texto foi escrito por uma pessoa."])
    """
    textos_limpos = [limparTexto(t) for t in textos]
    X_val = vetorizador.transform(textos_limpos)
    y_pred = modelo.predict(X_val)
    labels = le.inverse_transform(y_pred)

    for texto, label in zip(textos, labels):
        print(f"\nTexto: '{texto[:80]}...' \n→ Predição: {label}")

    return labels

# -------------------------------------------------------
# PIPELINE PRINCIPAL — CRISP-DM
# -------------------------------------------------------

# FASE 2: carregar dados
dados = carregarDados(NOMEARQUIVO)

if dados is not None:

    # FASE 3: preparar dados + limpeza
    dados, le = prepararDados(dados)

    # FASE 3 (opcional): visualizar distribuições
    # visualizarDados(dados)

    # FASE 3: engenharia de features (texto → vetores numéricos)
    X, Y, vetorizador = extrairFeatures(dados)

    # FASE 3: separar treino e teste
    x_train, x_test, y_train, y_test = separarDados(X, Y)

    # FASE 4: treinar modelos
    listaModelos = treinarModelo(x_train, y_train)

    # FASE 5: avaliar e escolher melhor modelo
    melhorModelo = avaliarListaModelos(listaModelos, x_test, y_test, le)

    # PERSISTÊNCIA: salvar modelo, vetorizador e LabelEncoder
    salvarModelo(NOMEMODELO, NOME_VETORIZADOR, NOME_ENCODER, melhorModelo, vetorizador, le)

    # INFERÊNCIA: carregar e testar com textos novos
    modeloCarregado, vetorizadorCarregado, leCarregado = carregarModelo(NOMEMODELO, NOME_VETORIZADOR, NOME_ENCODER)

    textos_exemplo = [
        "I went to the market today and bought some bread for my family.",
        "The integration of large language models into production environments requires careful consideration of latency, throughput, and cost-efficiency trade-offs.",
    ]
    validarTexto(modeloCarregado, vetorizadorCarregado, leCarregado, textos_exemplo)