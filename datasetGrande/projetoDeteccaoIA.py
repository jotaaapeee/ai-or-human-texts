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
from sklearn.model_selection import GroupShuffleSplit

# CRISP-DM: avaliação - fase 5
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# -------------------------------------------------------
# CONFIGURAÇÕES GLOBAIS
# -------------------------------------------------------
NOMEARQUIVO      = "model_training_dataset.parquet"  # Parquet com human_text / ai_text
NOMEMODELO       = "detectorIA.pickle"
NOME_VETORIZADOR = "vetorizador.pickle"
NOME_ENCODER     = "labelEncoder.pickle"

N_TREINO = 30_000 # numero de linhass do treino, mudar de acordo com o hardware

def contextualizarDataset():
    print("="*60)
    print("CONTEXTUALIZAÇÃO DO DATASET")
    print("="*60)
    print("""
        Dataset: Human vs AI Generated Text

        Colunas:
        - id: identificador único
        - human_text: texto escrito por humano
        - ai_text: texto gerado por IA
        - instructions: instrução dada para ambos

        Objetivo:
        Classificar se um texto foi escrito por um humano ou por uma IA.
""")

# Descobrindo um possivel dataleak
def diagnosticar_dataset(nomeArquivo):
    """Verifica se há padrões suspeitos no dataset"""
    df_full = pd.read_parquet(nomeArquivo, engine="pyarrow")
    
    print("="*60)
    print("DIAGNÓSTICO DO DATASET")
    print("="*60)
    
    # 1- Verificar colunas
    print(f"\nColunas disponíveis: {list(df_full.columns)}")
    
    # 2- Verificar se há marcadores nos textos
    print("\n--- Verificando marcadores nos primeiros textos ---")
    for i in range(min(5, len(df_full))):
        print(f"\nExemplo {i}:")
        print(f" human_text (início): '{df_full.iloc[i]['human_text'][:100]}'")
        print(f" ai_text (início): '{df_full.iloc[i]['ai_text'][:100]}'")
        if 'instructions' in df_full.columns:
            print(f" instructions: '{df_full.iloc[i]['instructions'][:100]}'")
    
    # 3- Verificar se a instrução é a mesma para humanos e IA
    if 'instructions' in df_full.columns:
        print("\n--- Analisando coluna instructions ---")
        unique_instructions = df_full['instructions'].nunique()
        print(f"Instruções únicas: {unique_instructions}")
        
        # Ver se humanos e IA recebem instruções diferentes
        sample = df_full.head(100)
        print(f"Exemplo de instruções nas primeiras 100 linhas:")
        print(sample['instructions'].value_counts().head())
    
    # 4- Verificar padrões de pontuação/capitalização
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

def carregarDados(nomeArquivo):
    try:
        print(f"[INFO] Lendo Parquet '{nomeArquivo}' (colunas: human_text, ai_text)...")
        df = pd.read_parquet(nomeArquivo,
                             columns=["human_text", "ai_text"],
                             engine="pyarrow")
        print(f"[OK] Parquet carregado: {len(df)} linhas")

        return df

    except Exception as e:
        print(f"[ERRO] Não foi possível carregar os dados: {e}")
        print("[DICA] Instale o engine: pip install pyarrow")
        return None
    
def montarDataset(df):
    """
    Monta o dataset long-format mantendo o group_id original
    para evitar vazamento entre splits.
    """
    humanos = pd.DataFrame({
        "text":     df["human_text"].values,
        "author":   "Human",
        "group_id": df.index  # <<< mantém o par de origem
    })
    ias = pd.DataFrame({
        "text":     df["ai_text"].values,
        "author":   "AI",
        "group_id": df.index  # <<< mesmo par = mesmo grupo
    })
    dados = pd.concat([humanos, ias], ignore_index=True)
    return dados.sample(frac=1, random_state=42).reset_index(drop=True)

def limparTexto(texto):
    """
    Limpeza básica do texto:
    - converte para minúsculas
    - remove caracteres especiais e números isolados
    - remove espaços extras
    """
    if not isinstance(texto, str):
        return ""
    texto = texto.strip()
    texto = texto[:MAX_CHARS]
    texto = texto.lower()
    texto = re.sub(r"http\S+|www\S+", "", texto) # remove URLs
    texto = re.sub(r"[^a-záéíóúãõâêîôûàèìòùç\s]", " ", texto) # só letras
    texto = re.sub(r"\s+", " ", texto).strip() # espaços extras
    return texto

def prepararDados(dados, le=None):
    dados = dados.copy()
    dados.dropna(subset=["text", "author"], inplace=True)
    dados["text_limpo"] = dados["text"].apply(limparTexto)
    dados = dados[dados["text_limpo"].str.strip() != ""]

    if le is None:
        le = LabelEncoder()
        dados["label"] = le.fit_transform(dados["author"])
        print(f"\n[INFO] Mapeamento LabelEncoder:")
        for i, c in enumerate(le.classes_):
            print(f"  {c!r} → {i}")
    else:
        dados["label"] = le.transform(dados["author"])

    print(f"[OK] {dados.shape[0]} registros | classes: {dados['author'].value_counts().to_dict()}")
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
        ngram_range=(1, 1),
        sublinear_tf=True,
        min_df=2,
        max_df=0.8,
        strip_accents="unicode"
    )

    X = vetorizador.fit_transform(dados["text_limpo"])
    Y = dados["label"]

    print(f"[OK] Matriz de features: {X.shape[0]} textos x {X.shape[1]} features")
    return X, Y, vetorizador

def separarDados(df_original):
    """
    Divide em treino (70%), validação (15%) e teste (15%)
    ANTES de montar o long-format, usando o índice da linha
    (= par human/ai) como grupo real.
    """
    print("\n[INFO] Separando dados em treino/validação/teste...")

    # grupos = linhas do parquet (pares human/ai)
    groups = np.arange(len(df_original))

    gss = GroupShuffleSplit(n_splits=1, test_size=0.3, random_state=42)
    train_idx, temp_idx = next(gss.split(df_original, groups=groups))

    df_train = df_original.iloc[train_idx].reset_index(drop=True)
    df_temp  = df_original.iloc[temp_idx].reset_index(drop=True)

    gss2 = GroupShuffleSplit(n_splits=1, test_size=0.5, random_state=42)
    val_idx, test_idx = next(gss2.split(df_temp, groups=np.arange(len(df_temp))))

    df_val  = df_temp.iloc[val_idx].reset_index(drop=True)
    df_test = df_temp.iloc[test_idx].reset_index(drop=True)

    print(f"[OK] Treino:    {len(df_train)} pares → {len(df_train)*2} textos")
    print(f"[OK] Validação: {len(df_val)}  pares → {len(df_val)*2} textos")
    print(f"[OK] Teste:     {len(df_test)} pares → {len(df_test)*2} textos")

    return df_train, df_val, df_test

def treinarModelo(x_train, y_train):
    listaModelos = []

    listaAlgoritmos = [
        LogisticRegression(
            max_iter=1000
        ),
        MLPClassifier(
            solver='lbfgs', 
            alpha=1e-5, 
            hidden_layer_sizes=(15, ), 
            max_iter=1000)
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

def avaliarListaModelos(listaModelos, x_test, y_test, le):
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
          f"- F1 macro: {max(listaF1):.4f}")

    return melhorModelo

def salvarModelo(nomeModelo, nomeVetorizador, nomeEncoder, modelo, vetorizador, le):
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
    with open(nomeModelo, "rb") as f:
        modelo = pickle.load(f)
    with open(nomeVetorizador, "rb") as f:
        vetorizador = pickle.load(f)
    with open(nomeEncoder, "rb") as f:
        le = pickle.load(f)
    print(f"[OK] Modelo, vetorizador e LabelEncoder carregados.")
    print(f"[INFO] Classes do modelo: { {c: i for i, c in enumerate(le.classes_)} }")
    return modelo, vetorizador, le

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

# =======================================================
# 0º CONTEXTUALIZAÇÃO DO DATASET
# =======================================================

contextualizarDataset()

# =======================================================
# 1º CARREGAMENTO DO DATASET
# =======================================================

df_original = carregarDados(NOMEARQUIVO)

limite = int(df_original['ai_text'].str.len().quantile(0.75))  # ~1505
print(f"Truncando textos em {limite} chars")

MAX_CHARS = limite  # substitui o valor fixo

# =======================================================
# 2º TRATAMENTO DE DADOS
# =======================================================

if df_original is not None:

    # SPLIT primeiro (nos pares originais)
    df_train, df_val, df_test = separarDados(df_original)

    # Montar long-format depois
    dados_train = montarDataset(df_train)
    dados_val = montarDataset(df_val)
    dados_test = montarDataset(df_test)

    # prepararDados: fit só no treino, transform nos demais
    dados_train, le = prepararDados(dados_train, le=None)
    dados_val, _  = prepararDados(dados_val, le=le)
    dados_test, _  = prepararDados(dados_test, le=le)

    # =======================================================
    # 3º ANÁLISE GRÁFICA DOS DADOS
    # =======================================================
    visualizarDados(dados_train)

    # =======================================================
    # 4º DIVISÃO DOS DADOS
    # =======================================================

    # Já realizada anteriormente com:
    # - Treino: 70%
    # - Validação: 15%
    # - Teste: 15%
    # Utilizando GroupShuffleSplit para evitar vazamento

    # preparar dados
    dados_train, le = prepararDados(dados_train)
    dados_val, _ = prepararDados(dados_val)
    dados_test, _ = prepararDados(dados_test)

    # =======================================================
    # 5º TREINAMENTO DOS MODELOS
    # =======================================================

    # features (fit só no treino!)
    x_train, y_train, vetorizador = extrairFeatures(dados_train)

    x_val = vetorizador.transform(dados_val["text_limpo"])
    y_val = dados_val["label"]

    x_test = vetorizador.transform(dados_test["text_limpo"])
    y_test = dados_test["label"]

    listaModelos = treinarModelo(x_train, y_train)

    # =======================================================
    # 6º AVALIAÇÃO DOS MODELOS
    # =======================================================

    print("\n=== VALIDAÇÃO ===")
    melhorModelo = avaliarListaModelos(listaModelos, x_val, y_val, le)

    print("\n=== TESTE FINAL ===")
    avaliarListaModelos([("Melhor Modelo", melhorModelo)], x_test, y_test, le)

    # =======================================================
    # 7º IMPLANTAÇÃO DO MODELO
    # =======================================================

    # PERSISTÊNCIA: salvar modelo, vetorizador e LabelEncoder
    salvarModelo(NOMEMODELO, NOME_VETORIZADOR, NOME_ENCODER, melhorModelo, vetorizador, le)

    # INFERÊNCIA: carregar e testar com textos novos
    modeloCarregado, vetorizadorCarregado, leCarregado = carregarModelo(NOMEMODELO, NOME_VETORIZADOR, NOME_ENCODER)

    textos_exemplo = [
        "I went to the market today and bought some bread for my family.",
        "The integration of large language models into production environments requires careful consideration of latency, throughput, and cost-efficiency trade-offs.",
    ]
    validarTexto(modeloCarregado, vetorizadorCarregado, leCarregado, textos_exemplo)

    print("TESTE COM DADOS REAIS")

    textos_reais = [
        "cara fui no mercado e comprei pao e leite pra casa",
        "Hoje eu tive um dia bem corrido no trabalho e quase não parei",
        "The system design requires careful consideration of scalability and latency trade-offs.",
        "In conclusion, it is essential to evaluate both perspectives before making a decision."
    ]

    validarTexto(melhorModelo, vetorizador, le, textos_reais)