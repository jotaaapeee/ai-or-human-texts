# ============================================================
# teste_predicao.py — Amostragem e teste do modelo treinado
# ============================================================
# Lê um arquivo Parquet com ~1M linhas de forma eficiente,
# amostra textos aleatórios (human_text e ai_text),
# e testa o pickle gerado pelo projeto_detector_ia.py
# ============================================================

import pickle
import re
import argparse
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# -------------------------------------------------------
# CONFIGURAÇÕES — ajuste conforme seu ambiente
# -------------------------------------------------------
NOME_PARQUET     = "model_training_dataset.parquet"
NOME_MODELO      = "detectorIA.pickle"
NOME_VETORIZADOR = "vetorizador.pickle"
NOME_ENCODER     = "labelEncoder.pickle"  # gerado pelo projeto_detector_ia.py

N_AMOSTRAS_POR_CLASSE = 500
RANDOM_SEED = 42

# -------------------------------------------------------
# CARREGAR MODELO, VETORIZADOR E LABELENCODER
# -------------------------------------------------------

def carregarArtefatos(nomeModelo, nomeVetorizador, nomeEncoder):
    """
    Carrega os três artefatos gerados pelo treinamento.
    O LabelEncoder é essencial: ele guarda o mapeamento real
    entre string ('Human', 'IA') e inteiro (0, 1), evitando
    suposições que causam predições 100% erradas.
    """
    try:
        with open(nomeModelo, "rb") as f:
            modelo = pickle.load(f)
        with open(nomeVetorizador, "rb") as f:
            vetorizador = pickle.load(f)
        with open(nomeEncoder, "rb") as f:
            le = pickle.load(f)

        mapa = {c: i for i, c in enumerate(le.classes_)}
        print(f"[OK] Modelo carregado: {type(modelo).__name__}")
        print(f"[OK] Mapeamento do LabelEncoder: {mapa}")
        return modelo, vetorizador, le

    except FileNotFoundError as e:
        print(f"[ERRO] Arquivo não encontrado: {e}")
        print("[DICA] Execute primeiro o projeto_detector_ia.py para gerar os pickles.")
        return None, None, None

# -------------------------------------------------------
# LEITURA EFICIENTE DO PARQUET
# -------------------------------------------------------

def carregarAmostraParquet(nomeArquivo, n_por_classe, seed, le):
    """
    Lê o Parquet carregando APENAS as colunas necessárias (formato colunar),
    amostra n_por_classe linhas e monta labels usando o LabelEncoder real.
    """
    # mapeamento vem do LabelEncoder salvo — sem hardcode
    mapa = {c: i for i, c in enumerate(le.classes_)}

    # identifica as classes de forma flexível (case-insensitive)
    classes = list(le.classes_)
    classe_human = next((c for c in classes if "human" in c.lower()), classes[0])
    classe_ia    = next((c for c in classes if c != classe_human), classes[1])
    label_human  = mapa[classe_human]
    label_ia     = mapa[classe_ia]

    print(f"\n[INFO] Lendo Parquet: '{nomeArquivo}'...")
    print(f"[INFO] Human='{classe_human}'({label_human}) | IA='{classe_ia}'({label_ia})")

    try:
        # lê só as colunas usadas — ignora 'id' e 'instructions' no disco
        df = pd.read_parquet(
            nomeArquivo,
            columns=["human_text", "ai_text"],
            engine="pyarrow"
        )
    except Exception as e:
        print(f"[ERRO] Não foi possível ler o Parquet: {e}")
        print("[DICA] Instale o engine: pip install pyarrow")
        return None

    print(f"[OK] Parquet carregado: {len(df)} linhas")

    df.dropna(subset=["human_text", "ai_text"], inplace=True)
    print(f"[OK] Após remover nulos: {len(df)} linhas")

    n_linhas = min(n_por_classe, len(df))
    amostra  = df.sample(n=n_linhas, random_state=seed)

    textos = list(amostra["human_text"]) + list(amostra["ai_text"])
    labels = [label_human] * n_linhas + [label_ia] * n_linhas  # inteiros reais

    print(f"[OK] Amostra: {len(textos)} textos ({n_linhas} Human + {n_linhas} IA)")
    return textos, labels

# -------------------------------------------------------
# PRÉ-PROCESSAMENTO (deve ser idêntico ao do treinamento)
# -------------------------------------------------------

def limparTexto(texto):
    if not isinstance(texto, str):
        return ""
    texto = texto.lower()
    texto = re.sub(r"http\S+|www\S+", "", texto)
    texto = re.sub(r"[^a-záéíóúãõâêîôûàèìòùç\s]", " ", texto)
    texto = re.sub(r"\s+", " ", texto).strip()
    return texto

# -------------------------------------------------------
# PREDIÇÃO
# -------------------------------------------------------

def preverTextos(modelo, vetorizador, textos):
    """Limpa, vetoriza e prediz. Retorna lista de inteiros (0 ou 1)."""
    textos_limpos = [limparTexto(t) for t in textos]
    X = vetorizador.transform(textos_limpos)
    return list(modelo.predict(X))

# -------------------------------------------------------
# AVALIAÇÃO
# -------------------------------------------------------

def avaliarResultados(labels_reais, labels_preditos, le):
    """Avalia e imprime métricas. Usa le.classes_ para os nomes reais."""
    acuracia = accuracy_score(labels_reais, labels_preditos)
    nomes_classes = list(le.classes_)

    print("\n" + "=" * 55)
    print("        RESULTADO DO TESTE DE PREDIÇÃO")
    print("=" * 55)
    print(f"  Total testado  : {len(labels_reais)} textos")
    print(f"  Acurácia geral : {acuracia:.4f}  ({acuracia*100:.2f}%)")
    print("=" * 55)

    print("\nRelatório de Classificação:")
    print(classification_report(
        labels_reais,
        labels_preditos,
        labels=list(range(len(nomes_classes))),
        target_names=nomes_classes,
        zero_division=0    # silencia o UndefinedMetricWarning
    ))

    print("Matriz de Confusão:")
    cm = confusion_matrix(labels_reais, labels_preditos,
                          labels=list(range(len(nomes_classes))))
    idx_cols = [f"Real: {c}" for c in nomes_classes]
    df_cm = pd.DataFrame(cm,
                         index=[f"Real: {c}" for c in nomes_classes],
                         columns=[f"Pred: {c}" for c in nomes_classes])
    print(df_cm)

def mostrarExemplos(textos, labels_reais, labels_preditos, le, n=10):
    """Exibe n exemplos aleatórios marcando erros com [✗]."""
    mapa_inv = {i: c for i, c in enumerate(le.classes_)}
    print(f"\n--- {n} exemplos aleatórios ---")
    indices = np.random.choice(len(textos), size=min(n, len(textos)), replace=False)

    for i in indices:
        real    = mapa_inv[labels_reais[i]]
        predito = mapa_inv[labels_preditos[i]]
        icone   = "✓" if real == predito else "✗"
        trecho  = textos[i][:100].replace("\n", " ")
        print(f"\n[{icone}] Real: {real:6s} | Predito: {predito:6s}")
        print(f"    \"{trecho}...\"")

# -------------------------------------------------------
# PIPELINE PRINCIPAL
# -------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Testa o modelo detector IA vs Humano com amostra do Parquet."
    )
    parser.add_argument("--parquet",     default=NOME_PARQUET)
    parser.add_argument("--modelo",      default=NOME_MODELO)
    parser.add_argument("--vetorizador", default=NOME_VETORIZADOR)
    parser.add_argument("--encoder",     default=NOME_ENCODER)
    parser.add_argument("--n",    type=int, default=N_AMOSTRAS_POR_CLASSE,
                        help="Nº de textos por classe (default: 500)")
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    parser.add_argument("--exemplos", type=int, default=10)
    args = parser.parse_args()

    print("\n" + "=" * 55)
    print("  TESTE DE PREDIÇÃO — Detector IA vs Humano")
    print("=" * 55)

    # 1. carregar artefatos (modelo + vetorizador + LabelEncoder)
    modelo, vetorizador, le = carregarArtefatos(args.modelo, args.vetorizador, args.encoder)
    if modelo is None:
        return

    # 2. carregar amostra do parquet com mapeamento correto
    resultado = carregarAmostraParquet(args.parquet, args.n, args.seed, le)
    if resultado is None:
        return
    textos, labels_reais = resultado

    # 3. predição
    print("\n[INFO] Realizando predições...")
    labels_preditos = preverTextos(modelo, vetorizador, textos)

    # 4. avaliação
    avaliarResultados(labels_reais, labels_preditos, le)

    # 5. exemplos individuais
    mostrarExemplos(textos, labels_reais, labels_preditos, le, n=args.exemplos)

    print("\n[DONE] Teste concluído.")

if __name__ == "__main__":
    main()