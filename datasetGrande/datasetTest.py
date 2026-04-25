import pandas as pd
from collections import Counter
import re

df = pd.read_parquet("model_training_dataset.parquet", engine="pyarrow")

print("=== HUMAN TEXT ===")
for i in range(3):
    print(f"\n[{i}] '{df['human_text'].iloc[i][:300]}'")

print("\n=== AI TEXT ===")
for i in range(3):
    print(f"\n[{i}] '{df['ai_text'].iloc[i][:300]}'")

# 1. Primeiros e últimos caracteres
print("=== PADRÕES DE INÍCIO ===")
print("AI - primeiros 5 chars (repr):")
for i in range(5):
    print(f"  [{i}] {repr(df['ai_text'].iloc[i][:5])}")

print("\nHuman - primeiros 5 chars (repr):")
for i in range(5):
    print(f"  [{i}] {repr(df['human_text'].iloc[i][:5])}")

# 2. Palavras mais comuns exclusivas de cada lado
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np

ai_texts = df['ai_text'].fillna('').str.strip().str.lower().tolist()
human_texts = df['human_text'].fillna('').str.strip().str.lower().tolist()

# top palavras por frequência bruta
ai_words = Counter(" ".join(ai_texts[:500]).split())
human_words = Counter(" ".join(human_texts[:500]).split())

print("\n=== TOP 20 PALAVRAS MAIS COMUNS EM AI ===")
print(ai_words.most_common(20))

print("\n=== TOP 20 PALAVRAS MAIS COMUNS EM HUMAN ===")
print(human_words.most_common(20))

# 3. Comprimento médio
print(f"\n=== COMPRIMENTO MÉDIO ===")
print(f"AI:    {df['ai_text'].str.len().mean():.0f} chars")
print(f"Human: {df['human_text'].str.len().mean():.0f} chars")

# 4. Verificar se instructions vaza info
print("\n=== INSTRUCTIONS (primeiras 3) ===")
for i in range(3):
    print(f"[{i}] {repr(df['instructions'].iloc[i][:200])}")



print("=== DISTRIBUIÇÃO DE COMPRIMENTO ===")
print("AI:")
print(df['ai_text'].str.len().describe())
print("\nHuman:")
print(df['human_text'].str.len().describe())

ai_ends = df['ai_text'].str.strip().str[-1]
human_ends = df['human_text'].str.strip().str[-1]

print("\nAI - últimos chars mais comuns:")
print(ai_ends.value_counts().head(10))
print("\nHuman - últimos chars mais comuns:")
print(human_ends.value_counts().head(10))