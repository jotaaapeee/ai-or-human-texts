# ai-or-human-texts

# Detector de Texto: IA vs Humano

Projeto de Machine Learning para classificar textos como sendo escritos por humanos ou por inteligência artificial.

---

## Objetivo

Desenvolver e avaliar modelos de machine learning capazes de distinguir entre textos gerados por IA e textos escritos por humanos, analisando padrões linguísticos, estruturais e estatísticos.

---

## Datasets

Foram utilizados os seguintes datasets:

- Hugging Face:  
  https://huggingface.co/datasets/dmitva/human_ai_generated_text  

- Kaggle:  
  https://www.kaggle.com/datasets/hasanyiitakbulut/ai-and-human-text-dataset  

Os dados contêm pares de textos (`human_text` e `ai_text`) baseados na mesma instrução (`instructions`), geralmente no formato de redações argumentativas.

---

## Pipeline do Projeto

O projeto segue as etapas clássicas de Machine Learning:

1. Contextualização do dataset  
2. Carregamento dos dados (Pandas)  
3. Tratamento e limpeza  
4. Análise exploratória  
5. Divisão em treino, validação e teste  
6. Treinamento de modelos  
7. Avaliação e seleção  
8. Persistência e inferência  

---

## Pré-processamento

- Normalização de texto (minúsculas, remoção de caracteres especiais)  
- Remoção de dados inválidos  
- Truncamento de textos (controle de viés de tamanho)  
- Vetorização com TF-IDF  

---

## Modelos Utilizados

- Regressão Logística  
- Rede Neural (MLP)  

---

## Resultados

- Acurácia: ~99.97% – 100%  
- F1-score macro: ~0.999  

Apesar da alta performance, análises adicionais mostraram que o modelo se apoia fortemente em padrões superficiais do dataset.

---

## Análise Crítica

O modelo apresenta alta acurácia devido a características específicas dos dados, como:

- Diferença de tamanho entre textos  
- Estilo de escrita (formal vs informal)  
- Frequência de palavras  

Isso indica que o modelo pode não generalizar bem para cenários reais.

---

## Insights Importantes

- Um modelo simples usando apenas o tamanho do texto já atinge ~78% de acurácia  
- Mesmo após reduzir esse viés, o modelo mantém alta performance  
- O problema está mais relacionado ao dataset do que à complexidade do modelo  

---