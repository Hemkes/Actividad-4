# Comparativa de Modelos NLP para Clasificación de Sentimientos

## Descripción del Proyecto
Este repositorio contiene un análisis técnico comparativo de tres modelos de Procesamiento de Lenguaje Natural (NLP) basados en la arquitectura Transformer: **DistilBERT**, **BERT** y **RoBERTa**. El objetivo es evaluar su rendimiento en la tarea de clasificación de sentimientos (positivo/negativo) utilizando el dataset **SST-2** (Stanford Sentiment Treebank). Se evalúan métricas de precisión (Accuracy, F1-Score) frente a la latencia de inferencia y consumo de recursos para determinar el mejor modelo para entornos de producción.

## Entorno de Ejecución
- **Lenguaje:** Python 3.9+
- **Hardware Recomendado:** GPU NVIDIA (ej. T4 en Google Colab) para una inferencia acelerada. También es compatible con CPU (aunque con mayor latencia).
- **Librerías principales:** `transformers`, `datasets`, `torch`, `scikit-learn`, `pandas`.

## Pasos de Ejecución

1. **Clonar el repositorio:**
   ```bash
   git clone [https://github.com/TU_USUARIO/comparativa-modelos-nlp.git](https://github.com/TU_USUARIO/comparativa-modelos-nlp.git)
   cd comparativa-modelos-nlp
