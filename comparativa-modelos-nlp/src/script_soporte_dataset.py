# script_soporte_dataset.py

# 1. Instalación de dependencias (si se ejecuta en un entorno nuevo)
# pip install datasets pandas

import pandas as pd
from datasets import load_dataset

def preparar_dataset_sst2(export_path="sst2_validation_sample.csv"):
    """
    Descarga el dataset SST-2, imprime información de su estructura 
    y lo exporta a un archivo CSV.
    """
    print("Iniciando la descarga del dataset SST-2 (versión Stanford)...")
    
    # Cargamos el split de validación para evaluación rápida
    # Usamos el namespace de stanfordnlp para evitar errores de URI obsoletos
    dataset = load_dataset("stanfordnlp/sst2", split="validation")
    
    print("\n--- Estructura del Dataset ---")
    print(dataset)
    print("\nCaracterísticas (Columnas):")
    print(dataset.features)
    
    # Convertimos a un DataFrame de pandas para facilitar la manipulación y exportación  
    df = dataset.to_pandas()
    
    # Mapear las etiquetas numéricas a texto para mayor claridad humana
    mapa_etiquetas = {0: "Negativo", 1: "Positivo"}
    df['label_text'] = df['label'].map(mapa_etiquetas)
    
    print("\n--- Muestra de los primeros 5 registros ---")
    print(df[['idx', 'sentence', 'label', 'label_text']].head())
    
    # Distribución de clases para verificar balance
    print("\n--- Distribución de Clases ---")
    print(df['label_text'].value_counts())
    
    # Exportar a CSV
    df.to_csv(export_path, index=False, encoding='utf-8')
    print(f"\nDataset exportado exitosamente a: {export_path}")

if __name__ == "__main__":
    preparar_dataset_sst2()