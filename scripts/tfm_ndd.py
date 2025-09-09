import pandas as pd
import csv
from pathlib import Path

INPUT = "ndd_datos.csv"
OUTPUT = "delitos_semanal.csv"

# Mapeo de columnas esperadas desde exportaciones de Elasticsearch/Kibana
COLMAP = {
    "d_PROVINCIA_INCIDENTE.keyword: Descending": "d_provincia_incidente",
    "d_PROVINCIA_INCIDENTE_COD.keyword: Descending": "d_provincia_incidente_cod",
    "d_CANTON_INCIDENTE.keyword: Descending": "d_canton_incidente",
    "d_CANTON_INCIDENTE_COD.keyword: Descending": "d_canton_incidente_cod",
    "d_DELITO.keyword: Descending": "d_delito",
    "PERIODO": "Periodo",
    "TOTAL": "Total_delitos",
}

def main():
    if not Path(INPUT).exists():
        raise FileNotFoundError(f"No se encuentra el archivo: {INPUT}")

    # Detectar delimitador
    with open(INPUT, "r", encoding="utf-8") as f:
        sample = f.read(2048)
        dialect = csv.Sniffer().sniff(sample)

    df = pd.read_csv(INPUT, delimiter=dialect.delimiter)

    # Renombrar columnas según el mapeo
    faltantes = [c for c in COLMAP if c not in df.columns]
    if faltantes:
        raise ValueError(f"Columnas faltantes en el archivo de entrada: {faltantes}")

    df = df.rename(columns=COLMAP)

    # Tipos de datos
    df["Periodo"] = pd.to_datetime(df["Periodo"], errors="coerce")
    df = df.dropna(subset=["Periodo"])

    df["d_provincia_incidente_cod"] = df["d_provincia_incidente_cod"].astype(str).str.zfill(2)
    df["d_canton_incidente_cod"] = df["d_canton_incidente_cod"].astype(str).str.zfill(4)

    # Normalizar textos
    for col in ["d_provincia_incidente", "d_canton_incidente", "d_delito"]:
        df[col] = df[col].astype(str).str.strip()

    # Reagrupar por seguridad
    group_cols = [
        "Periodo",
        "d_provincia_incidente", "d_provincia_incidente_cod",
        "d_canton_incidente", "d_canton_incidente_cod",
        "d_delito",
    ]
    df = (
        df.groupby(group_cols, as_index=False)
          .agg(Total_delitos=("Total_delitos", "sum"))
    )

    # Compatibilidad con pipelines anteriores
    df["Estado_procesal_dominante"] = pd.NA

    # Guardar
    df.to_csv(OUTPUT, index=False)
    print(f"✅ Archivo generado: {OUTPUT} (filas: {len(df)})")

if __name__ == "__main__":
    main()