# -*- coding: utf-8 -*-
"""
Unificación: delitos_semanal.csv + poblacion_proyectada_semanal.csv
- Normaliza PROVINCIA y CANTÓN (mayúsculas, sin tildes, equivalencias).
- Si población no tiene 'Periodo', lo calcula a partir de Año/Semana o de FECHA/PERIODO.
- Fuzzy matching (threshold=85) para resolver variantes de nombres.
- Silencia el warning de fuzzywuzzy cuando no está python-Levenshtein.
Salida: delitos_poblacion_semanal.csv
"""

import pandas as pd
import numpy as np
import warnings
from pathlib import Path
from unidecode import unidecode

# Silenciar warning de fuzzywuzzy sobre SequenceMatcher lento
warnings.filterwarnings('ignore', message='Using slow pure-python SequenceMatcher.*')

try:
    from fuzzywuzzy import process, fuzz
except Exception as e:
    raise ImportError(
        "Falta 'fuzzywuzzy'. Instala con: pip install fuzzywuzzy python-Levenshtein\n"
        f"Detalle: {e}"
    )

BASE_DIR = Path(__file__).resolve().parent
IN_DELITOS = BASE_DIR / "delitos_semanal.csv"
IN_POB = BASE_DIR / "poblacion_proyectada_semanal.csv"
OUT_FILE = BASE_DIR / "delitos_poblacion_semanal.csv"

# ---------------------------
# Helpers
# ---------------------------
def norm(text: str) -> str:
    if pd.isna(text):
        return ""
    t = unidecode(str(text)).upper().strip()
    rep = {
        "  ": " ",
        "(DISTRITO METROPOLITANO)": "",
        "( DISTRITO METROPOLITANO )": "",
        "( NUEVA LOJA )": "",
        "( PUYO )": "",
        "( FRANCISCO DE ORELLANA )": "",
        " CANTON ": " ",
    }
    for a, b in rep.items():
        t = t.replace(a, b)
    return " ".join(t.split())

# Equivalencias manuales (se aplican DESPUÉS de norm)
PROV_EQ = {
    "QUITO DISTRITO METROPOLITANO": "PICHINCHA",
    "DISTRITO METROPOLITANO DE QUITO": "PICHINCHA",
    "GALAPAGOS": "GALÁPAGOS",
}

CANTON_EQ = {
    "QUITO DISTRITO METROPOLITANO": "QUITO",
    "DISTRITO METROPOLITANO DE QUITO": "QUITO",
    "PUERTO FRANCISCO DE ORELLANA": "ORELLANA",
    "FRANCISCO DE ORELLANA": "ORELLANA",
    "LAGO AGRIO NUEVA LOJA": "NUEVA LOJA",
    "LAGO AGRIO": "NUEVA LOJA",
    "SAN MIGUEL DE SALCEDO": "SALCEDO",
    "PASTAZA PUYO": "PUYO",
    "SANTO DOMINGO DE LOS TSACHILAS": "SANTO DOMINGO",
    "RUMIÑAHUI": "RUMINAHUI",
}

def apply_equivalences(prov: str, cant: str) -> tuple[str, str]:
    p = PROV_EQ.get(prov, prov)
    c = CANTON_EQ.get(cant, cant)
    return p, c

def fuzzy_match(value: str, choices: list[str], threshold: int = 85, scorer=None) -> str | None:
    if not value or not choices:
        return None
    scorer = scorer or fuzz.token_sort_ratio
    match = process.extractOne(value, choices, scorer=scorer)
    if match and match[1] >= threshold:
        return match[0]
    return None

def ensure_periodo_in_pob(pob: pd.DataFrame) -> pd.DataFrame:
    """
    Asegura la columna 'Periodo' en el DF de población.
    - Si existe FECHA/PERIODO -> to_datetime.
    - Si existe Año (+ Semana) -> calcula el lunes de la semana ISO.
    """
    cols_upper = {c.upper(): c for c in pob.columns}

    # Caso 1: ya hay fecha directamente
    for cand in ["PERIODO", "FECHA", "FECHA_INICIO", "PERIODO_INICIO", "PERIODO_SEMANA"]:
        if cand in cols_upper:
            pob["Periodo"] = pd.to_datetime(pob[cols_upper[cand]], errors="coerce")
            break

    # Caso 2: construir desde Año+Semana
    if "Periodo" not in pob.columns or pob["Periodo"].isna().all():
        y_col = cols_upper.get("AÑO") or cols_upper.get("ANIO") or cols_upper.get("ANO") or cols_upper.get("YEAR")
        w_col = cols_upper.get("SEMANA") or cols_upper.get("WEEK") or cols_upper.get("N_SEMANA")
        if y_col and w_col:
            pob[y_col] = pd.to_numeric(pob[y_col], errors="coerce").astype("Int64")
            pob[w_col] = pd.to_numeric(pob[w_col], errors="coerce").astype("Int64")
            def week_start(row):
                if pd.isna(row[y_col]) or pd.isna(row[w_col]):
                    return pd.NaT
                try:
                    # YYYY-Www-1 (1=Lunes), ISO week
                    return pd.to_datetime(f"{int(row[y_col])}-W{int(row[w_col]):02d}-1",
                                          format="%G-W%V-%u", errors="coerce")
                except Exception:
                    return pd.NaT
            pob["Periodo"] = pob.apply(week_start, axis=1)

    if "Periodo" not in pob.columns or pob["Periodo"].isna().all():
        raise KeyError(
            "No se pudo construir 'Periodo' en población. "
            "Asegura tener (Año y Semana) o una columna de fecha tipo 'PERIODO'/'FECHA'."
        )
    return pob

# ---------------------------
# Carga
# ---------------------------
if not IN_DELITOS.exists():
    raise FileNotFoundError(f"No se encuentra {IN_DELITOS.name}. Ejecute primero tfm_ndd.py")
if not IN_POB.exists():
    raise FileNotFoundError(f"No se encuentra {IN_POB.name}. Ejecute primero tfm_varpob.py")

delitos = pd.read_csv(IN_DELITOS)
pob = pd.read_excel(IN_POB) if IN_POB.suffix.lower() in (".xls", ".xlsx") else pd.read_csv(IN_POB)

# Asegurar Periodo en ambos
delitos["Periodo"] = pd.to_datetime(delitos["Periodo"], errors="coerce")
delitos = delitos.dropna(subset=["Periodo"])

pob = ensure_periodo_in_pob(pob)
pob["Periodo"] = pd.to_datetime(pob["Periodo"], errors="coerce")
pob = pob.dropna(subset=["Periodo"])

# Normalizar y aplicar equivalencias en DELITOS
for col in ["d_provincia_incidente", "d_canton_incidente"]:
    if col not in delitos.columns:
        raise KeyError(f"Falta columna en delitos: {col}")

delitos["prov_norm"] = delitos["d_provincia_incidente"].map(norm)
delitos["cant_norm"] = delitos["d_canton_incidente"].map(norm)
delitos[["prov_norm", "cant_norm"]] = delitos.apply(
    lambda r: pd.Series(apply_equivalences(r["prov_norm"], r["cant_norm"])), axis=1
)

# Detectar columnas de Provincia/Cantón en POBLACIÓN
pob_cols = {c.upper(): c for c in pob.columns}
prov_col = pob_cols.get("PROVINCIA") or pob_cols.get("D_PROVINCIA_INCIDENTE") or pob_cols.get("PROV")
canton_col = pob_cols.get("CANTON") or pob_cols.get("D_CANTON_INCIDENTE") or pob_cols.get("CANT")

if not prov_col or not canton_col:
    # último intento: buscar nombres que contengan "PROV" y "CANT"
    prov_col = prov_col or next((pob_cols[c] for c in pob_cols if "PROV" in c), None)
    canton_col = canton_col or next((pob_cols[c] for c in pob_cols if "CANT" in c), None)

if not prov_col or not canton_col:
    raise KeyError("No se detectaron columnas de provincia/cantón en población. Renombre a 'Provincia' y 'Canton'.")

# Normalizar y equivalencias en POBLACIÓN
pob["prov_norm"] = pob[prov_col].map(norm)
pob["cant_norm"] = pob[canton_col].map(norm)
pob[["prov_norm", "cant_norm"]] = pob.apply(
    lambda r: pd.Series(apply_equivalences(r["prov_norm"], r["cant_norm"])), axis=1
)

# Listas de referencia para fuzzy
prov_choices = sorted(pob["prov_norm"].dropna().unique().tolist())
cant_choices = sorted(pob["cant_norm"].dropna().unique().tolist())

# Fuzzy match (threshold=85)
delitos["prov_match"] = delitos["prov_norm"].apply(lambda x: fuzzy_match(x, prov_choices, threshold=85))
delitos["cant_match"] = delitos["cant_norm"].apply(lambda x: fuzzy_match(x, cant_choices, threshold=85))

# Diagnóstico
prov_unmatched = delitos["prov_match"].isna().sum()
cant_unmatched = delitos["cant_match"].isna().sum()
if prov_unmatched or cant_unmatched:
    print(f"""[AVISO] Coincidencias no encontradas (threshold=85):
    - Provincias sin match: {prov_unmatched}
    - Cantones sin match:   {cant_unmatched}
    Sugerencia: agregue equivalencias a PROV_EQ / CANTON_EQ o baje el umbral a 80.
    """)

# Preparar merge por Periodo + matches difusos
left = delitos.rename(columns={"prov_match": "prov_norm", "cant_match": "cant_norm"})
right = pob.copy()

# Determinar columna de población a exportar (si existe)
pob_pop_col = None
for candidate in ["Poblacion", "POBLACION", "poblacion", "Total_Poblacion", "TOTAL_POBLACION", "POBLACION_TOTAL"]:
    if candidate in right.columns:
        pob_pop_col = candidate
        break

# Si hay múltiples filas por Periodo-prov-cant (p.ej., por grupos), agregamos población
if pob_pop_col:
    right = (
        right.groupby(["Periodo", "prov_norm", "cant_norm"], as_index=False)
             .agg({pob_pop_col: "sum"})
    )
else:
    right = right.drop_duplicates(subset=["Periodo", "prov_norm", "cant_norm"])

# Merge final
merged = pd.merge(
    left,
    right,
    on=["Periodo", "prov_norm", "cant_norm"],
    how="left",
    validate="m:1"
)

# Orden y salida
cols_out = [
    "Periodo",
    "d_provincia_incidente", "d_provincia_incidente_cod",
    "d_canton_incidente", "d_canton_incidente_cod",
    "d_delito", "Total_delitos"
]
if pob_pop_col and pob_pop_col in merged.columns:
    cols_out.append(pob_pop_col)

# Agregar columnas de control
cols_out += ["prov_norm", "cant_norm"]

# Filtrar columnas existentes
cols_out = [c for c in cols_out if c in merged.columns]

merged.sort_values(
    ["Periodo", "d_provincia_incidente", "d_canton_incidente", "d_delito"],
    inplace=True,
    na_position="last"
)
merged.to_csv(OUT_FILE, index=False, encoding="utf-8")
print(f"✅ Unificación lista: {OUT_FILE.name} (filas: {len(merged)})")
