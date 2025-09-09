# -*- coding: utf-8 -*-
"""
TFM - Unificación semanal por Semana–Provincia–Cantón–Delito con variables exógenas (versión optimizada)
Autor: (tu nombre)
Fecha: 2025-08-13

Objetivo
--------
Unificar datasets semanales con clave final:
    [Periodo (lunes ISO), Provincia, Canton, Delito]
e integrar variables exógenas (población, ENEMDU, PIB), más ingeniería de rasgos
(tasas, lags, rolling y calendarios), sin nulos y con reporte de calidad.

Cambios clave (esta versión)
----------------------------
- ENEMDU ahora prioriza **enemu_semanal.csv** (unificado semanal) para simplificar y robustecer.
- Fuzzy matching optimizado (trabaja con únicos, exact match previo, caché).
- Solo usa 'poblacion_proyectada_semanal.csv' para población (se eliminó XLSX en población).
- Parche en ENEMDU: forward-fill con **groupby(...).transform(...)** (sin KeyError).
- Mantiene: PIB, lags/rolling, calendario, imputación y reporte.

Requisitos recomendados:
    pip install fuzzywuzzy python-Levenshtein holidays unidecode
"""

import warnings
from pathlib import Path
from typing import List, Tuple, Optional, Dict

import numpy as np
import pandas as pd
from unidecode import unidecode

warnings.filterwarnings('ignore', message='Using slow pure-python SequenceMatcher.*')

# -------------------------------------------------------------
# Librerías opcionales: feriados y fuzzywuzzy
# -------------------------------------------------------------
try:
    import holidays
    HAVE_HOLIDAYS = True
except Exception:
    HAVE_HOLIDAYS = False

try:
    from fuzzywuzzy import process, fuzz
    HAVE_FUZZY = True
except Exception:
    HAVE_FUZZY = False

# -------------------------------------------------------------
# Configuración de rutas y parámetros
# -------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent

# Entradas
IN_DELITOS = BASE_DIR / "delitos_semanal.csv"
IN_POB_CSV = BASE_DIR / "poblacion_proyectada_semanal.csv"  # <- ÚNICA FUENTE DE POBLACIÓN
IN_ENEMDU_PRIOR = BASE_DIR / "enemu_semanal.csv"            # <- NUEVA PRIORIDAD
IN_ENEMDU_FALLB = BASE_DIR / "ENEMDU_UNIFICADO_SEMANAL.csv" # <- respaldo si falta el prior
IN_PIB_SEM = BASE_DIR / "PIB_semanal.csv"
IN_PIB_ANUAL = BASE_DIR / "PBI_1965_2023.xlsx"

# Salidas
OUT_MASTER = BASE_DIR / "del_master_tfm.csv"
OUT_REPORT_MD = BASE_DIR / "report_calidad.md"
OUT_REPORT_HTML = BASE_DIR / "report_calidad.html"

# Parámetros de features
FUZZY_THRESHOLD = 85        # umbral para fuzzy matching
WINDOWS_MEAN = [4, 8, 12]   # ventanas de rolling mean
WINDOWS_STD = [4, 8, 12]    # ventanas de rolling std
LAGS = [1, 2, 4]            # lags de semanas

# Equivalencias manuales (se aplican DESPUÉS de normalizar texto)
PROV_EQ = {
    "QUITO DISTRITO METROPOLITANO": "PICHINCHA",
    "DISTRITO METROPOLITANO DE QUITO": "PICHINCHA",
    "GALAPAGOS": "GALAPAGOS",
    "GALÁPAGOS": "GALAPAGOS",
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
    "ORELLA": "ORELLANA",
}

# -------------------------------------------------------------
# Utilidades de normalización y fechas
# -------------------------------------------------------------
def norm_text(text: str) -> str:
    """Normaliza texto: sin tildes, MAYÚSCULAS, colapsa espacios y limpia patrones comunes."""
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

def apply_equivalences(prov: str, cant: str) -> Tuple[str, str]:
    """Aplica equivalencias manuales a provincia y cantón ya normalizados."""
    p = PROV_EQ.get(prov, prov)
    c = CANTON_EQ.get(cant, cant)
    return p, c

def to_monday_period(date_like) -> pd.Timestamp:
    """Convierte fecha a lunes ISO (inicio de semana)."""
    dt = pd.to_datetime(date_like, errors="coerce")
    if pd.isna(dt):
        return pd.NaT
    weekday = dt.weekday()  # Monday=0
    return (dt - pd.Timedelta(days=weekday)).normalize()

def ensure_periodo(
    df: pd.DataFrame,
    prefer_cols: List[str] = None,
    year_col_candidates=("AÑO","ANIO","ANO","YEAR","ANIO_CALENDARIO"),
    week_col_candidates=("SEMANA","WEEK","N_SEMANA","SEMANA_ISO","SEM")
) -> pd.DataFrame:
    """
    Asegura columna 'Periodo' (lunes ISO) con una estrategia en cascada:
      1) Columnas conocidas de fecha.
      2) Heurística: columna con mayor % parseable a fecha.
      3) Campo tipo YYYYWW (6 dígitos).
      4) Construir desde Año + Semana (ISO).
    """
    prefer_cols = prefer_cols or [
        "PERIODO","FECHA","FECHA_LUNES","PERIODOEMU","PERIODO_INICIO",
        "PERIODO_SEMANA","PERIODO_LUNES","FECHA_INI","LUNES","DATE"
    ]
    cols_upper = {c.upper(): c for c in df.columns}

    # 1) columna de fecha conocida
    chosen = None
    for cand in prefer_cols:
        if cand in cols_upper:
            chosen = cols_upper[cand]
            break
    if chosen:
        df["Periodo"] = pd.to_datetime(df[chosen], errors="coerce").map(to_monday_period)

    # 2) heurística de columna más "fecha"
    if "Periodo" not in df.columns or df["Periodo"].isna().all():
        best_col, best_rate = None, 0.0
        for c in df.columns:
            ser = pd.to_datetime(df[c], errors="coerce")
            rate = ser.notna().mean()
            if rate > best_rate:
                best_col, best_rate = c, rate
        if best_col and best_rate >= 0.7:
            df["Periodo"] = pd.to_datetime(df[best_col], errors="coerce").map(to_monday_period)

    # 3) columna estilo YYYYWW (6 dígitos)
    if "Periodo" not in df.columns or df["Periodo"].isna().all():
        for c in df.columns:
            s = pd.to_numeric(df[c], errors="coerce").astype("Int64")
            mask = s.notna() & s.astype(str).str.len().eq(6)
            if mask.mean() > 0.7:
                def parse_yw(v):
                    try:
                        v = int(v); y = v // 100; w = v % 100
                        return pd.to_datetime(f"{y}-W{int(w):02d}-1", format="%G-W%V-%u", errors="coerce")
                    except Exception:
                        return pd.NaT
                df["Periodo"] = s.map(parse_yw)
                break

    # 4) construir desde Año + Semana
    if "Periodo" not in df.columns or df["Periodo"].isna().all():
        y_col = next((cols_upper.get(c) for c in year_col_candidates if c in cols_upper), None)
        w_col = next((cols_upper.get(c) for c in week_col_candidates if c in cols_upper), None)
        if y_col and w_col:
            df[y_col] = pd.to_numeric(df[y_col], errors="coerce").astype("Int64")
            df[w_col] = pd.to_numeric(df[w_col], errors="coerce").astype("Int64")
            def week_start(row):
                if pd.isna(row[y_col]) or pd.isna(row[w_col]):
                    return pd.NaT
                try:
                    return pd.to_datetime(f"{int(row[y_col])}-W{int(row[w_col]):02d}-1",
                                          format="%G-W%V-%u", errors="coerce")
                except Exception:
                    return pd.NaT
            df["Periodo"] = df.apply(week_start, axis=1)

    if "Periodo" not in df.columns or df["Periodo"].isna().all():
        raise KeyError("No se pudo construir 'Periodo'. Incluye fecha, (Año+Semana) o campo tipo YYYYWW.")
    return df

def infer_geo_columns(df: pd.DataFrame) -> Tuple[str, str]:
    """Detecta columnas de provincia y cantón por alias; lanza error si no encuentra."""
    cols_upper = {c.upper(): c for c in df.columns}
    prov_col = cols_upper.get("PROVINCIA") or cols_upper.get("D_PROVINCIA_INCIDENTE") or cols_upper.get("PROV")
    cant_col = cols_upper.get("CANTON") or cols_upper.get("D_CANTON_INCIDENTE") or cols_upper.get("CANT")
    if not prov_col:
        prov_col = next((cols_upper[c] for c in cols_upper if "PROV" in c), None)
    if not cant_col:
        cant_col = next((cols_upper[c] for c in cols_upper if "CANT" in c), None)
    if not prov_col or not cant_col:
        raise KeyError("No se detectaron columnas de 'Provincia' y/o 'Canton'. Renombra tu archivo.")
    return prov_col, cant_col

# -------------------------------------------------------------
# Features de calendario y de series (lags/rolling)
# -------------------------------------------------------------
def add_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    """Agrega año ISO, semana ISO, mes, trimestre, fin_de_mes, feriados (EC) y vacaciones (aprox)."""
    df["anio"] = df["Periodo"].dt.isocalendar().year.astype(int)
    df["semana_anio"] = df["Periodo"].dt.isocalendar().week.astype(int)
    df["mes"] = df["Periodo"].dt.month.astype(int)
    df["trimestre"] = df["Periodo"].dt.quarter.astype(int)
    df["fin_de_mes"] = (df["Periodo"] == (df["Periodo"] + pd.offsets.MonthEnd(0))).astype(int)

    if HAVE_HOLIDAYS:
        start = df["Periodo"].min().date()
        end = df["Periodo"].max().date()
        ec_holidays = holidays.country_holidays("EC", years=range(start.year, end.year+1))
        df["es_feriado"] = df["Periodo"].dt.date.apply(lambda d: int(d in ec_holidays))
    else:
        df["es_feriado"] = 0

    df["vacaciones_escolares"] = 0
    costa = df["mes"].between(2, 4)   # feb-abr
    sierra = df["mes"].between(7, 9)  # jul-sep
    df.loc[costa | sierra, "vacaciones_escolares"] = 1
    return df

def add_lags_and_rollings(df: pd.DataFrame, group_cols: List[str], target_cols: List[str]) -> pd.DataFrame:
    """Calcula lags (1,2,4) y rolling mean/std (4,8,12) por grupo."""
    df = df.sort_values(group_cols + ["Periodo"])
    g = df.groupby(group_cols, group_keys=False)

    for col in target_cols:
        for l in LAGS:
            df[f"{col}_lag{l}"] = g[col].shift(l)
        for w in WINDOWS_MEAN:
            df[f"{col}_roll{w}_mean"] = g[col].rolling(w, min_periods=max(1, w//2)).mean().reset_index(level=group_cols, drop=True)
        for w in WINDOWS_STD:
            df[f"{col}_roll{w}_std"] = g[col].rolling(w, min_periods=max(1, w//2)).std().reset_index(level=group_cols, drop=True)
    return df

def impute_numeric(df: pd.DataFrame, group_cols: List[str]) -> pd.DataFrame:
    """Imputa numéricos: ffill/bfill por grupo y mediana global como último recurso."""
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if not num_cols:
        return df
    df[num_cols] = df.groupby(group_cols)[num_cols].transform(lambda g: g.ffill().bfill())
    for c in num_cols:
        if df[c].isna().any():
            df[c] = df[c].fillna(df[c].median())
    return df

def quality_report(df: pd.DataFrame, out_md: Path, out_html: Path):
    """Reporte simple de calidad: % nulos por columna y duplicados clave."""
    lines = []
    lines.append("# Reporte de Calidad de Datos\n")
    lines.append(f"- Filas: {len(df)}")
    lines.append(f"- Columnas: {len(df.columns)}\n")

    nulls = df.isna().mean().sort_values(ascending=False)
    lines.append("## Porcentaje de nulos por columna (después de imputación)\n")
    lines.append(nulls.to_frame("pct_null").to_markdown())

    dup_count = df.duplicated(subset=["Periodo","Provincia","Canton","Delito"]).sum()
    lines.append(f"\n## Duplicados clave [Periodo, Provincia, Canton, Delito]\n- Duplicados: {dup_count}")

    content = "\n".join(lines)
    out_md.write_text(content, encoding="utf-8")

    html = f"""<!DOCTYPE html>
<html lang="es"><head><meta charset="utf-8"><title>Reporte de Calidad</title>
<style>body{{font-family:Arial, sans-serif; margin:24px;}} pre{{white-space:pre-wrap}}</style></head>
<body><pre>{content}</pre></body></html>"""
    out_html.write_text(html, encoding="utf-8")

# -------------------------------------------------------------
# Carga y normalización de DELITOS
# -------------------------------------------------------------
def load_delitos(path: Path) -> pd.DataFrame:
    """Lee 'delitos_semanal.csv', asegura Periodo (lunes ISO) y normaliza geo."""
    if not path.exists():
        raise FileNotFoundError(f"No se encuentra {path.name}")
    df = pd.read_csv(path)
    df = ensure_periodo(df, prefer_cols=["PERIODO","FECHA"])

    req = ["d_provincia_incidente","d_canton_incidente","d_delito","Total_delitos"]
    faltan = [c for c in req if c not in df.columns]
    if faltan:
        raise KeyError(f"Faltan columnas en delitos: {faltan}")

    df["prov_norm"] = df["d_provincia_incidente"].map(norm_text)
    df["cant_norm"] = df["d_canton_incidente"].map(norm_text)
    df[["prov_norm","cant_norm"]] = df.apply(
        lambda r: pd.Series(apply_equivalences(r["prov_norm"], r["cant_norm"])), axis=1
    )

    out = df.rename(columns={"d_delito": "Delito"})[
        ["Periodo","prov_norm","cant_norm","Delito","Total_delitos",
         "d_provincia_incidente","d_canton_incidente"]
    ].copy()
    out = out.dropna(subset=["Periodo"])
    return out

# -------------------------------------------------------------
# Carga y normalización de POBLACIÓN (solo CSV)
# -------------------------------------------------------------
def load_poblacion_only_csv(p_csv: Path) -> pd.DataFrame:
    """Lee 'poblacion_proyectada_semanal.csv', asegura Periodo y detecta Población; agrega por Periodo-prov-cant."""
    if not p_csv.exists():
        raise FileNotFoundError("No se encontró 'poblacion_proyectada_semanal.csv'.")
    print("[POB] Cargado desde: poblacion_proyectada_semanal.csv")

    pob = pd.read_csv(p_csv)
    pob = ensure_periodo(pob, prefer_cols=["PERIODO","FECHA","FECHA_INICIO","PERIODO_INICIO","PERIODO_SEMANA","FECHA_LUNES"])
    prov_col, cant_col = infer_geo_columns(pob)

    pob["prov_norm"] = pob[prov_col].map(norm_text)
    pob["cant_norm"] = pob[cant_col].map(norm_text)
    pob[["prov_norm","cant_norm"]] = pob.apply(
        lambda r: pd.Series(apply_equivalences(r["prov_norm"], r["cant_norm"])), axis=1
    )

    # Detectar columna de población
    pob_col = None
    for c in ["POBLACION","POBLACION_TOTAL","Poblacion","Total_Poblacion","TOTAL_POBLACION","poblacion"]:
        if c in pob.columns:
            pob_col = c; break
    if not pob_col:
        num_cols = pob.select_dtypes(include=[np.number]).columns.tolist()
        if not num_cols:
            raise KeyError("No se detectó columna de población. Renombra a 'Poblacion'.")
        medians = pob[num_cols].median().sort_values(ascending=False)
        pob_col = medians.index[0]
        print(f"[POB] Usando columna '{pob_col}' como población (heurística).")

    pob = (pob.groupby(["Periodo","prov_norm","cant_norm"], as_index=False)
              .agg({pob_col: "sum"})
              .rename(columns={pob_col: "Poblacion"}))
    return pob

# -------------------------------------------------------------
# Carga y normalización de ENEMDU (prioriza enemu_semanal.csv)
# -------------------------------------------------------------
def load_enemdu_prioritized(path_prior: Path, path_fallback: Path) -> pd.DataFrame:
    """
    Prioriza 'enemu_semanal.csv' (con todas las exógenas ya unificadas semanalmente).
    Si no existe, intenta 'ENEMDU_UNIFICADO_SEMANAL.csv'.
    Asegura Periodo, normaliza geo, y hace forward fill por grupo con transform().
    """
    src = None
    if path_prior.exists():
        src = path_prior
        print("[ENEMDU] Usando fuente prioritaria: enemu_semanal.csv")
        en = pd.read_csv(path_prior)
    elif path_fallback.exists():
        src = path_fallback
        print("[ENEMDU] Usando fuente de respaldo: ENEMDU_UNIFICADO_SEMANAL.csv")
        en = pd.read_csv(path_fallback)
    else:
        print("[ENEMDU] No se encontró fuente (ni enemu_semanal.csv ni ENEMDU_UNIFICADO_SEMANAL.csv).")
        return pd.DataFrame(columns=["Periodo","prov_norm","cant_norm"])

    # Asegurar Periodo (lunes ISO)
    en = ensure_periodo(en, prefer_cols=["FECHA_LUNES","PERIODOEMU","PERIODO","FECHA"])

    # Detectar columnas de geo
    try:
        prov_col, cant_col = infer_geo_columns(en)
    except KeyError:
        cols_upper = {c.upper(): c for c in en.columns}
        prov_col = cols_upper.get("PROVINCIA") or next((cols_upper[c] for c in cols_upper if "PROV" in c), None)
        cant_col = None

    en["prov_norm"] = en[prov_col].map(norm_text) if prov_col else ""
    en["prov_norm"] = en["prov_norm"].apply(lambda x: PROV_EQ.get(x, x))
    if cant_col:
        en["cant_norm"] = en[cant_col].map(norm_text).apply(lambda x: CANTON_EQ.get(x, x))
    else:
        en["cant_norm"] = ""  # algunos ENEMDU vienen a nivel provincia

    # Tomar solo indicadores con prefijos estándar
    indicator_cols = [c for c in en.columns if c.startswith(("edu_","lab_","pob_","viv_"))]
    keep = ["Periodo","prov_norm","cant_norm"] + indicator_cols
    en = en[keep].drop_duplicates(subset=["Periodo","prov_norm","cant_norm"])

    # Forward fill por grupo (provincia o prov-cant) con transform (evita KeyError/FutureWarning)
    if indicator_cols:
        gcols = ["prov_norm"] if en["cant_norm"].eq("").all() else ["prov_norm","cant_norm"]
        en = en.sort_values(gcols + ["Periodo"])
        en[indicator_cols] = en.groupby(gcols)[indicator_cols].transform(lambda g: g.ffill().bfill())

    return en

# -------------------------------------------------------------
# Carga y normalización de PIB
# -------------------------------------------------------------
def load_pib(pib_sem: Path, pib_anual: Path) -> pd.DataFrame:
    """Usa PIB semanal si está disponible; si no, interpola desde anual."""
    if pib_sem.exists():
        pib = pd.read_csv(pib_sem)
        # buscar columna fecha
        date_col = None
        for c in pib.columns:
            if c.lower() in ("fecha","periodo","date"):
                date_col = c; break
        if not date_col:
            date_col = pib.columns[0]
        pib["Periodo"] = pd.to_datetime(pib[date_col], errors="coerce").map(to_monday_period)
        # Detectar valor de PIB
        val_col = None
        for c in pib.columns:
            if c.upper().startswith("TOTAL_PIB") or c.upper() == "TOTAL_PIB":
                val_col = c; break
        if not val_col:
            num_cols = pib.select_dtypes(include=[np.number]).columns.tolist()
            if not num_cols:
                raise KeyError("PIB semanal no tiene columna numérica detectable.")
            val_col = num_cols[0]
        pib = pib[["Periodo", val_col]].rename(columns={val_col: "TOTAL_PIB"})
        pib = pib.dropna(subset=["Periodo"]).groupby("Periodo", as_index=False).agg({"TOTAL_PIB":"mean"})
        print("[PIB] Usando PIB semanal.")
        return pib

    elif pib_anual.exists():
        anual = pd.read_excel(pib_anual)
        # Detectar columnas
        year_col = None
        for c in anual.columns:
            if c.upper() in ("AÑO","ANIO","ANO","YEAR"):
                year_col = c; break
        if not year_col:
            raise KeyError("No se detectó columna de año en PBI anual.")
        anual[year_col] = pd.to_numeric(anual[year_col], errors="coerce")
        anual = anual.dropna(subset=[year_col])
        val_cols = [c for c in anual.columns if c != year_col]
        if not val_cols:
            raise KeyError("No hay columnas de valores en PBI anual.")
        val_col = val_cols[0]

        years = anual[year_col].astype(int)
        start_date = f"{years.min()}-01-01"; end_date = f"{years.max()}-12-31"
        date_range = pd.date_range(start=start_date, end=end_date, freq="W-MON")
        x = np.linspace(years.min(), years.max(), len(date_range))
        y = np.interp(x, years, anual[val_col])
        pib = pd.DataFrame({"Periodo": date_range, "TOTAL_PIB": y})
        print("[PIB] Interpolado desde anual.")
        return pib

    else:
        print("[PIB] No se encontró fuente. Continuo sin PIB.")
        return pd.DataFrame(columns=["Periodo","TOTAL_PIB"])

# -------------------------------------------------------------
# Fuzzy matching optimizado (únicos + exact match + "caché")
# -------------------------------------------------------------
def optimized_fuzzy_map(values: pd.Series, choices: List[str], threshold: int = FUZZY_THRESHOLD) -> Dict[str, str]:
    """
    Devuelve un dict value->match usando:
      1) match exacto (rápido)
      2) fuzzy SOLO sobre los únicos restantes (reduce comparaciones)
    Si no hay fuzzy disponible, retorna solo los exact matches.
    """
    values_unique = pd.Series(sorted(set(values.dropna().tolist())))
    choice_set = set([c for c in choices if isinstance(c, str)])
    mapping: Dict[str, str] = {}

    # 1) Exact match
    exact = values_unique[values_unique.isin(choice_set)]
    for v in exact:
        mapping[v] = v

    # 2) Fuzzy solo para los no mapeados
    to_match = [v for v in values_unique.tolist() if v not in mapping]
    if not to_match or not HAVE_FUZZY:
        return mapping

    for v in to_match:
        m = process.extractOne(v, choices, scorer=fuzz.token_sort_ratio)
        if m and m[1] >= threshold:
            mapping[v] = m[0]
    return mapping

# -------------------------------------------------------------
# Orquestación: Merge maestro y features
# -------------------------------------------------------------
def main():
    # 1) Delitos
    print("Cargando DELITOS...")
    delitos = load_delitos(IN_DELITOS)

    # 2) Población (solo CSV)
    print("Cargando POBLACIÓN (solo CSV)...")
    pob = load_poblacion_only_csv(IN_POB_CSV)

    # Catálogo de referencia desde población (mejor que desde delitos)
    prov_choices = sorted(pob["prov_norm"].dropna().unique().tolist())
    cant_choices = sorted(pob["cant_norm"].dropna().unique().tolist())

    # 3) Fuzzy matching optimizado sobre únicos
    print("Aplicando fuzzy (opt.) Provincias únicas de DELITOS...")
    prov_map = optimized_fuzzy_map(delitos["prov_norm"], prov_choices, FUZZY_THRESHOLD)
    print(f"[Fuzzy Provincias] únicos delitos: {delitos['prov_norm'].nunique()} | mapeados: {len(prov_map)}")

    print("Aplicando fuzzy (opt.) Cantones únicos de DELITOS...")
    cant_map = optimized_fuzzy_map(delitos["cant_norm"], cant_choices, FUZZY_THRESHOLD)
    print(f"[Fuzzy Cantones] únicos delitos: {delitos['cant_norm'].nunique()} | mapeados: {len(cant_map)}")

    # Aplicar mapping; fallback: valor original normalizado si no hubo match
    delitos["Provincia"] = delitos["prov_norm"].map(lambda x: prov_map.get(x, x))
    delitos["Canton"] = delitos["cant_norm"].map(lambda x: cant_map.get(x, x))

    # 4) Agregar delitos por Periodo-Provincia-Canton-Delito
    delitos_agg = (delitos.groupby(["Periodo","Provincia","Canton","Delito"], as_index=False)
                          .agg({"Total_delitos":"sum"}))

    # 5) Unir población a nivel cantonal
    pob = pob.rename(columns={"prov_norm":"Provincia","cant_norm":"Canton"})
    print("Uniendo DELITOS + POBLACIÓN...")
    df = pd.merge(delitos_agg, pob, on=["Periodo","Provincia","Canton"], how="left", validate="m:1")

    # 6) Tasa por 100k
    df["Tasa_delitos_100k"] = (df["Total_delitos"] / df["Poblacion"]) * 100000

    # 7) ENEMDU priorizado (enemu_semanal.csv)
    print("Cargando ENEMDU...")
    enemdu = load_enemdu_prioritized(IN_ENEMDU_PRIOR, IN_ENEMDU_FALLB)
    if len(enemdu):
        enemdu = enemdu.rename(columns={"prov_norm":"Provincia","cant_norm":"Canton"})
        indicator_cols = [c for c in enemdu.columns if c.startswith(("edu_","lab_","pob_","viv_"))]
        df = pd.merge(
            df,
            enemdu[["Periodo","Provincia"] + indicator_cols].drop_duplicates(),
            on=["Periodo","Provincia"],
            how="left"
        )
    else:
        print("AVISO: No se encontró ENEMDU; se continúa sin estas exógenas.")

    # 8) PIB (distribución proporcional a población cantonal por semana)
    pib = load_pib(IN_PIB_SEM, IN_PIB_ANUAL)
    if len(pib):
        print("Distribuyendo PIB proporcional a población cantonal...")
        df = pd.merge(df, pib, on="Periodo", how="left")
        pop_week = df.groupby(["Periodo"], as_index=False)["Poblacion"].sum().rename(columns={"Poblacion":"Poblacion_total_semana"})
        df = pd.merge(df, pop_week, on="Periodo", how="left")
        df["PIB_por_canton"] = (df["TOTAL_PIB"] * (df["Poblacion"] / df["Poblacion_total_semana"]))
    else:
        print("AVISO: No se encontró PIB; se continúa sin PIB.")

    # 9) Features de calendario
    print("Añadiendo features de calendario...")
    df = add_calendar_features(df)

    # 10) Lags y rolling
    print("Añadiendo lags y rolling...")
    df = add_lags_and_rollings(
        df,
        group_cols=["Provincia","Canton","Delito"],
        target_cols=["Total_delitos","Tasa_delitos_100k"]
    )

    # 11) Imputación numérica para evitar nulos
    print("Imputando numéricos...")
    df = impute_numeric(df, group_cols=["Provincia","Canton","Delito"])

    # 12) Orden y exportación
    base_cols = ["Periodo","Provincia","Canton","Delito","Total_delitos","Poblacion","Tasa_delitos_100k","PIB_por_canton"]
    base_cols = [c for c in base_cols if c in df.columns]
    other_cols = [c for c in df.columns if c not in base_cols + ["Poblacion_total_semana","TOTAL_PIB"]]
    df = df[base_cols + other_cols]

    df = df.sort_values(["Periodo","Provincia","Canton","Delito"])
    df.to_csv(OUT_MASTER, index=False, encoding="utf-8")
    print(f"✅ Archivo maestro guardado: {OUT_MASTER} (filas: {len(df)})")

    # 13) Reporte de calidad
    print("Generando reporte de calidad...")
    quality_report(df, OUT_REPORT_MD, OUT_REPORT_HTML)
    print(f"📄 Reporte MD: {OUT_REPORT_MD.name}")
    print(f"🌐 Reporte HTML: {OUT_REPORT_HTML.name}")
    print("Listo.")

# -------------------------------------------------------------
# Punto de entrada
# -------------------------------------------------------------
if __name__ == "__main__":
    main()
