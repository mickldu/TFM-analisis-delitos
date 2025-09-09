import pandas as pd
import unicodedata

# 1. Cargar archivos
pib = pd.read_csv("del_pob_pib.csv")
enemu = pd.read_csv("enemu_semanal.csv")

# 2. Normalizar texto (mayúsculas y sin tildes)
def normaliza(texto):
    texto = str(texto).strip().upper()
    texto = ''.join(c for c in unicodedata.normalize('NFD', texto) if unicodedata.category(c) != 'Mn')
    return texto

pib['Provincia_norm'] = pib['Provincia'].apply(normaliza)
if 'Canton' in pib.columns:
    pib['Canton_norm'] = pib['Canton'].apply(normaliza)
else:
    pib['Canton_norm'] = ""
pib['Periodo_norm'] = pd.to_datetime(pib['Periodo'], errors='coerce', dayfirst=False)  # ejemplo: 2012-06-18

# 3. Estandarizar fecha de ENEMDU (varios formatos posibles)
def parse_periodoemu(txt):
    for fmt in ("%d/%m/%Y", "%Y-%m-%d", "%Y/%m/%d"):
        try:
            return pd.to_datetime(str(txt), format=fmt, errors='raise')
        except:
            continue
    return pd.NaT

enemu['Provincia_norm'] = enemu['Provincia'].apply(normaliza)
enemu['PeriodoEMU_norm'] = enemu['PeriodoEMU'].apply(parse_periodoemu)

# 4. Identificar columnas de ENEMDU a agregar
cols_enemu = [c for c in enemu.columns if c not in ['Provincia', 'PeriodoEMU', 'Provincia_norm', 'PeriodoEMU_norm']]

# 5. Para cada fila del base, buscar registro ENEMDU más cercano en fecha por provincia
def encontrar_mas_cercano(row):
    df_prov = enemu[enemu['Provincia_norm'] == row['Provincia_norm']]
    if df_prov.empty:
        return [None]*len(cols_enemu)
    idx = (df_prov['PeriodoEMU_norm'] - row['Periodo_norm']).abs().idxmin()
    return df_prov.loc[idx, cols_enemu].tolist()

# 6. Aplicar la función y formar DataFrame
enemu_match = pib.apply(encontrar_mas_cercano, axis=1, result_type='expand')
enemu_match.columns = [f"{col}_ENEMDU" for col in cols_enemu]

# 7. Concatenar con el base y limpiar auxiliares
final = pd.concat([pib, enemu_match], axis=1)
final.drop(['Provincia_norm', 'Canton_norm', 'Periodo_norm'], axis=1, inplace=True)

# 8. Guardar archivo final
final.to_csv("data_tfm.csv", index=False)

print("¡Archivo generado con match por provincia y fecha más cercana!")
