"""Diagnóstico exploratorio para EAC_CIFRAS_2024_ANONIMIZADA_FINAL.csv.

Replica las revisiones principales de banco_mundial.py adaptadas a la EAC:
faltantes, posibles atípicos, duplicados y caracteres especiales.
No corrige ni modifica el archivo fuente. Guarda reportes en ./diagnostico_EAC.
"""

from collections import Counter
from pathlib import Path
import unicodedata

import numpy as np
import pandas as pd


CARPETA = Path(__file__).resolve().parent
ARCHIVO = CARPETA / "EAC_CIFRAS_2024_ANONIMIZADA_FINAL.csv"
SALIDA = CARPETA / "diagnostico_EAC"
SEPARADOR = ";"
CODIFICACION = "utf-8-sig"
DOMINIO = "CORRELA_16"
ID_EMPRESA = "V1"
MIN_OBSERVACIONES = 8

# Estas son variables de identificación/clasificación, no magnitudes monetarias.
# IDAIO es el año de inicio de operaciones, no un período de observación de panel.
EXCLUIR_ATIPICOS = {ID_EMPRESA, "CORRELA_16", "CORRE_9", "IDOJ1", "IDAIO"}
CARACTERES_PERMITIDOS = set(" -_.,;:/()[]{}'\"%&+@")


def principales_atipicos(datos, columnas_numericas, agrupar_por=None):
    """Devuelve filas candidatas y estadísticas de cada distribución evaluada."""
    hallazgos = []
    resumenes = []
    columnas_grupo = [agrupar_por] if agrupar_por else []
    grupos = datos.groupby(columnas_grupo, dropna=False, sort=False) if columnas_grupo else [("Todos", datos)]

    for etiqueta_grupo, bloque in grupos:
        if columnas_grupo and not isinstance(etiqueta_grupo, tuple):
            etiqueta_grupo = (etiqueta_grupo,)

        for columna in columnas_numericas:
            valores = pd.to_numeric(bloque[columna], errors="coerce").dropna()
            if len(valores) < MIN_OBSERVACIONES:
                continue

            q1 = valores.quantile(0.25)
            q3 = valores.quantile(0.75)
            iqr = q3 - q1
            inferior = q1 - 1.5 * iqr
            superior = q3 + 1.5 * iqr
            mediana = valores.median()
            mad = (valores - mediana).abs().median()
            serie_numerica = pd.to_numeric(bloque[columna], errors="coerce")
            mascara = (serie_numerica < inferior) | (serie_numerica > superior)
            filas = bloque.loc[mascara]

            resumen = {
                "variable": columna,
                "observaciones_evaluadas": int(len(valores)),
                "atipicos_iqr": int(mascara.sum()),
                "porcentaje_atipico": round(100 * int(mascara.sum()) / len(valores), 3),
                "atipicos_inferiores": int((serie_numerica < inferior).sum()),
                "atipicos_superiores": int((serie_numerica > superior).sum()),
                "minimo": valores.min(),
                "percentil_5": valores.quantile(0.05),
                "q1": q1,
                "mediana": mediana,
                "q3": q3,
                "percentil_95": valores.quantile(0.95),
                "maximo": valores.max(),
                "iqr": iqr,
                "mad": mad,
                "limite_inferior_iqr": inferior,
                "limite_superior_iqr": superior,
            }
            if columnas_grupo:
                resumen[agrupar_por] = etiqueta_grupo[0]
            if ID_EMPRESA in datos.columns:
                resumen["empresas_unicas_evaluadas"] = int(bloque.loc[serie_numerica.notna(), ID_EMPRESA].nunique())
                resumen["empresas_unicas_atipicas"] = int(filas[ID_EMPRESA].nunique()) if ID_EMPRESA in filas else 0
            resumenes.append(resumen)

            for indice, fila in filas.iterrows():
                registro = {
                    "fila_csv": int(indice) + 2,
                    "variable": columna,
                    "valor": fila[columna],
                    "limite_inferior": inferior,
                    "limite_superior": superior,
                    "n_grupo": int(len(valores)),
                    "mediana_grupo": mediana,
                    "mad_grupo": mad,
                }
                if columnas_grupo:
                    registro[agrupar_por] = fila[agrupar_por]
                if ID_EMPRESA in datos.columns:
                    registro[ID_EMPRESA] = fila[ID_EMPRESA]
                hallazgos.append(registro)

    return pd.DataFrame(hallazgos), pd.DataFrame(resumenes)


def revisar_caracteres(datos):
    """Cuenta símbolos inusuales en columnas de texto y conserva ejemplos."""
    conteos = Counter()
    ejemplos = []
    columnas_texto = datos.select_dtypes(include=["object", "string"]).columns

    for columna in columnas_texto:
        for indice, valor in datos[columna].dropna().items():
            texto = str(valor)
            for caracter in texto:
                if not caracter.isalnum() and not caracter.isspace() and caracter not in CARACTERES_PERMITIDOS:
                    conteos[(columna, caracter)] += 1
                    if len(ejemplos) < 500:
                        ejemplos.append({
                            "fila_csv": int(indice) + 2,
                            "columna": columna,
                            "valor": texto,
                            "caracter": caracter,
                            "categoria_unicode": unicodedata.category(caracter),
                        })

    resumen = pd.DataFrame([
        {"columna": columna, "caracter": caracter, "apariciones": cantidad}
        for (columna, caracter), cantidad in conteos.most_common()
    ], columns=["columna", "caracter", "apariciones"])
    return resumen, pd.DataFrame(ejemplos)


def main():
    if not ARCHIVO.exists():
        raise FileNotFoundError(f"No se encontró el CSV: {ARCHIVO}")
    SALIDA.mkdir(exist_ok=True)

    # El archivo usa punto y coma entre campos y coma decimal en los números.
    df = pd.read_csv(
        ARCHIVO,
        sep=SEPARADOR,
        decimal=",",
        encoding=CODIFICACION,
        low_memory=False,
    )
    df.columns = [str(col).strip() for col in df.columns]

    # 1. Faltantes por variable
    faltantes = pd.DataFrame({
        "variable": df.columns,
        "faltantes": df.isna().sum().values,
        "porcentaje": (df.isna().mean().values * 100).round(3),
        "tipo_leido": [str(df[col].dtype) for col in df.columns],
    }).sort_values("porcentaje", ascending=False)
    faltantes.to_csv(SALIDA / "faltantes_por_variable.csv", index=False, encoding="utf-8-sig")

    # Se excluyen identificadores y categorías. IDAIO se conserva para revisar
    # su rango, pero no se trata como magnitud continua ni como año de panel.
    columnas_numericas = [
        col for col in df.select_dtypes(include=[np.number]).columns
        if col not in EXCLUIR_ATIPICOS
    ]

    # 2. Candidatos IQR globales por variable
    atipicos_globales, resumen_global = principales_atipicos(df, columnas_numericas)
    atipicos_globales.to_csv(SALIDA / "atipicos_globales.csv", index=False, encoding="utf-8-sig")
    resumen_global.to_csv(SALIDA / "resumen_atipicos_globales_por_variable.csv", index=False, encoding="utf-8-sig")

    # 3. Candidatos IQR dentro de cada dominio CIIU anonimizado (CORRELA_16)
    if DOMINIO in df.columns:
        atipicos_dominio, resumen_dominio = principales_atipicos(df, columnas_numericas, agrupar_por=DOMINIO)
    else:
        atipicos_dominio = pd.DataFrame()
        resumen_dominio = pd.DataFrame()
    atipicos_dominio.to_csv(SALIDA / "atipicos_por_dominio.csv", index=False, encoding="utf-8-sig")
    resumen_dominio.to_csv(SALIDA / "resumen_atipicos_por_variable_dominio.csv", index=False, encoding="utf-8-sig")

    # 4. Duplicados: filas idénticas y repetición del identificador anonimizado.
    duplicados_exactos = df[df.duplicated(keep=False)].copy()
    duplicados_exactos.insert(0, "fila_csv", duplicados_exactos.index + 2)
    duplicados_exactos.to_csv(SALIDA / "filas_duplicadas_exactas.csv", index=False, encoding="utf-8-sig")

    if ID_EMPRESA in df.columns:
        duplicados_id = df[df.duplicated(subset=[ID_EMPRESA], keep=False)].copy()
        duplicados_id.insert(0, "fila_csv", duplicados_id.index + 2)
    else:
        duplicados_id = pd.DataFrame()
    duplicados_id.to_csv(SALIDA / "identificadores_duplicados.csv", index=False, encoding="utf-8-sig")

    # 5. Caracteres especiales en campos textuales, como códigos de dominio.
    resumen_caracteres, ejemplos_caracteres = revisar_caracteres(df)
    resumen_caracteres.to_csv(SALIDA / "caracteres_especiales.csv", index=False, encoding="utf-8-sig")
    ejemplos_caracteres.to_csv(SALIDA / "ejemplos_caracteres_especiales.csv", index=False, encoding="utf-8-sig")

    print(f"Archivo leído: {ARCHIVO}")
    print(f"Dimensiones: {len(df):,} filas × {len(df.columns)} columnas")
    print(f"Faltantes totales: {int(df.isna().sum().sum()):,}")
    print(f"Variables numéricas evaluadas: {len(columnas_numericas)}")
    print(f"Atípicos globales candidatos: {len(atipicos_globales):,}")
    print(f"Atípicos candidatos por dominio: {len(atipicos_dominio):,}")
    print(f"Empresas únicas con algún atípico por dominio: {atipicos_dominio[ID_EMPRESA].nunique() if ID_EMPRESA in atipicos_dominio else 0:,}")
    if not resumen_dominio.empty:
        print("\nVariables con mayor porcentaje de candidatos por dominio:")
        top = resumen_dominio.groupby("variable", as_index=False).agg(
            observaciones=("observaciones_evaluadas", "sum"),
            atipicos=("atipicos_iqr", "sum"),
            empresas_atipicas=("empresas_unicas_atipicas", "sum"),
        )
        top["porcentaje"] = 100 * top["atipicos"] / top["observaciones"]
        print(top.sort_values("porcentaje", ascending=False).head(10).to_string(index=False))
    print(f"Filas duplicadas exactas: {len(duplicados_exactos):,}")
    print(f"Filas con identificador V1 repetido: {len(duplicados_id):,}")
    print(f"Tipos de caracteres inusuales: {len(resumen_caracteres):,}")
    print(f"\nReportes guardados en: {SALIDA}")
    print("Nota: los atípicos son candidatos estadísticos. DANE advierte que los microdatos anonimizados pueden incluir microagregación y perturbación.")


if __name__ == "__main__":
    main()
