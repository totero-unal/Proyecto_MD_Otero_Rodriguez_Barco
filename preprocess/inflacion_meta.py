"""Diagnóstico exploratorio de Inflación y meta.csv.

Revisa faltantes, candidatos a atípicos, fechas, duplicados y caracteres
inusuales. No corrige ni modifica la base original. Produce reportes CSV en
la carpeta diagnostico_inflacion_meta, junto al archivo de entrada.
"""

from collections import Counter
from pathlib import Path
import unicodedata

import numpy as np
import pandas as pd


CARPETA = Path(__file__).resolve().parent
ARCHIVO = CARPETA / "Inflación y meta.csv"
SALIDA = CARPETA / "diagnostico_inflacion_meta"
SEPARADOR = ";"
DECIMAL = ","
CODIFICACION = "utf-8-sig"
FECHA = "Periodo(MMM, AAAA)"
MIN_OBSERVACIONES = 8
CARACTERES_PERMITIDOS = set(" -_.,;:/()[]{}'\"%&+@")
MARCADORES_MOJIBAKE = ("Ã", "Â", "â€", "ðŸ", "�")


def crear_fila_atipico(variable, indice, fecha, valor, inferior, superior, valores):
    fila = {
        "fila_csv": int(indice) + 2,
        "variable": variable,
        "periodo": fecha,
        "valor": valor,
        "limite_inferior_iqr": inferior,
        "limite_superior_iqr": superior,
        "n_observaciones": int(valores.count()),
    }
    return fila


def evaluar_iqr(df, columnas, por_decada=False):
    """IQR por serie; opcionalmente calcula límites independientes por década."""
    hallazgos, resumenes = [], []
    grupos_tiempo = None
    if por_decada:
        grupos_tiempo = (df[FECHA_PARSED].dt.year // 10) * 10

    for variable in columnas:
        agrupaciones = [("Toda_la_serie", df)]
        if por_decada:
            agrupaciones = [
                (f"{int(decada)}s", bloque)
                for decada, bloque in df.groupby(grupos_tiempo, dropna=False, sort=True)
            ]

        for periodo_grupo, bloque in agrupaciones:
            valores = pd.to_numeric(bloque[variable], errors="coerce").dropna()
            if len(valores) < MIN_OBSERVACIONES:
                continue

            q1, q3 = valores.quantile([0.25, 0.75])
            mediana = valores.median()
            iqr = q3 - q1
            inferior, superior = q1 - 1.5 * iqr, q3 + 1.5 * iqr
            serie = pd.to_numeric(bloque[variable], errors="coerce")
            mascara = (serie < inferior) | (serie > superior)
            marcados = bloque.loc[mascara]
            mad = (valores - mediana).abs().median()

            resumenes.append({
                "variable": variable,
                "periodo_grupo": periodo_grupo,
                "observaciones": int(len(valores)),
                "atipicos_iqr": int(mascara.sum()),
                "porcentaje_atipico": round(100 * int(mascara.sum()) / len(valores), 3),
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
            })

            for indice, fila in marcados.iterrows():
                registro = crear_fila_atipico(
                    variable, indice, fila[FECHA], fila[variable], inferior, superior, valores
                )
                registro["grupo_temporal"] = periodo_grupo
                registro["mediana_grupo"] = mediana
                registro["mad_grupo"] = mad
                hallazgos.append(registro)

    return pd.DataFrame(hallazgos), pd.DataFrame(resumenes)


def revisar_caracteres(df):
    conteos = Counter()
    ejemplos = []
    textos = df.select_dtypes(include=["object", "string"])

    for columna in textos.columns:
        for indice, valor in textos[columna].dropna().items():
            texto = str(valor)
            for caracter in texto:
                if not caracter.isalnum() and not caracter.isspace() and caracter not in CARACTERES_PERMITIDOS:
                    conteos[(columna, caracter)] += 1
                    if len(ejemplos) < 300:
                        ejemplos.append({
                            "fila_csv": int(indice) + 2,
                            "columna": columna,
                            "valor": texto,
                            "caracter": caracter,
                            "categoria_unicode": unicodedata.category(caracter),
                        })

    resumen = pd.DataFrame([
        {"columna": c, "caracter": x, "apariciones": n}
        for (c, x), n in conteos.most_common()
    ], columns=["columna", "caracter", "apariciones"])
    return resumen, pd.DataFrame(ejemplos)


def main():
    if not ARCHIVO.exists():
        raise FileNotFoundError(f"No se encontró el archivo: {ARCHIVO}")
    SALIDA.mkdir(exist_ok=True)

    # La base es CSV separado por ; y usa coma como separador decimal.
    df = pd.read_csv(
        ARCHIVO,
        sep=SEPARADOR,
        decimal=DECIMAL,
        encoding=CODIFICACION,
        low_memory=False,
    )
    df.columns = [str(c).strip() for c in df.columns]
    if FECHA not in df.columns:
        raise ValueError(f"No se encontró la columna de período esperada: {FECHA}")

    df[FECHA_PARSED] = pd.to_datetime(df[FECHA], format="%Y/%m/%d", errors="coerce")
    columnas_valor = [c for c in df.select_dtypes(include=[np.number]).columns]

    # 1. Faltantes por campo, incluyendo el porcentaje.
    faltantes = pd.DataFrame({
        "variable": df.columns.drop(FECHA_PARSED),
        "faltantes": df.drop(columns=[FECHA_PARSED]).isna().sum().values,
        "porcentaje": (df.drop(columns=[FECHA_PARSED]).isna().mean().values * 100).round(3),
    }).sort_values("porcentaje", ascending=False)
    faltantes.to_csv(SALIDA / "faltantes_por_variable.csv", index=False, encoding="utf-8-sig")

    # Detalle de faltantes por período y resumen específico para la meta.
    faltante_meta = df.loc[df["Meta de inflación"].isna(), [FECHA]].copy()
    faltante_meta.to_csv(SALIDA / "periodos_sin_meta.csv", index=False, encoding="utf-8-sig")
    fechas_meta = df.loc[df["Meta de inflación"].notna(), FECHA_PARSED]
    primera_meta = fechas_meta.min() if not fechas_meta.empty else pd.NaT
    resumen_meta = pd.DataFrame([{
        "filas_totales": len(df),
        "faltantes_meta": int(df["Meta de inflación"].isna().sum()) if "Meta de inflación" in df else None,
        "primera_fecha_con_meta": primera_meta,
        "faltantes_antes_de_primera_meta": int(
            (df[FECHA_PARSED] < primera_meta).sum()
        ) if pd.notna(primera_meta) else None,
        "faltantes_desde_primera_meta": int(
            (df.loc[df[FECHA_PARSED] >= primera_meta, "Meta de inflación"].isna()).sum()
        ) if pd.notna(primera_meta) else None,
    }])
    resumen_meta.to_csv(SALIDA / "resumen_cobertura_meta.csv", index=False, encoding="utf-8-sig")

    # 2. Formato/calendario: fechas inválidas, meses duplicados o ausentes.
    fechas_invalidas = df[df[FECHA_PARSED].isna()].copy()
    fechas_invalidas[[FECHA]].to_csv(SALIDA / "fechas_invalidas.csv", index=False, encoding="utf-8-sig")
    duplicados_periodo = df[df.duplicated(subset=[FECHA], keep=False)].copy()
    duplicados_periodo.to_csv(SALIDA / "periodos_duplicados.csv", index=False, encoding="utf-8-sig")
    validas = df[FECHA_PARSED].dropna()
    meses_ausentes = pd.DataFrame(columns=["periodo_ausente"])
    if not validas.empty:
        calendario = pd.date_range(validas.min(), validas.max(), freq="ME")
        presentes = pd.DatetimeIndex(validas.dt.to_period("M").dt.to_timestamp("M"))
        meses_ausentes = pd.DataFrame({"periodo_ausente": calendario.difference(presentes).strftime("%Y/%m/%d")})
    meses_ausentes.to_csv(SALIDA / "meses_ausentes_en_calendario.csv", index=False, encoding="utf-8-sig")

    # 3. Candidatos IQR para cada una de las series y por década.
    atipicos_globales, resumen_global = evaluar_iqr(df, columnas_valor)
    atipicos_globales.to_csv(SALIDA / "atipicos_por_serie.csv", index=False, encoding="utf-8-sig")
    resumen_global.to_csv(SALIDA / "resumen_atipicos_por_serie.csv", index=False, encoding="utf-8-sig")
    atipicos_decada, resumen_decada = evaluar_iqr(df, columnas_valor, por_decada=True)
    atipicos_decada.to_csv(SALIDA / "atipicos_por_serie_y_decada.csv", index=False, encoding="utf-8-sig")
    resumen_decada.to_csv(SALIDA / "resumen_atipicos_por_serie_y_decada.csv", index=False, encoding="utf-8-sig")

    # 4. Filas exactamente duplicadas (la columna auxiliar de fecha se excluye).
    datos_fuente = df.drop(columns=[FECHA_PARSED])
    duplicados_exactos = datos_fuente[datos_fuente.duplicated(keep=False)].copy()
    duplicados_exactos.insert(0, "fila_csv", duplicados_exactos.index + 2)
    duplicados_exactos.to_csv(SALIDA / "filas_duplicadas_exactas.csv", index=False, encoding="utf-8-sig")

    # 5. Caracteres inusuales y señales típicas de texto mal decodificado.
    resumen_caracteres, ejemplos_caracteres = revisar_caracteres(datos_fuente)
    resumen_caracteres.to_csv(SALIDA / "caracteres_especiales.csv", index=False, encoding="utf-8-sig")
    ejemplos_caracteres.to_csv(SALIDA / "ejemplos_caracteres_especiales.csv", index=False, encoding="utf-8-sig")
    mojibake = []
    for columna in datos_fuente.select_dtypes(include=["object", "string"]).columns:
        for indice, valor in datos_fuente[columna].dropna().items():
            texto = str(valor)
            if any(marca in texto for marca in MARCADORES_MOJIBAKE):
                mojibake.append({"fila_csv": int(indice) + 2, "columna": columna, "valor": texto})
    pd.DataFrame(mojibake).to_csv(SALIDA / "posible_mojibake.csv", index=False, encoding="utf-8-sig")

    # 6. Reporte resumido en consola.
    print(f"Archivo: {ARCHIVO}")
    print(f"Dimensiones: {len(df):,} filas × {len(datos_fuente.columns)} columnas")
    print(f"Fechas inválidas: {len(fechas_invalidas):,}; períodos duplicados: {len(duplicados_periodo):,}; meses ausentes: {len(meses_ausentes):,}")
    print(f"Faltantes en Meta de inflación: {int(df['Meta de inflación'].isna().sum()):,} / {len(df):,}")
    print(f"Primera observación con meta: {primera_meta.date() if pd.notna(primera_meta) else 'sin datos'}")
    print(f"Candidatos IQR en serie completa: {len(atipicos_globales):,}")
    print(f"Candidatos IQR dentro de década: {len(atipicos_decada):,}")
    print(f"Filas exactamente duplicadas: {len(duplicados_exactos):,}")
    print(f"Tipos de caracteres inusuales: {len(resumen_caracteres):,}; posibles casos de codificación: {len(mojibake):,}")
    print(f"\nReportes guardados en: {SALIDA}")
    print("Los candidatos IQR son alertas estadísticas, no errores confirmados.")


if __name__ == "__main__":
    FECHA_PARSED = "_fecha_convertida"
    main()
