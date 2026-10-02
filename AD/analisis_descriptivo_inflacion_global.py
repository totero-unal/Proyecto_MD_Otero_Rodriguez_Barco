"""Descriptivos, gráficos y reglas para la base global de inflación en largo.

Usa Inflation-data-largo-v2.csv, conserva solo observaciones anuales por país,
y separa las notas estructurales sin país/serie/tipo. No modifica la fuente.
"""
from pathlib import Path
import numpy as np
import pandas as pd
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns
    HAS_PLOTS = True
except ImportError:
    HAS_PLOTS = False

ROOT = Path(__file__).resolve().parent
INPUT = ROOT / "Inflation-data-largo-v2.csv"
OUT = ROOT / "analisis_inflacion_global"
SERIES_KEYS = ["Hoja", "Country Code", "Indicator Type", "Series Name"]


def stats(x):
    x = pd.to_numeric(x, errors="coerce").dropna()
    return {"n": len(x), "media": x.mean(), "mediana": x.median(), "desv_std": x.std(),
        "min": x.min(), "p01": x.quantile(.01), "p05": x.quantile(.05), "q1": x.quantile(.25),
        "q3": x.quantile(.75), "p95": x.quantile(.95), "p99": x.quantile(.99), "max": x.max(),
        "iqr": x.quantile(.75)-x.quantile(.25), "asimetria": x.skew(), "negativos": int(x.lt(0).sum())}


def outliers_by_series(df):
    events, summaries = [], []
    for key, part in df.groupby(SERIES_KEYS, dropna=False, sort=False):
        x = pd.to_numeric(part["Valor"], errors="coerce").dropna()
        if len(x) < 8: continue
        q1, q3 = x.quantile([.25, .75]); iqr = q3-q1; median = x.median(); mad = (x-median).abs().median()
        lo, hi = q1-1.5*iqr, q3+1.5*iqr
        y = pd.to_numeric(part["Valor"], errors="coerce")
        mask = y.lt(lo) | y.gt(hi)
        summaries.append({**dict(zip(SERIES_KEYS, key)), "n_observado": len(x), "atipicos_iqr": int(mask.sum()),
            "porcentaje": 100*mask.sum()/len(x), "min": x.min(), "p05": x.quantile(.05), "q1": q1,
            "mediana": median, "q3": q3, "p95": x.quantile(.95), "max": x.max(), "iqr": iqr,
            "mad": mad, "limite_inf": lo, "limite_sup": hi})
        for idx, row in part.loc[mask].iterrows():
            events.append({"fila_csv": idx+2, **dict(zip(SERIES_KEYS, key)), "Periodo": row["Periodo"],
                "Valor": row["Valor"], "limite_inf": lo, "limite_sup": hi, "mediana_serie": median, "mad_serie": mad})
    return pd.DataFrame(events), pd.DataFrame(summaries)


def main():
    OUT.mkdir(exist_ok=True)
    raw = pd.read_csv(INPUT, encoding="utf-8-sig", low_memory=False)
    if "Frecuencia" not in raw or "Valor" not in raw:
        raise ValueError("Se esperaba un CSV largo con columnas Frecuencia y Valor.")
    annual_all = raw.loc[raw["Frecuencia"].astype(str).str.casefold().eq("anual")].copy()
    note_rows = annual_all[annual_all[["Country Code", "Indicator Type", "Series Name"]].isna().any(axis=1)].copy()
    note_rows.to_csv(OUT / "filas_notas_o_sin_identificadores.csv", index=False, encoding="utf-8-sig")
    df = annual_all.dropna(subset=["Country Code", "Indicator Type", "Series Name"]).copy()
    df["Periodo"] = pd.to_numeric(df["Periodo"], errors="coerce")
    df["Valor"] = pd.to_numeric(df["Valor"], errors="coerce")

    # 1. Estadísticos robustos y convencionales, por hoja/tipo.
    records = []
    for key, block in df.groupby(["Hoja", "Indicador", "Indicator Type"], dropna=False):
        records.append({"Hoja": key[0], "Indicador": key[1], "Indicator Type": key[2], **stats(block["Valor"]),
                        "faltantes_valor": int(block["Valor"].isna().sum()), "pct_faltante_valor": 100*block["Valor"].isna().mean(),
                        "paises": int(block["Country Code"].nunique()), "periodo_min": block["Periodo"].min(), "periodo_max": block["Periodo"].max()})
    desc = pd.DataFrame(records)
    desc.to_csv(OUT / "descriptivos_por_indicador_tipo.csv", index=False, encoding="utf-8-sig")

    missing = (df.groupby(["Hoja", "Indicador", "Indicator Type"], dropna=False)
        .agg(filas=("Valor", "size"), faltantes=("Valor", lambda x: int(x.isna().sum())),
             no_faltantes=("Valor", "count"), paises=("Country Code", "nunique"))
        .reset_index())
    missing["porcentaje_faltante"] = 100*missing["faltantes"]/missing["filas"]
    missing.to_csv(OUT / "faltantes_por_indicador_tipo.csv", index=False, encoding="utf-8-sig")

    # Cobertura y distribución por año: cuenta observaciones y países con valor.
    coverage = (df.groupby(["Hoja", "Indicator Type", "Periodo"], dropna=False)
        .agg(filas=("Valor", "size"), observaciones=("Valor", "count"), paises_con_dato=("Country Code", lambda x: x[df.loc[x.index, "Valor"].notna()].nunique()))
        .reset_index())
    coverage["porcentaje_faltante"] = 100*(coverage["filas"]-coverage["observaciones"])/coverage["filas"]
    coverage.to_csv(OUT / "cobertura_por_anio.csv", index=False, encoding="utf-8-sig")

    # 2. Posibles atípicos longitudinales por país y serie; faltantes se omiten solo del cálculo.
    events, series_summary = outliers_by_series(df)
    events.to_csv(OUT / "candidatos_atipicos_por_serie.csv", index=False, encoding="utf-8-sig")
    series_summary.to_csv(OUT / "resumen_atipicos_por_serie.csv", index=False, encoding="utf-8-sig")

    # Agregados de eventos por variable/indicador para interpretar el conteo.
    if not events.empty:
        (events.groupby(["Hoja", "Indicator Type"], dropna=False)
            .agg(eventos_atipicos=("Valor", "size"), series_afectadas=("Country Code", "nunique"))
            .reset_index().to_csv(OUT / "resumen_atipicos_por_indicador.csv", index=False, encoding="utf-8-sig"))

    # 3. Duplicados: observación clave y fila exactamente repetida.
    exact = df[df.duplicated(keep=False)].copy(); exact.insert(0, "fila_csv", exact.index+2)
    exact.to_csv(OUT / "filas_duplicadas_exactas.csv", index=False, encoding="utf-8-sig")
    observation_key = SERIES_KEYS + ["Periodo"]
    key_duplicates = df[df.duplicated(observation_key, keep=False)].copy(); key_duplicates.insert(0, "fila_csv", key_duplicates.index+2)
    key_duplicates.to_csv(OUT / "observaciones_duplicadas_por_clave.csv", index=False, encoding="utf-8-sig")
    invalid_period = df[df["Periodo"].isna() | ~df["Periodo"].between(1900, 2100)].copy()
    invalid_period.to_csv(OUT / "periodos_invalidos.csv", index=False, encoding="utf-8-sig")

    rules = [
        ["Observación", "Una fila por país, indicador, tipo de serie y año", "Validar clave Hoja + Country Code + Indicator Type + Series Name + Periodo como única."],
        ["Notas", "Filas con identificadores de serie vacíos pueden ser notas al pie", "Separar notas de observaciones; no contarlas como valores faltantes del panel."],
        ["Frecuencia", "Anual para esta descripción", "Filtrar Frecuencia='anual'; no mezclar con trimestral/mensual."],
        ["Valor faltante", "Cobertura desigual por país/serie/año", "Mantener nulos y reportar cobertura por año; no imputar automáticamente."],
        ["Valor", "Inflación/tasas con colas extremas y posibles episodios de hiperinflación", "Admitir valores negativos; no fijar tope universal ni eliminar extremos IQR sin validación histórica."],
        ["Atípicos", "IQR dentro de cada país-serie, mínimo 8 datos", "Usar candidatos para revisión; reportar mediana/IQR/MAD y hacer sensibilidad con/sin extremos."],
        ["Año final", "La fuente puede incluir estimaciones/proyecciones para algunos países en 2024", "Revisar notas/fuente del registro y marcar el estado de 2024 antes de compararlo con históricos."],
    ]
    pd.DataFrame(rules, columns=["tema", "contexto", "regla_sugerida"]).to_csv(OUT / "reglas_calidad_sugeridas.csv", index=False, encoding="utf-8-sig")

    # 4. Figuras: tendencia mediana, amplitud intercuartílica, cobertura y distribución comprimida.
    if HAS_PLOTS:
        sns.set_theme(style="whitegrid")
        yearly = (df.dropna(subset=["Valor", "Periodo"]).groupby(["Hoja", "Indicator Type", "Periodo"])
            .agg(mediana=("Valor", "median"), q1=("Valor", lambda x: x.quantile(.25)), q3=("Valor", lambda x: x.quantile(.75)))
            .reset_index())
        yearly["serie"] = yearly["Hoja"] + " / " + yearly["Indicator Type"]
        fig, ax = plt.subplots(figsize=(14, 7))
        for name, g in yearly.groupby("serie"):
            ax.plot(g["Periodo"], g["mediana"], lw=1.2, label=name)
        ax.set_title("Inflación anual: mediana entre países por indicador y tipo")
        ax.set_xlabel("Año"); ax.set_ylabel("Valor reportado"); ax.legend(fontsize=7, ncol=2)
        fig.tight_layout(); fig.savefig(OUT / "mediana_anual_por_indicador.png", dpi=170); plt.close(fig)

        coverage["serie"] = coverage["Hoja"] + " / " + coverage["Indicator Type"].astype(str)
        fig, ax = plt.subplots(figsize=(14, 6))
        for name, g in coverage.groupby("serie"):
            ax.plot(g["Periodo"], g["paises_con_dato"], lw=1.0, label=name)
        ax.set_title("Cobertura anual: países con valor observado"); ax.set_xlabel("Año"); ax.set_ylabel("Países")
        ax.legend(fontsize=7, ncol=2); fig.tight_layout(); fig.savefig(OUT / "cobertura_paises_por_anio.png", dpi=170); plt.close(fig)

        dist = df.dropna(subset=["Valor"]).copy()
        dist["valor_signed_log"] = np.sign(dist["Valor"]) * np.log1p(np.abs(dist["Valor"]))
        dist["serie"] = dist["Hoja"] + " / " + dist["Indicator Type"]
        fig, ax = plt.subplots(figsize=(14, 6))
        sns.boxplot(data=dist, x="serie", y="valor_signed_log", ax=ax, showfliers=False)
        ax.set_title("Distribuciones de valores (transformación signed log; sin puntos extremos)")
        ax.set_xlabel(""); ax.set_ylabel("signo(x) × log(1 + |x|)"); ax.tick_params(axis="x", rotation=45)
        fig.tight_layout(); fig.savefig(OUT / "distribuciones_signed_log.png", dpi=170); plt.close(fig)

    note_count = len(note_rows)
    print(f"Base anual limpia: {len(df):,} filas; notas/sin identificadores separadas: {note_count:,}.")
    print(f"Faltantes de Valor: {int(df['Valor'].isna().sum()):,} ({100*df['Valor'].isna().mean():.2f}%).")
    print(f"Indicador/tipo analizados: {len(desc)}; países: {df['Country Code'].nunique():,}; años {df['Periodo'].min():.0f}-{df['Periodo'].max():.0f}.")
    print(f"Candidatos IQR por país-serie (mínimo 8 valores): {len(events):,}; filas con clave duplicada: {len(key_duplicates):,}.")
    print(f"Salidas: {OUT}")
    if not HAS_PLOTS:
        print("Gráficos omitidos en este entorno; instala matplotlib y seaborn para generarlos.")


if __name__ == "__main__":
    main()
