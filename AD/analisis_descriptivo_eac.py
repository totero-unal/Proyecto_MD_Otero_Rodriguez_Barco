"""Descriptivos, visualizaciones y reglas sugeridas para EAC 2024 anonimizada.

Entrada: EAC_CIFRAS_2024_ANONIMIZADA_FINAL.csv
Salidas: carpeta analisis_EAC_descriptivo (tablas CSV y gráficos PNG).
El archivo de entrada nunca se modifica.
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
INPUT = ROOT / "EAC_CIFRAS_2024_ANONIMIZADA_FINAL.csv"
OUT = ROOT / "analisis_EAC_descriptivo"
SEP, DECIMAL, ENCODING = ";", ",", "utf-8-sig"
ID, DOMAIN, LEGAL, START_YEAR = "V1", "CORRELA_16", "IDOJ1", "IDAIO"
LEGAL_CODES = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 99}
WORKFORCE = {
    "PERREMUN",
    "SOCIOS",
    "PERSONOM",
    "DIRECTO",
    "AGENCIA",
    "APRENDIZ",
    "TOTPERSO",
    "PROMUJ",
    "PROHOM",
    "PERMUJ",
    "PERHOM",
    "DIRMUJ",
    "DIRHOM",
    "AGEMUJ",
    "AGEHOM",
    "APRENMUJ",
    "APRENHOM",
    "TOTMUJ",
    "TOTHOM",
}
MONEY_PLOT = ["BRUTA", "VENTA", "AGREGA", "GASTOS", "GASTOSNOP", "GASTOPNOP"]


def write_rules(df, metrics):
    rules = [
        [
            "Identificación",
            ID,
            "Un identificador por fila; no calcular media ni marcar como magnitud atípica.",
            "V1 único y no nulo; investigar cualquier repetición.",
        ],
        [
            "Clasificación",
            DOMAIN,
            "Código categórico de dominio CIIU anonimizado.",
            "Validar contra catálogo/versionado DANE; no tratar el código como magnitud numérica.",
        ],
        [
            "Clasificación",
            LEGAL,
            "Tipo de organización jurídica; códigos documentados 1–14 y 99.",
            "Aceptar solo códigos del catálogo; los faltantes se dejan como faltantes o categoría analítica separada, nunca imputar moda por defecto.",
        ],
        [
            "Fecha/antigüedad",
            START_YEAR,
            "Año de inicio de operaciones.",
            "Requerir entero y no posterior a 2024; valores anteriores a 1800 deben revisarse. No usar como año de encuesta/panel.",
        ],
        [
            "Magnitudes",
            "Variables financieras y de personal",
            "Montos publicados en miles de pesos y conteos de personal.",
            "Exigir valores numéricos finitos; marcar negativos para revisión semántica, no fijar topes máximos globales.",
        ],
        [
            "Atípicos",
            "Variables numéricas por CORRELA_16",
            "Se reporta IQR por dominio y variable; no equivale a error.",
            "Revisar primero ceros estructurales, IQR=0 y consistencia entre totales/componentes; conservar extremos plausibles.",
        ],
        [
            "Modelado",
            "Montos sesgados",
            "Distribuciones con cola derecha y muchos ceros.",
            "Preferir mediana/IQR o estimadores robustos; para modelos, evaluar log1p solo en variables no negativas y guardar la transformación aparte.",
        ],
        [
            "Confidencialidad",
            "Microdatos EAC",
            "La anonimización puede incluir microagregación y perturbación.",
            "No corregir un candidato estadístico usando supuestos de empresa real; para evolución, priorizar agregados comparables publicados por DANE.",
        ],
    ]
    pd.DataFrame(
        rules, columns=["tema", "campo", "evidencia", "regla_sugerida"]
    ).to_csv(OUT / "reglas_calidad_sugeridas.csv", index=False, encoding="utf-8-sig")


def main():
    OUT.mkdir(exist_ok=True)
    df = pd.read_csv(
        INPUT, sep=SEP, decimal=DECIMAL, encoding=ENCODING, low_memory=False
    )
    df.columns = df.columns.astype(str).str.strip()
    numeric = [
        c
        for c in df.select_dtypes(include=np.number).columns
        if c not in {ID, LEGAL, START_YEAR}
    ]

    missing = pd.DataFrame(
        {
            "variable": df.columns,
            "faltantes": df.isna().sum().values,
            "porcentaje": (df.isna().mean().values * 100).round(3),
            "tipo": [str(df[c].dtype) for c in df.columns],
        }
    )
    missing.to_csv(OUT / "faltantes.csv", index=False, encoding="utf-8-sig")
    records = []
    for c in numeric:
        x = pd.to_numeric(df[c], errors="coerce").dropna()
        records.append(
            {
                "variable": c,
                "n": len(x),
                "faltantes": int(df[c].isna().sum()),
                "media": x.mean(),
                "mediana": x.median(),
                "desv_std": x.std(),
                "q1": x.quantile(0.25),
                "q3": x.quantile(0.75),
                "iqr": x.quantile(0.75) - x.quantile(0.25),
                "p95": x.quantile(0.95),
                "p99": x.quantile(0.99),
                "min": x.min(),
                "max": x.max(),
                "asimetria": x.skew(),
                "ceros_pct": 100 * (x.eq(0).mean()),
                "negativos": int(x.lt(0).sum()),
            }
        )
    stats = pd.DataFrame(records)
    stats.to_csv(
        OUT / "descriptivos_variables_numericas.csv", index=False, encoding="utf-8-sig"
    )

    cats = []
    for c in [DOMAIN, "CORRE_9", LEGAL]:
        if c in df:
            vc = df[c].value_counts(dropna=False)
            cats.extend(
                {
                    "variable": c,
                    "categoria": str(k),
                    "n": int(v),
                    "porcentaje": 100 * v / len(df),
                }
                for k, v in vc.items()
            )
    pd.DataFrame(cats).to_csv(
        OUT / "frecuencias_categoricas.csv", index=False, encoding="utf-8-sig"
    )

    # Edad de operación y códigos legales: validación de rangos/catálogo.
    if START_YEAR in df:
        years = pd.to_numeric(df[START_YEAR], errors="coerce")
        pd.DataFrame(
            {
                "fila_csv": np.arange(len(df)) + 2,
                ID: df[ID],
                START_YEAR: years,
                "fuera_rango_sugerido": years.notna()
                & ((years < 1800) | (years > 2024)),
            }
        ).query("fuera_rango_sugerido").to_csv(
            OUT / "revisar_anios_inicio.csv", index=False, encoding="utf-8-sig"
        )
    if LEGAL in df:
        code = pd.to_numeric(df[LEGAL], errors="coerce")
        invalid = df[code.notna() & ~code.isin(LEGAL_CODES)].copy()
        invalid.insert(0, "fila_csv", invalid.index + 2)
        invalid.to_csv(
            OUT / "codigos_idoj1_fuera_catalogo.csv", index=False, encoding="utf-8-sig"
        )

    # Atípicos empíricos por dominio: resumen completo y detalle; no filtra la base.
    out_rows, summary = [], []
    for domain, block in df.groupby(DOMAIN, dropna=False, sort=False):
        for c in numeric:
            x = pd.to_numeric(block[c], errors="coerce")
            valid = x.dropna()
            if len(valid) < 8:
                continue
            q1, q3 = valid.quantile([0.25, 0.75])
            iqr = q3 - q1
            lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
            mask = x.lt(lo) | x.gt(hi)
            med = valid.median()
            mad = (valid - med).abs().median()
            summary.append(
                {
                    "CORRELA_16": domain,
                    "variable": c,
                    "n": len(valid),
                    "candidatos_iqr": int(mask.sum()),
                    "pct_candidatos": 100 * mask.sum() / len(valid),
                    "min": valid.min(),
                    "p5": valid.quantile(0.05),
                    "q1": q1,
                    "mediana": med,
                    "q3": q3,
                    "p95": valid.quantile(0.95),
                    "max": valid.max(),
                    "iqr": iqr,
                    "mad": mad,
                    "limite_inf": lo,
                    "limite_sup": hi,
                }
            )
            for idx in block.index[mask]:
                out_rows.append(
                    {
                        "fila_csv": idx + 2,
                        ID: df.at[idx, ID],
                        "CORRELA_16": domain,
                        "variable": c,
                        "valor": df.at[idx, c],
                        "limite_inf": lo,
                        "limite_sup": hi,
                        "mediana_grupo": med,
                        "mad_grupo": mad,
                        "iqr_cero": bool(iqr == 0),
                    }
                )
    sdom = pd.DataFrame(summary)
    outliers = pd.DataFrame(out_rows)
    sdom.to_csv(
        OUT / "resumen_atipicos_variable_dominio.csv", index=False, encoding="utf-8-sig"
    )
    outliers.to_csv(
        OUT / "candidatos_atipicos_variable_dominio.csv",
        index=False,
        encoding="utf-8-sig",
    )
    if not outliers.empty:
        outliers.groupby("variable").agg(
            observaciones_atipicas=("valor", "size"),
            empresas=(ID, "nunique"),
            grupos_iqr_cero=("iqr_cero", "sum"),
        ).sort_values("observaciones_atipicas", ascending=False).to_csv(
            OUT / "resumen_atipicos_por_variable.csv", encoding="utf-8-sig"
        )

    duplicates = df[df.duplicated(keep=False)].copy()
    duplicates.insert(0, "fila_csv", duplicates.index + 2)
    duplicates.to_csv(
        OUT / "filas_duplicadas_exactas.csv", index=False, encoding="utf-8-sig"
    )
    iddups = df[df.duplicated(ID, keep=False)].copy() if ID in df else pd.DataFrame()
    if len(iddups):
        iddups.insert(0, "fila_csv", iddups.index + 2)
    iddups.to_csv(
        OUT / "identificadores_duplicados.csv", index=False, encoding="utf-8-sig"
    )

    rules = [
        [
            "V1",
            "Clave de registro",
            "Requerir no nulo y único; no usar como variable explicativa.",
        ],
        [
            "CORRELA_16 / CORRE_9",
            "Dominio de actividad categórico",
            "Validar con catálogo DANE; no tratar como número continuo.",
        ],
        [
            "IDOJ1",
            "Código legal categórico",
            "Permitir solo códigos del diccionario {1..14, 99}; conservar faltantes explícitos.",
        ],
        [
            "IDAIO",
            "Año inicio actividad",
            "Entero no posterior a 2024; revisar años anteriores a 1800; no confundir con año de encuesta.",
        ],
        [
            "Variables financieras/personal",
            "Montos en miles de pesos y conteos",
            "Numérico finito; revisar negativos con definiciones; no imponer máximo global ni borrar outliers IQR automáticamente.",
        ],
        [
            "Comparabilidad",
            "Microdatos anonimizados",
            "Evitar inferir evolución longitudinal a nivel V1; DANE advierte microagregación/perturbación.",
        ],
        [
            "Distribución",
            "Variables financieras sesgadas y con ceros",
            "Describir con mediana/IQR y cuantiles; para modelar comparar estimadores robustos y log1p en columnas no negativas.",
        ],
    ]
    pd.DataFrame(rules, columns=["campo", "contexto", "regla_sugerida"]).to_csv(
        OUT / "reglas_calidad_sugeridas.csv", index=False, encoding="utf-8-sig"
    )

    if HAS_PLOTS:
        sns.set_theme(style="whitegrid")
        cols = [c for c in MONEY_PLOT if c in df]
        if cols:
            fig, axes = plt.subplots(2, 3, figsize=(14, 8))
            axes = axes.ravel()
            for ax, c in zip(axes, cols):
                x = pd.to_numeric(df[c], errors="coerce").dropna()
                x = x[x >= 0]
                sns.histplot(np.log10(x + 1), bins=35, ax=ax)
                ax.set_title(c)
                ax.set_xlabel("log10(valor en miles de pesos + 1)")
            for ax in axes[len(cols) :]:
                ax.remove()
            fig.suptitle("EAC 2024: distribución de montos seleccionados (escala log)")
            fig.tight_layout()
            fig.savefig(OUT / "distribuciones_montos_log.png", dpi=160)
            plt.close(fig)
        workforce_cols = [
            c
            for c in ["TOTPERSO", "PERSONOM", "DIRECTO", "AGENCIA", "APRENDIZ"]
            if c in df
        ]
        if workforce_cols:
            fig, ax = plt.subplots(figsize=(10, 5))
            vals = [
                pd.to_numeric(df[c], errors="coerce").dropna() for c in workforce_cols
            ]
            ax.boxplot(vals, label=workforce_cols, showfliers=False)
            ax.set_yscale("symlog", linthresh=1)
            ax.set_title("Personal ocupado (sin mostrar puntos extremos)")
            ax.set_ylabel("Personas, escala symlog")
            fig.tight_layout()
            fig.savefig(OUT / "distribucion_personal.png", dpi=160)
            plt.close(fig)
        if LEGAL in df:
            fig, ax = plt.subplots(figsize=(9, 5))
            df[LEGAL].value_counts(dropna=False).sort_index().plot.bar(ax=ax)
            ax.set_title("Tipo de organización jurídica (IDOJ1)")
            ax.set_xlabel("Código DANE")
            ax.set_ylabel("Empresas")
            fig.tight_layout()
            fig.savefig(OUT / "frecuencia_idoj1.png", dpi=160)
            plt.close(fig)

    print(
        f"EAC: {len(df):,} filas; {len(numeric)} variables cuantitativas; faltantes totales {int(df.isna().sum().sum()):,}."
    )
    print(
        f"Candidatos IQR por dominio: {len(outliers):,} eventos en {outliers[ID].nunique() if not outliers.empty else 0:,} identificadores V1."
    )
    print(
        f"Filas duplicadas exactas: {len(duplicates):,}; V1 repetidos: {len(iddups):,}. Salidas: {OUT}"
    )
    if not HAS_PLOTS:
        print(
            "Gráficos omitidos en este entorno; instala matplotlib y seaborn para generarlos."
        )


if __name__ == "__main__":
    main()
