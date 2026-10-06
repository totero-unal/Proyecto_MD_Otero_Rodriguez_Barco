"""Descriptivos, gráficos y reglas para Inflación total y Meta (BanRep).

Lee Inflación y meta.csv y escribe reportes en analisis_inflacion_meta.
No edita la base original.
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

ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / "data" / "raw" / "Inflación y meta.csv"
OUT = ROOT / "reports" / "analisis_inflacion_meta"
DATE = "Periodo(MMM, AAAA)"
META = "Meta de inflación"
INFL = "Inflación total anual"


def stats(x, name):
    x = pd.to_numeric(x, errors="coerce").dropna()
    return {
        "variable": name,
        "n": len(x),
        "media": x.mean(),
        "mediana": x.median(),
        "desv_std": x.std(),
        "min": x.min(),
        "p05": x.quantile(0.05),
        "q1": x.quantile(0.25),
        "q3": x.quantile(0.75),
        "p95": x.quantile(0.95),
        "max": x.max(),
        "iqr": x.quantile(0.75) - x.quantile(0.25),
        "asimetria": x.skew(),
        "negativos": int(x.lt(0).sum()),
    }


def iqr_reports(df, variable, period_group=None):
    detail, summary = [], []
    group_iter = (
        [("Serie completa", df)]
        if period_group is None
        else list(df.groupby(period_group, sort=True, dropna=False))
    )
    for label, part in group_iter:
        x = pd.to_numeric(part[variable], errors="coerce").dropna()
        if len(x) < 8:
            continue
        q1, q3 = x.quantile([0.25, 0.75])
        iqr = q3 - q1
        med = x.median()
        mad = (x - med).abs().median()
        lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        mask = part[variable].lt(lo) | part[variable].gt(hi)
        summary.append(
            {
                "variable": variable,
                "grupo": label,
                "n": len(x),
                "candidatos": int(mask.sum()),
                "porcentaje": 100 * mask.sum() / len(x),
                "min": x.min(),
                "p05": x.quantile(0.05),
                "q1": q1,
                "mediana": med,
                "q3": q3,
                "p95": x.quantile(0.95),
                "max": x.max(),
                "iqr": iqr,
                "mad": mad,
                "limite_inf": lo,
                "limite_sup": hi,
            }
        )
        for idx, row in part.loc[mask].iterrows():
            detail.append(
                {
                    "fila_csv": idx + 2,
                    "periodo": row[DATE],
                    "variable": variable,
                    "valor": row[variable],
                    "grupo": label,
                    "limite_inf": lo,
                    "limite_sup": hi,
                }
            )
    return pd.DataFrame(detail), pd.DataFrame(summary)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(
        INPUT, sep=";", decimal=",", encoding="utf-8-sig", low_memory=False
    )
    df["_fecha"] = pd.to_datetime(df[DATE], format="%Y/%m/%d", errors="coerce")
    df = df.sort_values("_fecha").reset_index(drop=True)

    vars_num = [META, INFL]
    pd.DataFrame([stats(df[v], v) for v in vars_num]).to_csv(
        OUT / "descriptivos.csv", index=False, encoding="utf-8-sig"
    )
    missing = pd.DataFrame(
        {
            "variable": vars_num,
            "faltantes": [df[v].isna().sum() for v in vars_num],
            "porcentaje": [100 * df[v].isna().mean() for v in vars_num],
            "no_faltantes": [df[v].notna().sum() for v in vars_num],
        }
    )
    missing.to_csv(OUT / "faltantes.csv", index=False, encoding="utf-8-sig")

    first_meta = df.loc[df[META].notna(), "_fecha"].min()
    coverage = pd.DataFrame(
        [
            {
                "filas": len(df),
                "inicio": df["_fecha"].min(),
                "fin": df["_fecha"].max(),
                "primera_meta": first_meta,
                "faltantes_meta": int(df[META].isna().sum()),
                "faltantes_meta_antes_primera": int((df["_fecha"] < first_meta).sum()),
                "faltantes_meta_desde_primera": int(
                    df.loc[df["_fecha"] >= first_meta, META].isna().sum()
                ),
                "fechas_invalidas": int(df["_fecha"].isna().sum()),
                "periodos_duplicados": int(df[DATE].duplicated().sum()),
            }
        ]
    )
    coverage.to_csv(
        OUT / "cobertura_y_calendario.csv", index=False, encoding="utf-8-sig"
    )
    absent_months = pd.date_range(
        df["_fecha"].min(), df["_fecha"].max(), freq="ME"
    ).difference(
        pd.DatetimeIndex(df["_fecha"].dropna().dt.to_period("M").dt.to_timestamp("M"))
    )
    pd.DataFrame({"mes_ausente": absent_months.strftime("%Y-%m-%d")}).to_csv(
        OUT / "meses_ausentes.csv", index=False, encoding="utf-8-sig"
    )

    # Resumen por año; conserva el número de meses observados y toma la meta anual no faltante.
    annual = (
        df.groupby(df["_fecha"].dt.year)
        .agg(
            inflacion_media=(INFL, "mean"),
            inflacion_mediana=(INFL, "median"),
            inflacion_min=(INFL, "min"),
            inflacion_max=(INFL, "max"),
            meses_inflacion=(INFL, "count"),
            meta_anual=(META, "median"),
            meses_con_meta=(META, "count"),
        )
        .rename_axis("anio")
        .reset_index()
    )
    annual.to_csv(OUT / "resumen_anual.csv", index=False, encoding="utf-8-sig")

    whole, summaries, by_decade, decade_summaries = [], [], [], []
    decade = (df["_fecha"].dt.year // 10) * 10
    for v in vars_num:
        d, s = iqr_reports(df, v)
        whole.append(d)
        summaries.append(s)
        d, s = iqr_reports(df, v, decade)
        by_decade.append(d)
        decade_summaries.append(s)
    pd.concat(whole, ignore_index=True).to_csv(
        OUT / "atipicos_iqr_serie_completa.csv", index=False, encoding="utf-8-sig"
    )
    pd.concat(summaries, ignore_index=True).to_csv(
        OUT / "resumen_iqr_serie_completa.csv", index=False, encoding="utf-8-sig"
    )
    pd.concat(by_decade, ignore_index=True).to_csv(
        OUT / "atipicos_iqr_por_decada.csv", index=False, encoding="utf-8-sig"
    )
    pd.concat(decade_summaries, ignore_index=True).to_csv(
        OUT / "resumen_iqr_por_decada.csv", index=False, encoding="utf-8-sig"
    )

    rules = [
        [
            DATE,
            "Fecha mensual",
            "Convertir a fecha válida; una observación por mes; marcar meses ausentes y duplicados.",
        ],
        [
            INFL,
            "Tasa de inflación anual observada mensualmente",
            "Numérica; admitir negativos (deflación). No aplicar límite máximo global ni eliminar extremos sin verificar contexto.",
        ],
        [
            META,
            "Meta oficial disponible solo para parte del período",
            "No imputar como cero ni rellenar antes de la primera observación. Comparar solo en fechas con ambas variables presentes.",
        ],
        [
            "2026",
            "Año parcial en el archivo (hasta agosto)",
            "No comparar su promedio anual con años completos sin etiquetarlo como parcial.",
        ],
        [
            "Atípicos",
            "Serie larga con cambios de régimen",
            "Usar cuantiles/mediana junto al promedio; revisar IQR global y por década. Conservar episodios económicamente plausibles.",
        ],
    ]
    pd.DataFrame(rules, columns=["campo", "contexto", "regla_sugerida"]).to_csv(
        OUT / "reglas_calidad_sugeridas.csv", index=False, encoding="utf-8-sig"
    )

    if HAS_PLOTS:
        sns.set_theme(style="whitegrid")
        fig, ax = plt.subplots(figsize=(13, 5))
        ax.plot(df["_fecha"], df[INFL], label="Inflación total anual", lw=1.2)
        ax.plot(df["_fecha"], df[META], label="Meta de inflación", lw=1.4)
        ax.set_ylabel("Porcentaje")
        ax.set_title("Inflación total y meta mensual reportada")
        ax.legend()
        fig.tight_layout()
        fig.savefig(OUT / "serie_inflacion_meta.png", dpi=170)
        plt.close(fig)
        fig, axes = plt.subplots(1, 2, figsize=(11, 4))
        sns.histplot(df[INFL].dropna(), bins=35, ax=axes[0])
        axes[0].set_title("Inflación total: distribución")
        sns.boxplot(x=df[INFL], ax=axes[1], showfliers=True)
        axes[1].set_title("Inflación total: caja y bigotes")
        fig.tight_layout()
        fig.savefig(OUT / "distribucion_inflacion.png", dpi=170)
        plt.close(fig)
        fig, ax = plt.subplots(figsize=(12, 5))
        annual.plot(x="anio", y=["inflacion_media", "meta_anual"], ax=ax)
        ax.set_title("Promedio anual de inflación y meta anual")
        ax.set_ylabel("Porcentaje")
        fig.tight_layout()
        fig.savefig(OUT / "comparacion_anual_meta.png", dpi=170)
        plt.close(fig)

    print(
        f"Inflación/meta: {len(df):,} meses; faltantes de meta {int(df[META].isna().sum()):,} ({100 * df[META].isna().mean():.2f}%)."
    )
    print(
        f"Primera meta: {first_meta:%Y-%m}; meses faltantes en calendario: {len(absent_months)}."
    )
    print(
        f"Atípicos IQR serie completa: {sum(len(x) for x in whole):,}; por década: {sum(len(x) for x in by_decade):,}."
    )
    print(f"Salidas: {OUT}")
    if not HAS_PLOTS:
        print(
            "Gráficos omitidos en este entorno; instala matplotlib y seaborn para generarlos."
        )


if __name__ == "__main__":
    main()
