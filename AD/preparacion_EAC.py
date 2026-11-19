"""Diagnóstico de clases, remuestreo y selección por impureza para la EAC 2024."""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from imblearn.over_sampling import RandomOverSampler, SMOTENC
from imblearn.pipeline import Pipeline
from imblearn.under_sampling import RandomUnderSampler
from scipy.stats import ks_2samp
from sklearn.compose import ColumnTransformer
from sklearn.metrics import balanced_accuracy_score, f1_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier


ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / "data/raw/EAC_CIFRAS_2024_ANONIMIZADA_FINAL.csv"
OUTPUT = ROOT / "reports/preparacion_EAC"
TARGET = "CORRE_9"
CATEGORY = "IDOJ1"
SEED = 42
ORIGINAL = "Train original"
UNDER = "Submuestreo"
OVER = "Sobremuestreo aleatorio"
SMOTE = "SMOTENC"
COLORS = ["#254c68", "#ad591b", "#217f75", "#74568c"]


def save_chart(fig, out, filename):
    fig.tight_layout()
    fig.savefig(out / filename, dpi=175, bbox_inches="tight")
    plt.close(fig)


def class_summary(y, train, test, out):
    counts = y.value_counts()
    summary = pd.DataFrame({
        "total": counts,
        "train": train.value_counts(),
        "test": test.value_counts(),
    }).reindex(counts.index)
    for column in ["total", "train", "test"]:
        summary[f"pct_{column}"] = 100 * summary[column] / summary[column].sum()
    summary["brecha_train_test_pp"] = summary["pct_train"] - summary["pct_test"]
    summary.to_csv(out / "distribucion_clases.csv", encoding="utf-8-sig")

    fig, ax = plt.subplots(figsize=(10, 4.8))
    bars = ax.bar(summary.index, summary["total"], color=COLORS[0])
    ax.bar_label(bars, padding=3)
    ax.set(ylim=(0, summary["total"].max() * 1.13), xlabel="Clase CORRE_9",
           ylabel="Registros", title="Distribución original por actividad económica")
    save_chart(fig, out, "01_clases_originales.png")

    fig, ax = plt.subplots(figsize=(10, 4.8))
    summary[["pct_train", "pct_test"]].rename(
        columns={"pct_train": f"Train (n={len(train)})", "pct_test": f"Test (n={len(test)})"}
    ).plot.bar(ax=ax, rot=0, color=[COLORS[0], "#77b6b0"])
    ax.set(xlabel="Clase CORRE_9", ylabel="% dentro de cada partición",
           title="Distribución después de la partición estratificada 80/20")
    save_chart(fig, out, "02_particion_estratificada.png")
    return summary


def joined(x, y):
    return x.reset_index(drop=True).assign(**{TARGET: y.to_numpy()})


def resample(train, numeric):
    x = train.drop(columns=TARGET)
    y = train[TARGET]
    under = RandomUnderSampler(random_state=SEED, replacement=False)
    over = RandomOverSampler(random_state=SEED)
    under_x, under_y = under.fit_resample(x, y)
    over_x, over_y = over.fit_resample(x, y)

    # Dividir por la desviación estándar mantiene los ceros exactos.
    scaler = StandardScaler(with_mean=False)
    scaled = x.copy()
    scaled[numeric] = scaler.fit_transform(x[numeric])
    sampler = SMOTENC(categorical_features=[CATEGORY], random_state=SEED, k_neighbors=5)
    smote_x, smote_y = sampler.fit_resample(scaled, y)
    # imbalanced-learn devuelve primero las filas originales y luego las sintéticas.
    synthetic = smote_x.iloc[len(x):].copy()
    synthetic[numeric] = scaler.inverse_transform(synthetic[numeric])
    # Las cuentas de personal de la EAC anonimizada admiten fracciones.
    # El año de inicio de operaciones conserva su unidad entera.
    synthetic["IDAIO"] = synthetic["IDAIO"].round()
    generated = joined(synthetic, smote_y.iloc[len(x):])

    return {
        ORIGINAL: train.reset_index(drop=True),
        UNDER: joined(under_x, under_y),
        OVER: joined(over_x, over_y),
        SMOTE: pd.concat([train, generated], ignore_index=True),
    }


def compare_structure(samples, numeric, classes, out):
    reference = samples[ORIGINAL]
    counts = pd.DataFrame({
        name: frame[TARGET].value_counts().reindex(classes)
        for name, frame in samples.items()
    })
    counts.to_csv(out / "clases_por_tecnica.csv", encoding="utf-8-sig")
    fig, ax = plt.subplots(figsize=(11, 5))
    counts.plot.bar(ax=ax, rot=0, color=COLORS)
    ax.set(xlabel="Clase CORRE_9", ylabel="Registros en train",
           title="Tamaño de cada clase con las tres técnicas")
    save_chart(fig, out, "03_clases_por_tecnica.png")

    distributions = []
    relations = []
    for label in classes:
        original = reference.loc[reference[TARGET].eq(label)]
        corr_original = original[numeric].corr(method="spearman").to_numpy()
        upper_triangle = np.triu_indices(len(numeric), k=1)
        corr_original = corr_original[upper_triangle]
        for method in [UNDER, OVER, SMOTE]:
            frame = samples[method]
            group = frame.loc[frame[TARGET].eq(label)]
            for feature in numeric:
                distance = ks_2samp(original[feature], group[feature], method="asymp").statistic
                distributions.append((method, label, feature, "KS", distance))
            proportions = pd.concat([
                original[CATEGORY].value_counts(normalize=True),
                group[CATEGORY].value_counts(normalize=True),
            ], axis=1).fillna(0)
            distance = (proportions.iloc[:, 0] - proportions.iloc[:, 1]).abs().sum() / 2
            distributions.append((method, label, CATEGORY, "variación total", distance))

            corr = group[numeric].corr(method="spearman").to_numpy()[upper_triangle]
            valid = np.isfinite(corr) & np.isfinite(corr_original)
            relations.append((method, label, valid.sum(), np.isfinite(corr_original).sum(),
                              np.abs(corr[valid] - corr_original[valid]).mean()))

    detail = pd.DataFrame(distributions, columns=["metodo", TARGET, "variable", "medida", "distancia"])
    detail.to_csv(out / "cambios_distribuciones.csv", index=False, encoding="utf-8-sig")
    correlations = pd.DataFrame(relations, columns=[
        "metodo", TARGET, "pares_comparables", "pares_en_train", "cambio_medio_spearman",
    ])
    correlations.to_csv(out / "cambios_correlaciones.csv", index=False, encoding="utf-8-sig")

    original_n = len(reference)
    summary = []
    for method in [UNDER, OVER, SMOTE]:
        n = len(samples[method])
        numeric_shifts = detail.loc[detail["metodo"].eq(method) & detail["medida"].eq("KS"), "distancia"]
        category_shifts = detail.loc[detail["metodo"].eq(method) & detail["variable"].eq(CATEGORY), "distancia"]
        changes = correlations.loc[correlations["metodo"].eq(method)]
        summary.append({
            "metodo": method,
            "registros_por_clase": int(counts[method].iloc[0]),
            "total": n,
            "originales_conservados": min(n, original_n),
            "originales_descartados": max(0, original_n - n),
            "copias_agregadas": n - original_n if method == OVER else 0,
            "sinteticos_agregados": n - original_n if method == SMOTE else 0,
            "ks_mediana": numeric_shifts.median(),
            "ks_p90": numeric_shifts.quantile(.90),
            "idoj1_variacion_total_mediana": category_shifts.median(),
            "cambio_spearman_medio_por_clase": changes["cambio_medio_spearman"].mean(),
            "pares_comparables_pct": 100 * changes["pares_comparables"].sum() / changes["pares_en_train"].sum(),
        })
    summary = pd.DataFrame(summary)
    summary.to_csv(out / "comparacion_estructura.csv", index=False, encoding="utf-8-sig")

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))
    for ax, metric, title, ylabel in [
        (axes[0], "ks_mediana", "Distribuciones numéricas", "Distancia KS mediana"),
        (axes[1], "cambio_spearman_medio_por_clase", "Relaciones entre predictores", "Cambio absoluto medio de ρ"),
    ]:
        bars = ax.bar(["Submuestreo", "Sobremuestreo\naleatorio", "SMOTENC"], summary[metric], color=COLORS[1:])
        ax.bar_label(bars, fmt="%.3f", padding=3)
        ax.set(title=title, ylabel=ylabel, ylim=(0, max(summary[metric].max() * 1.22, .01)))
    fig.suptitle("Cambios dentro de cada clase respecto a train original")
    save_chart(fig, out, "04_cambio_estructura.png")
    return summary


def tree_pipeline(features):
    numeric = [feature for feature in features if feature != CATEGORY]
    categorical = [feature for feature in features if feature == CATEGORY]
    encoding = ColumnTransformer([
        ("numericas", "passthrough", numeric),
        ("categoricas", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical),
    ])
    return Pipeline([
        ("remuestreo", RandomOverSampler(random_state=SEED)),
        ("codificar", encoding),
        ("arbol", DecisionTreeClassifier(random_state=SEED)),
    ])


def select_predictors(train, test, out):
    x_train, y_train = train.drop(columns=TARGET), train[TARGET]
    x_test, y_test = test.drop(columns=TARGET), test[TARGET]
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    search = GridSearchCV(
        tree_pipeline(x_train.columns),
        param_grid={
            "arbol__criterion": ["gini", "entropy"],
            "arbol__max_depth": [3, 5, 8, 12, 16, None],
            "arbol__min_samples_leaf": [5, 15, 30, 60],
        },
        scoring={"balanced_accuracy": "balanced_accuracy", "f1_macro": "f1_macro"},
        refit="balanced_accuracy",
        cv=cv,
        n_jobs=2,
        error_score="raise",
    )
    search.fit(x_train, y_train)
    cv_results = pd.DataFrame(search.cv_results_)
    columns = [
        "param_arbol__criterion", "param_arbol__max_depth", "param_arbol__min_samples_leaf",
        "mean_test_balanced_accuracy", "std_test_balanced_accuracy", "mean_test_f1_macro",
        "rank_test_balanced_accuracy",
    ]
    cv_results[columns].sort_values("rank_test_balanced_accuracy", kind="stable").to_csv(
        out / "busqueda_arbol_cv.csv", index=False, encoding="utf-8-sig"
    )

    best = search.best_estimator_
    encoder = best.named_steps["codificar"]
    tree = best.named_steps["arbol"]
    category_slice = encoder.output_indices_["categoricas"]
    numeric_features = [feature for feature in x_train if feature != CATEGORY]
    importance = pd.Series(tree.feature_importances_[encoder.output_indices_["numericas"]],
                           index=numeric_features, name="importancia")
    importance.loc[CATEGORY] = tree.feature_importances_[category_slice].sum()
    ranking = importance.sort_values(ascending=False, kind="stable").to_frame()
    ranking.index.name = "variable"
    ranking["importancia_acumulada"] = ranking["importancia"].cumsum()
    # Conservar el menor prefijo del ranking que cubre el 80 % de la importancia.
    target_importance = 0.80
    if ranking["importancia"].sum() == 0:
        raise ValueError("El árbol elegido no realizó divisiones.")
    ranking["seleccionada"] = (
        ranking["importancia_acumulada"].shift(fill_value=0).lt(target_importance)
        & ranking["importancia"].gt(0)
    )
    selected = ranking.index[ranking["seleccionada"]].tolist()
    ranking.to_csv(out / "importancia_impureza.csv", encoding="utf-8-sig")

    # El subconjunto se fija con train y ambos árboles se evalúan luego en el mismo test.
    reduced = tree_pipeline(selected).set_params(**search.best_params_)
    reduced.fit(x_train[selected], y_train)
    predictions = [best.predict(x_test), reduced.predict(x_test[selected])]
    models = [best, reduced]
    metrics = pd.DataFrame([
        {
            "atributos": label,
            "n_atributos": n,
            "criterio": model.named_steps["arbol"].criterion,
            "max_depth": model.named_steps["arbol"].max_depth,
            "min_samples_leaf": model.named_steps["arbol"].min_samples_leaf,
            "profundidad_real": model.named_steps["arbol"].get_depth(),
            "hojas": model.named_steps["arbol"].get_n_leaves(),
            "balanced_accuracy_cv": cv_score,
            "balanced_accuracy_test": balanced_accuracy_score(y_test, pred),
            "f1_macro_test": f1_score(y_test, pred, average="macro"),
        }
        for label, n, model, pred, cv_score in zip(
            ["Todos", "Seleccionados"], [x_train.shape[1], len(selected)],
            models, predictions, [search.best_score_, np.nan],
        )
    ])
    metrics.to_csv(out / "evaluacion_arbol.csv", index=False, encoding="utf-8-sig")

    top = ranking.head(max(10, len(selected))).iloc[::-1]
    fig, ax = plt.subplots(figsize=(9, 5.1))
    values = top["importancia"] * 100
    bars = ax.barh(top.index, values, color=np.where(top["seleccionada"], COLORS[0], "#a7b7c0"))
    ax.bar_label(bars, fmt="%.1f %%", padding=5)
    ax.set(xlabel="Porcentaje de la reducción ponderada de impureza",
           title="Importancia de predictores en el árbol seleccionado",
           xlim=(0, values.max() * 1.23))
    ax.legend(handles=[
        matplotlib.patches.Patch(color=COLORS[0], label="Seleccionado"),
        matplotlib.patches.Patch(color="#a7b7c0", label="Fuera del subconjunto"),
    ], loc="lower right")
    save_chart(fig, out, "05_importancia_impureza.png")
    return ranking, metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--entrada", type=Path, default=INPUT)
    parser.add_argument("--salida", type=Path, default=OUTPUT)
    parser.add_argument("--guardar-muestras", action="store_true",
                        help="Guarda las filas de train, test y las tres muestras para uso local.")
    args = parser.parse_args()

    df = pd.read_csv(args.entrada, sep=";", decimal=",", encoding="utf-8-sig", dtype={TARGET: str})
    df[TARGET] = df[TARGET].str.strip()
    if df[TARGET].isna().any() or df[TARGET].eq("").any():
        raise ValueError("La base contiene registros sin clasificación CORRE_9.")
    if df["V1"].isna().any() or df["V1"].duplicated().any():
        raise ValueError("Se esperaba un identificador V1 presente y único por registro.")
    df = df.drop(columns=["V1", "CORRELA_16"])
    df[CATEGORY] = df[CATEGORY].astype("Int64").astype("string").fillna("Sin dato").astype(object)
    numeric = [column for column in df if column not in {TARGET, CATEGORY}]
    df[numeric] = df[numeric].apply(pd.to_numeric).astype(float)
    if not np.isfinite(df[numeric].to_numpy()).all():
        raise ValueError("La comparación requiere predictores numéricos completos y finitos.")

    train, test = train_test_split(df, test_size=.20, stratify=df[TARGET], random_state=SEED)
    args.salida.mkdir(parents=True, exist_ok=True)
    distribution = class_summary(df[TARGET], train[TARGET], test[TARGET], args.salida)
    samples = resample(train, numeric)
    comparison = compare_structure(samples, numeric, distribution.index, args.salida)
    ranking, tree_metrics = select_predictors(train, test, args.salida)
    if args.guardar_muestras:
        private = args.salida / "muestras"
        private.mkdir(exist_ok=True)
        for filename, frame in zip(
            ["train", "submuestreo", "sobremuestreo_aleatorio", "smotenc"], samples.values()
        ):
            frame.to_csv(private / f"{filename}.csv", index=False, encoding="utf-8-sig")
        test.to_csv(private / "test.csv", index=False, encoding="utf-8-sig")

    print(f"{len(df)} registros, {len(distribution)} clases, train={len(train)}, test={len(test)}")
    print(f"Mayor brecha train/test: {distribution['brecha_train_test_pp'].abs().max():.4f} pp")
    print(comparison.to_string(index=False))
    print(tree_metrics.to_string(index=False))
    print(ranking.loc[ranking["seleccionada"]].to_string())


if __name__ == "__main__":
    main()
