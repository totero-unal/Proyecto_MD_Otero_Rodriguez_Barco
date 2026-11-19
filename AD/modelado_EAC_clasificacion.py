"""Clasifica la actividad económica agregada (CORRE_9) en la EAC 2024.

Usa una partición estratificada 80/20. Las clases se ponderan durante el
entrenamiento; la prueba conserva la distribución original. La importancia
de atributos se estima con permutaciones sobre entrenamiento para evitar
seleccionar atributos con información del conjunto de prueba.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / "data" / "raw" / "EAC_CIFRAS_2024_ANONIMIZADA_FINAL.csv"
OUT = ROOT / "reports" / "modelo_EAC_clasificacion"
TARGET = "CORRE_9"
SEED = 42


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(INPUT, sep=";", decimal=",", encoding="utf-8-sig", low_memory=False)

    # CORRELA_16 es la versión más detallada del mismo sector y causaría fuga.
    # V1 solo identifica la empresa; no debe servir como predictor.
    df = df.dropna(subset=[TARGET]).copy()
    y = df[TARGET].astype(str)
    X = df.drop(columns=[TARGET, "CORRELA_16", "V1"], errors="ignore")

    # CORRE_9 ya se separó como objetivo; IDOJ1 es categórica, no continua.
    categorical = [column for column in X.columns if column == "IDOJ1"]
    for column in categorical:
        X[column] = X[column].map(lambda value: str(int(value)) if pd.notna(value) else np.nan).astype(object)
    numeric = [column for column in X.columns if column not in categorical]

    class_counts = y.value_counts().sort_index()
    class_table = class_counts.rename("observaciones").to_frame()
    class_table["porcentaje"] = 100 * class_table["observaciones"] / len(y)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=SEED, stratify=y
    )
    train_counts = y_train.value_counts().reindex(class_table.index, fill_value=0)
    test_counts = y_test.value_counts().reindex(class_table.index, fill_value=0)
    class_table["train"] = train_counts
    class_table["test"] = test_counts
    class_table["peso_train_balanceado"] = len(y_train) / (len(class_counts) * train_counts)
    class_table.to_csv(OUT / "balance_clases.csv", encoding="utf-8-sig")

    prep = ColumnTransformer(
        transformers=[
            ("numericas", Pipeline([("imputar", SimpleImputer(strategy="median"))]), numeric),
            ("categoricas", Pipeline([
                ("imputar", SimpleImputer(strategy="most_frequent")),
                ("onehot", OneHotEncoder(handle_unknown="ignore")),
            ]), categorical),
        ],
        remainder="drop",
    )
    model = Pipeline([
        ("preparar", prep),
        ("clasificador", RandomForestClassifier(
            n_estimators=400,
            min_samples_leaf=2,
            class_weight="balanced_subsample",
            random_state=SEED,
            n_jobs=-1,
        )),
    ])
    model.fit(X_train, y_train)

    pred = model.predict(X_test)
    metrics = pd.DataFrame([{
        "objetivo": TARGET,
        "filas": len(df),
        "clases": y.nunique(),
        "train": len(X_train),
        "test": len(X_test),
        "accuracy": accuracy_score(y_test, pred),
        "balanced_accuracy": balanced_accuracy_score(y_test, pred),
        "f1_macro": f1_score(y_test, pred, average="macro"),
        "clase_mayor": class_counts.idxmax(),
        "clase_menor": class_counts.idxmin(),
        "razon_mayor_menor": class_counts.max() / class_counts.min(),
    }])
    metrics.to_csv(OUT / "metricas_prueba.csv", index=False, encoding="utf-8-sig")

    report = classification_report(y_test, pred, output_dict=True, zero_division=0)
    pd.DataFrame(report).transpose().to_csv(OUT / "metricas_por_clase.csv", encoding="utf-8-sig")
    labels = sorted(y.unique())
    pd.DataFrame(
        confusion_matrix(y_test, pred, labels=labels), index=labels, columns=labels
    ).to_csv(OUT / "matriz_confusion.csv", encoding="utf-8-sig")

    importance = permutation_importance(
        model,
        X_train,
        y_train,
        scoring="balanced_accuracy",
        n_repeats=5,
        random_state=SEED,
        n_jobs=-1,
    )
    importance_table = pd.DataFrame({
        "atributo": X.columns,
        "importancia_media": importance.importances_mean,
        "desviacion": importance.importances_std,
    }).sort_values("importancia_media", ascending=False)
    importance_table.to_csv(OUT / "importancia_atributos.csv", index=False, encoding="utf-8-sig")
    selected = importance_table.head(10)["atributo"].tolist()
    selected_categorical = [column for column in categorical if column in selected]
    selected_numeric = [column for column in numeric if column in selected]
    selected_prep = ColumnTransformer(
        transformers=[
            ("numericas", Pipeline([("imputar", SimpleImputer(strategy="median"))]), selected_numeric),
            ("categoricas", Pipeline([
                ("imputar", SimpleImputer(strategy="most_frequent")),
                ("onehot", OneHotEncoder(handle_unknown="ignore")),
            ]), selected_categorical),
        ],
        remainder="drop",
    )
    selected_model = Pipeline([
        ("preparar", selected_prep),
        ("clasificador", RandomForestClassifier(
            n_estimators=400,
            min_samples_leaf=2,
            class_weight="balanced_subsample",
            random_state=SEED,
            n_jobs=-1,
        )),
    ])
    selected_model.fit(X_train[selected], y_train)
    selected_pred = selected_model.predict(X_test[selected])
    pd.DataFrame([{
        "objetivo": TARGET,
        "predictores_seleccionados": len(selected),
        "train": len(X_train),
        "test": len(X_test),
        "accuracy": accuracy_score(y_test, selected_pred),
        "balanced_accuracy": balanced_accuracy_score(y_test, selected_pred),
        "f1_macro": f1_score(y_test, selected_pred, average="macro"),
    }]).to_csv(OUT / "metricas_prueba.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(classification_report(y_test, selected_pred, output_dict=True, zero_division=0)).transpose().to_csv(
        OUT / "metricas_por_clase.csv", encoding="utf-8-sig"
    )
    pd.DataFrame(
        confusion_matrix(y_test, selected_pred, labels=labels), index=labels, columns=labels
    ).to_csv(OUT / "matriz_confusion.csv", encoding="utf-8-sig")
    importance_table.head(10).sort_values("importancia_media").plot.barh(
        x="atributo", y="importancia_media", legend=False, figsize=(9, 6)
    )
    plt.title("Diez predictores más importantes (permutación en train)")
    plt.xlabel("Caída de balanced accuracy al permutar")
    plt.ylabel("")
    plt.tight_layout()
    plt.savefig(OUT / "importancia_atributos.png", dpi=160)
    plt.close()

    split_table = pd.DataFrame({
        "conjunto": ["train", "test"],
        "filas": [len(y_train), len(y_test)],
    })
    split_table.to_csv(OUT / "particion.csv", index=False, encoding="utf-8-sig")
    print(f"Objetivo: {TARGET}; {y.nunique()} clases; razón mayor/menor {class_counts.max()/class_counts.min():.2f}.")
    print(f"Partición estratificada: train {len(X_train):,} ({len(X_train)/len(df):.0%}), test {len(X_test):,} ({len(X_test)/len(df):.0%}).")
    print(f"Balance: class_weight=balanced_subsample; balanced accuracy de prueba con los 10 seleccionados {balanced_accuracy_score(y_test, selected_pred):.3f}; F1 macro {f1_score(y_test, selected_pred, average='macro'):.3f}.")
    print("Atributos seleccionados:", ", ".join(selected))
    print(f"Resultados: {OUT}")


if __name__ == "__main__":
    main()
