import argparse
import logging
from pathlib import Path
import sys

import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, f1_score, precision_score, recall_score
from sklearn.model_selection import ParameterGrid, train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import QuantileTransformer
from sklearn.tree import DecisionTreeClassifier

logging.getLogger("mlflow.sklearn").setLevel(logging.ERROR)

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from AD.preparacion_EAC import CATEGORY, INPUT, OUTPUT, OVER, SEED, TARGET, resample


def construir_caracteristicas(df: pd.DataFrame, mediana_idaio: float | None = None) -> pd.DataFrame:
    eps = 1.0
    venta = df["VENTA"].clip(lower=eps)
    cto = df["CTO"].clip(lower=eps)
    gastos = df["GASTOS"].clip(lower=eps)
    totperso = df["TOTPERSO"].clip(lower=0.1)
    invpro = df["INVPRO"].clip(lower=eps)

    idaio_num = pd.to_numeric(df["IDAIO"], errors="coerce")
    if mediana_idaio is None:
        mediana_idaio = float(idaio_num.median())

    X = pd.DataFrame(index=df.index)

    # 1. Margen comercial y rotación real de inventario
    X["MARGEN_BRUTO"] = (df["VENTA"] - df["CTO"]) / venta
    X["ROTACION_INV_REAL"] = df["CTO"] / invpro

    # 2. Estructura relativa de costos y gastos operativos
    X["ENERGIA_PCT_GASTOS"] = df["ENERGIA"] / gastos
    X["TOTREM_PCT_GASTOS"] = df["TOTREM"] / gastos
    X["ENERGIA_PCT_VENTA"] = df["ENERGIA"] / venta
    X["ARRIENDO_PCT_VENTA"] = df["ARRIENDO"] / venta
    X["COMUNICA_PCT_GASTOS"] = df["COMUNICA"] / gastos
    X["EMPAQUE_PCT_VENTA"] = df["EMPAQUE"] / venta
    X["GASTOSNOP_PCT_GASTOS"] = df["GASTOSNOP"] / gastos
    X["FLETES_PCT_VENTA"] = df["FLETES"] / venta
    X["PUBLICO_PCT_VENTA"] = df["PUBLICO"] / venta
    X["HONORA_PCT_VENTA"] = df["HONORA"] / venta

    # 3. Productividad y composición laboral
    X["VENTA_POR_PERSONA_LOG"] = np.log1p((df["VENTA"] / totperso).clip(lower=0))
    X["REMUN_POR_PERSONA_LOG"] = np.log1p((df["TOTREM"] / totperso).clip(lower=0))
    X["PROP_MUJERES"] = df["TOTMUJ"] / totperso
    X["PROP_AGENCIA"] = df["AGENCIA"] / totperso

    # 4. Demografía empresarial y organización jurídica
    X["IDAIO"] = idaio_num.fillna(mediana_idaio)
    X["IDOJ1"] = pd.to_numeric(df["IDOJ1"], errors="coerce").fillna(-1.0)

    return X


def train_test():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--entrada", type=Path, default=INPUT)
    parser.add_argument("--salida", type=Path, default=OUTPUT)
    parser.add_argument(
        "--guardar-muestras",
        action="store_true",
        help="Guarda las filas de train, test y las tres muestras para uso local.",
    )
    args = parser.parse_args()

    df = pd.read_csv(args.entrada, sep=";", decimal=",", encoding="utf-8-sig", dtype={TARGET: str})
    df[TARGET] = df[TARGET].str.strip()
    df = df.drop(columns=["V1", "CORRELA_16"])
    df[CATEGORY] = df[CATEGORY].astype("Int64").astype("string").fillna("Sin dato").astype(object)
    numeric = [column for column in df if column not in {TARGET, CATEGORY}]
    df[numeric] = df[numeric].apply(pd.to_numeric).astype(float)

    train, test = train_test_split(df, test_size=0.20, stratify=df[TARGET], random_state=SEED)
    args.salida.mkdir(parents=True, exist_ok=True)
    samples = resample(train, numeric)
    samples_over = samples[OVER]

    mediana_idaio = float(pd.to_numeric(samples_over["IDAIO"], errors="coerce").median())

    X_train = construir_caracteristicas(samples_over, mediana_idaio).reset_index(drop=True)
    Y_train = samples_over[TARGET].reset_index(drop=True)
    X_test = construir_caracteristicas(test, mediana_idaio).reset_index(drop=True)
    Y_test = test[TARGET].reset_index(drop=True)

    return X_train, Y_train, X_test, Y_test


def entrenar_knn(X_train, Y_train, X_test, Y_test, root_dir=ROOT):
    db_path = (root_dir / "mlflow.db").as_posix()
    mlflow.set_tracking_uri(f"sqlite:///{db_path}")
    mlflow.set_experiment("EAC_2024_KNN_Ratios")

    scaler = QuantileTransformer(output_distribution="normal", random_state=SEED)
    X_train_esc = scaler.fit_transform(X_train)
    X_test_esc = scaler.transform(X_test)
    rejilla = {
        "n_neighbors": [51, 101, 501, 1001],
        "weights": ["uniform", "distance"],
        "metric": ["euclidean", "manhattan"],
    }

    for i, params in enumerate(ParameterGrid(rejilla), start=1):
        nombre_run = f"knn_{i:03d}_k{params['n_neighbors']}_{params['weights']}_{params['metric']}"

        with mlflow.start_run(run_name=nombre_run):
            clf = KNeighborsClassifier(**params, n_jobs=-1)
            clf.fit(X_train_esc, Y_train)
            pred_test = clf.predict(X_test_esc)

            mlflow.log_params({
                **params,
                "scaler": "QuantileTransformer_normal",
                "feature_set": "ratios_18",
                "n_features": X_train.shape[1],
                "remuestreo": OVER,
            })

            mlflow.log_metric("acurracy", accuracy_score(Y_test, pred_test))
            mlflow.log_metric("precision", precision_score(Y_test, pred_test, average="macro", zero_division=0))
            mlflow.log_metric("recall", recall_score(Y_test, pred_test, average="macro", zero_division=0))
            mlflow.log_metric("f1", f1_score(Y_test, pred_test, average="macro", zero_division=0))

            mlflow.log_text(classification_report(Y_test, pred_test, zero_division=0), "classification_report.txt")

            modelo_completo = make_pipeline(scaler, clf)
            mlflow.sklearn.log_model(
                sk_model=modelo_completo,
                name="modelo_knn",
                serialization_format="cloudpickle",
            )


def entrenar_arbol(X_train, Y_train, X_test, Y_test, root_dir=ROOT):
    db_path = (root_dir / "mlflow.db").as_posix()
    mlflow.set_tracking_uri(f"sqlite:///{db_path}")
    mlflow.set_experiment("EAC_2024_Arbol_Decision_Ratios")

    rejilla = {
        "criterion": ["gini", "entropy"],
        "max_depth": [8, 16, 32],
        "min_samples_leaf": [30, 60, 120, 250, 500],
        "ccp_alpha": [0.0, 0.0002, 0.001],
    }

    for i, params in enumerate(ParameterGrid(rejilla), start=1):
        nombre_run = f"dt_{i:03d}_{params['criterion']}_d{params['max_depth']}_leaf{params['min_samples_leaf']}"

        with mlflow.start_run(run_name=nombre_run):
            clf = DecisionTreeClassifier(**params, random_state=SEED)
            clf.fit(X_train, Y_train)
            pred_test = clf.predict(X_test)

            mlflow.log_params({
                **params,
                "random_state": SEED,
                "feature_set": "ratios_18",
                "n_features": X_train.shape[1],
                "remuestreo": OVER,
            })

            mlflow.log_metric("acurracy", accuracy_score(Y_test, pred_test))
            mlflow.log_metric("precision", precision_score(Y_test, pred_test, average="macro", zero_division=0))
            mlflow.log_metric("recall", recall_score(Y_test, pred_test, average="macro", zero_division=0))
            mlflow.log_metric("f1", f1_score(Y_test, pred_test, average="macro", zero_division=0))

            mlflow.log_text(classification_report(Y_test, pred_test, zero_division=0), "classification_report.txt")
            mlflow.sklearn.log_model(
                sk_model=clf,
                name="modelo_arbol",
                serialization_format="cloudpickle",
            )


def entrenar_naive_bayes(X_train, Y_train, X_test, Y_test, root_dir=ROOT):
    db_path = (root_dir / "mlflow.db").as_posix()
    mlflow.set_tracking_uri(f"sqlite:///{db_path}")
    mlflow.set_experiment("EAC_2024_Naive_Bayes_Ratios")

    scaler = QuantileTransformer(output_distribution="normal", random_state=SEED)
    X_train_esc = scaler.fit_transform(X_train)
    X_test_esc = scaler.transform(X_test)

    with mlflow.start_run(run_name="nb_default"):
        clf = GaussianNB()
        clf.fit(X_train_esc, Y_train)
        pred_test = clf.predict(X_test_esc)

        f1_val = f1_score(Y_test, pred_test, average="macro", zero_division=0)
        acc_val = accuracy_score(Y_test, pred_test)

        mlflow.log_params({
            "scaler": "QuantileTransformer_normal",
            "feature_set": "ratios_18",
            "n_features": X_train.shape[1],
            "remuestreo": "ninguno",
        })

        mlflow.log_metric("acurracy", acc_val)
        mlflow.log_metric("precision", precision_score(Y_test, pred_test, average="macro", zero_division=0))
        mlflow.log_metric("recall", recall_score(Y_test, pred_test, average="macro", zero_division=0))
        mlflow.log_metric("f1", f1_val)

        mlflow.log_text(classification_report(Y_test, pred_test, zero_division=0), "classification_report.txt")

        modelo_completo = make_pipeline(scaler, clf)
        mlflow.sklearn.log_model(
            sk_model=modelo_completo,
            name="modelo_nb",
            serialization_format="cloudpickle",
        )
        print(f"[NB 01/01] nb_default -> F1: {f1_val:.4f} | Acc: {acc_val:.4f}", flush=True)
    


def main():
    X_train, Y_train, X_test, Y_test = train_test()
    entrenar_arbol(X_train, Y_train, X_test, Y_test, ROOT)
    #entrenar_knn(X_train, Y_train, X_test, Y_test, ROOT)
    #entrenar_naive_bayes(X_train, Y_train, X_test, Y_test, ROOT)


if __name__ == "__main__":
    main()