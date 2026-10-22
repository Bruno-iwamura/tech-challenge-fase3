"""Pipelines de pré-processamento + modelo.

Cada pipeline é um único objeto: o pré-processamento é ajustado apenas nos
dados de treino de cada fold, evitando vazamento de estatísticas (medianas,
médias) do conjunto de validação/teste.

Todos os pipelines terminam em uma etapa chamada "modelo", para que o ajuste
ponderado seja uniforme: pipeline.fit(X, y, modelo__sample_weight=pesos).

O transformador categórico só entra quando há features categóricas (nas
ablações sem a família de território, não há nenhuma).
"""

import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, OrdinalEncoder, StandardScaler

from src.modeling import config


def _log1p_seguro(X):
    """log(1 + x) com valores negativos truncados em zero."""
    return np.log1p(np.clip(X, 0, None))


def _separar_colunas(features: list[str]) -> dict[str, list[str]]:
    categoricas = [c for c in features if c in config.FEATURES_CATEGORICAS]
    numericas = [c for c in features if c not in categoricas]
    return {"categoricas": categoricas, "numericas": numericas}


def construir_baseline() -> Pipeline:
    """Prevê a taxa média ponderada do treino para todos (AUC = 0,5 por definição)."""
    return Pipeline([("modelo", DummyClassifier(strategy="prior"))])


def construir_logistica(features: list[str], C: float = 1.0) -> Pipeline:
    """Regressão logística com regularização L2 e pré-processamento por tipo de coluna."""
    colunas = _separar_colunas(features)
    numericas = [c for c in colunas["numericas"] if c not in config.REFERENCIAS_COMPOSICIONAIS]

    log = [c for c in numericas if c in config.FEATURES_LOG]
    estruturais = [c for c in numericas if c in config.NULOS_ESTRUTURAIS]
    demais = [c for c in numericas if c not in log and c not in estruturais]

    transformadores = [
        ("log", Pipeline([
            ("imputar", SimpleImputer(strategy="median")),
            ("log1p", FunctionTransformer(_log1p_seguro, feature_names_out="one-to-one")),
            ("padronizar", StandardScaler()),
        ]), log),
        ("estrutural", Pipeline([
            ("imputar", SimpleImputer(strategy="median", add_indicator=True)),
            ("padronizar", StandardScaler()),
        ]), estruturais),
        ("numerico", Pipeline([
            ("imputar", SimpleImputer(strategy="median")),
            ("padronizar", StandardScaler()),
        ]), demais),
    ]
    if colunas["categoricas"]:
        transformadores.append(
            ("categorico", OneHotEncoder(handle_unknown="ignore", sparse_output=False), colunas["categoricas"])
        )

    pipeline = Pipeline([
        ("prep", ColumnTransformer(transformadores, remainder="drop")),
        ("modelo", LogisticRegression(C=C, max_iter=2000)),
    ])
    return pipeline.set_output(transform="pandas")


def construir_hgb(
    features: list[str],
    learning_rate: float = 0.05,
    max_iter: int = 300,
    max_leaf_nodes: int = 15,
    min_samples_leaf: int = 40,
    l2_regularization: float = 1.0,
) -> Pipeline:
    """HistGradientBoosting com nulos tratados nativamente e categóricas declaradas.

    Early stopping interno desligado: ele separaria linhas aleatórias para
    validação, misturando registros do mesmo município entre treino e
    validação internos. Hiperparâmetros são ajustados pela validação cruzada
    agrupada por município.
    """
    colunas = _separar_colunas(features)

    transformadores = []
    if colunas["categoricas"]:
        transformadores.append(
            ("categorico",
             OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=np.nan),
             colunas["categoricas"])
        )
    transformadores.append(("numerico", "passthrough", colunas["numericas"]))

    modelo = HistGradientBoostingClassifier(
        learning_rate=learning_rate,
        max_iter=max_iter,
        max_leaf_nodes=max_leaf_nodes,
        min_samples_leaf=min_samples_leaf,
        l2_regularization=l2_regularization,
        categorical_features=colunas["categoricas"] or None,
        early_stopping=False,
        random_state=config.SEED,
    )

    pipeline = Pipeline([
        ("prep", ColumnTransformer(transformadores, remainder="drop", verbose_feature_names_out=False)),
        ("modelo", modelo),
    ])
    return pipeline.set_output(transform="pandas")