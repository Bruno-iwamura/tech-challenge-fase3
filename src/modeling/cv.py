"""Validação cruzada no nível do município.

Os municípios são divididos em folds (estratificados por UF) ANTES da
expansão no nível do aluno: os registros de um município nunca ficam em treino
e validação ao mesmo tempo. A cada fold, um pipeline novo é construído,
ajustado com os pesos de contagem de alunos e avaliado nos municípios de
validação (uma previsão por município).
"""

from collections.abc import Callable

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline

from src.evaluation.metrics import avaliar_predicoes
from src.modeling import config
from src.modeling.data import expandir_alunos


def _folds(df: pd.DataFrame, n_splits: int, seed: int):
    divisor = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    return divisor.split(df, df[config.COL_ESTRATO])


def _ajustar(
    construir_modelo: Callable[[list[str]], Pipeline],
    treino: pd.DataFrame,
    features: list[str],
) -> Pipeline:
    """Constrói um pipeline novo e o ajusta no treino expandido, com pesos de alunos."""
    modelo = construir_modelo(features)
    X, y, pesos, _ = expandir_alunos(treino, features)
    modelo.fit(X, y, modelo__sample_weight=pesos)
    return modelo


def validacao_cruzada(
    construir_modelo: Callable[[list[str]], Pipeline],
    df: pd.DataFrame,
    features: list[str],
    n_splits: int = config.N_SPLITS,
    seed: int = config.SEED,
) -> pd.DataFrame:
    """Uma linha de métricas por fold."""
    linhas = []
    for fold, (idx_treino, idx_valid) in enumerate(_folds(df, n_splits, seed)):
        treino, valid = df.iloc[idx_treino], df.iloc[idx_valid]
        modelo = _ajustar(construir_modelo, treino, features)
        prob = modelo.predict_proba(valid[features])[:, 1]
        metricas = avaliar_predicoes(valid[config.COL_ALVO], prob, valid[config.COL_PESO])
        linhas.append({"fold": fold, **metricas})
    return pd.DataFrame(linhas)


def previsoes_fora_do_fold(
    construir_modelo: Callable[[list[str]], Pipeline],
    df: pd.DataFrame,
    features: list[str],
    n_splits: int = config.N_SPLITS,
    seed: int = config.SEED,
) -> pd.Series:
    """Probabilidade prevista para cada município por um modelo que não o viu (mesmos folds da CV)."""
    prob = pd.Series(np.nan, index=df.index, dtype=float)
    for idx_treino, idx_valid in _folds(df, n_splits, seed):
        treino, valid = df.iloc[idx_treino], df.iloc[idx_valid]
        modelo = _ajustar(construir_modelo, treino, features)
        prob.iloc[idx_valid] = modelo.predict_proba(valid[features])[:, 1]
    return prob


def resumir(resultados: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Média e desvio-padrão por métrica, um modelo por linha."""
    linhas = {}
    for nome, df in resultados.items():
        metricas = df.drop(columns="fold")
        linhas[nome] = {c: f"{metricas[c].mean():.4f} ± {metricas[c].std():.4f}" for c in metricas.columns}
    return pd.DataFrame(linhas).T