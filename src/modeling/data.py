"""Carga da base de modelagem, divisão treino/teste e expansão no nível do aluno."""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from src.modeling import config

ROOT = Path(__file__).resolve().parents[2]
BASE_PATH = ROOT / "data" / "processed" / "base_modelagem.parquet"


def carregar_base() -> pd.DataFrame:
    """Base de modelagem (uma linha por município)."""
    return pd.read_parquet(BASE_PATH)


def dividir_treino_teste(
    base: pd.DataFrame,
    test_size: float = config.TEST_SIZE,
    seed: int = config.SEED,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Separa municípios em treino e teste, estratificado por UF.

    A divisão acontece no nível do município, antes da expansão: os registros
    de um município nunca ficam dos dois lados.
    """
    treino, teste = train_test_split(
        base,
        test_size=test_size,
        stratify=base[config.COL_ESTRATO],
        random_state=seed,
    )
    return treino.reset_index(drop=True), teste.reset_index(drop=True)


def expandir_alunos(
    df: pd.DataFrame,
    features: list[str],
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray]:
    """Expande cada município em dois registros no nível do aluno.

    - rótulo 1 (alfabetizados),     peso = n_alunos × taxa
    - rótulo 0 (não alfabetizados), peso = n_alunos × (1 − taxa)

    As features são as do município, repetidas nos dois registros. Ajustar um
    classificador ponderado nesses registros equivale a modelar a probabilidade
    de um aluno do município ser alfabetizado.

    Retorna X, y, pesos e grupos (id do município, para manter os registros juntos).
    """
    taxa = df[config.COL_ALVO].astype(float) / 100
    n_alunos = df[config.COL_PESO].astype(float)

    alfabetizados = df.assign(_rotulo=1, _peso=n_alunos * taxa)
    nao_alfabetizados = df.assign(_rotulo=0, _peso=n_alunos * (1 - taxa))
    expandido = pd.concat([alfabetizados, nao_alfabetizados], ignore_index=True)

    X = expandido[features]
    y = expandido["_rotulo"].to_numpy()
    pesos = expandido["_peso"].to_numpy()
    grupos = expandido[config.COL_ID].to_numpy()
    return X, y, pesos, grupos