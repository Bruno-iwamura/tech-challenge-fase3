"""Interpretabilidade do modelo: importância por permutação e valores SHAP.

Importância por permutação: queda de desempenho (pct_teto, log-loss) ao
embaralhar uma feature — ou um grupo de features em conjunto, com a mesma
permutação, o que evita que features correlacionadas dividam o crédito.

SHAP: PermutationExplainer (agnóstico ao modelo) sobre a matriz
pré-processada. O TreeExplainer não é usado porque tem um bug conhecido com
as categóricas nativas do HistGradientBoosting (issue #3436 do shap). A
matriz é convertida para float64 (o explicador não aceita os tipos anuláveis
do pandas); o HGB trata NaN nativamente.

Ambas medem ASSOCIAÇÃO no modelo, não efeito causal.
"""

import warnings

import numpy as np
import pandas as pd
import shap
from sklearn.pipeline import Pipeline

from src.evaluation.metrics import avaliar_predicoes
from src.modeling import config


def grupos_interpretacao(features: list[str]) -> dict[str, list[str]]:
    """Agrupa as features em temas para a leitura de importância."""
    grupos = {
        "Território (UF e região)": [f for f in features if f.startswith("ter_")],
        "Fluxo escolar": [f for f in features if f in config.FEATURES_FLUXO],
        "Formação e regularidade docente": [
            f for f in features if f.startswith(("ce_adeq_docente", "ce_regularidade"))
        ],
        "Turmas e jornada": [f for f in features if f.startswith(("ce_alunos_turma", "ce_horas_aula"))],
        "Socioeconômico e demográfico": [f for f in features if f.startswith("soc_")],
        "Financiamento da educação": [f for f in features if f.startswith("fin_")],
    }
    cobertas = [f for colunas in grupos.values() for f in colunas]
    faltando = set(features) - set(cobertas)
    repetidas = {f for f in cobertas if cobertas.count(f) > 1}
    if faltando or repetidas:
        raise ValueError(f"Grupos inconsistentes. Fora de grupo: {faltando}; em mais de um: {repetidas}")
    return {nome: colunas for nome, colunas in grupos.items() if colunas}


def importancia_permutacao(
    modelo: Pipeline,
    df: pd.DataFrame,
    features: list[str],
    grupos: dict[str, list[str]] | None = None,
    n_repeticoes: int = 5,
    seed: int = config.SEED,
) -> pd.DataFrame:
    """Queda de desempenho ao embaralhar cada feature (ou cada grupo, em conjunto)."""
    grupos = grupos or {f: [f] for f in features}
    rng = np.random.default_rng(seed)

    X = df[features]
    alvo, pesos = df[config.COL_ALVO], df[config.COL_PESO]
    referencia = avaliar_predicoes(alvo, modelo.predict_proba(X)[:, 1], pesos)

    linhas = []
    for nome, colunas in grupos.items():
        quedas_teto, aumentos_log_loss = [], []
        for _ in range(n_repeticoes):
            ordem = rng.permutation(len(X))
            X_perm = X.copy()
            for coluna in colunas:  # mesma permutação para todo o grupo
                X_perm[coluna] = X[coluna].to_numpy()[ordem]
            metricas = avaliar_predicoes(alvo, modelo.predict_proba(X_perm)[:, 1], pesos)
            quedas_teto.append(referencia["pct_teto"] - metricas["pct_teto"])
            aumentos_log_loss.append(metricas["log_loss"] - referencia["log_loss"])
        linhas.append({
            "nome": nome,
            "n_features": len(colunas),
            "queda_pct_teto": float(np.mean(quedas_teto)),
            "queda_pct_teto_dp": float(np.std(quedas_teto)),
            "aumento_log_loss": float(np.mean(aumentos_log_loss)),
        })

    return pd.DataFrame(linhas).sort_values("queda_pct_teto", ascending=False).reset_index(drop=True)


def _para_float(df: pd.DataFrame) -> np.ndarray:
    return df.to_numpy(dtype=float, na_value=np.nan)


def explicar_shap(
    modelo: Pipeline,
    fundo: pd.DataFrame,
    explicar: pd.DataFrame,
    features: list[str],
    seed: int = config.SEED,
) -> shap.Explanation:
    """Valores SHAP (escala de probabilidade) para os municípios em `explicar`.

    `fundo` é a distribuição de referência (municípios de treino) usada para
    "ausentar" features. As categóricas aparecem com os códigos do
    OrdinalEncoder do pipeline.
    """
    prep, estimador = modelo.named_steps["prep"], modelo.named_steps["modelo"]
    colunas = list(prep.get_feature_names_out())

    matriz_fundo = _para_float(prep.transform(fundo[features]))
    matriz_explicar = _para_float(prep.transform(explicar[features]))

    def prever(X: np.ndarray) -> np.ndarray:
        return estimador.predict_proba(pd.DataFrame(X, columns=colunas))[:, 1]

    explicador = shap.PermutationExplainer(prever, shap.maskers.Independent(matriz_fundo), seed=seed)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        valores = explicador(matriz_explicar)
    valores.feature_names = colunas
    return valores