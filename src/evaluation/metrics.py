"""Métricas de avaliação no nível do aluno e do município.

Todas as métricas são calculadas a partir de três vetores por município: a
taxa real (0–100), a probabilidade prevista e o número de alunos. A expansão
no nível do aluno é reconstruída aqui:
    rótulo 1, peso n × taxa;  rótulo 0, peso n × (1 − taxa);
    score = probabilidade prevista do município (igual nos dois registros).

O "oráculo" usa a taxa real do município como score: nenhuma feature
municipal ordena alunos melhor do que isso, então o seu AUC é o teto
alcançável. O desempenho é reportado também como % desse teto.
"""

import numpy as np
from sklearn.metrics import log_loss, roc_auc_score

from src.modeling import config

EPS = 1e-6


def _expandir(taxa_real_pct, score, n_alunos) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Reconstrói rótulos, scores e pesos no nível do aluno a partir dos vetores municipais."""
    taxa = np.asarray(taxa_real_pct, dtype=float) / 100
    s = np.asarray(score, dtype=float)
    n = np.asarray(n_alunos, dtype=float)
    rotulos = np.r_[np.ones_like(taxa), np.zeros_like(taxa)]
    pesos = np.r_[n * taxa, n * (1 - taxa)]
    scores = np.r_[s, s]
    return rotulos, scores, pesos


def avaliar_predicoes(
    taxa_real_pct,
    prob_prevista,
    n_alunos,
    limiar_risco: float = config.LIMIAR_RISCO,
) -> dict[str, float]:
    """Métricas no nível do aluno (ponderadas) e do município."""
    taxa_real = np.asarray(taxa_real_pct, dtype=float)
    prob = np.asarray(prob_prevista, dtype=float)
    n = np.asarray(n_alunos, dtype=float)

    rotulos, scores, pesos = _expandir(taxa_real, prob, n)
    _, scores_oraculo, _ = _expandir(taxa_real, taxa_real / 100, n)

    auc = roc_auc_score(rotulos, scores, sample_weight=pesos)
    auc_teto = roc_auc_score(rotulos, scores_oraculo, sample_weight=pesos)

    taxa_prevista = 100 * prob
    em_risco = (taxa_real < limiar_risco).astype(int)

    return {
        # Nível do aluno (ponderado pelo número de alunos)
        "auc_aluno": auc,
        "auc_teto": auc_teto,
        "pct_teto": (auc - 0.5) / (auc_teto - 0.5),
        "log_loss": log_loss(rotulos, np.clip(scores, EPS, 1 - EPS), sample_weight=pesos),
        # Nível do município
        "mae_pp": float(np.average(np.abs(taxa_prevista - taxa_real), weights=n)),
        # Menor taxa prevista = maior risco; cada município pesa igual nesta métrica.
        "auc_risco_municipio": roc_auc_score(em_risco, -prob),
    }