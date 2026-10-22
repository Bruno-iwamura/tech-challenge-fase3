"""Treino dos modelos finais e avaliação única no conjunto de teste.

Todas as escolhas (features, modelo, hiperparâmetros, ausência de
recalibração) foram feitas na validação cruzada do treino (notebook 03).
Este script ajusta os modelos no treino completo e os avalia UMA vez no
teste, que não foi usado em nenhuma decisão.

Saídas:
    models/hgb_final.joblib        -> modelo principal (fora do Git; regenerável)
    reports/metricas_teste.json    -> métricas no teste + metadados de reprodução

Uso:
    python -m src.modeling.train
"""

import json
import platform
from pathlib import Path

import joblib
import pandas as pd
import sklearn

from src.evaluation.metrics import avaliar_predicoes
from src.modeling import config
from src.modeling.data import carregar_base, dividir_treino_teste, expandir_alunos
from src.modeling.pipelines import construir_hgb, construir_logistica

ROOT = Path(__file__).resolve().parents[2]
MODELS = ROOT / "models"
REPORTS = ROOT / "reports"

MODELO_PRINCIPAL = "hgb"
CONSTRUTORES = {
    "hgb": construir_hgb,  # principal
    "logistica": construir_logistica,  # referência interpretável
}


def treinar(construtor, treino: pd.DataFrame, features: list[str]):
    """Ajusta um pipeline novo no treino expandido no nível do aluno, com pesos de alunos."""
    modelo = construtor(features)
    X, y, pesos, _ = expandir_alunos(treino, features)
    modelo.fit(X, y, modelo__sample_weight=pesos)
    return modelo


def main() -> None:
    base = carregar_base()
    features = config.selecionar_features(base.columns)
    treino, teste = dividir_treino_teste(base)

    metricas = {}
    for nome, construtor in CONSTRUTORES.items():
        modelo = treinar(construtor, treino, features)
        prob = modelo.predict_proba(teste[features])[:, 1]
        metricas[nome] = avaliar_predicoes(teste[config.COL_ALVO], prob, teste[config.COL_PESO])

        if nome == MODELO_PRINCIPAL:
            MODELS.mkdir(parents=True, exist_ok=True)
            joblib.dump(modelo, MODELS / f"{nome}_final.joblib")

    relatorio = {
        "modelo_principal": MODELO_PRINCIPAL,
        "n_municipios_treino": len(treino),
        "n_municipios_teste": len(teste),
        "n_features": len(features),
        "features": features,
        "seed": config.SEED,
        "test_size": config.TEST_SIZE,
        "limiar_risco": config.LIMIAR_RISCO,
        "versoes": {
            "python": platform.python_version(),
            "scikit-learn": sklearn.__version__,
            "pandas": pd.__version__,
        },
        "metricas_teste": metricas,
    }
    REPORTS.mkdir(parents=True, exist_ok=True)
    (REPORTS / "metricas_teste.json").write_text(json.dumps(relatorio, indent=2, ensure_ascii=False))

    print(f"Treino: {len(treino):,} municípios | Teste: {len(teste):,} municípios | {len(features)} features\n")
    print(pd.DataFrame(metricas).T.round(4).to_string())
    print(f"\nSalvo: models/{MODELO_PRINCIPAL}_final.joblib")
    print("Salvo: reports/metricas_teste.json")


if __name__ == "__main__":
    main()