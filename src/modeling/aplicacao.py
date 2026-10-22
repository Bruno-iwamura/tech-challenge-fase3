"""Aplicação: taxa esperada pelo contexto, quadrantes de risco e municípios acima/abaixo do esperado.

Previsões cross-fitted — cada município é previsto por um modelo que não o viu:
    treino: previsão fora do fold (mesmos folds da validação cruzada);
    teste:  modelo final (models/hgb_final.joblib), ajustado só no treino.

Quadrantes (limiar = meta nacional de 60%):
    real < 60 e prevista < 60  -> abaixo da meta, como esperado pelo contexto
    real < 60 e prevista >= 60 -> abaixo da meta, apesar do contexto favorável
    real >= 60 e prevista < 60 -> acima da meta, apesar do contexto desfavorável
    real >= 60 e prevista >= 60 -> acima da meta, como esperado

Destaque: o resíduo (real − prevista) é comparado à incerteza TOTAL, que soma
    - a variância amostral, que depende do número de alunos: p(1−p)/n;
    - a variância que o contexto não explica (comum a todos os municípios),
      estimada como variância dos resíduos menos a variância amostral média.
z = resíduo / sqrt(variância amostral + variância não explicada); |z| >= 2 é destaque.
Sem o segundo termo, municípios grandes virariam destaque com resíduos mínimos
(o ruído amostral deles é quase nulo). Um destaque indica diferença em relação
ao que o contexto medido prevê, a ser investigada — não um veredito.

Saída: reports/ranking_municipios.csv (ordenado da menor para a maior taxa prevista).

Uso:
    python -m src.modeling.aplicacao
"""

from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src.modeling import config
from src.modeling.cv import previsoes_fora_do_fold
from src.modeling.data import carregar_base, dividir_treino_teste
from src.modeling.pipelines import construir_hgb

ROOT = Path(__file__).resolve().parents[2]
MODELO = ROOT / "models" / "hgb_final.joblib"
SAIDA = ROOT / "reports" / "ranking_municipios.csv"

Z_DESTAQUE = 2.0

QUADRANTES = {
    (True, True): "Abaixo da meta, como esperado pelo contexto",
    (True, False): "Abaixo da meta, apesar do contexto favorável",
    (False, True): "Acima da meta, apesar do contexto desfavorável",
    (False, False): "Acima da meta, como esperado",
}


def previsoes_cross_fitted(base: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    """Probabilidade prevista para todos os municípios, sempre por um modelo que não os viu."""
    treino, teste = dividir_treino_teste(base)

    prob_treino = previsoes_fora_do_fold(construir_hgb, treino, features)
    modelo_final = joblib.load(MODELO)
    prob_teste = modelo_final.predict_proba(teste[features])[:, 1]

    treino = treino.assign(prob_prevista=prob_treino.to_numpy(), conjunto="treino (fora do fold)")
    teste = teste.assign(prob_prevista=prob_teste, conjunto="teste")
    return pd.concat([treino, teste], ignore_index=True)


def classificar(df: pd.DataFrame, limiar: float = config.LIMIAR_RISCO) -> tuple[pd.DataFrame, float]:
    """Taxa prevista, resíduo, z (amostral e total), quadrante e destaque por município.

    Retorna também o desvio-padrão estimado da variação não explicada pelo contexto (em p.p.).
    """
    real = df[config.COL_ALVO].astype(float)
    prevista = 100 * df["prob_prevista"].astype(float)
    n_alunos = df[config.COL_PESO].astype(float)

    residuo = real - prevista
    p = prevista / 100
    var_amostral = 100**2 * p * (1 - p) / n_alunos
    var_nao_explicada = max(float(np.mean(residuo**2) - np.mean(var_amostral)), 0.0)

    z_amostral = residuo / np.sqrt(var_amostral)
    z_total = residuo / np.sqrt(var_amostral + var_nao_explicada)

    quadrante = [QUADRANTES[(r < limiar, v < limiar)] for r, v in zip(real, prevista)]
    destaque = np.select(
        [z_total >= Z_DESTAQUE, z_total <= -Z_DESTAQUE],
        ["Supera o esperado", "Abaixo do esperado"],
        default="Dentro da variação esperada",
    )

    saida = pd.DataFrame({
        "id_municipio": df[config.COL_ID],
        "nome_municipio": df["nome_municipio"],
        "uf": df["ter_uf"],
        "regiao": df["ter_regiao"],
        "n_alunos": n_alunos.astype(int),
        "taxa_real": real.round(1),
        "taxa_prevista": prevista.round(1),
        "residuo_pp": residuo.round(1),
        "z_amostral": z_amostral.round(2),
        "z": z_total.round(2),
        "quadrante": quadrante,
        "destaque": destaque,
        "conjunto": df["conjunto"],
    })
    return saida.sort_values("taxa_prevista").reset_index(drop=True), float(np.sqrt(var_nao_explicada))


def resumo_por_uf(ranking: pd.DataFrame) -> pd.DataFrame:
    """Indicadores por UF: taxas ponderadas, % abaixo da meta (real e prevista) e destaques."""
    def _resumo(g: pd.DataFrame) -> pd.Series:
        return pd.Series({
            "municipios": len(g),
            "alunos": g["n_alunos"].sum(),
            "taxa_real_ponderada": np.average(g["taxa_real"], weights=g["n_alunos"]),
            "taxa_prevista_ponderada": np.average(g["taxa_prevista"], weights=g["n_alunos"]),
            "pct_abaixo_meta_real": 100 * (g["taxa_real"] < config.LIMIAR_RISCO).mean(),
            "pct_abaixo_meta_prevista": 100 * (g["taxa_prevista"] < config.LIMIAR_RISCO).mean(),
            "n_supera_esperado": int((g["destaque"] == "Supera o esperado").sum()),
            "n_abaixo_esperado": int((g["destaque"] == "Abaixo do esperado").sum()),
        })

    return ranking.groupby("uf").apply(_resumo).sort_values("taxa_real_ponderada")


def main() -> None:
    base = carregar_base()
    features = config.selecionar_features(base.columns)

    ranking, sigma_nao_explicado = classificar(previsoes_cross_fitted(base, features))
    SAIDA.parent.mkdir(parents=True, exist_ok=True)
    ranking.to_csv(SAIDA, index=False)

    print(f"{len(ranking):,} municípios previstos (todos por modelos que não os viram)")
    print(f"Desvio-padrão da variação não explicada pelo contexto: {sigma_nao_explicado:.1f} p.p.\n")
    print("Quadrantes:")
    print(ranking["quadrante"].value_counts().to_string())
    print("\nDestaques (|z| >= 2, incerteza total):")
    print(ranking["destaque"].value_counts().to_string())
    print(f"\nSalvo: reports/{SAIDA.name}")


if __name__ == "__main__":
    main()