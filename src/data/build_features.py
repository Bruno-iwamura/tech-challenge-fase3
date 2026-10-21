"""Construção da base de modelagem.

Lê as fontes brutas de data/raw/ (geradas por src.data.extract, src.data.download_inep
e src.data.inep_oficial), padroniza o identificador de município, deriva as
features e salva:

    data/processed/base_modelagem.parquet   -> uma linha por município
    reports/dicionario_base_modelagem.csv   -> dicionário de dados (versionado)

População: municípios com resultado da rede municipal (rede 3) em 2024 na Silver.
Os municípios sem indicador são descartados. A Gold é usada apenas para as metas
municipais (ausentes nos municípios que ficaram fora dela por não terem meta).

Famílias de features (prefixos):
    ce_   Censo Escolar 2023 (planilhas oficiais do INEP; localização Total,
          dependência Municipal — mesma rede do alvo)
    soc_  socioeconômicas e demográficas (Censo 2022 e PIB per capita 2023, IBGE)
    fin_  financiamento da educação (SIOPE/FUNDEB 2023, consolidado anual)
    ter_  território (região e UF, derivadas do código IBGE)

Colunas que não são features:
    id_municipio, nome_municipio      identificação
    indicador_alfabetizacao           alvo (0–100, rede municipal, 2024)
    meta_alfabetizacao, atingiu_meta  enquadramento com metas / análise
                                      (vazamento no enquadramento só-contexto)
    indicador_2023                    histórico (só no enquadramento com defasagem)
    n_alunos                          peso de treino (alunos avaliados, rede municipal, 2024)

Nulos NÃO são preenchidos aqui: a imputação acontece dentro do pipeline de
modelagem, ajustada apenas nos dados de treino de cada fold.

Validações: notebooks 00 e 01.

Uso:
    python -m src.data.build_features
"""

from pathlib import Path

import numpy as np
import pandas as pd

from src.data.inep_oficial import DESCRICOES as DESCRICOES_INEP
from src.data.inep_oficial import MAPA as MAPA_INEP

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "raw"
PROCESSED = ROOT / "data" / "processed"
REPORTS = ROOT / "reports"

ANO_ALVO = 2024
ANO_HISTORICO = 2023
ANO_PIB = 2023  # ano que reproduz a Gold (notebook 01, seção 7)

FONTE_INEP = "INEP — Indicadores Educacionais 2023 (planilhas oficiais; localização Total, dependência Municipal)"
FONTE_CENSO = "IBGE — Censo Demográfico 2022 (Base dos Dados)"
FONTE_PIB = "IBGE — PIB dos Municípios 2023 ÷ população 2023 (Base dos Dados)"
FONTE_SIOPE = "FNDE/SIOPE — indicadores municipais 2023, bimestre 6 (Base dos Dados)"

# Código IBGE: os 2 primeiros dígitos identificam a UF; o primeiro, a grande região.
UF_POR_CODIGO = {
    "11": "RO", "12": "AC", "13": "AM", "14": "RR", "15": "PA", "16": "AP", "17": "TO",
    "21": "MA", "22": "PI", "23": "CE", "24": "RN", "25": "PB", "26": "PE", "27": "AL",
    "28": "SE", "29": "BA", "31": "MG", "32": "ES", "33": "RJ", "35": "SP", "41": "PR",
    "42": "SC", "43": "RS", "50": "MS", "51": "MT", "52": "GO", "53": "DF",
}
REGIAO_POR_DIGITO = {"1": "Norte", "2": "Nordeste", "3": "Sudeste", "4": "Sul", "5": "Centro-Oeste"}

# Alfabetização da geração dos pais das crianças do 2º ano.
FAIXAS_PAIS = ["25 a 34 anos", "35 a 44 anos"]

# Esgotamento: o agregado é a soma exata de dois componentes (validado no notebook 01).
ESGOTO_AGREGADO = "Rede geral, rede pluvial ou fossa ligada à rede"
ESGOTO_PRECARIO = [
    "Fossa rudimentar ou buraco",
    "Vala",
    "Rio, lago, córrego ou mar",
    "Outra forma",
    "Não tinham banheiro nem sanitário",
]

# SIOPE/FUNDEB: valores acumulados no ano; o bimestre 6 é o consolidado anual.
BIMESTRE_ANUAL = 6
# id -> (nome da feature, coluna de valor, descrição)
# Indicador 61 descartado (mediana 0,00 em todos os bimestres).
# Indicadores 65/66 (IDEB) nem são extraídos: embutem desempenho escolar.
FUNDEB_FEATURES = {
    "56": ("fin_invest_aluno_educacao_basica", "valor_real",
           "Investimento educacional por aluno da educação básica (R$)"),
    "45": ("fin_invest_aluno_fundamental", "valor_real",
           "Investimento educacional por aluno do ensino fundamental (R$)"),
    "44": ("fin_invest_aluno_infantil", "valor_real",
           "Investimento educacional por aluno da educação infantil (R$)"),
    "85": ("fin_invest_aluno_pre_escola", "valor_real",
           "Investimento educacional por aluno da pré-escola (R$)"),
    "58": ("fin_despesa_professores_aluno", "valor_real",
           "Despesa com professores por aluno da educação básica (R$)"),
    "35": ("fin_pct_despesa_educacao", "valor_percentual",
           "% das despesas municipais totais destinadas à educação"),
}


def _padronizar_id(serie: pd.Series) -> pd.Series:
    """Código IBGE do município como texto de 7 dígitos."""
    return serie.astype("string").str.strip().str.zfill(7)


def _ler(nome: str) -> pd.DataFrame:
    return pd.read_parquet(RAW / f"{nome}.parquet")


def _mapa_censo_escolar() -> dict[str, tuple[str, str]]:
    """Mapa LEGADO: coluna da Gold (enriquecimento anterior) -> (nome da feature, descrição).

    Não é usado na construção da base. Mantido porque o notebook 01 (seções 6.1
    e 6.2) o utiliza para comparar a Gold com as planilhas oficiais do INEP.
    """
    mapa: dict[str, tuple[str, str]] = {}

    mat = "mat_media_de_alunos_por_turma_etapas_de_ensino_"
    mapa.update({
        f"{mat}educacao_infantil_total_1_ed_inf_cat_0":
            ("ce_alunos_turma_infantil", "Média de alunos por turma — educação infantil"),
        f"{mat}educacao_infantil_creche_cre_cat_0":
            ("ce_alunos_turma_creche", "Média de alunos por turma — creche"),
        f"{mat}educacao_infantil_pre_escola_pre_cat_0":
            ("ce_alunos_turma_pre_escola", "Média de alunos por turma — pré-escola"),
        f"{mat}ensino_fundamental_anos_iniciais_fun_ai_cat_0":
            ("ce_alunos_turma_anos_iniciais", "Média de alunos por turma — anos iniciais"),
        f"{mat}ensino_fundamental_1o_ano_fun_01_cat_0":
            ("ce_alunos_turma_1ano", "Média de alunos por turma — 1º ano"),
        f"{mat}ensino_fundamental_2_ano_fun_02_cat_0":
            ("ce_alunos_turma_2ano", "Média de alunos por turma — 2º ano"),
        f"{mat}ensino_fundamental_turmas_multietapa_multi_ou_correcao_de_fluxo_2_mult_eta_cat_0":
            ("ce_alunos_turma_multietapa", "Média de alunos por turma — turmas multietapa"),
    })

    had = "had_"
    mapa.update({
        f"{had}educacao_infantil_total_ed_inf_cat_0":
            ("ce_horas_aula_infantil", "Média de horas-aula diárias — educação infantil"),
        f"{had}educacao_infantil_creche_cre_cat_0":
            ("ce_horas_aula_creche", "Média de horas-aula diárias — creche"),
        f"{had}educacao_infantil_pre_escola_pre_cat_0":
            ("ce_horas_aula_pre_escola", "Média de horas-aula diárias — pré-escola"),
        f"{had}ensino_fundamental_anos_iniciais_fun_ai_cat_0":
            ("ce_horas_aula_anos_iniciais", "Média de horas-aula diárias — anos iniciais"),
        f"{had}ensino_fundamental_1o_ano_fun_01_cat_0":
            ("ce_horas_aula_1ano", "Média de horas-aula diárias — 1º ano"),
        f"{had}ensino_fundamental_2o_ano_fun_02_cat_0":
            ("ce_horas_aula_2ano", "Média de horas-aula diárias — 2º ano"),
    })

    afd = (
        "afd_percentual_de_docentes1_2_por_grupo4_de_adequacao_da_formacao_"
        "a_disciplina_que_leciona_e_etapa_modalidade_de_ensino_"
    )
    for k in range(1, 6):
        mapa[f"{afd}educacao_infantil_grupo_{k}_ed_inf_cat_{k}"] = (
            f"ce_adeq_docente_infantil_g{k}",
            f"% de docentes no grupo {k} de adequação da formação — educação infantil",
        )
        mapa[f"{afd}ensino_fundamental_anos_iniciais_grupo_{k}_fun_ai_cat_{k}"] = (
            f"ce_adeq_docente_anos_iniciais_g{k}",
            f"% de docentes no grupo {k} de adequação da formação — anos iniciais",
        )

    ird = (
        "ird_percentual_de_escolas_segundo_faixa_do_indicador_de_regularidade_"
        "do_docente_ird_1_2_"
    )
    mapa.update({
        f"{ird}baixa_regularidade_0_2_edu_bas_cat_1":
            ("ce_regularidade_baixa", "% de escolas com regularidade docente baixa (0 a 2)"),
        f"{ird}media_baixa_2_3_edu_bas_cat_2":
            ("ce_regularidade_media_baixa", "% de escolas com regularidade docente média-baixa (2 a 3)"),
        f"{ird}media_alta_3_4_edu_bas_cat_3":
            ("ce_regularidade_media_alta", "% de escolas com regularidade docente média-alta (3 a 4)"),
        f"{ird}alta_4_5_edu_bas_cat_4":
            ("ce_regularidade_alta", "% de escolas com regularidade docente alta (4 a 5)"),
    })

    tdi = "tdi_ensino_fundamental_"
    mapa.update({
        f"{tdi}anos_iniciais_fun_ai_cat_0":
            ("ce_distorcao_anos_iniciais", "Taxa de distorção idade-série — anos iniciais"),
        f"{tdi}1o_ano_fun_01_cat_0":
            ("ce_distorcao_1ano", "Taxa de distorção idade-série — 1º ano"),
        f"{tdi}2o_ano_fun_02_cat_0":
            ("ce_distorcao_2ano", "Taxa de distorção idade-série — 2º ano"),
    })

    tre = "tre_taxa_de_{}_ensino_fundamental_de_8_e_9_anos_anos_iniciais_{}_cat_fun_ai"
    mapa.update({
        tre.format("aprovacao", 1):
            ("ce_aprovacao_anos_iniciais", "Taxa de aprovação — anos iniciais"),
        tre.format("reprovacao", 2):
            ("ce_reprovacao_anos_iniciais", "Taxa de reprovação — anos iniciais"),
        tre.format("abandono", 3):
            ("ce_abandono_anos_iniciais", "Taxa de abandono — anos iniciais"),
    })

    dsu = "dsu_percentual_de_funcoes_docentes_com_curso_superior_por_etapa_modalidade_de_ensino_"
    mapa.update({
        f"{dsu}educacao_infantil_total_ed_inf_cat_0":
            ("ce_docentes_superior_infantil", "% de funções docentes com curso superior — educação infantil"),
        f"{dsu}educacao_infantil_creche_cre_cat_0":
            ("ce_docentes_superior_creche", "% de funções docentes com curso superior — creche"),
        f"{dsu}educacao_infantil_pre_escola_pre_cat_0":
            ("ce_docentes_superior_pre_escola", "% de funções docentes com curso superior — pré-escola"),
        f"{dsu}ensino_fundamental_anos_iniciais_fun_ai_cat_0":
            ("ce_docentes_superior_anos_iniciais", "% de funções docentes com curso superior — anos iniciais"),
    })

    return mapa


def _base_alvo() -> tuple[pd.DataFrame, int]:
    """Alvo, número de alunos (2024) e histórico (2023) da rede municipal, a partir da Silver."""
    silver = _ler("silver_alunos")
    silver["id_municipio"] = _padronizar_id(silver["id_municipio"])

    alvo = silver.loc[silver["ano"] == ANO_ALVO, ["id_municipio", "indicador_alfabetizacao", "qtd_alunos"]]
    alvo = alvo.rename(columns={"qtd_alunos": "n_alunos"})
    sem_alvo = int(alvo["indicador_alfabetizacao"].isna().sum())
    alvo = alvo.dropna(subset=["indicador_alfabetizacao"])

    historico = silver.loc[silver["ano"] == ANO_HISTORICO, ["id_municipio", "indicador_alfabetizacao"]]
    historico = historico.rename(columns={"indicador_alfabetizacao": "indicador_2023"})

    base = alvo.merge(historico, on="id_municipio", how="left", validate="one_to_one")
    return base, sem_alvo


def _territorio(ids: pd.Series) -> pd.DataFrame:
    """UF e grande região derivadas do código IBGE."""
    return pd.DataFrame({
        "id_municipio": ids.to_numpy(),
        "ter_regiao": ids.str[0].map(REGIAO_POR_DIGITO).to_numpy(),
        "ter_uf": ids.str[:2].map(UF_POR_CODIGO).to_numpy(),
    })


def _features_censo_escolar() -> tuple[pd.DataFrame, list[tuple]]:
    """Nome do município + 37 indicadores das planilhas oficiais do INEP."""
    inep = _ler("inep_oficial_2023")
    inep["id_municipio"] = _padronizar_id(inep["id_municipio"])

    nomes = [nome for codigos in MAPA_INEP.values() for nome in codigos.values()]
    df = inep[["id_municipio", "nome_municipio", *nomes]]
    dicionario = [(nome, "ce", DESCRICOES_INEP[nome], FONTE_INEP) for nome in nomes]
    return df, dicionario


def _features_socioeconomicas(
    censo: pd.DataFrame,
    alf: pd.DataFrame,
    esg: pd.DataFrame,
    pib: pd.DataFrame,
    populacao: pd.DataFrame,
) -> tuple[pd.DataFrame, list[tuple]]:
    # Censo 2022 — tabela municipal
    c = censo.copy()
    c["id_municipio"] = _padronizar_id(c["id_municipio"])
    pop_censo = c["populacao"].astype("float")
    soc = pd.DataFrame({
        "id_municipio": c["id_municipio"],
        "soc_populacao": pop_censo,
        "soc_densidade": pop_censo / c["area"].astype("float").replace(0, np.nan),
        "soc_moradores_domicilio": pop_censo / c["domicilios"].astype("float").replace(0, np.nan),
        "soc_alfab_15_mais": 100 * c["taxa_alfabetizacao"],  # tabela usa escala 0–1
        "soc_idade_mediana": c["idade_mediana"],
        "soc_indice_envelhecimento": c["indice_envelhecimento"],
        "soc_pct_indigena": 100 * c["populacao_indigena"].astype("float") / pop_censo,
        "soc_pct_quilombola": 100 * c["populacao_quilombola"].astype("float") / pop_censo,
    })

    # Alfabetização da geração dos pais (25 a 44 anos)
    a = alf[alf["grupo_idade"].isin(FAIXAS_PAIS)].copy()
    a["alfabetizadas"] = a["populacao"].where(a["alfabetizacao"] == "Alfabetizadas", 0)
    pais = a.groupby("id_municipio")[["alfabetizadas", "populacao"]].sum().reset_index()
    pais["soc_alfab_25_44"] = 100 * pais["alfabetizadas"].astype("float") / pais["populacao"]
    pais["id_municipio"] = _padronizar_id(pais["id_municipio"])
    pais = pais[["id_municipio", "soc_alfab_25_44"]]

    # Esgotamento sanitário (total exclui o agregado para não contar duas vezes)
    e = esg.pivot_table(
        index="id_municipio",
        columns="tipo_esgotamento_sanitario",
        values="populacao",
        aggfunc="sum",
    ).fillna(0)
    total = e.drop(columns=ESGOTO_AGREGADO).sum(axis=1)
    esgoto = pd.DataFrame({
        "soc_pct_esgoto_rede": 100 * e[ESGOTO_AGREGADO] / total,
        "soc_pct_esgoto_precario": 100 * e[ESGOTO_PRECARIO].sum(axis=1) / total,
    }).reset_index()
    esgoto["id_municipio"] = _padronizar_id(esgoto["id_municipio"])

    # PIB per capita (IBGE): PIB do ano ÷ população do mesmo ano
    p = pib[pib["ano"] == ANO_PIB].merge(
        populacao[populacao["ano"] == ANO_PIB], on=["ano", "id_municipio"], validate="one_to_one"
    )
    p["id_municipio"] = _padronizar_id(p["id_municipio"])
    p["soc_pib_per_capita"] = p["pib"].astype("float") / p["populacao"].astype("float")
    pib_pc = p[["id_municipio", "soc_pib_per_capita"]]

    df = (
        soc.merge(pais, on="id_municipio", how="left", validate="one_to_one")
        .merge(esgoto, on="id_municipio", how="left", validate="one_to_one")
        .merge(pib_pc, on="id_municipio", how="left", validate="one_to_one")
    )

    dicionario = [
        ("soc_populacao", "soc", "População residente", FONTE_CENSO),
        ("soc_densidade", "soc", "Densidade demográfica (hab/km²)", FONTE_CENSO),
        ("soc_moradores_domicilio", "soc", "Moradores por domicílio", FONTE_CENSO),
        ("soc_alfab_15_mais", "soc", "Taxa de alfabetização da população de 15 anos ou mais (%)", FONTE_CENSO),
        ("soc_alfab_25_44", "soc", "Taxa de alfabetização de 25 a 44 anos — geração dos pais (%)", FONTE_CENSO),
        ("soc_idade_mediana", "soc", "Idade mediana da população", FONTE_CENSO),
        ("soc_indice_envelhecimento", "soc", "Índice de envelhecimento", FONTE_CENSO),
        ("soc_pct_indigena", "soc",
         "% da população indígena (contexto de política educacional; não interpretar como causa)", FONTE_CENSO),
        ("soc_pct_quilombola", "soc",
         "% da população quilombola (contexto de política educacional; não interpretar como causa)", FONTE_CENSO),
        ("soc_pct_esgoto_rede", "soc", "% da população com esgoto por rede geral, pluvial ou fossa ligada à rede", FONTE_CENSO),
        ("soc_pct_esgoto_precario", "soc",
         "% da população com esgotamento precário (fossa rudimentar, vala, rio/lago/mar, outra forma, sem banheiro)",
         FONTE_CENSO),
        ("soc_pib_per_capita", "soc", "PIB per capita do município, 2023 (R$)", FONTE_PIB),
    ]
    return df, dicionario


def _features_financiamento(fundeb: pd.DataFrame) -> tuple[pd.DataFrame, list[tuple]]:
    f = fundeb[
        (fundeb["bimestre"] == BIMESTRE_ANUAL) & fundeb["id_indicador"].isin(FUNDEB_FEATURES)
    ].copy()
    usa_valor_real = f["id_indicador"].map(lambda i: FUNDEB_FEATURES[i][1] == "valor_real")
    f["valor"] = f["valor_real"].where(usa_valor_real, f["valor_percentual"])

    largo = f.pivot(index="id_municipio", columns="id_indicador", values="valor")
    largo = largo.rename(columns={i: nome for i, (nome, _, _) in FUNDEB_FEATURES.items()})
    largo.columns.name = None
    largo = largo.reset_index()
    largo["id_municipio"] = _padronizar_id(largo["id_municipio"])

    dicionario = [(nome, "fin", desc, FONTE_SIOPE) for nome, _, desc in FUNDEB_FEATURES.values()]
    return largo, dicionario


def _metas_gold() -> pd.DataFrame:
    """Metas municipais de 2024 (só existem para os municípios da Gold)."""
    gold = _ler("gold_base")
    gold["id_municipio"] = _padronizar_id(gold["id_municipio"])
    return gold[["id_municipio", "meta_alfabetizacao", "atingiu_meta"]]


def construir_base() -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Monta a base de modelagem, o dicionário de dados e um resumo da construção."""
    base, sem_alvo = _base_alvo()
    n_inicial = len(base)

    ce, dic_ce = _features_censo_escolar()
    soc, dic_soc = _features_socioeconomicas(
        _ler("censo2022_municipio"),
        _ler("censo2022_alfabetizacao"),
        _ler("censo2022_esgotamento"),
        _ler("ibge_pib"),
        _ler("ibge_populacao"),
    )
    fin, dic_fin = _features_financiamento(_ler("fundeb_2023"))

    for parte in [_territorio(base["id_municipio"]), _metas_gold(), ce, soc, fin]:
        base = base.merge(parte, on="id_municipio", how="left", validate="one_to_one")

    if len(base) != n_inicial:
        raise ValueError(f"Número de linhas mudou na junção: {n_inicial} -> {len(base)}")

    nao_features = [
        "id_municipio", "nome_municipio",
        "indicador_alfabetizacao", "meta_alfabetizacao", "atingiu_meta",
        "indicador_2023", "n_alunos",
    ]
    features = (
        ["ter_regiao", "ter_uf"]
        + [nome for nome, *_ in dic_ce]
        + [nome for nome, *_ in dic_soc]
        + [nome for nome, *_ in dic_fin]
    )
    base = base[nao_features + features]

    dic_base = [
        ("id_municipio", "id", "Código IBGE do município (7 dígitos)", "IBGE"),
        ("nome_municipio", "id", "Nome do município", "INEP — planilhas oficiais"),
        ("indicador_alfabetizacao", "alvo",
         "% de alunos alfabetizados ao fim do 2º ano, rede municipal (0–100), 2024",
         "INEP/MEC — Indicador Criança Alfabetizada (Silver, rede municipal)"),
        ("meta_alfabetizacao", "meta",
         "Meta municipal do indicador para 2024 (0–100); ausente nos municípios sem meta publicada",
         "MEC (via Gold)"),
        ("atingiu_meta", "meta",
         "Indica se o município atingiu a meta municipal de 2024; ausente sem meta", "Gold"),
        ("indicador_2023", "historico", "Indicador de 2023, rede municipal (ausente para SP)",
         "INEP/MEC (Silver, rede municipal)"),
        ("n_alunos", "peso", "Alunos avaliados na rede municipal em 2024", "Silver (deduplicada, rede municipal)"),
        ("ter_regiao", "ter", "Grande região (derivada do código IBGE)", "IBGE"),
        ("ter_uf", "ter", "Sigla da UF (derivada do código IBGE)", "IBGE"),
    ]
    dicionario = pd.DataFrame(
        dic_base + dic_ce + dic_soc + dic_fin,
        columns=["coluna", "familia", "descricao", "fonte"],
    )
    resumo = {"sem_alvo_descartados": sem_alvo, "com_meta": int(base["meta_alfabetizacao"].notna().sum())}
    return base, dicionario, resumo


def main() -> None:
    base, dicionario, resumo = construir_base()

    PROCESSED.mkdir(parents=True, exist_ok=True)
    REPORTS.mkdir(parents=True, exist_ok=True)
    base.to_parquet(PROCESSED / "base_modelagem.parquet", index=False)
    dicionario.to_csv(REPORTS / "dicionario_base_modelagem.csv", index=False)

    print(f"Base: {base.shape[0]:,} municípios × {base.shape[1]} colunas")
    print(f"  municípios sem indicador descartados: {resumo['sem_alvo_descartados']}")
    print(f"  municípios com meta municipal: {resumo['com_meta']:,}")
    print(f"  UFs presentes: {base['ter_uf'].nunique()}")
    for prefixo in ["ce_", "soc_", "fin_", "ter_"]:
        n = sum(col.startswith(prefixo) for col in base.columns)
        print(f"  {prefixo:<5} {n} features")

    nulos = base.isna().sum()
    nulos = nulos[nulos > 0].sort_values(ascending=False)
    print("\nColunas com nulos:")
    print(nulos.to_string() if len(nulos) else "  nenhuma")

    print("\nSalvo: data/processed/base_modelagem.parquet")
    print("Salvo: reports/dicionario_base_modelagem.csv")


if __name__ == "__main__":
    main()