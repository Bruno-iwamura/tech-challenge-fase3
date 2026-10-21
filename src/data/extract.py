"""Extração das fontes brutas para data/raw/ (Parquet).

Cada fonte é consultada no BigQuery uma única vez, já agregada no grão
necessário (município, ou município × categoria), e salva localmente.
Execuções seguintes reaproveitam o arquivo salvo: o BigQuery só é consultado
de novo com --forcar. Toda exploração e construção de features acontece
depois, localmente, sem custo.

Controle de custo:
    - Antes de cada consulta é feita uma estimativa (dry run). Algumas tabelas
      não informam a estimativa; isso é sinalizado, e o teto continua valendo.
    - --apenas-estimar mostra o custo das fontes sem executar nada.
    - --fontes restringe a extração às fontes indicadas.
    - Toda consulta tem o teto de bytes faturados definido em bq_client.

Uso:
    python -m src.data.extract --apenas-estimar                 # só estimativas
    python -m src.data.extract                                  # extrai o que ainda não existe
    python -m src.data.extract --forcar --fontes silver_alunos  # reextrai fontes específicas
"""

import argparse
from pathlib import Path

from src.data.bq_client import (
    DATASET_GOLD,
    MAX_BYTES_BILLED,
    PROJECT_ID,
    estimate_bytes,
    format_bytes,
    get_client,
    run_query,
)
from src.data.inep import ANO as ANO_INEP
from src.data.inep import COLUNAS_INEP, LOCALIZACAO, REDES_EXTRAIDAS

RAW_DIR = Path(__file__).resolve().parents[2] / "data" / "raw"

BD = "basedosdados"
T_GOLD_BASE = f"`{PROJECT_ID}.{DATASET_GOLD}.metas_resultados_municipio_enriquecido_2024`"
T_GOLD_EVOL = f"`{PROJECT_ID}.{DATASET_GOLD}.evolucao_indicador_municipio`"
T_SILVER = f"`{PROJECT_ID}.silver.consolidado_municipio`"

# Indicadores municipais do SIOPE/FUNDEB (ver basedosdados.br_fnde_fundeb.dicionario),
# ano-base 2023 (precede o alvo de 2024).
# Excluídos de propósito: 65 e 66 (IDEB) — embutem proficiência no Saeb, ou seja,
# desempenho escolar; usá-los para prever alfabetização seria vazamento conceitual.
INDICADORES_FUNDEB = {
    "56": "invest_aluno_educacao_basica",
    "45": "invest_aluno_fundamental",
    "44": "invest_aluno_educacao_infantil",
    "85": "invest_aluno_pre_escola",
    "58": "despesa_professores_aluno",
    "35": "pct_despesa_educacao_total",
    "61": "pct_invest_aluno_pib_pc",
}
_IDS_FUNDEB_SQL = ", ".join(f"'{i}'" for i in INDICADORES_FUNDEB)
_COLUNAS_INEP_SQL = ",\n          ".join(COLUNAS_INEP)
_REDES_INEP_SQL = ", ".join(f"'{r}'" for r in REDES_EXTRAIDAS)

FONTES = {
    # Gold: metas municipais de 2024 e o enriquecimento anterior (usado só para validação).
    "gold_base": f"SELECT * FROM {T_GOLD_BASE}",
    # Histórico da Gold (mantido por rastreabilidade do notebook 01; a base não depende dele).
    "gold_evolucao": f"SELECT id_municipio, indicador_2023 FROM {T_GOLD_EVOL}",
    # Resultados da rede municipal (rede 3) por município e ano: alvo, histórico e alunos.
    # Silver deduplicada: as cópias por join têm valores idênticos (ver notebook 00).
    "silver_alunos": f"""
        SELECT
          id_municipio,
          ano,
          ANY_VALUE(indicador_alfabetizacao) AS indicador_alfabetizacao,
          ANY_VALUE(qtd_alunos) AS qtd_alunos,
          ANY_VALUE(peso_total) AS peso_total
        FROM {T_SILVER}
        WHERE rede = '3'
        GROUP BY id_municipio, ano
    """,
    # NÃO USAR COMO FEATURE: a carga de 2023 desta tabela tem colunas deslocadas
    # (identidades composicionais violadas; notebook 01, seção 6.1). Mantida apenas
    # como evidência. As features do Censo Escolar vêm das planilhas oficiais do INEP
    # (src.data.download_inep + src.data.inep_oficial).
    "inep_indicadores_2023": f"""
        SELECT
          id_municipio,
          rede,
          {_COLUNAS_INEP_SQL}
        FROM `{BD}.br_inep_indicadores_educacionais.municipio`
        WHERE ano = {ANO_INEP}
          AND localizacao = '{LOCALIZACAO}'
          AND rede IN ({_REDES_INEP_SQL})
    """,
    # PIB municipal (IBGE), vários anos: o ano usado é identificado na validação.
    "ibge_pib": f"""
        SELECT ano, id_municipio, pib
        FROM `{BD}.br_ibge_pib.municipio`
        WHERE ano >= 2019
    """,
    # População municipal (IBGE), vários anos: denominador do PIB per capita.
    "ibge_populacao": f"""
        SELECT ano, id_municipio, populacao
        FROM `{BD}.br_ibge_populacao.municipio`
        WHERE ano >= 2019
    """,
    # Censo 2022 agregado por município (uma linha por município).
    "censo2022_municipio": f"""
        SELECT
          id_municipio,
          populacao,
          area,
          domicilios,
          taxa_alfabetizacao,
          idade_mediana,
          indice_envelhecimento,
          razao_sexo,
          populacao_indigena,
          populacao_quilombola
        FROM `{BD}.br_ibge_censo_2022.municipio`
    """,
    # Alfabetização 15+ por faixa de idade (permite taxa 15+ e taxa da geração dos pais).
    "censo2022_alfabetizacao": f"""
        SELECT
          id_municipio,
          grupo_idade,
          alfabetizacao,
          SUM(populacao) AS populacao
        FROM `{BD}.br_ibge_censo_2022.alfabetizacao_grupo_idade_sexo_raca`
        GROUP BY id_municipio, grupo_idade, alfabetizacao
    """,
    # Esgotamento sanitário por tipo (tipos com sobreposição; ver notebook 01).
    "censo2022_esgotamento": f"""
        SELECT
          id_municipio,
          tipo_esgotamento_sanitario,
          SUM(populacao) AS populacao
        FROM `{BD}.br_ibge_censo_2022.caracteristica_domicilio_grupo_idade_raca_esgotamento_sanitario`
        GROUP BY id_municipio, tipo_esgotamento_sanitario
    """,
    # Indicadores SIOPE/FUNDEB de 2023 (tabela particionada por ano: lê só a partição 2023).
    # Pior caso conhecido pelos metadados: a tabela inteira tem ~318 MiB.
    "fundeb_2023": f"""
        SELECT
          id_municipio,
          bimestre,
          id_indicador,
          valor_real,
          valor_percentual
        FROM `{BD}.br_fnde_fundeb.indicador_municipal`
        WHERE ano = 2023
          AND id_indicador IN ({_IDS_FUNDEB_SQL})
    """,
}


def extrair(
    nome: str,
    sql: str,
    client,
    forcar: bool = False,
    apenas_estimar: bool = False,
) -> int | None:
    """Extrai uma fonte para data/raw/<nome>.parquet.

    Retorna os bytes estimados, 0 se a fonte foi pulada (já existe), ou None
    se o BigQuery não informou a estimativa.
    """
    destino = RAW_DIR / f"{nome}.parquet"

    if destino.exists() and not forcar:
        print(f"[{nome}] já existe, pulando ({destino.name})")
        return 0

    n_bytes = estimate_bytes(sql, client=client)
    if n_bytes is None:
        print(
            f"[{nome}] estimativa indisponível (o BigQuery não informou); "
            f"o teto de {format_bytes(MAX_BYTES_BILLED)} por query continua valendo"
        )
    else:
        print(f"[{nome}] estimativa: {format_bytes(n_bytes)}")

    if apenas_estimar:
        return n_bytes

    df = run_query(sql, client=client)
    df.to_parquet(destino, index=False)
    print(f"[{nome}] {len(df):,} linhas salvas em data/raw/{destino.name}")
    return n_bytes


def main() -> None:
    parser = argparse.ArgumentParser(description="Extrai as fontes brutas para data/raw/.")
    parser.add_argument("--forcar", action="store_true", help="reextrai mesmo se o arquivo existir")
    parser.add_argument(
        "--apenas-estimar",
        action="store_true",
        help="mostra as estimativas de custo sem executar nenhuma consulta",
    )
    parser.add_argument(
        "--fontes",
        nargs="+",
        choices=list(FONTES),
        help="extrai apenas as fontes indicadas (padrão: todas)",
    )
    args = parser.parse_args()

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    client = get_client()
    fontes = args.fontes or list(FONTES)

    total = 0
    sem_estimativa = []
    for nome in fontes:
        n_bytes = extrair(
            nome, FONTES[nome], client, forcar=args.forcar, apenas_estimar=args.apenas_estimar
        )
        if n_bytes is None:
            sem_estimativa.append(nome)
        else:
            total += n_bytes

    rotulo = "Total estimado" if args.apenas_estimar else "Total processado (estimado)"
    print(f"\n{rotulo}: {format_bytes(total)}")
    if sem_estimativa:
        print(f"Sem estimativa disponível: {', '.join(sem_estimativa)}")


if __name__ == "__main__":
    main()