"""Leitura das planilhas oficiais dos Indicadores Educacionais do INEP (2023, municípios).

Fonte: planilhas baixadas em data/raw/inep_2023/ (ver src.data.download_inep).
Motivo: a tabela da Base dos Dados tem colunas deslocadas em 2023 (notebook 01,
seção 6.1); as planilhas oficiais são a fonte íntegra.

Estrutura das planilhas: linhas de título e notas, cabeçalho em vários níveis e
uma linha de códigos técnicos iniciada por NU_ANO_CENSO, imediatamente antes
dos dados. O leitor ancora o cabeçalho nessa linha (a posição varia entre
arquivos). Ausência é marcada com "--".

Linha extraída: localização Total e dependência administrativa Municipal — a
mesma rede do alvo (rede municipal). É também a linha usada no enriquecimento
anterior (Gold), verificado empiricamente no notebook 01, seção 6.2: em todos
os municípios em que apenas uma linha da planilha coincide com a Gold, essa
linha é Total/Municipal.

Saída: data/raw/inep_oficial_2023.parquet (uma linha por município, com o nome
do município e os 37 indicadores).

Uso:
    python -m src.data.inep_oficial
"""

from pathlib import Path

import pandas as pd

from src.data.inep import COLUNAS_INEP

ROOT = Path(__file__).resolve().parents[2]
PASTA = ROOT / "data" / "raw" / "inep_2023"
SAIDA = ROOT / "data" / "raw" / "inep_oficial_2023.parquet"

ANO = 2023
LOCALIZACAO = "Total"
DEPENDENCIA = "Municipal"
MARCADORES_AUSENCIA = {"--", ""}
INDICADOR_COM_NOME = "atu"  # o nome do município é lido desta planilha

# indicador -> {código na planilha: nome da feature}
MAPA = {
    "atu": {
        "ED_INF_CAT_0": "ce_alunos_turma_infantil",
        "CRE_CAT_0": "ce_alunos_turma_creche",
        "PRE_CAT_0": "ce_alunos_turma_pre_escola",
        "FUN_AI_CAT_0": "ce_alunos_turma_anos_iniciais",
        "FUN_01_CAT_0": "ce_alunos_turma_1ano",
        "FUN_02_CAT_0": "ce_alunos_turma_2ano",
        "MULT_ETA_CAT_0": "ce_alunos_turma_multietapa",
    },
    "had": {
        "ED_INF_CAT_0": "ce_horas_aula_infantil",
        "CRE_CAT_0": "ce_horas_aula_creche",
        "PRE_CAT_0": "ce_horas_aula_pre_escola",
        "FUN_AI_CAT_0": "ce_horas_aula_anos_iniciais",
        "FUN_01_CAT_0": "ce_horas_aula_1ano",
        "FUN_02_CAT_0": "ce_horas_aula_2ano",
    },
    "afd": {
        **{f"ED_INF_CAT_{k}": f"ce_adeq_docente_infantil_g{k}" for k in range(1, 6)},
        **{f"FUN_AI_CAT_{k}": f"ce_adeq_docente_anos_iniciais_g{k}" for k in range(1, 6)},
    },
    "ird": {
        "EDU_BAS_CAT_1": "ce_regularidade_baixa",
        "EDU_BAS_CAT_2": "ce_regularidade_media_baixa",
        "EDU_BAS_CAT_3": "ce_regularidade_media_alta",
        "EDU_BAS_CAT_4": "ce_regularidade_alta",
    },
    "tdi": {
        "FUN_AI_CAT_0": "ce_distorcao_anos_iniciais",
        "FUN_01_CAT_0": "ce_distorcao_1ano",
        "FUN_02_CAT_0": "ce_distorcao_2ano",
    },
    "tre": {
        "1_CAT_FUN_AI": "ce_aprovacao_anos_iniciais",
        "2_CAT_FUN_AI": "ce_reprovacao_anos_iniciais",
        "3_CAT_FUN_AI": "ce_abandono_anos_iniciais",
    },
    "dsu": {
        "ED_INF_CAT_0": "ce_docentes_superior_infantil",
        "CRE_CAT_0": "ce_docentes_superior_creche",
        "PRE_CAT_0": "ce_docentes_superior_pre_escola",
        "FUN_AI_CAT_0": "ce_docentes_superior_anos_iniciais",
    },
}

# Descrições reaproveitadas do mapeamento existente (mesmos nomes de feature).
DESCRICOES = {nome: descricao for nome, descricao in COLUNAS_INEP.values()}


def _planilha(sigla: str) -> Path:
    arquivos = sorted((PASTA / sigla).rglob("*.xlsx"))
    if len(arquivos) != 1:
        raise FileNotFoundError(f"[{sigla}] esperado 1 arquivo .xlsx em {PASTA / sigla}, encontrados {len(arquivos)}")
    return arquivos[0]


def ler_indicador(sigla: str) -> pd.DataFrame:
    """Lê uma planilha e devolve id_municipio, nome_municipio e as features da linha escolhida."""
    bruto = pd.read_excel(_planilha(sigla), sheet_name=0, header=None, dtype=object)

    primeira_coluna = bruto.iloc[:, 0].astype(str).str.strip()
    linhas_codigo = primeira_coluna.index[primeira_coluna == "NU_ANO_CENSO"]
    if len(linhas_codigo) != 1:
        raise ValueError(f"[{sigla}] linha de códigos (NU_ANO_CENSO) não encontrada de forma única")
    linha = linhas_codigo[0]

    dados = bruto.iloc[linha + 1:].copy()
    dados.columns = bruto.iloc[linha].astype(str).str.strip()

    # Mantém só linhas de dados do ano (descarta notas de rodapé) e a linha escolhida.
    dados = dados[pd.to_numeric(dados["NU_ANO_CENSO"], errors="coerce") == ANO]
    dados = dados[
        (dados["NO_CATEGORIA"].astype(str).str.strip() == LOCALIZACAO)
        & (dados["NO_DEPENDENCIA"].astype(str).str.strip() == DEPENDENCIA)
    ]

    codigos = MAPA[sigla]
    faltando = [c for c in codigos if c not in dados.columns]
    if faltando:
        disponiveis = [c for c in dados.columns if "CAT" in c]
        raise KeyError(f"[{sigla}] códigos não encontrados: {faltando}. Disponíveis: {disponiveis}")

    saida = pd.DataFrame({
        "id_municipio": pd.to_numeric(dados["CO_MUNICIPIO"]).astype("int64").astype(str).str.zfill(7).to_numpy(),
        "nome_municipio": dados["NO_MUNICIPIO"].astype(str).str.strip().to_numpy(),
    })
    for codigo, nome in codigos.items():
        valores = dados[codigo]
        numerico = pd.to_numeric(valores, errors="coerce")
        texto = valores.astype(str).str.strip()
        inesperados = texto[numerico.isna() & valores.notna() & ~texto.isin(MARCADORES_AUSENCIA)]
        if len(inesperados):
            print(f"  aviso [{sigla}/{codigo}]: {len(inesperados)} valores não numéricos inesperados, "
                  f"ex.: {inesperados.unique()[:5].tolist()}")
        saida[nome] = numerico.to_numpy()

    if saida["id_municipio"].duplicated().any():
        raise ValueError(f"[{sigla}] municípios duplicados após o filtro {LOCALIZACAO}/{DEPENDENCIA}")
    return saida


def construir() -> pd.DataFrame:
    """Junta os 7 indicadores em uma linha por município (nome lido de uma única planilha)."""
    base = None
    for sigla in MAPA:
        df = ler_indicador(sigla)
        print(f"[{sigla}] {len(df):,} municípios")
        if sigla != INDICADOR_COM_NOME:
            df = df.drop(columns="nome_municipio")
        base = df if base is None else base.merge(df, on="id_municipio", how="outer", validate="one_to_one")
    return base


def main() -> None:
    df = construir()
    df.to_parquet(SAIDA, index=False)
    n_indicadores = df.shape[1] - 2  # menos id e nome
    print(f"\n{len(df):,} municípios × {n_indicadores} indicadores -> data/raw/{SAIDA.name}")

    nulos = df.isna().sum()
    nulos = nulos[nulos > 0].sort_values(ascending=False)
    print("\nNulos por coluna:")
    print(nulos.to_string() if len(nulos) else "  nenhum")


if __name__ == "__main__":
    main()