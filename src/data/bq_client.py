"""Cliente BigQuery do projeto.

Autenticação via ADC (Application Default Credentials): a biblioteca encontra
sozinha as credenciais geradas por `gcloud auth application-default login`,
sem necessidade de chave JSON.

Download via endpoint REST (padrão): a BigQuery Storage Read API só compensa
para volumes grandes; para o tamanho desta base, REST é suficiente e evita uma
dependência extra.

Controle de custo:
    - estimate_bytes(): dry run — informa quantos bytes a query leria, sem
      executar e sem custo. Algumas tabelas não informam a estimativa; nesse
      caso a função retorna None.
    - run_query(): toda execução tem um teto de bytes faturados
      (maximum_bytes_billed). Se a query ultrapassaria o teto, ela falha
      antes de rodar e nada é cobrado. O teto vale mesmo quando a estimativa
      não está disponível.

Configuração (opcional, via variáveis de ambiente):
    GCP_PROJECT_ID       -> ID do projeto GCP (padrão: projeto do grupo)
    BQ_DATASET_GOLD      -> dataset da camada Gold (padrão: "gold")
    BQ_MAX_BYTES_BILLED  -> teto de bytes por query (padrão: 1 GiB)
"""

import os

import pandas as pd
from google.cloud import bigquery

PROJECT_ID = os.getenv("GCP_PROJECT_ID", "grupo-pos-fase-2-alfabetizacao")
DATASET_GOLD = os.getenv("BQ_DATASET_GOLD", "gold")
MAX_BYTES_BILLED = int(os.getenv("BQ_MAX_BYTES_BILLED", str(1024**3)))  # 1 GiB


def get_client() -> bigquery.Client:
    """Retorna um cliente BigQuery autenticado via ADC."""
    return bigquery.Client(project=PROJECT_ID)


def format_bytes(n_bytes: int | None) -> str:
    """Formata uma quantidade de bytes em unidade legível ('indisponível' se None)."""
    if n_bytes is None:
        return "indisponível"
    size = float(n_bytes)
    for unit in ["B", "KiB", "MiB", "GiB", "TiB"]:
        if size < 1024:
            return f"{size:,.1f} {unit}"
        size /= 1024
    return f"{size:,.1f} PiB"


def estimate_bytes(sql: str, client: bigquery.Client | None = None) -> int | None:
    """Dry run: bytes que a query processaria, sem executar e sem custo.

    Retorna None quando o BigQuery não informa a estimativa para as tabelas
    envolvidas. O teto de run_query() continua protegendo a execução.
    """
    client = client or get_client()
    job_config = bigquery.QueryJobConfig(dry_run=True, use_query_cache=False)
    job = client.query(sql, job_config=job_config)
    return job.total_bytes_processed


def run_query(
    sql: str,
    client: bigquery.Client | None = None,
    max_bytes_billed: int = MAX_BYTES_BILLED,
) -> pd.DataFrame:
    """Executa uma query com teto de bytes faturados e retorna um DataFrame (via REST)."""
    client = client or get_client()
    job_config = bigquery.QueryJobConfig(maximum_bytes_billed=max_bytes_billed)
    job = client.query(sql, job_config=job_config)
    return job.to_dataframe(create_bqstorage_client=False)


def _smoke_test() -> None:
    """Teste rápido de conexão: lista as tabelas da Gold e roda uma query trivial."""
    client = get_client()
    print(f"Projeto: {client.project}")
    print(f"Teto por query: {format_bytes(MAX_BYTES_BILLED)}")

    dataset_ref = f"{PROJECT_ID}.{DATASET_GOLD}"
    print(f"Tabelas em {dataset_ref}:")
    for table in client.list_tables(dataset_ref):
        print(f"  - {table.table_id}")

    sql = "SELECT 1 AS ok"
    print("Estimativa (dry run):", format_bytes(estimate_bytes(sql, client=client)))
    df = run_query(sql, client=client)
    print("Query de teste:", df.to_dict("records"))


if __name__ == "__main__":
    _smoke_test()