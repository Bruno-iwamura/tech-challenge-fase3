"""Cliente BigQuery do projeto.

Autenticação via ADC (Application Default Credentials): a biblioteca encontra
sozinha as credenciais geradas por `gcloud auth application-default login`,
sem necessidade de chave JSON.

Download via endpoint REST (padrão): a BigQuery Storage Read API só compensa
para volumes grandes; para o tamanho desta base, REST é suficiente e evita uma
dependência extra.

Configuração (opcional, via variáveis de ambiente):
    GCP_PROJECT_ID   -> ID do projeto GCP (padrão: projeto do grupo)
    BQ_DATASET_GOLD  -> dataset da camada Gold (padrão: "gold")
"""

import os

import pandas as pd
from google.cloud import bigquery

PROJECT_ID = os.getenv("GCP_PROJECT_ID", "grupo-pos-fase-2-alfabetizacao")
DATASET_GOLD = os.getenv("BQ_DATASET_GOLD", "gold")


def get_client() -> bigquery.Client:
    """Retorna um cliente BigQuery autenticado via ADC."""
    return bigquery.Client(project=PROJECT_ID)


def run_query(sql: str, client: bigquery.Client | None = None) -> pd.DataFrame:
    """Executa uma query e retorna o resultado como DataFrame (via REST)."""
    client = client or get_client()
    return client.query(sql).to_dataframe(create_bqstorage_client=False)


def _smoke_test() -> None:
    """Teste rápido de conexão: lista as tabelas da Gold e roda uma query trivial."""
    client = get_client()
    print(f"Projeto: {client.project}")

    dataset_ref = f"{PROJECT_ID}.{DATASET_GOLD}"
    print(f"Tabelas em {dataset_ref}:")
    for table in client.list_tables(dataset_ref):
        print(f"  - {table.table_id}")

    df = run_query("SELECT 1 AS ok", client=client)
    print("Query de teste:", df.to_dict("records"))


if __name__ == "__main__":
    _smoke_test()