"""Download dos Indicadores Educacionais do INEP (2023, nível municipal).

Fonte oficial: https://www.gov.br/inep/pt-br/acesso-a-informacao/dados-abertos/indicadores-educacionais

Motivo: a tabela basedosdados.br_inep_indicadores_educacionais.municipio tem
colunas deslocadas na carga de 2023 (identidades composicionais violadas; ver
notebook 01, seção 6.1). As planilhas oficiais do INEP são a fonte íntegra.

Os arquivos .zip são salvos e extraídos em data/raw/inep_2023/ (fora do Git).

Salvaguardas:
    - Cache: um .zip já presente não é baixado de novo (use --forcar para
      rebaixar). Também permite o caminho manual: baixar pelo navegador e
      colocar o arquivo na pasta.
    - Download atômico: grava em .part e renomeia ao final.
    - Falhas isoladas: um arquivo com erro não interrompe os demais.

Uso:
    python -m src.data.download_inep
    python -m src.data.download_inep --forcar
"""

import argparse
import shutil
import urllib.error
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DESTINO = ROOT / "data" / "raw" / "inep_2023"

BASE_URL = "https://download.inep.gov.br/informacoes_estatisticas/indicadores_educacionais/2023"
ARQUIVOS = {
    "atu": f"{BASE_URL}/ATU_2023_MUNICIPIOS.zip",  # média de alunos por turma
    "had": f"{BASE_URL}/HAD_2023_MUNICIPIOS.zip",  # média de horas-aula diária (URL inferida pelo padrão)
    "afd": f"{BASE_URL}/AFD_2023_MUNICIPIOS.zip",  # adequação da formação docente
    "ird": f"{BASE_URL}/IRD_2023_MUNICIPIOS.zip",  # regularidade do corpo docente
    "tdi": f"{BASE_URL}/TDI_2023_MUNICIPIOS.zip",  # taxas de distorção idade-série
    "tre": f"{BASE_URL}/tx_rend_municipios_2023.zip",  # taxas de rendimento
    "dsu": f"{BASE_URL}/DSU_2023_MUNICIPIOS.zip",  # docentes com curso superior
}

TIMEOUT_S = 120
CABECALHOS = {"User-Agent": "Mozilla/5.0 (tech-challenge-fase3; download de dados abertos)"}


def baixar(nome: str, url: str, forcar: bool = False) -> Path | None:
    """Baixa um .zip para DESTINO. Retorna o caminho, ou None em caso de falha."""
    destino = DESTINO / Path(url).name

    if destino.exists() and not forcar:
        print(f"[{nome}] já existe, pulando ({destino.name})")
        return destino

    parcial = destino.with_suffix(destino.suffix + ".part")
    requisicao = urllib.request.Request(url, headers=CABECALHOS)
    try:
        with urllib.request.urlopen(requisicao, timeout=TIMEOUT_S) as resposta, open(parcial, "wb") as arquivo:
            shutil.copyfileobj(resposta, arquivo)
    except urllib.error.HTTPError as erro:
        parcial.unlink(missing_ok=True)
        print(f"[{nome}] ERRO HTTP {erro.code} em {url}")
        return None
    except urllib.error.URLError as erro:
        parcial.unlink(missing_ok=True)
        print(f"[{nome}] ERRO de conexão: {erro.reason}")
        print(f"         Caminho manual: baixe {url} pelo navegador e salve em {DESTINO}")
        return None

    parcial.rename(destino)
    print(f"[{nome}] baixado: {destino.name} ({destino.stat().st_size / 1024**2:,.1f} MiB)")
    return destino


def extrair(nome: str, caminho_zip: Path) -> None:
    """Extrai o .zip em DESTINO/<nome>/ e lista os arquivos extraídos."""
    pasta = DESTINO / nome
    with zipfile.ZipFile(caminho_zip) as arquivo_zip:
        arquivo_zip.extractall(pasta)
    for arquivo in sorted(p for p in pasta.rglob("*") if p.is_file()):
        tamanho = arquivo.stat().st_size / 1024**2
        print(f"         {arquivo.relative_to(DESTINO)} ({tamanho:,.1f} MiB)")


def main() -> None:
    parser = argparse.ArgumentParser(description="Baixa os Indicadores Educacionais do INEP (2023, municípios).")
    parser.add_argument("--forcar", action="store_true", help="rebaixa mesmo se o arquivo existir")
    args = parser.parse_args()

    DESTINO.mkdir(parents=True, exist_ok=True)

    falhas = []
    for nome, url in ARQUIVOS.items():
        caminho = baixar(nome, url, forcar=args.forcar)
        if caminho is None:
            falhas.append(nome)
            continue
        extrair(nome, caminho)

    print()
    if falhas:
        print(f"Falharam: {', '.join(falhas)}")
    else:
        print(f"Todos os {len(ARQUIVOS)} arquivos disponíveis em data/raw/inep_2023/")


if __name__ == "__main__":
    main()