"""Configuração da modelagem: features curadas, famílias e parâmetros.

Fonte única das decisões de modelagem. Justificativas no notebook 02
(EDA de modelagem) e no README.
"""

SEED = 42
TEST_SIZE = 0.20
N_SPLITS = 5
LIMIAR_RISCO = 60.0  # meta nacional de 2024 (%): município "em risco" se a taxa < 60

COL_ID = "id_municipio"
COL_ALVO = "indicador_alfabetizacao"  # 0–100, rede municipal, 2024
COL_PESO = "n_alunos"  # alunos avaliados na rede municipal, 2024
COL_ESTRATO = "ter_uf"  # estratificação das divisões (todas as UFs em todos os folds)

PREFIXOS_FEATURES = ("ce_", "soc_", "fin_", "ter_")

# Retiradas por colinearidade (|rho| de Spearman >= 0,85; notebook 02, seção 5).
REMOVIDAS_COLINEARIDADE = [
    "ce_reprovacao_anos_iniciais",  # complemento de aprovação (rho = -0,99)
    "ce_docentes_superior_infantil",  # espelho do grupo 5 de adequação (rho = -0,99)
    "ce_docentes_superior_creche",
    "ce_docentes_superior_pre_escola",
    "ce_docentes_superior_anos_iniciais",  # espelho do grupo 5 de adequação (rho = -0,98)
    "ce_horas_aula_1ano",  # redundante com anos iniciais (rho = 0,92)
    "ce_horas_aula_2ano",  # redundante com anos iniciais (rho = 0,95)
    "ce_horas_aula_creche",  # redundante com infantil (rho = 0,90)
    "ce_alunos_turma_1ano",  # redundante com anos iniciais (rho = 0,86)
    "ce_alunos_turma_2ano",  # redundante com anos iniciais (rho = 0,87)
    "soc_indice_envelhecimento",  # redundante com idade mediana (rho = 0,96)
    "soc_alfab_15_mais",  # redundante com 25–44, que é mais forte (rho = 0,95)
    "fin_invest_aluno_infantil",  # redundante com pré-escola (rho = 0,85)
]
N_FEATURES_ESPERADAS = 44

# Famílias para as ablações.
FEATURES_FLUXO = [  # limite entre contexto e histórico de desempenho (decisão D3)
    "ce_aprovacao_anos_iniciais",
    "ce_abandono_anos_iniciais",
    "ce_distorcao_anos_iniciais",
    "ce_distorcao_1ano",
    "ce_distorcao_2ano",
]
FEATURES_TERRITORIO = ["ter_regiao", "ter_uf"]  # mistura política estadual e instrumento de medida
FAMILIAS_ABLACAO = {
    "fluxo": FEATURES_FLUXO,
    "territorio": FEATURES_TERRITORIO,
}

FEATURES_CATEGORICAS = ["ter_regiao", "ter_uf"]

# Nulos estruturais (ausência da modalidade): imputação + indicador de ausência.
NULOS_ESTRUTURAIS = ["ce_alunos_turma_multietapa", "ce_alunos_turma_creche"]

# Modelo linear: coluna de referência retirada de cada grupo composicional (soma ~100%).
# Escolhida a categoria com relação mais fraca com o alvo (notebook 02, seção 4).
REFERENCIAS_COMPOSICIONAIS = [
    "ce_adeq_docente_infantil_g4",
    "ce_adeq_docente_anos_iniciais_g4",
    "ce_regularidade_media_alta",
]

# Modelo linear: variáveis muito assimétricas recebem log1p.
FEATURES_LOG = [
    "soc_populacao",
    "soc_densidade",
    "soc_pib_per_capita",
    "fin_invest_aluno_educacao_basica",
    "fin_invest_aluno_fundamental",
    "fin_invest_aluno_pre_escola",
    "fin_despesa_professores_aluno",
]


def selecionar_features(colunas, excluir_familias=()) -> list[str]:
    """Features curadas presentes em `colunas`, opcionalmente sem famílias de ablação."""
    features = [
        c for c in colunas
        if c.startswith(PREFIXOS_FEATURES) and c not in REMOVIDAS_COLINEARIDADE
    ]
    excluir = set()
    for familia in excluir_familias:
        excluir |= set(FAMILIAS_ABLACAO[familia])
    return [c for c in features if c not in excluir]