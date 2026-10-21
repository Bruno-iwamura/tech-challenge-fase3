"""Indicadores educacionais do INEP (2023) usados como features do Censo Escolar.

Fonte: basedosdados.br_inep_indicadores_educacionais.municipio, com uma linha
por município × rede × localização. Usamos localização 'Total'.

A rede usada nas features é a 'Municipal', a mesma rede do alvo. As redes
'Pública' e 'Total' são extraídas apenas para validação: a comparação com o
enriquecimento anterior (Gold) mostra qual linha foi usada nele.

Corte por etapa (decisão D1): educação infantil, anos iniciais, 1º e 2º ano e
turmas multietapa. Grupos composicionais completos (decisão D2).
"""

ANO = 2023
LOCALIZACAO = "Total"
REDE_FEATURES = "Municipal"
REDES_EXTRAIDAS = ["Municipal", "Pública", "Total"]

# coluna na Base dos Dados -> (nome da feature, descrição)
COLUNAS_INEP: dict[str, tuple[str, str]] = {
    # Média de alunos por turma
    "atu_ei": ("ce_alunos_turma_infantil", "Média de alunos por turma — educação infantil"),
    "atu_ei_creche": ("ce_alunos_turma_creche", "Média de alunos por turma — creche"),
    "atu_ei_pre_escola": ("ce_alunos_turma_pre_escola", "Média de alunos por turma — pré-escola"),
    "atu_ef_anos_iniciais": ("ce_alunos_turma_anos_iniciais", "Média de alunos por turma — anos iniciais"),
    "atu_ef_1_ano": ("ce_alunos_turma_1ano", "Média de alunos por turma — 1º ano"),
    "atu_ef_2_ano": ("ce_alunos_turma_2ano", "Média de alunos por turma — 2º ano"),
    "atu_ef_turmas_unif_multi_fluxo": (
        "ce_alunos_turma_multietapa",
        "Média de alunos por turma — turmas unificadas, multietapa ou de correção de fluxo",
    ),
    # Média de horas-aula diárias
    "had_ei": ("ce_horas_aula_infantil", "Média de horas-aula diárias — educação infantil"),
    "had_ei_creche": ("ce_horas_aula_creche", "Média de horas-aula diárias — creche"),
    "had_ei_pre_escola": ("ce_horas_aula_pre_escola", "Média de horas-aula diárias — pré-escola"),
    "had_ef_anos_iniciais": ("ce_horas_aula_anos_iniciais", "Média de horas-aula diárias — anos iniciais"),
    "had_ef_1_ano": ("ce_horas_aula_1ano", "Média de horas-aula diárias — 1º ano"),
    "had_ef_2_ano": ("ce_horas_aula_2ano", "Média de horas-aula diárias — 2º ano"),
    # Regularidade do corpo docente (% de escolas por faixa)
    "ird_baixa_regularidade": ("ce_regularidade_baixa", "% de escolas com regularidade docente baixa (0 a 2)"),
    "ird_media_baixa": ("ce_regularidade_media_baixa", "% de escolas com regularidade docente média-baixa (2 a 3)"),
    "ird_media_alta": ("ce_regularidade_media_alta", "% de escolas com regularidade docente média-alta (3 a 4)"),
    "ird_alta": ("ce_regularidade_alta", "% de escolas com regularidade docente alta (4 a 5)"),
    # Distorção idade-série
    "tdi_ef_anos_iniciais": ("ce_distorcao_anos_iniciais", "Taxa de distorção idade-série — anos iniciais"),
    "tdi_ef_1_ano": ("ce_distorcao_1ano", "Taxa de distorção idade-série — 1º ano"),
    "tdi_ef_2_ano": ("ce_distorcao_2ano", "Taxa de distorção idade-série — 2º ano"),
    # Rendimento escolar
    "taxa_aprovacao_ef_anos_iniciais": ("ce_aprovacao_anos_iniciais", "Taxa de aprovação — anos iniciais"),
    "taxa_reprovacao_ef_anos_iniciais": ("ce_reprovacao_anos_iniciais", "Taxa de reprovação — anos iniciais"),
    "taxa_abandono_ef_anos_iniciais": ("ce_abandono_anos_iniciais", "Taxa de abandono — anos iniciais"),
    # Docentes com curso superior
    "dsu_ei": ("ce_docentes_superior_infantil", "% de funções docentes com curso superior — educação infantil"),
    "dsu_ei_creche": ("ce_docentes_superior_creche", "% de funções docentes com curso superior — creche"),
    "dsu_ei_pre_escola": ("ce_docentes_superior_pre_escola", "% de funções docentes com curso superior — pré-escola"),
    "dsu_ef_anos_iniciais": (
        "ce_docentes_superior_anos_iniciais",
        "% de funções docentes com curso superior — anos iniciais",
    ),
}

# Adequação da formação docente (grupos 1 a 5)
for _k in range(1, 6):
    COLUNAS_INEP[f"afd_ei_grupo_{_k}"] = (
        f"ce_adeq_docente_infantil_g{_k}",
        f"% de docentes no grupo {_k} de adequação da formação — educação infantil",
    )
    COLUNAS_INEP[f"afd_ef_anos_iniciais_grupo_{_k}"] = (
        f"ce_adeq_docente_anos_iniciais_g{_k}",
        f"% de docentes no grupo {_k} de adequação da formação — anos iniciais",
    )