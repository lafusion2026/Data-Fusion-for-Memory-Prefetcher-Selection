"""
Script para:
1. Carregar um CSV com estatísticas de desempenho (cache, TLB, DRAM etc.)
2. Selecionar um subconjunto de colunas de interesse
3. Calcular os quartis (min, 25%, 50%, 75%, max) dessas colunas
4. Usar esses quartis para construir os "Antecedents" (conjuntos fuzzy)
   de um sistema de lógica fuzzy, usando scikit-fuzzy (skfuzzy)
5. Opcionalmente, tratar um subconjunto das colunas (COLUNAS_INDEPENDENTES_POR_PREFETCHER)
   como variáveis fuzzy independentes, gerando para cada uma delas uma regra
   fuzzy própria, separada da regra conjunta (E/AND) das demais colunas

Requisitos:
    pip install pandas numpy scikit-fuzzy
"""

import itertools
import os
import glob

import pandas as pd
import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# 1. Configurações gerais
# ---------------------------------------------------------------------------

# Diretório contendo os arquivos .csv a processar. TODOS devem ter o mesmo
# formato de 'bancada_bingo.csv': as colunas de COLUNAS_TODAS disponíveis, e
# a ÚLTIMA coluna do CSV com o cabeçalho (primeira linha) contendo o nome do
# prefetcher daquele arquivo (ex.: 'bingo', 'stride', 'spp' etc.) — os valores
# dessa coluna indicam a colocação/rank do prefetcher em cada linha (ver
# VALORES_VENCEDORES / VALORES_INTERMEDIARIOS / VALORES_PERDEDORES). O
# subconjunto de colunas usado para montar os antecedentes é escolhido
# automaticamente por prefetcher (ver COLUNAS_SUBCONJUNTO_POR_PREFETCHER).
DIRETORIO_CSV_ENTRADA = './database_calib'

# Diretório onde é gravado o arquivo de saída de CADA CSV processado (a
# tabela com as colunas de entrada + a adequabilidade calculada linha a linha).
# Criado automaticamente se não existir.
DIRETORIO_CSV_SAIDA = './saida'

# Padrão do nome do arquivo de saída para cada CSV de entrada. '{nome}' é
# substituído pelo nome do arquivo de entrada sem extensão (ex.:
# 'bancada_bingo.csv' -> 'bancada_bingo_resultado.csv').
PADRAO_NOME_ARQUIVO_SAIDA = '{nome}_resultado.csv'

# Nome da coluna de IPC (instruções por ciclo) nos CSVs de ENTRADA
# (DIRETORIO_CSV_ENTRADA) — não é calculada pelo script, precisa já existir
# nesses arquivos. Usada por `adicionar_coluna_ipc_aos_resultados` para
# copiá-la para os arquivos de RESULTADO depois que eles já foram gerados.
NOME_COLUNA_IPC = 'ipc'

# Nome da coluna que identifica o trace (workload/benchmark) de cada linha
# nos CSVs de ENTRADA (DIRETORIO_CSV_ENTRADA). Usada por
# `filtrar_linhas_por_traces` para restringir quais linhas entram no
# seletor de prefetcher — ver TRACES_SELECIONADOS e USAR_TODOS_OS_TRACES
# logo abaixo.
NOME_COLUNA_TRACE = 'trace'

# Lista de traces (valores da coluna NOME_COLUNA_TRACE) a USAR na EXECUÇÃO
# do seletor de prefetcher (cálculo de adequabilidade linha a linha e o CSV
# de saída) para avaliar o desempenho do sistema fuzzy. Só tem efeito
# quando USAR_TODOS_OS_TRACES == False (ver abaixo).
#
# O sistema fuzzy em si (quartis, antecedentes, consequente, regras e as
# medianas dos grupos vencedoras/intermediárias/perdedoras) é SEMPRE
# construído com TODAS as linhas do CSV, de TODOS os traces, independente
# desta configuração — o filtro só restringe quais linhas são avaliadas
# (recebem adequabilidade e vão para o CSV de resultado), não como o
# sistema fuzzy é calibrado.
#
# Preencha com os valores exatos que aparecem na coluna 'trace' dos CSVs de
# DIRETORIO_CSV_ENTRADA (ex.: nomes de benchmark do SPEC/traces do
# ChampSim). Lista vazia + USAR_TODOS_OS_TRACES = False é um erro de
# configuração (ver `filtrar_linhas_por_traces`).
TRACES_SELECIONADOS = [
#	'450.soplex-247B.champsimtrace.xz',
#	'471.omnetpp-188B.champsimtrace.xz',
#	'602.gcc_s-2226B.champsimtrace.xz',
#	'602.gcc_s-734B.champsimtrace.xz',
#	'603.bwaves_s-891B.champsimtrace.xz',
#	'605.mcf_s-1554B.champsimtrace.xz',
#	'605.mcf_s-472B.champsimtrace.xz',
#	'619.lbm_s-2676B.champsimtrace.xz',
#	'619.lbm_s-3766B.champsimtrace.xz',
#	'623.xalancbmk_s-202B.champsimtrace.xz',
#	'623.xalancbmk_s-700B.champsimtrace.xz',
#	'627.cam4_s-573B.champsimtrace.xz',
#	'649.fotonik3d_s-10881B.champsimtrace.xz',
#	'compute_fp_45.champsim.gz',
#	'compute_int_12.champsim.gz',
#	'ligra_BFS.com-lj.ungraph.gcc_6.3.0_O3.drop_500M.length_250M.champsimtrace.xz',
#	'ligra_PageRankDelta.com-lj.ungraph.gcc_6.3.0_O3.drop_1250M.length_250M.champsimtrace.xz',
#	'parsec_2.1.canneal.simlarge.prebuilt.drop_4750M.length_250M.champsimtrace.xz',
#	'parsec_2.1.streamcluster.simlarge.prebuilt.drop_250M.length_250M.champsimtrace.xz',


#	TRACES TESTE
	'602.gcc_s-1850B.champsimtrace.xz',
#	'603.bwaves_s-2931B.champsimtrace.xz',
#	'605.mcf_s-994B.champsimtrace.xz',
#	'619.lbm_s-4268B.champsimtrace.xz',
#	'620.omnetpp_s-874B.champsimtrace.xz',
#	'623.xalancbmk_s-592B.champsimtrace.xz',
#	'649.fotonik3d_s-7084B.champsimtrace.xz',

]

# Se True, IGNORA TRACES_SELECIONADOS e EXECUTA (calcula adequabilidade e
# grava no CSV de saída) TODAS as linhas de TODOS os traces de cada CSV
# (comportamento original do script, sem filtro por trace). Se False, só as
# linhas cujo valor em NOME_COLUNA_TRACE estiver em TRACES_SELECIONADOS são
# executadas — a criação do sistema fuzzy não é afetada por este flag.
USAR_TODOS_OS_TRACES = False

# Diretório alternativo de onde ler os dados da ETAPA DE EXECUÇÃO (cálculo
# de adequabilidade linha a linha / CSV de saída), usado somente quando
# USAR_DIRETORIO_EXECUCAO_SEPARADO (abaixo) for True.
#
# Cada arquivo .csv aqui dentro precisa ter a MESMA ÚLTIMA coluna (nome do
# prefetcher) de algum arquivo em DIRETORIO_CSV_ENTRADA — é por esse nome,
# não pelo nome do arquivo, que `processar_arquivo_csv` encontra o par
# entrada/execução de cada prefetcher (mesmo critério já usado por
# `_encontrar_arquivo_entrada_por_prefetcher`). Também precisa ter as
# colunas do subconjunto usado pelos antecedentes daquele prefetcher (ver
# COLUNAS_SUBCONJUNTO_POR_PREFETCHER) e, se USAR_TODOS_OS_TRACES for False,
# a coluna NOME_COLUNA_TRACE.
DIRETORIO_CSV_EXECUCAO = './database_testes'

# Se True, a ETAPA DE EXECUÇÃO usa os dados de DIRETORIO_CSV_EXECUCAO (em
# vez do próprio arquivo de DIRETORIO_CSV_ENTRADA) para calcular a
# adequabilidade linha a linha e gerar o CSV de saída.
#
# O sistema fuzzy (quartis, antecedentes, consequente, regras e medianas
# dos grupos) continua SEMPRE construído a partir de DIRETORIO_CSV_ENTRADA,
# independente deste flag — só a etapa de execução muda de fonte. Se False
# (padrão), a execução usa o mesmo arquivo de DIRETORIO_CSV_ENTRADA usado
# para montar o sistema (comportamento original).
USAR_DIRETORIO_EXECUCAO_SEPARADO = True

# Se True, `main()` chama `adicionar_coluna_ipc_aos_resultados()`
# automaticamente logo depois de processar todos os CSVs de entrada (e antes
# de combinar as saídas) — copiando, para CADA prefetcher, a coluna de IPC
# do CSV de entrada correspondente para o CSV de resultado daquele
# prefetcher, inserida logo antes da coluna 'adequabilidade'.
ADICIONAR_COLUNA_IPC_AUTOMATICAMENTE = True

# Caminho do arquivo .csv COMBINADO, gerado por `combinar_saidas_por_coluna`
# a partir de TODOS os arquivos de DIRETORIO_CSV_SAIDA: primeiro a primeira
# coluna de cada um (sequencialmente), depois a última coluna de cada um
# (sequencialmente), renomeada para 'adequabilidade_<prefetcher>', e por fim
# uma coluna de RANKING previsto por prefetcher (ver `adicionar_ranking_adequabilidade`).
#
# Fica DENTRO de DIRETORIO_CSV_SAIDA (junto dos demais resultados). Isso não
# causa autoinclusão numa próxima combinação: `combinar_saidas_por_coluna`
# sempre exclui este arquivo (por caminho absoluto) da varredura de '*.csv'
# que ela faz nesse mesmo diretório.
ARQUIVO_SAIDA_COMBINADO = os.path.join(DIRETORIO_CSV_SAIDA, 'saida_combinada.csv')

# Se True, `main()` chama `combinar_saidas_por_coluna()` automaticamente ao
# final, depois de processar todos os CSVs de DIRETORIO_CSV_ENTRADA, gerando
# ARQUIVO_SAIDA_COMBINADO. Se False, a combinação não roda sozinha — chame
# `combinar_saidas_por_coluna()` manualmente quando quiser.
COMBINAR_SAIDAS_AUTOMATICAMENTE = True

# Se True, `main()` chama `calcular_percentual_acerto()` automaticamente
# logo depois de gerar ARQUIVO_SAIDA_COMBINADO, imprimindo o percentual de
# acerto do sistema fuzzy (previsão de vencedor) por prefetcher.
CALCULAR_PERCENTUAL_ACERTO_AUTOMATICAMENTE = True

# Limiar (em pontos percentuais) usado pela métrica GERAL POR DIFERENÇA
# PERCENTUAL de `calcular_percentual_acerto`: se o prefetcher previsto como
# vencedor não for o vencedor medido de verdade, mas sua coluna
# '%dif_<prefetcher_previsto>' (a diferença percentual de IPC dele em
# relação ao vencedor medido — ver `adicionar_percentual_diferenca_ipc`)
# ficar ABAIXO deste limiar, a previsão ainda é considerada correta ("quase
# empatou" com o vencedor real em desempenho, mesmo não sendo ele).
# Ex.: 5.0 aceita como acerto qualquer previsão cujo IPC fique até 5% abaixo
# do IPC do vencedor real daquela linha.
LIMIAR_DIFERENCA_PERCENTUAL_ACERTO = 1.0

# Se True, `main()` chama `calcular_percentual_vitorias_reais()`
# automaticamente logo depois de `calcular_percentual_acerto()`, imprimindo
# o percentual de vitórias REAIS (rank medido = 1, sem envolver o sistema
# fuzzy) de cada prefetcher no arquivo combinado.
CALCULAR_PERCENTUAL_VITORIAS_REAIS_AUTOMATICAMENTE = True

# Se True, exibe os gráficos (Antecedent.view() / Consequent.view()) para
# CADA arquivo processado. Em lote (vários CSVs), isso abre uma janela por
# arquivo — deixe False para rodar em lote sem interrupções.
MOSTRAR_GRAFICOS = False

# Se True, `processar_arquivo_csv` chama `sugerir_colunas_sinergicas`
# automaticamente para CADA arquivo, logo no início (antes de montar
# antecedentes/regras), usando como alvo a PRÓPRIA COLUNA DE RANK do
# prefetcher (valores 1 a 7, direto do CSV de entrada — "opção 1" discutida:
# rank bruto, já disponível, sem precisar calcular adequabilidade antes).
#
# Por padrão é só uma sugestão IMPRESSA no relatório — ver
# MODO_SELECAO_COLUNAS logo abaixo para de fato usá-la.
SUGERIR_COLUNAS_SINERGICAS_AUTOMATICAMENTE = False

# Modo de seleção do subconjunto de colunas e da classificação em
# independentes/sinérgicas usado para montar os antecedentes fuzzy de CADA
# prefetcher. Condensa em UMA escolha o que antes eram duas flags booleanas
# combinadas por "OU" (USAR_COLUNAS_SINERGICAS_AUTOMATICAMENTE e
# USAR_SHAP_TODAS_COLUNAS_AUTOMATICAMENTE) — mais fácil de configurar sem
# ambiguidade sobre qual flag "vence" quando as duas estão ligadas.
#
# Valores possíveis:
#
#   'manual'
#       Nada automático. `colunas_subconjunto`, `colunas_independentes` e
#       `colunas_sinergicas` vêm inteiramente de COLUNAS_SUBCONJUNTO_POR_PREFETCHER,
#       COLUNAS_INDEPENDENTES_POR_PREFETCHER e COLUNAS_SINERGICAS_POR_PREFETCHER.
#
#   'automatico_subconjunto'
#       O subconjunto de colunas continua vindo de
#       COLUNAS_SUBCONJUNTO_POR_PREFETCHER, mas a classificação em
#       independentes/sinérgicas DENTRO dele é decidida automaticamente pela
#       sugestão (SHAP ou regressão — ver USAR_SHAP_PARA_SUGESTAO, abaixo).
#
#   'automatico_todas_colunas'
#       O próprio SUBCONJUNTO de colunas passa a ser decidido
#       automaticamente a partir de COLUNAS_TODAS (não só do subconjunto
#       manual), junto com a classificação em independentes/sinérgicas.
#       Usa `_gerar_sugestao_colunas_independentes_sinergicas` (SHAP ou
#       regressão) SEM descartar nenhuma coluna por relevância — colunas
#       "sem classificação clara" viram independentes (ver mais abaixo).
#
#   'automatico_shap_unificado'
#       Como 'automatico_todas_colunas', mas usando
#       `selecionar_e_classificar_colunas_shap` — que primeiro SELECIONA as
#       colunas de maior importância total via SHAP (`NUMERO_COLUNAS_SHAP_UNIFICADO`
#       e/ou `LIMIAR_RELEVANCIA_MINIMA_SHAP_UNIFICADO`), DESCARTANDO as
#       demais por completo (não viram independentes, saem do subconjunto),
#       e só depois classifica as selecionadas em independentes/sinérgicas.
#       Sempre usa SHAP (não tem fallback para regressão nesta função — se o
#       SHAP falhar, cai para a config manual do prefetcher, como nos outros
#       modos automáticos).
#
# Em qualquer um dos modos automáticos, se a sugestão falhar por qualquer
# motivo (ex.: poucas linhas sem NaN, shap indisponível), o script imprime
# um aviso e usa a config manual daquele prefetcher como fallback, em vez de
# interromper o processamento do arquivo.
MODO_SELECAO_COLUNAS = 'automatico_shap_unificado'

# Usados apenas quando MODO_SELECAO_COLUNAS == 'automatico_shap_unificado'
# (ver `selecionar_e_classificar_colunas_shap`).
#
# Quantas colunas manter, pela maior importância total (soma da linha
# inteira da matriz de interação SHAP). None = não corta por quantidade, só
# pelo limiar de relevância abaixo.
NUMERO_COLUNAS_SHAP_UNIFICADO = 10

# Importância total mínima (escala normalizada do SHAP, 0 a 1) para uma
# coluna ser selecionada. Default 0.0 (sem corte — mantém todas com
# importância > 0, úteis pra alguma coisa, e ordena por importância).
LIMIAR_RELEVANCIA_MINIMA_SHAP_UNIFICADO = 0.09

# Limiares usados pela sugestão automática (mesmos defaults de
# `sugerir_colunas_sinergicas` — ver a função para o significado de cada um).
LIMIAR_SINERGIA_SUGESTAO = 0.01
LIMIAR_INDEPENDENCIA_SUGESTAO = 0.005
TAMANHO_MAXIMO_GRUPO_SUGESTAO = 3

# Se True, a sugestão automática de colunas independentes/sinérgicas (em
# qualquer um dos modos automáticos de MODO_SELECAO_COLUNAS) usa
# `sugerir_colunas_sinergicas_shap` (SHAP interaction values, via árvore de
# decisão) EM VEZ DE `sugerir_colunas_sinergicas` (regressão linear com
# termo de interação).
#
# Requer as bibliotecas opcionais `shap` e (`xgboost` OU `scikit-learn`).
# Se não estiverem instaladas (ou o cálculo falhar por qualquer motivo), o
# script imprime um aviso e cai automaticamente de volta para a versão por
# regressão — nunca interrompe o processamento do arquivo por causa disso.
USAR_SHAP_PARA_SUGESTAO = True

# Limiares usados pela sugestão via SHAP. Ficam numa escala DIFERENTE dos
# limiares da regressão (LIMIAR_SINERGIA_SUGESTAO/LIMIAR_INDEPENDENCIA_SUGESTAO):
# aqui os valores são a fração (0 a 1) do total de |interação| explicada por
# aquele efeito, depois de normalizar a matriz de interação inteira para
# somar 1 — não são comparáveis diretamente com o R² usado na regressão.
LIMIAR_SINERGIA_SHAP = 0.02
LIMIAR_INDEPENDENCIA_SHAP = 0.01

# Todas as colunas disponíveis no CSV (mesma lista do script original)
COLUNAS_TODAS = [
    'l1i_mpki', 'l1i_hitrate', 'l1d_mpki', 'l1d_hitrate',
    'l2c_mpki', 'l2c_hitrate', 'llc_mpki', 'llc_hitrate',
    'itlb_mpki', 'dtlb_mpki', 'stlb_mpki', 'branch_mpki',
    'dram_congest_pki', 'dram_rq_rbmissrate', 'l1d_avg_miss_latency',
    'l2c_avg_miss_latency', 'llc_avg_miss_latency', 'dram_rq_pki',
    'l2c_pf_requested', 'l2c_pf_dropped', 'l2c_pf_issued',
    'l2c_pf_filled', 'l2c_pf_useful', 'l2c_pf_useless',
    'l2c_pf_polluted_misses', 'l2c_pf_late', 'l2c_stall', 'ipc'
]

# Subconjunto de colunas que você quer transformar em variáveis fuzzy, UM
# POR PREFETCHER. Como cada arquivo .csv processado tem seu próprio
# prefetcher (detectado automaticamente pela última coluna — ver
# `processar_arquivo_csv`), cada um pode usar um conjunto diferente de
# colunas/estatísticas para montar os antecedentes.
#
# Por padrão, todas começam iguais ao subconjunto original do script — ajuste
# cada lista individualmente conforme o que fizer sentido para aquele
# prefetcher (as colunas disponíveis estão em COLUNAS_TODAS, acima).
COLUNAS_SUBCONJUNTO_AMPM = [
    'llc_mpki',
    'llc_hitrate',
    'l2c_hitrate'
]

COLUNAS_SUBCONJUNTO_BINGO = [
    'llc_mpki',
    'llc_hitrate',
]

COLUNAS_SUBCONJUNTO_MLOP = [
    'l2c_hitrate',
    'llc_mpki',
    'llc_hitrate',
    'itlb_mpki',
]

COLUNAS_SUBCONJUNTO_PYTHIA = [
    'l2c_mpki',
    'l2c_hitrate',
    'llc_mpki',
    'llc_hitrate',
    'itlb_mpki',
    'dtlb_mpki',
    'stlb_mpki',
    'branch_mpki',
]

COLUNAS_SUBCONJUNTO_SPP = [
    'llc_mpki',
    'llc_hitrate',
    'dram_congest_pki',
]

COLUNAS_SUBCONJUNTO_STRIDE = [
    'llc_mpki',
    'llc_hitrate',
    'dram_congest_pki',
    'dram_rq_rbmissrate',
]

COLUNAS_SUBCONJUNTO_NOPREF = [
    'llc_mpki',
    'llc_hitrate',
    'dram_congest_pki',
    'dram_rq_rbmissrate',
]

# Mapeia o nome do prefetcher (detectado na última coluna de cada CSV) para o
# COLUNAS_SUBCONJUNTO_<PREFETCHER> correspondente, usado em
# `processar_arquivo_csv` para escolher o subconjunto certo automaticamente.
COLUNAS_SUBCONJUNTO_POR_PREFETCHER = {
    'ampm': COLUNAS_SUBCONJUNTO_AMPM,
    'bingo': COLUNAS_SUBCONJUNTO_BINGO,
    'mlop': COLUNAS_SUBCONJUNTO_MLOP,
    'pythia': COLUNAS_SUBCONJUNTO_PYTHIA,
    'spp': COLUNAS_SUBCONJUNTO_SPP,
    'stride': COLUNAS_SUBCONJUNTO_STRIDE,
    'nopref': COLUNAS_SUBCONJUNTO_NOPREF,
}

# Subconjunto de COLUNAS_SUBCONJUNTO_POR_PREFETCHER[<prefetcher>] que deve ser
# tratado como variáveis fuzzy INDEPENDENTES, UM POR PREFETCHER (mesma ideia
# de COLUNAS_SUBCONJUNTO_POR_PREFETCHER — cada prefetcher pode ter um
# conjunto diferente de colunas independentes). Cada coluna listada gera sua
# PRÓPRIA regra fuzzy separada:
#
#     SE <coluna> é <termo> ENTÃO adequabilidade é <termo_consequente>
#
# em vez de entrar na regra conjunta (combinada por E/AND) junto com as
# demais colunas do subconjunto daquele prefetcher. As colunas que NÃO
# estiverem aqui continuam sendo combinadas normalmente em uma única regra
# "E" (comportamento original do script). Colunas listadas aqui que não
# existam no subconjunto de um determinado prefetcher são simplesmente
# ignoradas para aquele arquivo.
#
# Deixe a lista de um prefetcher vazia ([]) para manter, só para ele, o
# comportamento original (todas as colunas combinadas em uma única regra por
# grupo).
COLUNAS_INDEPENDENTES_AMPM = [
    'llc_hitrate',
]

COLUNAS_INDEPENDENTES_BINGO = [
    'dram_congest_pki',
]

COLUNAS_INDEPENDENTES_MLOP = [
    'llc_hitrate',
]

COLUNAS_INDEPENDENTES_PYTHIA = [
    'stlb_mpki',
    'branch_mpki',
    'dram_congest_pki',
    'dram_rq_rbmissrate',
]

COLUNAS_INDEPENDENTES_SPP = [
    'dram_congest_pki',
]

COLUNAS_INDEPENDENTES_STRIDE = [
    'dram_congest_pki',
]

COLUNAS_INDEPENDENTES_NOPREF = [
    'dram_congest_pki',
]

# Mapeia o nome do prefetcher (detectado na última coluna de cada CSV) para o
# COLUNAS_INDEPENDENTES_<PREFETCHER> correspondente, usado em
# `processar_arquivo_csv` para escolher as colunas independentes certas
# automaticamente.
COLUNAS_INDEPENDENTES_POR_PREFETCHER = {
    'ampm': COLUNAS_INDEPENDENTES_AMPM,
    'bingo': COLUNAS_INDEPENDENTES_BINGO,
    'mlop': COLUNAS_INDEPENDENTES_MLOP,
    'pythia': COLUNAS_INDEPENDENTES_PYTHIA,
    'spp': COLUNAS_INDEPENDENTES_SPP,
    'stride': COLUNAS_INDEPENDENTES_STRIDE,
    'nopref': COLUNAS_INDEPENDENTES_NOPREF,
}

# Agrupa colunas de COLUNAS_SUBCONJUNTO_POR_PREFETCHER[<prefetcher>] em
# GRUPOS SINÉRGICOS — pares, trios, quartetos etc. de variáveis que só fazem
# sentido avaliar EM CONJUNTO (relação AND) entre si, SEM misturar com as
# demais colunas do subconjunto daquele prefetcher.
#
# Cada grupo listado gera sua PRÓPRIA regra fuzzy, combinando por E/AND
# SOMENTE as colunas daquele grupo:
#
#     SE <col_grupo_1> é <termo> E <col_grupo_2> é <termo> [E ...]
#     ENTÃO adequabilidade é <termo_consequente>
#
# Diferença para COLUNAS_INDEPENDENTES: uma coluna independente vira uma
# regra sozinha (sem E com ninguém); um grupo sinérgico tem 2+ colunas
# combinadas por E só entre elas.
#
# As colunas que NÃO estiverem em nenhum grupo sinérgico nem em
# COLUNAS_INDEPENDENTES continuam sendo combinadas normalmente em uma única
# regra "E" geral (comportamento original do script). Uma mesma coluna NÃO
# pode aparecer em mais de um grupo sinérgico, nem simultaneamente em
# COLUNAS_INDEPENDENTES e em um grupo sinérgico — isso gera erro ao montar as
# regras (ver `gerar_regra_fuzzy`).
#
# Formato: lista de grupos, cada grupo é uma lista/tupla com 2+ nomes de
# coluna. Deixe como lista vazia ([]) para não usar grupos sinérgicos para
# aquele prefetcher.
COLUNAS_SINERGICAS_AMPM = []

COLUNAS_SINERGICAS_BINGO = [
    ('llc_mpki', 'llc_hitrate'),
]

COLUNAS_SINERGICAS_MLOP = []

COLUNAS_SINERGICAS_PYTHIA = [
    ('l2c_mpki', 'l2c_hitrate'),
    ('itlb_mpki', 'dtlb_mpki'),
]

COLUNAS_SINERGICAS_SPP = []

COLUNAS_SINERGICAS_STRIDE = []

COLUNAS_SINERGICAS_NOPREF = []

# Mapeia o nome do prefetcher (detectado na última coluna de cada CSV) para o
# COLUNAS_SINERGICAS_<PREFETCHER> correspondente, usado em
# `processar_arquivo_csv` para escolher os grupos sinérgicos certos
# automaticamente.
COLUNAS_SINERGICAS_POR_PREFETCHER = {
    'ampm': COLUNAS_SINERGICAS_AMPM,
    'bingo': COLUNAS_SINERGICAS_BINGO,
    'mlop': COLUNAS_SINERGICAS_MLOP,
    'pythia': COLUNAS_SINERGICAS_PYTHIA,
    'spp': COLUNAS_SINERGICAS_SPP,
    'stride': COLUNAS_SINERGICAS_STRIDE,
    'nopref': COLUNAS_SINERGICAS_NOPREF,
}

# Nome do prefetcher da última coluna do CSV. NÃO precisa mais ser ajustado
# manualmente: como o script agora processa TODOS os .csv de
# DIRETORIO_CSV_ENTRADA em lote, e cada arquivo pode ter um prefetcher
# diferente, essa informação é detectada automaticamente para cada arquivo
# (ver `processar_arquivo_csv`, que usa `df.columns[-1]` — a última coluna do
# CSV — como o nome do prefetcher daquele arquivo específico).
#
# Esta constante fica só como referência/valor de fallback (ex.: para uso
# manual em testes fora do fluxo de `main()`); ela não é mais lida pelo
# pipeline principal.
PREFETCHER = 'bingo'

# Valores da coluna do prefetcher considerados "vencedor"
VALORES_VENCEDORES = [1]

# Valores da coluna do prefetcher considerados "intermediário"
VALORES_INTERMEDIARIOS = [2, 3, 4]

# Valores da coluna do prefetcher considerados "perdedor"
VALORES_PERDEDORES = [5, 6, 7]

# Passo (resolução) do universo de discurso, usado por TODOS os antecedentes.
# Quanto menor, mais fino/preciso o universo (e mais pontos ele terá).
PASSO_UNIVERSO = 0.01

# Largura do "núcleo" (platô com grau de pertinência = 1) de cada termo do
# consequente 'adequabilidade'.
#
# Os picos de baixo/medio/alto já ficam nos extremos do universo de discurso
# (0, 5 e 10) — não dá para afastá-los mais sem também mudar a escala da
# saída. O que este parâmetro controla é outra coisa: o quanto cada termo
# permanece com grau de pertinência = 1 ao REDOR do pico, em vez de cair
# imediatamente (trimf puro). Isso reduz a SOBREPOSIÇÃO entre termos vizinhos
# e diminui o "puxão para o meio" que a defuzzificação (centróide) sofre
# quando uma entrada bem definida (ex.: mediana 100% 'alta') ainda assim gera
# uma adequabilidade bem abaixo de 10 por causa da contribuição de regras
# vizinhas.
#
#   0.0  -> comportamento original: termos triangulares puros (trimf),
#           grau de pertinência = 1 só exatamente no pico.
#   > 0  -> termos viram trapézios (trapmf) com um platô dessa largura em
#           volta do pico. Precisa ser < 10 (e tipicamente bem menor, ex.:
#           1.0 a 3.0) para não invadir o platô do termo vizinho.
LARGURA_NUCLEO_CONSEQUENTE = 0.0

# Termo do consequente ('baixo', 'medio' ou 'alto') usado pela regra GERAL
# que cobre as combinações de termos sem regra específica (área cega).
# 'baixo' é o padrão mais conservador: como não há dados suficientes para
# decidir entre baixo/medio/alto nessas regiões do espaço de entrada, assume-se
# que o prefetcher não deve ser recomendado (adequabilidade baixa) em vez de
# arriscar uma saída otimista sem embasamento.
TERMO_PADRAO_AREA_CEGA = 'baixo'

# Número máximo de condições (uma por combinação/termo de área cega)
# encadeadas por OU dentro de UMA MESMA regra "geral" (catch-all).
#
# Prefetchers com muitas colunas (ex.: pythia) podem ter dezenas de
# combinações sem regra específica. Encadear todas com OU numa única regra
# cria uma árvore de expressão muito profunda, que o scikit-fuzzy percorre
# recursivamente ao montar a regra/sistema de controle — o que pode estourar
# o limite de recursão do Python ("RecursionError: maximum recursion depth
# exceeded"). Por isso, a regra geral é dividida em VÁRIAS regras menores
# (cada uma com até este número de condições), todas produzindo o mesmo
# termo do consequente — funcionalmente equivalente a uma única regra
# gigante, mas sem o risco de estourar a recursão.
TAMANHO_MAXIMO_CONDICOES_POR_REGRA_GERAL = 15

# Aviso (não bloqueia) se o grupo de variáveis "restantes" (nem
# independentes, nem em nenhum grupo sinérgico — as que entram na regra
# conjunta geral por E/AND) tiver mais colunas do que este limite. A área
# cega desse grupo cresce em 3^n (n = nº de colunas nele), então poucas
# colunas a mais aqui já bastam para gerar milhares de combinações sem
# regra — o que pode gerar centenas de regras catch-all e, em casos
# extremos, aproximar-se de limites internos do scikit-fuzzy mesmo com a
# regra geral já dividida em blocos (ver TAMANHO_MAXIMO_CONDICOES_POR_REGRA_GERAL).
LIMITE_VARIAVEIS_REGRA_CONJUNTA_GERAL = 6

# Estratégia usada para resolver CONFLITOS entre regras independentes (mesma
# variável, mesmo termo, apontando para consequentes diferentes — ex:
# "dram_congest_pki é medio" -> medio para 'intermediarias' mas -> baixo para
# 'perdedoras'). Opções:
#
#   'grau'       Vence a regra cujo grupo teve o MAIOR grau de pertinência
#                naquele termo (mediana "mais dentro" da categoria fuzzy).
#                As demais regras conflitantes são descartadas.
#
#   'prioridade' Vence a regra do grupo mais bem posicionado em
#                ORDEM_PRIORIDADE_GRUPOS (ex: perdedoras sempre tem
#                prioridade sobre intermediárias/vencedoras). As demais
#                regras conflitantes são descartadas.
#
#   'descartar'  TODAS as regras conflitantes daquele par (variável, termo)
#                são removidas — nenhuma "vence". A lacuna deixada fica para
#                a regra geral (catch-all) de área cega cobrir
#                (ver TERMO_PADRAO_AREA_CEGA / gerar_regra_geral_area_cega).
#
#   'peso'       Nenhuma regra é descartada: cada regra conflitante recebe um
#                peso (`regra.weight`) igual ao grau de pertinência do seu
#                grupo naquele termo, deixando a decisão para a agregação
#                fuzzy (regras com maior grau pesam mais na saída final).
#                Requer que a versão instalada do scikit-fuzzy suporte
#                `Rule.weight`; se não suportar, um aviso é impresso e o peso
#                é ignorado (a regra continua ativa, só que sem peso ajustado).
ESTRATEGIA_RESOLUCAO_CONFLITOS = 'grau'

# Usado apenas quando ESTRATEGIA_RESOLUCAO_CONFLITOS == 'prioridade'.
# Ordem de prioridade dos grupos, do mais para o menos prioritário — o
# primeiro da lista sempre vence em caso de conflito. Grupos que apareçam em
# um conflito mas não estejam nesta lista ficam com a menor prioridade.
ORDEM_PRIORIDADE_GRUPOS = ['perdedoras', 'intermediarias', 'vencedoras']

# Controla quantas linhas de CADA CSV são passadas pelo defuzzificador na
# etapa final (a que calcula e exibe/grava a adequabilidade linha a linha).
# Aplica-se a TODOS os arquivos processados em DIRETORIO_CSV_ENTRADA.
#
# ATENÇÃO: isso NÃO afeta os quartis/medianas usados para construir os
# antecedentes e as regras — esses sempre usam TODAS as linhas do CSV. Só
# afeta quais linhas são avaliadas pelo sistema fuzzy já pronto no final.
#
#   True  -> processa TODAS as linhas de cada CSV.
#   False -> processa apenas as primeiras NUMERO_LINHAS_PROCESSAR linhas de
#            cada CSV (útil para checar rapidamente o resultado em lotes com
#            CSVs grandes, sem esperar o cálculo linha a linha em cada um).
PROCESSAR_TODAS_LINHAS = True

# Usado apenas quando PROCESSAR_TODAS_LINHAS == False: quantas das primeiras
# linhas de CADA CSV serão avaliadas pelo defuzzificador.
NUMERO_LINHAS_PROCESSAR = 1000


# ---------------------------------------------------------------------------
# 2. Carrega o CSV e calcula os quartis do subconjunto (tabela inteira)
# ---------------------------------------------------------------------------

def calcular_quartis(df: pd.DataFrame, colunas: list) -> pd.DataFrame:
    """
    Retorna um DataFrame com min, 25%, 50% (mediana), 75% e max
    para cada coluna do subconjunto, considerando TODAS as linhas de `df`.
    """
    estatisticas = df[colunas].describe().loc[['min', '25%', '50%', '75%', 'max']]
    return estatisticas


def selecionar_linhas_por_valores(df: pd.DataFrame, nome_prefetcher: str, valores: list) -> pd.DataFrame:
    """
    Localiza a coluna com o mesmo nome de `nome_prefetcher` (ex: 'stride')
    e retorna apenas as linhas cujo valor nessa coluna esteja em `valores`
    (ex: [1] para vencedoras, [2, 3, 4] para intermediárias).
    """
    if nome_prefetcher not in df.columns:
        raise ValueError(
            f"Coluna de prefetcher '{nome_prefetcher}' não encontrada no CSV. "
            f"Colunas disponíveis: {list(df.columns)}"
        )

    linhas_selecionadas = df[df[nome_prefetcher].isin(valores)]

    if linhas_selecionadas.empty:
        valores_disponiveis = sorted(df[nome_prefetcher].dropna().unique().tolist())
        raise ValueError(
            f"Nenhuma linha encontrada com '{nome_prefetcher}' em {valores}. "
            f"Valores disponíveis nessa coluna: {valores_disponiveis}"
        )

    return linhas_selecionadas


def calcular_medianas(df_selecionado: pd.DataFrame, colunas: list) -> pd.Series:
    """
    Calcula a mediana de cada coluna do subconjunto, considerando apenas
    as linhas já filtradas em `df_selecionado`.
    """
    return df_selecionado[colunas].median()


# ---------------------------------------------------------------------------
# 3. Cria os antecedentes fuzzy a partir dos quartis
# ---------------------------------------------------------------------------

def criar_antecedente_trapezoidal(nome: str, stats_coluna: pd.Series, passo: float) -> ctrl.Antecedent:
    """
    Cria um Antecedent fuzzy com 3 conjuntos (baixo, medio, alto),
    usando os quartis da coluna para definir os pontos das funções
    de pertinência trapezoidais/triangulares.

    Mapeamento dos quartis:
        min  -> Q1  : região "baixo"
        Q1   -> mediana -> Q3 : região "medio" (triangular)
        Q3   -> max  : região "alto"

    O universo de discurso é criado com np.arange(min, max, passo), ou seja,
    com um espaçamento fixo (passo) igual para todos os antecedentes,
    em vez de uma quantidade fixa de pontos.
    """
    minimo = stats_coluna['min']
    q1 = stats_coluna['25%']
    mediana = stats_coluna['50%']
    q3 = stats_coluna['75%']
    maximo = stats_coluna['max']

    # Verifica se algum quartil veio NaN (coluna com dados faltantes/vazios)
    if pd.isna([minimo, q1, mediana, q3, maximo]).any():
        raise ValueError(
            f"Coluna '{nome}' tem quartil(is) NaN: "
            f"min={minimo}, Q1={q1}, mediana={mediana}, Q3={q3}, max={maximo}. "
            f"Verifique se a coluna existe e não tem valores ausentes/vazios no CSV."
        )

    # Evita universo degenerado (todas as amostras iguais)
    if minimo == maximo:
        maximo = minimo + passo

    # np.arange não inclui o valor final, então somamos um "passo" extra
    # ao limite superior para garantir que o máximo fique dentro do universo.
    universo = np.arange(minimo, maximo + passo, passo)
    antecedente = ctrl.Antecedent(universo, nome)

    # "baixo": trapézio decrescente de min até Q1 (achatado antes do min)
    antecedente['baixo'] = fuzz.trapmf(universo, [minimo, minimo, q1, mediana])

    # "medio": triângulo centrado na mediana, entre Q1 e Q3
    antecedente['medio'] = fuzz.trimf(universo, [q1, mediana, q3])

    # "alto": trapézio crescente de Q3 até max (achatado após o max)
    antecedente['alto'] = fuzz.trapmf(universo, [mediana, q3, maximo, maximo])

    return antecedente


def criar_antecedentes(estatisticas: pd.DataFrame, passo: float = PASSO_UNIVERSO) -> dict:
    """
    Recebe o DataFrame de estatísticas (colunas = variáveis, linhas = quartis)
    e retorna um dicionário {nome_da_coluna: Antecedent}.

    O mesmo `passo` é usado para o universo de discurso de todos os antecedentes.
    """
    antecedentes = {}
    for coluna in estatisticas.columns:
        try:
            antecedentes[coluna] = criar_antecedente_trapezoidal(coluna, estatisticas[coluna], passo)
        except Exception as erro:
            # Não interrompe as demais colunas: avisa qual falhou e o motivo, e segue.
            print(f"[AVISO] Falha ao criar antecedente para '{coluna}': {erro}")
    return antecedentes


# ---------------------------------------------------------------------------
# 3b. Cria o consequente fuzzy "adequabilidade"
# ---------------------------------------------------------------------------

def criar_consequente_adequabilidade(passo: float = PASSO_UNIVERSO,
                                      largura_nucleo: float = LARGURA_NUCLEO_CONSEQUENTE) -> ctrl.Consequent:
    """
    Cria o Consequent fuzzy 'adequabilidade', com universo de discurso fixo
    de 0 a 10 (mesmo passo usado nos antecedentes) e 3 conjuntos
    uniformemente distribuídos, com picos em 0 (baixo), 5 (medio) e 10 (alto).

    Se `largura_nucleo` == 0 (padrão), os termos são triângulos puros (trimf):

        baixo : pico em 0  (trimf [0, 0, 5])
        medio : pico em 5  (trimf [0, 5, 10])
        alto  : pico em 10 (trimf [5, 10, 10])

    Se `largura_nucleo` > 0, os termos viram trapézios (trapmf) com um platô
    (grau de pertinência = 1) dessa largura ao redor do pico, reduzindo a
    sobreposição com os termos vizinhos:

        baixo : platô [0, largura_nucleo/2],           desce até 5
        medio : platô [5 - largura_nucleo/2, 5 + largura_nucleo/2]
        alto  : platô [10 - largura_nucleo/2, 10],      sobe a partir de 5

    Ver LARGURA_NUCLEO_CONSEQUENTE no topo do script para mais detalhes.
    """
    if largura_nucleo < 0 or largura_nucleo >= 10:
        raise ValueError(
            f"LARGURA_NUCLEO_CONSEQUENTE inválida: {largura_nucleo}. "
            "Deve ser >= 0 e < 10 (tipicamente bem menor, ex.: 1.0 a 3.0)."
        )

    # np.arange não inclui o valor final, então somamos um "passo" extra
    # para garantir que o 10 fique dentro do universo (mesma lógica dos antecedentes).
    universo = np.arange(0, 10 + passo, passo)
    consequente = ctrl.Consequent(universo, 'adequabilidade')

    if largura_nucleo == 0:
        consequente['baixo'] = fuzz.trimf(universo, [0, 0, 5])
        consequente['medio'] = fuzz.trimf(universo, [0, 5, 10])
        consequente['alto'] = fuzz.trimf(universo, [5, 10, 10])
    else:
        meia_largura = largura_nucleo / 2
        consequente['baixo'] = fuzz.trapmf(universo, [0, 0, meia_largura, 5])
        consequente['medio'] = fuzz.trapmf(universo, [0, 5 - meia_largura, 5 + meia_largura, 10])
        consequente['alto'] = fuzz.trapmf(universo, [5, 10 - meia_largura, 10, 10])

    return consequente


# ---------------------------------------------------------------------------
# 3c. Calcula o grau de pertinência das medianas vencedoras nos antecedentes
# ---------------------------------------------------------------------------

def calcular_graus_pertinencia(antecedentes: dict, medianas: pd.Series) -> dict:
    """
    Para cada mediana em `medianas` (indexada pelo nome da variável, ex: 'llc_mpki'),
    localiza o Antecedent de mesmo nome em `antecedentes` e calcula o grau de
    pertinência desse valor em CADA função de pertinência do antecedente
    (baixo, medio, alto), usando fuzz.interp_membership.

    Retorna um dicionário aninhado:
        {
            'llc_mpki': {'baixo': 0.12, 'medio': 0.80, 'alto': 0.0},
            'llc_hitrate': {...},
            ...
        }
    """
    graus_por_variavel = {}

    for nome, valor in medianas.items():
        if nome not in antecedentes:
            print(f"[AVISO] Não há antecedente correspondente para '{nome}'; pulando.")
            continue

        antecedente = antecedentes[nome]
        graus = {}
        for termo, conjunto in antecedente.terms.items():
            grau = fuzz.interp_membership(antecedente.universe, conjunto.mf, valor)
            graus[termo] = float(grau)

        graus_por_variavel[nome] = graus

    return graus_por_variavel


# ---------------------------------------------------------------------------
# 3d. Gera as regras fuzzy automaticamente a partir dos graus de pertinência
# ---------------------------------------------------------------------------

def determinar_termo_dominante(graus_variavel: dict) -> str:
    """
    Recebe o dicionário de graus de uma variável (ex: {'baixo': 0.0, 'medio': 0.36,
    'alto': 0.64}) e retorna o nome do termo com maior grau de pertinência.
    """
    return max(graus_variavel, key=graus_variavel.get)


def validar_colunas_independentes_e_sinergicas(colunas_independentes: set, colunas_sinergicas: list) -> set:
    """
    Valida que:
    - nenhuma coluna apareça em mais de um grupo sinérgico;
    - nenhuma coluna apareça simultaneamente em `colunas_independentes` e em
      algum grupo sinérgico.

    Retorna o conjunto de todas as colunas que pertencem a algum grupo
    sinérgico (união de todos os grupos), para uso pelo chamador.

    Levanta ValueError com uma mensagem clara caso alguma das validações falhe.
    """
    colunas_em_algum_grupo_sinergico = set()
    for grupo in colunas_sinergicas:
        grupo_set = set(grupo)

        sobreposicao_independente = grupo_set & colunas_independentes
        if sobreposicao_independente:
            raise ValueError(
                f"Coluna(s) {sorted(sobreposicao_independente)} configurada(s) tanto em "
                f"COLUNAS_INDEPENDENTES quanto no grupo sinérgico {list(grupo)} — cada "
                f"coluna só pode pertencer a UMA das duas categorias."
            )

        sobreposicao_outro_grupo = grupo_set & colunas_em_algum_grupo_sinergico
        if sobreposicao_outro_grupo:
            raise ValueError(
                f"Coluna(s) {sorted(sobreposicao_outro_grupo)} aparecem em mais de um "
                f"grupo de COLUNAS_SINERGICAS — cada coluna só pode pertencer a UM grupo."
            )

        colunas_em_algum_grupo_sinergico.update(grupo_set)

    return colunas_em_algum_grupo_sinergico


def gerar_regra_fuzzy(antecedentes: dict, consequente: ctrl.Consequent,
                       graus: dict, termo_consequente: str,
                       colunas_independentes: list = None,
                       colunas_sinergicas: list = None) -> list:
    """
    A partir do dicionário de graus de pertinência de UM grupo (vencedoras,
    intermediárias ou perdedoras), monta a(s) regra(s) fuzzy correspondente(s).

    As variáveis se dividem em três categorias:

    1. INDEPENDENTES (`colunas_independentes`): cada uma gera sua PRÓPRIA
       regra, isolada das demais (sem E/AND):

           SE <var_independente> é <termo_dominante> ENTÃO adequabilidade é <termo_consequente>

    2. SINÉRGICAS (`colunas_sinergicas`, uma lista de grupos — pares, trios,
       quartetos etc.): cada grupo gera sua PRÓPRIA regra, combinando por
       E/AND SOMENTE as colunas daquele grupo (sem misturar com as demais):

           SE <col_grupo_1> é <termo> E <col_grupo_2> é <termo> [E ...]
           ENTÃO adequabilidade é <termo_consequente>

    3. RESTANTES (nem independentes, nem em nenhum grupo sinérgico): todas
       combinadas em uma única regra conjunta "geral":

           SE <var1> é <termo_dominante1> E <var2> é <termo_dominante2> E ...
           ENTÃO adequabilidade é <termo_consequente>

    O "termo dominante" de cada variável é o que teve o MAIOR grau de
    pertinência para a mediana daquele grupo (vencedora/intermediária/perdedora).

    Uma mesma coluna não pode estar simultaneamente em `colunas_independentes`
    e em um grupo de `colunas_sinergicas`, nem em mais de um grupo sinérgico
    — ver `validar_colunas_independentes_e_sinergicas`.

    Retorna uma LISTA de tuplas (regra_ctrl, descricao_textual, termos_dominantes)
    — 1 item por grupo sinérgico com 2+ colunas presentes, mais 1 item para a
    regra conjunta "restante" (se sobrar alguma variável), mais 1 item extra
    para cada variável independente presente no grupo.
    """
    colunas_independentes = set(colunas_independentes or [])
    colunas_sinergicas = [list(grupo) for grupo in (colunas_sinergicas or [])]

    colunas_em_algum_grupo_sinergico = validar_colunas_independentes_e_sinergicas(
        colunas_independentes, colunas_sinergicas
    )

    termos_dominantes = {
        nome: determinar_termo_dominante(graus_var)
        for nome, graus_var in graus.items()
        if nome in antecedentes
    }

    if not termos_dominantes:
        raise ValueError("Nenhuma variável em comum entre 'graus' e 'antecedentes' para montar a regra.")

    termos_independentes = {
        nome: termo for nome, termo in termos_dominantes.items() if nome in colunas_independentes
    }

    regras = []

    # --- Regras SINÉRGICAS: uma regra por grupo, combinando só as colunas do grupo ---
    colunas_cobertas_por_sinergia = set()
    for grupo in colunas_sinergicas:
        termos_do_grupo = {nome: termos_dominantes[nome] for nome in grupo if nome in termos_dominantes}
        if len(termos_do_grupo) < 2:
            # Menos de 2 colunas do grupo disponíveis para este subconjunto
            # de dados (ex.: prefetcher sem alguma dessas colunas) — não dá
            # para formar uma relação AND de verdade; pula esse grupo.
            continue

        condicoes = [antecedentes[nome][termo] for nome, termo in termos_do_grupo.items()]
        condicao_combinada = condicoes[0]
        for condicao in condicoes[1:]:
            condicao_combinada = condicao_combinada & condicao

        regra = ctrl.Rule(condicao_combinada, consequente[termo_consequente])
        descricao = " E ".join(f"{nome} é {termo}" for nome, termo in termos_do_grupo.items())
        descricao_completa = f"SE {descricao} ENTÃO adequabilidade é {termo_consequente}  [regra sinérgica]"

        regras.append((regra, descricao_completa, dict(termos_do_grupo)))
        colunas_cobertas_por_sinergia.update(termos_do_grupo.keys())

    # --- Regra conjunta (E/AND) com as variáveis restantes (nem independentes, nem sinérgicas) ---
    termos_combinados = {
        nome: termo for nome, termo in termos_dominantes.items()
        if nome not in colunas_independentes and nome not in colunas_cobertas_por_sinergia
    }
    if termos_combinados:
        condicoes = [antecedentes[nome][termo] for nome, termo in termos_combinados.items()]
        condicao_combinada = condicoes[0]
        for condicao in condicoes[1:]:
            condicao_combinada = condicao_combinada & condicao

        regra = ctrl.Rule(condicao_combinada, consequente[termo_consequente])
        descricao = " E ".join(f"{nome} é {termo}" for nome, termo in termos_combinados.items())
        descricao_completa = f"SE {descricao} ENTÃO adequabilidade é {termo_consequente}"

        regras.append((regra, descricao_completa, dict(termos_combinados)))

    # --- Regras independentes: uma regra separada por variável ---
    for nome, termo in termos_independentes.items():
        regra = ctrl.Rule(antecedentes[nome][termo], consequente[termo_consequente])
        descricao_completa = (
            f"SE {nome} é {termo} ENTÃO adequabilidade é {termo_consequente}  [regra independente]"
        )
        regras.append((regra, descricao_completa, {nome: termo}))

    return regras


def gerar_regras_fuzzy(antecedentes: dict, consequente: ctrl.Consequent,
                        graus_vencedoras: dict, graus_intermediarias: dict,
                        graus_perdedoras: dict,
                        colunas_independentes: list = None,
                        colunas_sinergicas: list = None) -> list:
    """
    Gera automaticamente as regras fuzzy (vencedoras -> alto,
    intermediárias -> medio, perdedoras -> baixo) a partir dos graus de
    pertinência já calculados para cada grupo.

    Se `colunas_independentes` for informado, as variáveis listadas geram
    regras próprias e separadas (ver `gerar_regra_fuzzy`). Se
    `colunas_sinergicas` for informado (lista de grupos — pares, trios,
    quartetos etc.), cada grupo gera sua própria regra combinando por E/AND
    somente as colunas daquele grupo. Assim, cada grupo (vencedoras/
    intermediárias/perdedoras) pode contribuir com MAIS de uma regra: 1 por
    grupo sinérgico + 1 regra conjunta "restante" + 1 por variável independente.

    Retorna uma lista de tuplas (nome_grupo, regra_ctrl, descricao_textual,
    termos_dominantes, termo_consequente).
    """
    grupos = [
        ('vencedoras', graus_vencedoras, 'alto'),
        ('intermediarias', graus_intermediarias, 'medio'),
        ('perdedoras', graus_perdedoras, 'baixo'),
    ]

    regras = []
    for nome_grupo, graus, termo_consequente in grupos:
        regras_do_grupo = gerar_regra_fuzzy(
            antecedentes, consequente, graus, termo_consequente,
            colunas_independentes, colunas_sinergicas
        )
        for regra, descricao, termos_dominantes in regras_do_grupo:
            regras.append((nome_grupo, regra, descricao, termos_dominantes, termo_consequente))

    return regras


# ---------------------------------------------------------------------------
# 3d-bis. Resolve conflitos entre regras independentes
# ---------------------------------------------------------------------------

def resolver_conflitos_regras_independentes(
    regras_geradas: list,
    graus_por_grupo: dict,
    estrategia: str = ESTRATEGIA_RESOLUCAO_CONFLITOS,
    ordem_prioridade_grupos: list = None,
) -> tuple:
    """
    Resolve conflitos entre regras INDEPENDENTES (aquelas cujo `termos_dominantes`
    define o termo de uma única variável) que apontam para o MESMO par
    (variável, termo) com consequentes diferentes, ex:

        [intermediarias] SE dram_congest_pki é medio ENTÃO adequabilidade é medio
        [perdedoras]      SE dram_congest_pki é medio ENTÃO adequabilidade é baixo

    Regras conjuntas (que combinam mais de uma variável por E/AND) nunca
    entram nesse conflito, pois cada uma delas já cobre uma combinação
    diferente de termos — só passam por aqui inalteradas.

    Estratégias suportadas (ver ESTRATEGIA_RESOLUCAO_CONFLITOS no topo do
    script para a descrição completa de cada uma):

        'grau'        Vence a regra com maior grau de pertinência no termo.
        'prioridade'  Vence a regra do grupo mais bem posicionado em
                      `ordem_prioridade_grupos`.
        'descartar'   Todas as regras conflitantes daquele par são removidas.
        'peso'        Todas são mantidas, cada uma com `regra.weight` = grau
                      de pertinência do seu grupo naquele termo.

    Parâmetros
    ----------
    regras_geradas : list
        Lista retornada por `gerar_regras_fuzzy`.
    graus_por_grupo : dict
        Dicionário {nome_grupo: graus_do_grupo}, ex.:
        {'vencedoras': graus_vencedoras, 'intermediarias': graus_intermediarias,
         'perdedoras': graus_perdedoras} — mesmo formato retornado por
        `calcular_graus_pertinencia`.
    estrategia : str
        Uma de 'grau', 'prioridade', 'descartar', 'peso'.
        Default: ESTRATEGIA_RESOLUCAO_CONFLITOS.
    ordem_prioridade_grupos : list
        Necessário apenas quando `estrategia == 'prioridade'`. Lista de nomes
        de grupo, do mais para o menos prioritário. Default: ORDEM_PRIORIDADE_GRUPOS.

    Retorna
    -------
    tuple (regras_resolvidas, conflitos_resolvidos)
        `regras_resolvidas`: nova lista de regras (mesma estrutura de
        `regras_geradas`) já refletindo a estratégia escolhida.
        `conflitos_resolvidos`: lista de dicionários descrevendo cada conflito
        encontrado e como foi tratado.
    """
    estrategias_validas = {'grau', 'prioridade', 'descartar', 'peso'}
    if estrategia not in estrategias_validas:
        raise ValueError(
            f"ESTRATEGIA_RESOLUCAO_CONFLITOS inválida: '{estrategia}'. "
            f"Use uma de: {sorted(estrategias_validas)}."
        )

    if estrategia == 'prioridade':
        ordem_prioridade_grupos = ordem_prioridade_grupos or ORDEM_PRIORIDADE_GRUPOS
        if not ordem_prioridade_grupos:
            raise ValueError(
                "estrategia='prioridade' requer 'ordem_prioridade_grupos' "
                "(ex: ['perdedoras', 'intermediarias', 'vencedoras'])."
            )

    # Agrupa as regras INDEPENDENTES por (nome_variavel, termo); as demais
    # (regras conjuntas) seguem direto para o resultado, sem checagem.
    grupos_por_par = {}
    regras_resolvidas = []
    for item in regras_geradas:
        _nome_grupo, _regra, _descricao, termos_dominantes, _termo_consequente = item
        if len(termos_dominantes) == 1:
            (nome_var, termo), = termos_dominantes.items()
            grupos_por_par.setdefault((nome_var, termo), []).append(item)
        else:
            regras_resolvidas.append(item)

    def grau_do_item(item, nome_var, termo):
        return graus_por_grupo[item[0]][nome_var][termo]

    conflitos_resolvidos = []

    for (nome_var, termo), itens in grupos_por_par.items():
        consequentes_distintos = {item[4] for item in itens}

        # Só uma regra, ou várias mas concordando no consequente: não há
        # conflito de verdade — mantém a de maior grau e segue.
        if len(itens) == 1 or len(consequentes_distintos) == 1:
            item_escolhido = max(itens, key=lambda item: grau_do_item(item, nome_var, termo))
            regras_resolvidas.append(item_escolhido)
            continue

        # --- Há conflito de verdade: consequentes diferentes para o mesmo (var, termo) ---
        info_conflito = {'variavel': nome_var, 'termo': termo, 'estrategia': estrategia}

        if estrategia == 'grau':
            item_vencedor = max(itens, key=lambda item: grau_do_item(item, nome_var, termo))
            itens_descartados = [item for item in itens if item is not item_vencedor]
            regras_resolvidas.append(item_vencedor)
            info_conflito.update({
                'grupo_vencedor': item_vencedor[0],
                'consequente_vencedor': item_vencedor[4],
                'grau_vencedor': grau_do_item(item_vencedor, nome_var, termo),
                'descartados': [
                    {'grupo': item[0], 'consequente': item[4], 'grau': grau_do_item(item, nome_var, termo)}
                    for item in itens_descartados
                ],
            })

        elif estrategia == 'prioridade':
            def posicao_prioridade(item):
                try:
                    return ordem_prioridade_grupos.index(item[0])
                except ValueError:
                    return len(ordem_prioridade_grupos)  # grupos não listados ficam por último

            item_vencedor = min(itens, key=posicao_prioridade)
            itens_descartados = [item for item in itens if item is not item_vencedor]
            regras_resolvidas.append(item_vencedor)
            info_conflito.update({
                'grupo_vencedor': item_vencedor[0],
                'consequente_vencedor': item_vencedor[4],
                'grau_vencedor': grau_do_item(item_vencedor, nome_var, termo),
                'descartados': [
                    {'grupo': item[0], 'consequente': item[4], 'grau': grau_do_item(item, nome_var, termo)}
                    for item in itens_descartados
                ],
            })

        elif estrategia == 'descartar':
            info_conflito.update({
                'grupo_vencedor': None,
                'consequente_vencedor': None,
                'grau_vencedor': None,
                'descartados': [
                    {'grupo': item[0], 'consequente': item[4], 'grau': grau_do_item(item, nome_var, termo)}
                    for item in itens
                ],
            })
            # Nenhum item entra em regras_resolvidas: a lacuna fica para a
            # regra geral (catch-all) de área cega cobrir.

        elif estrategia == 'peso':
            mantidos = []
            for item in itens:
                grau = grau_do_item(item, nome_var, termo)
                regra_obj = item[1]
                try:
                    regra_obj.weight = grau
                except Exception:
                    print(f"[AVISO] Não foi possível aplicar peso na regra '{item[2]}' "
                          f"(Rule.weight não suportado nesta versão do scikit-fuzzy).")
                mantidos.append({'grupo': item[0], 'consequente': item[4], 'grau': grau})
                regras_resolvidas.append(item)
            info_conflito.update({
                'grupo_vencedor': None,
                'consequente_vencedor': None,
                'grau_vencedor': None,
                'mantidos': mantidos,
                'descartados': [],
            })

        conflitos_resolvidos.append(info_conflito)

    return regras_resolvidas, conflitos_resolvidos


def imprimir_relatorio_conflitos_resolvidos(conflitos_resolvidos: list) -> None:
    """
    Imprime um relatório legível a partir da lista retornada por
    `resolver_conflitos_regras_independentes`.
    """
    if not conflitos_resolvidos:
        print("Nenhum conflito entre regras independentes foi encontrado.")
        return

    estrategia = conflitos_resolvidos[0]['estrategia']
    print(f"{len(conflitos_resolvidos)} conflito(s) encontrado(s) (estratégia: '{estrategia}'):")

    for conflito in conflitos_resolvidos:
        cabecalho = f"  - {conflito['variavel']}={conflito['termo']}"

        if conflito['estrategia'] == 'descartar':
            print(f"{cabecalho}: TODAS as regras descartadas (lacuna coberta pela regra geral)")
            for descartado in conflito['descartados']:
                print(f"      descartada: '{descartado['grupo']}' -> adequabilidade {descartado['consequente']} "
                      f"(grau={descartado['grau']:.4f})")

        elif conflito['estrategia'] == 'peso':
            print(f"{cabecalho}: TODAS as regras mantidas, com peso proporcional ao grau")
            for mantido in conflito['mantidos']:
                print(f"      mantida (peso={mantido['grau']:.4f}): '{mantido['grupo']}' -> "
                      f"adequabilidade {mantido['consequente']}")

        else:  # 'grau' ou 'prioridade'
            print(f"{cabecalho}: venceu '{conflito['grupo_vencedor']}' -> "
                  f"adequabilidade {conflito['consequente_vencedor']} "
                  f"(grau={conflito['grau_vencedor']:.4f})")
            for descartado in conflito['descartados']:
                print(f"      descartada: '{descartado['grupo']}' -> adequabilidade {descartado['consequente']} "
                      f"(grau={descartado['grau']:.4f})")


# ---------------------------------------------------------------------------
# 3e. Verifica áreas cegas (combinações de termos sem regra correspondente)
# ---------------------------------------------------------------------------

def verificar_areas_cegas(antecedentes: dict, regras_geradas: list,
                           colunas_independentes: list = None,
                           colunas_sinergicas: list = None) -> dict:
    """
    Verifica se as regras fuzzy geradas cobrem todas as combinações possíveis
    de termos entre os antecedentes.

    Como as variáveis em `colunas_independentes` e as colunas de cada grupo
    de `colunas_sinergicas` geram regras PRÓPRIAS (ver `gerar_regra_fuzzy`),
    elas são avaliadas SEPARADAMENTE e NÃO entram no produto combinatório das
    variáveis "restantes" (as que formam a regra conjunta geral E/AND).
    Misturá-las no mesmo produto combinatório é o que fazia a checagem antiga
    nunca encontrar nenhuma regra combinada.

    Três coisas são checadas:
    1. Área cega das variáveis RESTANTES: combinação de termos entre as
       variáveis que não são independentes nem pertencem a nenhum grupo
       sinérgico, para a qual nenhuma regra conjunta "geral" foi definida.
    2. Área cega de cada GRUPO SINÉRGICO: combinação de termos DENTRO daquele
       grupo (ex: llc_mpki=baixo E llc_hitrate=alto) para a qual nenhuma
       regra daquele grupo foi definida.
    3. Área cega das variáveis INDEPENDENTES: termo de uma variável
       independente (ex: dram_congest_pki=alto) para o qual nenhuma regra
       independente foi definida.

    Também verifica CONFLITOS nos três casos: a mesma combinação/termo
    aparecendo em mais de uma regra com consequentes diferentes.

    Retorna um dicionário com o resumo da verificação.
    """
    colunas_independentes = set(colunas_independentes or [])
    colunas_sinergicas = [list(grupo) for grupo in (colunas_sinergicas or [])]

    colunas_em_algum_grupo_sinergico = validar_colunas_independentes_e_sinergicas(
        colunas_independentes, colunas_sinergicas
    )

    # --- Variáveis restantes (nem independentes, nem sinérgicas) — entram na regra conjunta "geral" ---
    nomes_variaveis = [
        nome for nome in antecedentes
        if nome not in colunas_independentes and nome not in colunas_em_algum_grupo_sinergico
    ]
    termos_por_variavel = [list(antecedentes[nome].terms.keys()) for nome in nomes_variaveis]

    todas_combinacoes = list(itertools.product(*termos_por_variavel)) if nomes_variaveis else []
    total_combinacoes = len(todas_combinacoes)

    # Mapeia cada combinação coberta -> lista de (nome_grupo, termo_consequente)
    #
    # Só entram aqui as regras "conjuntas gerais", ou seja, aquelas cujo
    # `termos_dominantes` define EXATAMENTE as variáveis restantes (nenhuma a
    # mais, nenhuma a menos). Regras independentes e sinérgicas são ignoradas
    # aqui e tratadas nas seções seguintes.
    cobertura = {}
    for nome_grupo, _regra, _descricao, termos_dominantes, termo_consequente in regras_geradas:
        if set(termos_dominantes.keys()) != set(nomes_variaveis):
            continue
        combinacao = tuple(termos_dominantes[nome] for nome in nomes_variaveis)
        cobertura.setdefault(combinacao, []).append((nome_grupo, termo_consequente))

    combinacoes_cobertas = set(cobertura.keys())
    combinacoes_descobertas = [c for c in todas_combinacoes if c not in combinacoes_cobertas]

    # Conflitos: mesma combinação de termos levando a consequentes diferentes
    conflitos = {
        combinacao: grupos
        for combinacao, grupos in cobertura.items()
        if len({termo_consequente for _, termo_consequente in grupos}) > 1
    }

    percentual_coberto = (
        100.0 * len(combinacoes_cobertas) / total_combinacoes if total_combinacoes else 100.0
    )

    # --- Cada GRUPO SINÉRGICO avaliado isoladamente (combinatório só dentro do grupo) ---
    grupos_sinergicos = {}
    for grupo in colunas_sinergicas:
        nomes_grupo = [nome for nome in grupo if nome in antecedentes]
        if len(nomes_grupo) < 2:
            # Grupo com menos de 2 colunas presentes nos antecedentes deste
            # arquivo/prefetcher — não gera regra própria (ver `gerar_regra_fuzzy`),
            # então também não faz sentido avaliar cobertura para ele aqui.
            continue

        termos_por_variavel_grupo = [list(antecedentes[nome].terms.keys()) for nome in nomes_grupo]
        todas_combinacoes_grupo = list(itertools.product(*termos_por_variavel_grupo))

        cobertura_grupo = {}
        for nome_grupo_regra, _regra, _descricao, termos_dominantes, termo_consequente in regras_geradas:
            if set(termos_dominantes.keys()) != set(nomes_grupo):
                continue
            combinacao = tuple(termos_dominantes[nome] for nome in nomes_grupo)
            cobertura_grupo.setdefault(combinacao, []).append((nome_grupo_regra, termo_consequente))

        combinacoes_cobertas_grupo = set(cobertura_grupo.keys())
        combinacoes_descobertas_grupo = [
            c for c in todas_combinacoes_grupo if c not in combinacoes_cobertas_grupo
        ]
        conflitos_grupo = {
            combinacao: grupos_regra
            for combinacao, grupos_regra in cobertura_grupo.items()
            if len({tc for _, tc in grupos_regra}) > 1
        }
        total_combinacoes_grupo = len(todas_combinacoes_grupo)

        grupos_sinergicos[tuple(nomes_grupo)] = {
            'nomes_variaveis': nomes_grupo,
            'total_combinacoes': total_combinacoes_grupo,
            'combinacoes_cobertas': combinacoes_cobertas_grupo,
            'combinacoes_descobertas': combinacoes_descobertas_grupo,
            'percentual_coberto': (
                100.0 * len(combinacoes_cobertas_grupo) / total_combinacoes_grupo
                if total_combinacoes_grupo else 100.0
            ),
            'conflitos': conflitos_grupo,
        }

    # --- Variáveis independentes (cada uma avaliada isoladamente) ---
    variaveis_independentes = [nome for nome in antecedentes if nome in colunas_independentes]

    cobertura_independentes = {
        nome: {termo: [] for termo in antecedentes[nome].terms.keys()}
        for nome in variaveis_independentes
    }
    for nome_grupo, _regra, _descricao, termos_dominantes, termo_consequente in regras_geradas:
        if len(termos_dominantes) != 1:
            continue
        (nome_var, termo), = termos_dominantes.items()
        if nome_var in cobertura_independentes:
            cobertura_independentes[nome_var][termo].append((nome_grupo, termo_consequente))

    termos_descobertos_independentes = {
        nome: [termo for termo, grupos in termos.items() if not grupos]
        for nome, termos in cobertura_independentes.items()
    }
    conflitos_independentes = {
        (nome, termo): grupos
        for nome, termos in cobertura_independentes.items()
        for termo, grupos in termos.items()
        if len({tc for _, tc in grupos}) > 1
    }

    resumo = {
        'nomes_variaveis': nomes_variaveis,
        'total_combinacoes': total_combinacoes,
        'combinacoes_cobertas': combinacoes_cobertas,
        'combinacoes_descobertas': combinacoes_descobertas,
        'percentual_coberto': percentual_coberto,
        'conflitos': conflitos,
        'grupos_sinergicos': grupos_sinergicos,
        'variaveis_independentes': variaveis_independentes,
        'cobertura_independentes': cobertura_independentes,
        'termos_descobertos_independentes': termos_descobertos_independentes,
        'conflitos_independentes': conflitos_independentes,
    }
    return resumo


def imprimir_relatorio_areas_cegas(resumo: dict) -> None:
    """
    Imprime um relatório legível a partir do dicionário retornado por
    `verificar_areas_cegas`, agrupado em três seções:

    - VARIÁVEIS RESTANTES (REGRA CONJUNTA GERAL): variáveis que não são
      independentes nem pertencem a nenhum grupo sinérgico, combinadas por
      E/AND em uma única regra conjunta (cobertura avaliada sobre as
      combinações de termos entre elas).
    - GRUPOS SINÉRGICOS: cada grupo de COLUNAS_SINERGICAS avaliado
      isoladamente (cobertura avaliada só sobre as combinações de termos
      DENTRO daquele grupo).
    - VARIÁVEIS INDEPENDENTES: variáveis que geram regra própria (cobertura
      avaliada termo a termo, isoladamente).
    """
    # --- Seção 1: variáveis restantes (regra conjunta geral, E/AND) ---
    nomes_variaveis = resumo['nomes_variaveis']
    total = resumo['total_combinacoes']
    n_cobertas = len(resumo['combinacoes_cobertas'])
    n_descobertas = len(resumo['combinacoes_descobertas'])

    print("VARIÁVEIS RESTANTES (regra conjunta geral)")
    print(f"  Variáveis consideradas: {nomes_variaveis}")
    print(f"  Total de combinações possíveis de termos: {total}")
    print(f"  Combinações cobertas por alguma regra: {n_cobertas} ({resumo['percentual_coberto']:.1f}%)")
    print(f"  Combinações SEM nenhuma regra (áreas cegas): {n_descobertas} "
          f"({100 - resumo['percentual_coberto']:.1f}%)")

    if resumo['conflitos']:
        print(f"  [ALERTA] {len(resumo['conflitos'])} combinação(ões) com regras conflitantes "
              f"(mesma combinação de termos apontando para consequentes diferentes):")
        for combinacao, grupos in resumo['conflitos'].items():
            combinacao_fmt = ", ".join(f"{v}={t}" for v, t in zip(nomes_variaveis, combinacao))
            grupos_fmt = ", ".join(f"{g}->{c}" for g, c in grupos)
            print(f"    - ({combinacao_fmt}): {grupos_fmt}")

    # --- Seção 2: grupos sinérgicos (cada um avaliado isoladamente) ---
    grupos_sinergicos = resumo.get('grupos_sinergicos', {})
    print()
    print("GRUPOS SINÉRGICOS")
    if not grupos_sinergicos:
        print("  Nenhum grupo sinérgico configurado (ou nenhum com 2+ colunas disponíveis).")
    else:
        for nomes_grupo, info_grupo in grupos_sinergicos.items():
            n_cobertas_grupo = len(info_grupo['combinacoes_cobertas'])
            n_descobertas_grupo = len(info_grupo['combinacoes_descobertas'])
            print(f"  - {list(nomes_grupo)}: {n_cobertas_grupo}/{info_grupo['total_combinacoes']} "
                  f"combinação(ões) coberta(s) ({info_grupo['percentual_coberto']:.1f}%)")
            if n_descobertas_grupo:
                for combinacao in info_grupo['combinacoes_descobertas']:
                    combinacao_fmt = ", ".join(f"{v}={t}" for v, t in zip(nomes_grupo, combinacao))
                    print(f"      SEM regra (área cega): ({combinacao_fmt})")
            if info_grupo['conflitos']:
                print(f"      [ALERTA] {len(info_grupo['conflitos'])} combinação(ões) com regras "
                      f"conflitantes:")
                for combinacao, grupos_regra in info_grupo['conflitos'].items():
                    combinacao_fmt = ", ".join(f"{v}={t}" for v, t in zip(nomes_grupo, combinacao))
                    grupos_fmt = ", ".join(f"{g}->{c}" for g, c in grupos_regra)
                    print(f"        - ({combinacao_fmt}): {grupos_fmt}")

    # --- Seção 3: variáveis independentes (cada uma avaliada isoladamente) ---
    variaveis_independentes = resumo.get('variaveis_independentes', [])
    print()
    print("VARIÁVEIS INDEPENDENTES")
    if not variaveis_independentes:
        print("  Nenhuma variável independente configurada.")
    else:
        print(f"  Variáveis consideradas: {variaveis_independentes}")
        for nome in variaveis_independentes:
            cobertura_var = resumo['cobertura_independentes'][nome]
            termos_descobertos = resumo['termos_descobertos_independentes'][nome]
            n_termos = len(cobertura_var)
            n_termos_cobertos = n_termos - len(termos_descobertos)
            print(f"  - {nome}: {n_termos_cobertos}/{n_termos} termo(s) coberto(s)")
            if termos_descobertos:
                print(f"      termo(s) SEM regra (áreas cegas): {termos_descobertos}")

        if resumo.get('conflitos_independentes'):
            print(f"  [ALERTA] {len(resumo['conflitos_independentes'])} termo(s) de variável independente "
                  f"com regras conflitantes:")
            for (nome, termo), grupos in resumo['conflitos_independentes'].items():
                grupos_fmt = ", ".join(f"{g}->{c}" for g, c in grupos)
                print(f"    - ({nome}={termo}): {grupos_fmt}")


# ---------------------------------------------------------------------------
# 3f. Gera UMA regra geral (catch-all) para cobrir toda a área cega
# ---------------------------------------------------------------------------

def gerar_regra_geral_area_cega(antecedentes: dict, consequente: ctrl.Consequent,
                                 resumo_areas_cegas: dict,
                                 termo_consequente_padrao: str = TERMO_PADRAO_AREA_CEGA,
                                 tamanho_maximo_condicoes_por_regra: int = TAMANHO_MAXIMO_CONDICOES_POR_REGRA_GERAL):
    """
    Gera a(s) regra(s) fuzzy "geral(is)" (catch-all) que cobrem TODAS as
    combinações de termos identificadas como "área cega" por
    `verificar_areas_cegas` (combinações de antecedentes para as quais
    nenhuma regra específica foi gerada por `gerar_regras_fuzzy`).

    Cada condição é montada com E (operador `&` do scikit-fuzzy) entre os
    termos de uma combinação descoberta, e as condições são então
    encadeadas com OU (operador `|`) para formar a(s) regra(s):

        SE (var1=t1 E var2=t2 E ...) OU (var1=t1' E var2=t2' E ...) OU ...
        ENTÃO adequabilidade é <termo_consequente_padrao>

    IMPORTANTE — divisão em blocos: encadear TODAS as condições numa única
    regra cria uma árvore de expressão muito profunda quando há muitas áreas
    cegas (comum em prefetchers com muitas colunas, como o pythia), e o
    scikit-fuzzy percorre essa árvore recursivamente ao montar a regra —
    o que pode estourar o limite de recursão do Python. Por isso, as
    condições são divididas em blocos de até `tamanho_maximo_condicoes_por_regra`,
    e CADA BLOCO vira sua própria regra (todas produzindo o mesmo
    `termo_consequente_padrao`) — funcionalmente equivalente a uma única
    regra gigante (todas disparam a mesma saída), mas sem o risco de
    recursão excessiva.

    Por padrão, `termo_consequente_padrao` é 'baixo': como essas regiões do
    espaço de entrada não têm dados suficientes (nenhum grupo vencedor/
    intermediário/perdedor teve mediana caindo predominantemente ali), a
    saída mais conservadora é considerar adequabilidade baixa, em vez de
    deixar a região sem regra alguma (o que zeraria a saída do sistema fuzzy
    nesses casos) ou assumir uma saída otimista sem embasamento.

    Parâmetros
    ----------
    antecedentes : dict
        Dicionário {nome_da_variavel: Antecedent}, o mesmo usado nas demais
        etapas do script.
    consequente : ctrl.Consequent
        Consequent 'adequabilidade' criado por `criar_consequente_adequabilidade`.
    resumo_areas_cegas : dict
        Dicionário retornado por `verificar_areas_cegas`.
    termo_consequente_padrao : str
        Termo do consequente ('baixo', 'medio' ou 'alto') que a(s) regra(s)
        geral(is) devem produzir. Default: TERMO_PADRAO_AREA_CEGA.
    tamanho_maximo_condicoes_por_regra : int
        Máximo de condições OR-encadeadas por regra. Default:
        TAMANHO_MAXIMO_CONDICOES_POR_REGRA_GERAL.

    Retorna
    -------
    None
        Se não houver nenhuma área cega — nem nas combinações conjuntas
        restantes, nem nos grupos sinérgicos, nem nos termos de variáveis
        independentes — ou seja, não há nada para a regra geral cobrir.
    tuple (regras_gerais, descricao_geral, itens_cobertos)
        Caso contrário. `regras_gerais` é uma LISTA de `ctrl.Rule` (1 ou
        mais, conforme a divisão em blocos). `itens_cobertos` é um
        dicionário com as chaves 'combinacoes' (lista de combinações
        conjuntas restantes descobertas), 'sinergicas' (lista de tuplas
        (nomes_do_grupo, combinacao) descobertas por grupo sinérgico) e
        'independentes' (lista de tuplas (nome_variavel, termo)
        descobertas) que essas regras gerais passam a cobrir.
    """
    nomes_variaveis = resumo_areas_cegas['nomes_variaveis']
    combinacoes_descobertas = resumo_areas_cegas['combinacoes_descobertas']
    termos_descobertos_independentes = resumo_areas_cegas.get('termos_descobertos_independentes', {})
    grupos_sinergicos = resumo_areas_cegas.get('grupos_sinergicos', {})

    # Lista achatada de (nome_variavel_independente, termo) sem regra
    independentes_descobertos = [
        (nome, termo)
        for nome, termos in termos_descobertos_independentes.items()
        for termo in termos
    ]

    # Lista achatada de (nomes_do_grupo, combinacao) sem regra, por grupo sinérgico
    sinergicas_descobertas = [
        (nomes_grupo, combinacao)
        for nomes_grupo, info_grupo in grupos_sinergicos.items()
        for combinacao in info_grupo['combinacoes_descobertas']
    ]

    if not combinacoes_descobertas and not independentes_descobertos and not sinergicas_descobertas:
        return None

    condicoes_gerais = []

    # Uma condição por combinação conjunta (restante) descoberta (E entre as
    # variáveis daquela combinação)
    for combinacao in combinacoes_descobertas:
        condicoes_variaveis = [
            antecedentes[nome][termo]
            for nome, termo in zip(nomes_variaveis, combinacao)
        ]
        condicao_and = condicoes_variaveis[0]
        for condicao in condicoes_variaveis[1:]:
            condicao_and = condicao_and & condicao
        condicoes_gerais.append(condicao_and)

    # Uma condição por combinação sinérgica descoberta (E só entre as
    # variáveis daquele grupo)
    for nomes_grupo, combinacao in sinergicas_descobertas:
        condicoes_variaveis = [
            antecedentes[nome][termo]
            for nome, termo in zip(nomes_grupo, combinacao)
        ]
        condicao_and = condicoes_variaveis[0]
        for condicao in condicoes_variaveis[1:]:
            condicao_and = condicao_and & condicao
        condicoes_gerais.append(condicao_and)

    # Uma condição por termo descoberto de variável independente (termo
    # isolado, sem E com nenhuma outra variável)
    for nome, termo in independentes_descobertos:
        condicoes_gerais.append(antecedentes[nome][termo])

    # Divide as condições em blocos e monta UMA regra por bloco, encadeando
    # só as condições daquele bloco com OU — evita uma única árvore de
    # expressão profunda demais (ver docstring).
    regras_gerais = []
    for inicio in range(0, len(condicoes_gerais), tamanho_maximo_condicoes_por_regra):
        bloco = condicoes_gerais[inicio: inicio + tamanho_maximo_condicoes_por_regra]
        condicao_bloco = bloco[0]
        for condicao in bloco[1:]:
            condicao_bloco = condicao_bloco | condicao
        regras_gerais.append(ctrl.Rule(condicao_bloco, consequente[termo_consequente_padrao]))

    partes_descricao = []
    if combinacoes_descobertas:
        partes_descricao.append(f"{len(combinacoes_descobertas)} combinação(ões) conjunta(s)")
    if sinergicas_descobertas:
        partes_descricao.append(f"{len(sinergicas_descobertas)} combinação(ões) sinérgica(s)")
    if independentes_descobertos:
        partes_descricao.append(f"{len(independentes_descobertos)} termo(s) de variável(is) independente(s)")

    descricao_geral = (
        f"REGRA GERAL (área cega): cobre {' + '.join(partes_descricao)} "
        f"sem regra específica -> adequabilidade é {termo_consequente_padrao} "
        f"(dividida em {len(regras_gerais)} regra(s) de até "
        f"{tamanho_maximo_condicoes_por_regra} condição(ões) cada, para evitar recursão excessiva)"
    )

    itens_cobertos = {
        'combinacoes': combinacoes_descobertas,
        'sinergicas': sinergicas_descobertas,
        'independentes': independentes_descobertos,
    }

    return regras_gerais, descricao_geral, itens_cobertos


# ---------------------------------------------------------------------------
# 3g. Cria o defuzzificador (sistema de controle fuzzy) a partir das regras
# ---------------------------------------------------------------------------

def criar_defuzzificador(regras_geradas: list, regra_geral_area_cega=None) -> tuple:
    """
    Monta o sistema de controle fuzzy (Mamdani) — o "defuzzificador" — a
    partir das regras já geradas (e com conflitos resolvidos) e do consequente
    'adequabilidade', usando `skfuzzy.control`.

    O defuzzificador é o que transforma um conjunto de valores numéricos de
    entrada (um por antecedente) em UM único valor numérico de saída
    (adequabilidade, de 0 a 10): cada regra é avaliada, os graus de disparo
    são agregados nos conjuntos fuzzy do consequente, e o resultado é reduzido
    a um número real pelo método padrão do scikit-fuzzy (centróide).

    Parâmetros
    ----------
    regras_geradas : list
        Lista de regras no formato retornado por `gerar_regras_fuzzy` /
        `resolver_conflitos_regras_independentes`: tuplas
        (nome_grupo, regra_ctrl, descricao, termos_dominantes, termo_consequente).
    regra_geral_area_cega : tuple ou None
        Resultado de `gerar_regra_geral_area_cega`. Se fornecido (não None),
        a(s) regra(s) geral(is) (catch-all) — pode ser mais de uma, se a
        área cega foi dividida em blocos — são incluídas no sistema,
        garantindo que NENHUMA combinação de entradas fique sem regra
        alguma disparando.

    Retorna
    -------
    tuple (sistema_ctrl, simulador)
        `sistema_ctrl` : ctrl.ControlSystem montado com todas as regras.
        `simulador`    : ctrl.ControlSystemSimulation pronto para receber
                         entradas (`simulador.input[nome] = valor`),
                         computar (`simulador.compute()`) e ler a saída
                         (`simulador.output['adequabilidade']`).
    """
    regras_ctrl = [regra for (_nome_grupo, regra, _descricao, _termos, _tc) in regras_geradas]

    if regra_geral_area_cega is not None:
        regras_gerais_ctrl, _descricao_geral, _itens_cobertos = regra_geral_area_cega
        regras_ctrl.extend(regras_gerais_ctrl)

    if not regras_ctrl:
        raise ValueError("Nenhuma regra fornecida para montar o defuzzificador.")

    sistema_ctrl = ctrl.ControlSystem(regras_ctrl)
    simulador = ctrl.ControlSystemSimulation(sistema_ctrl)

    return sistema_ctrl, simulador


def calcular_adequabilidade(simulador: ctrl.ControlSystemSimulation, valores_entrada: dict) -> float:
    """
    Executa o defuzzificador para um conjunto de valores de entrada e
    retorna a adequabilidade resultante (valor numérico de 0 a 10, pelo
    método do centróide).

    Parâmetros
    ----------
    simulador : ctrl.ControlSystemSimulation
        Retornado por `criar_defuzzificador`.
    valores_entrada : dict
        {nome_da_variavel: valor_numerico} — precisa cobrir todas as
        variáveis usadas nos antecedentes (ver
        COLUNAS_SUBCONJUNTO_POR_PREFETCHER para o subconjunto de cada
        prefetcher).

    Retorna
    -------
    float
        Valor de adequabilidade (0 a 10) após a defuzzificação.
    """
    for nome, valor in valores_entrada.items():
        simulador.input[nome] = valor
    simulador.compute()
    return simulador.output['adequabilidade']


def calcular_adequabilidade_dataframe(df: pd.DataFrame, simulador: ctrl.ControlSystemSimulation,
                                       colunas: list) -> pd.Series:
    """
    Aplica o defuzzificador a CADA LINHA de `df`, usando os valores das
    colunas em `colunas` (mesmos nomes usados nos antecedentes, ex.: o
    subconjunto de COLUNAS_SUBCONJUNTO_POR_PREFETCHER daquele prefetcher)
    como entrada do sistema fuzzy.

    Retorna uma Series de adequabilidade (0 a 10), com o mesmo índice de `df`
    — uma linha do CSV, uma adequabilidade calculada.

    Linhas cujos valores de entrada não disparem NENHUMA regra do sistema
    (ex.: valor fora do universo de discurso dos antecedentes) resultam em
    NaN, e um único aviso é impresso ao final (em vez de um por linha).
    """
    resultados = []
    houve_erro = False
    for _, linha in df[colunas].iterrows():
        valores_entrada = linha.to_dict()
        try:
            resultados.append(calcular_adequabilidade(simulador, valores_entrada))
        except Exception:
            resultados.append(float('nan'))
            houve_erro = True

    if houve_erro:
        print("[AVISO] Uma ou mais linhas não puderam ser calculadas pelo defuzzificador "
              "(valor fora do universo de discurso ou nenhuma regra disparada) e ficaram "
              "com adequabilidade = NaN.")

    return pd.Series(resultados, index=df.index, name='adequabilidade')


# ---------------------------------------------------------------------------
# 3h. Sugestão automática de colunas independentes/sinérgicas (heurística estatística)
# ---------------------------------------------------------------------------

def _r2_regressao_linear(colunas_x: list, y: np.ndarray, dados: pd.DataFrame) -> float:
    """
    Ajusta y ~ 1 + colunas_x por mínimos quadrados (numpy puro, sem sklearn)
    e retorna o R² do ajuste. Usado como bloco básico para medir "quanto uma
    combinação de colunas explica de y".
    """
    matriz_design = np.column_stack(
        [np.ones(len(dados))] + [dados[coluna].to_numpy(dtype=float) for coluna in colunas_x]
    )
    coeficientes, _, _, _ = np.linalg.lstsq(matriz_design, y, rcond=None)
    y_previsto = matriz_design @ coeficientes

    soma_quadrados_residuo = np.sum((y - y_previsto) ** 2)
    soma_quadrados_total = np.sum((y - y.mean()) ** 2)

    if soma_quadrados_total <= 0:
        return 0.0
    return 1.0 - soma_quadrados_residuo / soma_quadrados_total


def _r2_grupo_com_interacao(colunas_grupo: list, y: np.ndarray, dados: pd.DataFrame) -> float:
    """
    R² de um modelo com os efeitos individuais de `colunas_grupo` MAIS um
    único termo de interação de ordem alta (produto de todas as colunas do
    grupo). Usado para medir o quanto um grupo sinérgico, como um todo,
    ganha ao ter suas colunas combinadas em vez de tratadas separadamente.
    """
    dados_com_interacao = dados.copy()
    termo_interacao = np.ones(len(dados_com_interacao))
    for coluna in colunas_grupo:
        termo_interacao = termo_interacao * dados_com_interacao[coluna].to_numpy(dtype=float)
    dados_com_interacao['_interacao_grupo'] = termo_interacao

    return _r2_regressao_linear(colunas_grupo + ['_interacao_grupo'], y, dados_com_interacao)


def sugerir_colunas_sinergicas(df: pd.DataFrame, colunas_subconjunto: list, coluna_alvo: str,
                                limiar_sinergia: float = 0.01,
                                limiar_independencia: float = 0.005,
                                max_tamanho_grupo: int = 3,
                                imprimir: bool = True) -> dict:
    """
    Sugere, A PARTIR DOS PRÓPRIOS DADOS, quais colunas de `colunas_subconjunto`
    parecem ser boas candidatas a INDEPENDENTES (efeito aditivo sobre
    `coluna_alvo`, sem interação relevante com as demais) e quais parecem
    formar GRUPOS SINÉRGICOS (interação real entre 2+ colunas, que juntas
    explicam mais de `coluna_alvo` do que a soma dos efeitos individuais).

    MÉTODO (regressão linear com termo de interação — o mesmo princípio por
    trás da H-statistic de Friedman e dos SHAP interaction values, mas
    implementado só com numpy/pandas, sem exigir instalar shap/xgboost):

    Para cada par de colunas (i, j), ajustam-se dois modelos por mínimos
    quadrados, usando `coluna_alvo` como variável dependente:

        aditivo:      y ~ 1 + x_i + x_j
        c/interação:  y ~ 1 + x_i + x_j + (x_i * x_j)

    O "ganho de interação" do par é a diferença de R² entre os dois — o
    quanto o termo (x_i * x_j) sozinho explica ALÉM do que os efeitos
    individuais já explicavam:
      - ganho ALTO  -> sinergia real entre as colunas (candidatas a grupo).
      - ganho BAIXO -> efeitos aditivos (a coluna pode ficar independente,
        ou na regra conjunta geral — esta função não distingue as duas,
        veja `colunas_sem_classificacao` no retorno).

    Pares com ganho >= `limiar_sinergia` são agrupados NUM PROCESSO GULOSO:
    ordena-se os pares do maior para o menor ganho e vai-se formando grupos
    sem reaproveitar colunas já usadas; cada grupo é então testado para
    EXTENSÃO (até `max_tamanho_grupo` colunas), aceitando uma nova coluna no
    grupo só se isso aumentar o R² do grupo (com termo de interação de
    ordem alta) em pelo menos `limiar_sinergia`.

    Colunas cujo MAIOR ganho de interação (com qualquer outra coluna) fica
    abaixo de `limiar_independencia` são sugeridas como independentes.

    IMPORTANTE: esta função só SUGERE — ela NÃO altera
    COLUNAS_INDEPENDENTES_POR_PREFETCHER nem COLUNAS_SINERGICAS_POR_PREFETCHER.
    Revise o resultado e copie manualmente para a config se fizer sentido.

    Parâmetros
    ----------
    df : pd.DataFrame
        Dados amostrados (ex.: o CSV de um prefetcher já carregado).
    colunas_subconjunto : list
        Colunas candidatas, ex.: COLUNAS_SUBCONJUNTO_POR_PREFETCHER[prefetcher].
    coluna_alvo : str
        Nome de uma coluna NUMÉRICA de `df` usada como alvo da análise (ex.:
        uma coluna de adequabilidade já calculada, ou a própria coluna de
        rank do prefetcher).
    limiar_sinergia : float
        Ganho mínimo de R² (0 a 1) para um par/grupo ser sugerido como
        sinérgico. Default 0.01 (1 ponto percentual de R²) — ajuste conforme
        o ruído típico dos seus dados.
    limiar_independencia : float
        Ganho de interação MÁXIMO (com qualquer outra coluna) para uma
        coluna ser sugerida como independente. Default 0.005.
    max_tamanho_grupo : int
        Tamanho máximo de grupo sinérgico considerado na extensão gulosa
        (pares -> trios -> quartetos ...).
    imprimir : bool
        Se True, imprime um relatório legível além de retornar o dicionário.

    Retorna
    -------
    dict com as chaves:
        'scores_individuais': {coluna: R² de y ~ coluna}
        'scores_interacao': {(col_i, col_j): ganho de R² do termo de interação}
        'sugestao_independentes': lista de colunas sugeridas como independentes
        'sugestao_sinergicas': lista de grupos (listas de colunas) sugeridos
        'colunas_sem_classificacao': colunas que não ficaram claramente em
            nenhuma das duas categorias — revise manualmente.
    """
    dados = df[colunas_subconjunto + [coluna_alvo]].dropna()
    if len(dados) < len(colunas_subconjunto) + 3:
        raise ValueError(
            f"Apenas {len(dados)} linha(s) sem NaN em {colunas_subconjunto + [coluna_alvo]} — "
            f"poucas para estimar as regressões com confiança."
        )

    y = dados[coluna_alvo].to_numpy(dtype=float)

    # --- Scores individuais (efeito principal de cada coluna sozinha) ---
    scores_individuais = {
        coluna: _r2_regressao_linear([coluna], y, dados) for coluna in colunas_subconjunto
    }

    # --- Scores de interação par a par ---
    scores_interacao = {}
    for indice, coluna_i in enumerate(colunas_subconjunto):
        for coluna_j in colunas_subconjunto[indice + 1:]:
            r2_aditivo = _r2_regressao_linear([coluna_i, coluna_j], y, dados)
            r2_interacao = _r2_grupo_com_interacao([coluna_i, coluna_j], y, dados)
            scores_interacao[(coluna_i, coluna_j)] = max(0.0, r2_interacao - r2_aditivo)

    # --- Sugestão de independentes: maior interação (com qualquer parceiro) é baixa ---
    maior_interacao_por_coluna = {coluna: 0.0 for coluna in colunas_subconjunto}
    for (coluna_i, coluna_j), ganho in scores_interacao.items():
        maior_interacao_por_coluna[coluna_i] = max(maior_interacao_por_coluna[coluna_i], ganho)
        maior_interacao_por_coluna[coluna_j] = max(maior_interacao_por_coluna[coluna_j], ganho)

    sugestao_independentes = [
        coluna for coluna in colunas_subconjunto
        if maior_interacao_por_coluna[coluna] <= limiar_independencia
    ]
    colunas_independentes_sugeridas = set(sugestao_independentes)

    # --- Sugestão de grupos sinérgicos: guloso por maior ganho, com extensão ---
    pares_ordenados = sorted(
        (item for item in scores_interacao.items() if item[1] >= limiar_sinergia),
        key=lambda item: item[1], reverse=True
    )

    colunas_usadas = set()
    sugestao_sinergicas = []

    for (coluna_i, coluna_j), _ganho in pares_ordenados:
        if coluna_i in colunas_usadas or coluna_j in colunas_usadas:
            continue
        if coluna_i in colunas_independentes_sugeridas or coluna_j in colunas_independentes_sugeridas:
            continue

        grupo = [coluna_i, coluna_j]
        r2_grupo_atual = _r2_grupo_com_interacao(grupo, y, dados)

        candidatas_restantes = [
            coluna for coluna in colunas_subconjunto
            if coluna not in grupo
            and coluna not in colunas_usadas
            and coluna not in colunas_independentes_sugeridas
        ]
        for candidata in candidatas_restantes:
            if len(grupo) >= max_tamanho_grupo:
                break
            novo_grupo = grupo + [candidata]
            r2_novo_grupo = _r2_grupo_com_interacao(novo_grupo, y, dados)
            if r2_novo_grupo - r2_grupo_atual >= limiar_sinergia:
                grupo = novo_grupo
                r2_grupo_atual = r2_novo_grupo

        sugestao_sinergicas.append(grupo)
        colunas_usadas.update(grupo)

    colunas_sem_classificacao = [
        coluna for coluna in colunas_subconjunto
        if coluna not in colunas_independentes_sugeridas and coluna not in colunas_usadas
    ]

    resultado = {
        'scores_individuais': scores_individuais,
        'scores_interacao': scores_interacao,
        'sugestao_independentes': sugestao_independentes,
        'sugestao_sinergicas': sugestao_sinergicas,
        'colunas_sem_classificacao': colunas_sem_classificacao,
    }

    if imprimir:
        print(f"SUGESTÃO DE COLUNAS INDEPENDENTES/SINÉRGICAS (alvo: '{coluna_alvo}', "
              f"{len(dados)} linha(s) usadas)")
        print()
        print("Scores individuais (R² de y ~ coluna):")
        for coluna, r2 in sorted(scores_individuais.items(), key=lambda item: item[1], reverse=True):
            print(f"  - {coluna}: R² = {r2:.4f}")

        print()
        print("Top ganhos de interação por par (R² c/ interação - R² aditivo):")
        top_pares = sorted(scores_interacao.items(), key=lambda item: item[1], reverse=True)[:10]
        for (coluna_i, coluna_j), ganho in top_pares:
            print(f"  - ({coluna_i}, {coluna_j}): ganho = {ganho:.4f}")

        print()
        print(f"Sugestão INDEPENDENTES (maior interação <= {limiar_independencia}):")
        print(f"  {sugestao_independentes if sugestao_independentes else '(nenhuma)'}")

        print()
        print(f"Sugestão GRUPOS SINÉRGICOS (ganho >= {limiar_sinergia}, "
              f"estendidos até {max_tamanho_grupo} coluna(s)):")
        if sugestao_sinergicas:
            for grupo in sugestao_sinergicas:
                print(f"  - {grupo}")
        else:
            print("  (nenhum)")

        if colunas_sem_classificacao:
            print()
            print(f"Colunas SEM classificação clara (revise manualmente): {colunas_sem_classificacao}")

        print()
        print("Lembrete: esta função apenas SUGERE. Copie manualmente o resultado para "
              "COLUNAS_INDEPENDENTES_POR_PREFETCHER / COLUNAS_SINERGICAS_POR_PREFETCHER "
              "se fizer sentido — nada é alterado automaticamente na configuração.")

    return resultado


def _detectar_grupos_por_uniao(pares_fortes: list, max_tamanho_grupo: int) -> list:
    """
    A partir de uma lista de tuplas (col_i, col_j, peso) com sinergia forte,
    agrupa colunas CONECTADAS por transitividade — se (A,B) e (B,C) são
    pares fortes, A/B/C entram no mesmo grupo — usando componentes conexos
    de um grafo simples (busca em profundidade).

    Componentes maiores que `max_tamanho_grupo` são truncados: mantêm-se as
    colunas com maior soma de peso de aresta DENTRO do componente, e as
    demais ficam de fora (voltam a ser candidatas a "sem classificação").

    Usado por `sugerir_colunas_sinergicas_shap`, já que SHAP interaction
    values só dá interações par a par diretamente — diferente da extensão
    gulosa usada em `sugerir_colunas_sinergicas` (que testa o ganho de R²
    de estender um grupo, algo que não temos disponível aqui).
    """
    grafo = {}
    pesos_aresta = {}
    for coluna_i, coluna_j, peso in pares_fortes:
        grafo.setdefault(coluna_i, set()).add(coluna_j)
        grafo.setdefault(coluna_j, set()).add(coluna_i)
        pesos_aresta[(coluna_i, coluna_j)] = peso
        pesos_aresta[(coluna_j, coluna_i)] = peso

    visitadas = set()
    grupos = []

    for coluna_inicial in grafo:
        if coluna_inicial in visitadas:
            continue

        pilha = [coluna_inicial]
        componente = set()
        while pilha:
            atual = pilha.pop()
            if atual in componente:
                continue
            componente.add(atual)
            pilha.extend(grafo.get(atual, set()) - componente)
        visitadas.update(componente)

        componente_lista = list(componente)
        if len(componente_lista) > max_tamanho_grupo:
            peso_por_coluna = {
                coluna: sum(
                    pesos_aresta.get((coluna, outra), 0.0)
                    for outra in componente_lista if outra != coluna
                )
                for coluna in componente_lista
            }
            componente_lista = sorted(
                componente_lista, key=lambda coluna: peso_por_coluna[coluna], reverse=True
            )[:max_tamanho_grupo]

        grupos.append(sorted(componente_lista))

    return grupos


def _calcular_matriz_interacao_shap(df: pd.DataFrame, colunas_candidatas: list, coluna_alvo: str) -> tuple:
    """
    Treina um modelo em árvore (XGBoost, com fallback para
    RandomForestRegressor) prevendo `coluna_alvo` a partir de
    `colunas_candidatas`, e calcula a matriz de interação SHAP normalizada
    (soma 1) entre elas — o mesmo cálculo usado por
    `sugerir_colunas_sinergicas_shap`, fatorado aqui para ser reaproveitado
    também por `selecionar_e_classificar_colunas_shap`.

    Retorna (matriz_interacao: pd.DataFrame k×k, nome_modelo: str, n_linhas: int).
    Levanta ImportError se `shap` (ou um modelo em árvore compatível) não
    estiver disponível.
    """
    try:
        import shap
    except ImportError as erro:
        raise ImportError(
            "A biblioteca 'shap' não está instalada. Instale com 'pip install shap'."
        ) from erro

    nome_modelo = None
    modelo = None
    try:
        import xgboost as xgb
        modelo = xgb.XGBRegressor(n_estimators=100, max_depth=4, random_state=0)
        nome_modelo = 'XGBRegressor'
    except ImportError:
        try:
            from sklearn.ensemble import RandomForestRegressor
            modelo = RandomForestRegressor(n_estimators=200, max_depth=6, random_state=0)
            nome_modelo = 'RandomForestRegressor'
        except ImportError as erro:
            raise ImportError(
                "Nenhum modelo em árvore disponível para o SHAP (nem 'xgboost' nem "
                "'scikit-learn' estão instalados)."
            ) from erro

    dados = df[colunas_candidatas + [coluna_alvo]].dropna()
    if len(dados) < len(colunas_candidatas) + 3:
        raise ValueError(
            f"Apenas {len(dados)} linha(s) sem NaN em {colunas_candidatas + [coluna_alvo]} — "
            f"poucas para treinar o modelo com confiança."
        )

    X = dados[colunas_candidatas]
    y = dados[coluna_alvo].to_numpy(dtype=float)

    modelo.fit(X, y)
    explicador = shap.TreeExplainer(modelo)
    interacoes = explicador.shap_interaction_values(X)

    interacao_media = np.abs(interacoes).mean(axis=0)  # matriz k x k
    soma_total = interacao_media.sum()
    interacao_normalizada = interacao_media / soma_total if soma_total > 0 else interacao_media

    matriz_interacao = pd.DataFrame(
        interacao_normalizada, index=colunas_candidatas, columns=colunas_candidatas
    )
    return matriz_interacao, nome_modelo, len(dados)


def _classificar_independentes_sinergicas_de_matriz(matriz_interacao: pd.DataFrame, colunas: list,
                                                      limiar_sinergia: float, limiar_independencia: float,
                                                      max_tamanho_grupo: int) -> tuple:
    """
    Dada uma matriz de interação já calculada (e normalizada), classifica
    `colunas` em independentes/sinérgicas/sem-classificação — a mesma lógica
    de classificação usada por `sugerir_colunas_sinergicas_shap`, fatorada
    aqui para ser reaproveitada também por `selecionar_e_classificar_colunas_shap`
    (que aplica essa mesma lógica só a um SUBCONJUNTO já selecionado das colunas).

    Retorna (scores_individuais, scores_interacao, sugestao_independentes,
    sugestao_sinergicas, colunas_sem_classificacao) — mesmo formato/nomes de
    campo do dicionário retornado por `sugerir_colunas_sinergicas_shap`.
    """
    scores_individuais = {coluna: float(matriz_interacao.loc[coluna, coluna]) for coluna in colunas}

    scores_interacao = {}
    for indice, coluna_i in enumerate(colunas):
        for coluna_j in colunas[indice + 1:]:
            scores_interacao[(coluna_i, coluna_j)] = float(matriz_interacao.loc[coluna_i, coluna_j])

    maior_interacao_por_coluna = {coluna: 0.0 for coluna in colunas}
    for (coluna_i, coluna_j), ganho in scores_interacao.items():
        maior_interacao_por_coluna[coluna_i] = max(maior_interacao_por_coluna[coluna_i], ganho)
        maior_interacao_por_coluna[coluna_j] = max(maior_interacao_por_coluna[coluna_j], ganho)

    sugestao_independentes = [
        coluna for coluna in colunas if maior_interacao_por_coluna[coluna] <= limiar_independencia
    ]
    colunas_independentes_sugeridas = set(sugestao_independentes)

    pares_fortes = [
        (coluna_i, coluna_j, ganho)
        for (coluna_i, coluna_j), ganho in scores_interacao.items()
        if ganho >= limiar_sinergia
        and coluna_i not in colunas_independentes_sugeridas
        and coluna_j not in colunas_independentes_sugeridas
    ]
    sugestao_sinergicas = _detectar_grupos_por_uniao(pares_fortes, max_tamanho_grupo)

    colunas_em_grupo = {coluna for grupo in sugestao_sinergicas for coluna in grupo}
    colunas_sem_classificacao = [
        coluna for coluna in colunas
        if coluna not in colunas_independentes_sugeridas and coluna not in colunas_em_grupo
    ]

    return (scores_individuais, scores_interacao, sugestao_independentes,
            sugestao_sinergicas, colunas_sem_classificacao)


def sugerir_colunas_sinergicas_shap(df: pd.DataFrame, colunas_subconjunto: list, coluna_alvo: str,
                                     limiar_sinergia: float = LIMIAR_SINERGIA_SHAP,
                                     limiar_independencia: float = LIMIAR_INDEPENDENCIA_SHAP,
                                     max_tamanho_grupo: int = 3,
                                     imprimir: bool = True) -> dict:
    """
    Mesma finalidade de `sugerir_colunas_sinergicas`, mas usando SHAP
    interaction values (via um modelo em árvore) em vez de regressão linear
    com termo de interação.

    MÉTODO:
    1. Treina um modelo em árvore (XGBoost, com fallback para
       RandomForestRegressor do scikit-learn) prevendo `coluna_alvo` a
       partir de `colunas_subconjunto`.
    2. Calcula `shap.TreeExplainer(modelo).shap_interaction_values(X)` — uma
       matriz k×k POR AMOSTRA: a DIAGONAL é o efeito principal de cada
       variável (já descontando qualquer interação); as células FORA da
       diagonal são o efeito de interação entre cada par.
    3. Tira a média (em módulo) dessa matriz por todas as amostras, e
       normaliza para somar 1 — assim os limiares ficam numa escala de
       "fração do total de |interação| explicada" (por isso os limiares do
       SHAP são configurados separadamente dos limiares da regressão: ver
       LIMIAR_SINERGIA_SHAP / LIMIAR_INDEPENDENCIA_SHAP no topo do script).
    4. Pares com interação normalizada >= `limiar_sinergia` viram arestas de
       um grafo; os componentes conexos (`_detectar_grupos_por_uniao`) viram
       os grupos sinérgicos sugeridos, truncados em `max_tamanho_grupo`
       colunas se necessário.
    5. Colunas cuja maior interação (com qualquer parceira) fica abaixo de
       `limiar_independencia` são sugeridas como independentes.

    IMPORTANTE: assim como a versão de regressão, esta função só SUGERE —
    nunca altera COLUNAS_INDEPENDENTES_POR_PREFETCHER nem
    COLUNAS_SINERGICAS_POR_PREFETCHER sozinha. Também NÃO seleciona/descarta
    colunas por relevância — ela pressupõe que `colunas_subconjunto` já é o
    conjunto que você quer classificar. Para selecionar as colunas mais
    representativas E classificá-las numa única passada, veja
    `selecionar_e_classificar_colunas_shap`.

    Levanta ImportError se `shap` (ou um modelo em árvore compatível — via
    `xgboost` ou `scikit-learn`) não estiver instalado, para que o chamador
    possa cair de volta em `sugerir_colunas_sinergicas` (ver
    `_gerar_sugestao_colunas_independentes_sinergicas`, que já faz isso
    automaticamente quando USAR_SHAP_PARA_SUGESTAO=True).

    Parâmetros
    ----------
    (mesmo significado de `sugerir_colunas_sinergicas`, exceto os limiares,
    que usam a escala normalizada do SHAP em vez de R²)

    Retorna
    -------
    dict com as mesmas chaves de `sugerir_colunas_sinergicas`, mais
    'matriz_interacao': DataFrame k×k com a matriz média normalizada (para
    inspeção/depuração).
    """
    matriz_interacao, nome_modelo, n_linhas = _calcular_matriz_interacao_shap(
        df, colunas_subconjunto, coluna_alvo
    )

    (scores_individuais, scores_interacao, sugestao_independentes,
     sugestao_sinergicas, colunas_sem_classificacao) = _classificar_independentes_sinergicas_de_matriz(
        matriz_interacao, colunas_subconjunto, limiar_sinergia, limiar_independencia, max_tamanho_grupo
    )

    resultado = {
        'scores_individuais': scores_individuais,
        'scores_interacao': scores_interacao,
        'sugestao_independentes': sugestao_independentes,
        'sugestao_sinergicas': sugestao_sinergicas,
        'colunas_sem_classificacao': colunas_sem_classificacao,
        'matriz_interacao': matriz_interacao,
    }

    if imprimir:
        print(f"SUGESTÃO DE COLUNAS INDEPENDENTES/SINÉRGICAS VIA SHAP (alvo: '{coluna_alvo}', "
              f"{n_linhas} linha(s) usadas, modelo: {nome_modelo})")
        print()
        print("Efeitos principais (diagonal da matriz de interação normalizada):")
        for coluna, valor in sorted(scores_individuais.items(), key=lambda item: item[1], reverse=True):
            print(f"  - {coluna}: {valor:.4f}")

        print()
        print("Top interações por par (fora da diagonal, normalizado):")
        top_pares = sorted(scores_interacao.items(), key=lambda item: item[1], reverse=True)[:10]
        for (coluna_i, coluna_j), ganho in top_pares:
            print(f"  - ({coluna_i}, {coluna_j}): {ganho:.4f}")

        print()
        print(f"Sugestão INDEPENDENTES (maior interação <= {limiar_independencia}):")
        print(f"  {sugestao_independentes if sugestao_independentes else '(nenhuma)'}")

        print()
        print(f"Sugestão GRUPOS SINÉRGICOS (interação >= {limiar_sinergia}, agrupados por "
              f"conectividade, truncados em {max_tamanho_grupo} coluna(s)):")
        if sugestao_sinergicas:
            for grupo in sugestao_sinergicas:
                print(f"  - {grupo}")
        else:
            print("  (nenhum)")

        if colunas_sem_classificacao:
            print()
            print(f"Colunas SEM classificação clara (revise manualmente): {colunas_sem_classificacao}")

        print()
        print("Lembrete: esta função apenas SUGERE. Copie manualmente o resultado para "
              "COLUNAS_INDEPENDENTES_POR_PREFETCHER / COLUNAS_SINERGICAS_POR_PREFETCHER "
              "se fizer sentido — nada é alterado automaticamente na configuração.")

    return resultado


def selecionar_e_classificar_colunas_shap(df: pd.DataFrame, colunas_candidatas: list, coluna_alvo: str,
                                           numero_colunas: int = None,
                                           limiar_relevancia_minima_shap: float = 0.0,
                                           limiar_sinergia: float = LIMIAR_SINERGIA_SHAP,
                                           limiar_independencia: float = LIMIAR_INDEPENDENCIA_SHAP,
                                           max_tamanho_grupo: int = 3,
                                           imprimir: bool = True) -> dict:
    """
    Usa SHAP para, numa ÚNICA passada, (1) SELECIONAR as colunas mais
    representativas de `colunas_candidatas` em relação a `coluna_alvo`, e
    (2) CLASSIFICAR as selecionadas em independentes/sinérgicas — tudo a
    partir da MESMA matriz de interação SHAP, sem precisar de um método de
    seleção separado (ex.: R²/correlação).

    MÉTODO:
    1. Treina um modelo em árvore e calcula a matriz de interação SHAP
       completa sobre TODAS as `colunas_candidatas` (mesmo procedimento de
       `sugerir_colunas_sinergicas_shap`, via `_calcular_matriz_interacao_shap`).
    2. Calcula a IMPORTÂNCIA TOTAL de cada coluna como a soma da LINHA
       INTEIRA da matriz (efeito principal + toda a interação dela com as
       demais) — o equivalente ao valor de SHAP "completo" da variável, não
       só o efeito principal isolado (que seria só a diagonal).
    3. Seleciona as colunas de maior importância total: até `numero_colunas`
       (se informado) e/ou acima de `limiar_relevancia_minima_shap`.
    4. Restringe a matriz de interação às colunas selecionadas e a
       RENORMALIZA para somar 1 — assim os limiares de sinergia/independência
       continuam na mesma escala relativa, não importa quantas colunas
       sobraram da seleção.
    5. Classifica as colunas selecionadas em independentes/sinérgicas DENTRO
       desse subconjunto (`_classificar_independentes_sinergicas_de_matriz`)
       — mesma lógica de `sugerir_colunas_sinergicas_shap`, só que restrita.

    IMPORTANTE: esta função só SUGERE — nunca altera
    COLUNAS_SUBCONJUNTO_POR_PREFETCHER, COLUNAS_INDEPENDENTES_POR_PREFETCHER
    nem COLUNAS_SINERGICAS_POR_PREFETCHER automaticamente.

    Levanta ImportError se `shap` (ou um modelo em árvore compatível) não
    estiver disponível, e ValueError se nenhuma coluna atingir o critério de
    seleção pedido.

    Parâmetros
    ----------
    df : pd.DataFrame
        Dados amostrados (ex.: o CSV de um prefetcher já carregado).
    colunas_candidatas : list
        Colunas candidatas à seleção (ex.: COLUNAS_TODAS).
    coluna_alvo : str
        Nome da coluna-alvo (ex.: a coluna de rank do prefetcher).
    numero_colunas : int ou None
        Quantas colunas manter, pela maior importância total. Default: None
        (não corta por quantidade — só pelo limiar de relevância abaixo).
    limiar_relevancia_minima_shap : float
        Importância total mínima (escala normalizada do SHAP, 0 a 1) para
        uma coluna ser selecionada. Default 0.0 (sem corte por relevância —
        útil se você só quer limitar por `numero_colunas`).
    limiar_sinergia, limiar_independencia, max_tamanho_grupo, imprimir :
        Mesmo significado de `sugerir_colunas_sinergicas_shap`, aplicados
        DENTRO do subconjunto já selecionado.

    Retorna
    -------
    dict com as chaves:
        'colunas_selecionadas': lista de dicts [{'coluna', 'importancia_total'}, ...],
        na ordem de importância (maior primeiro).
        'scores_individuais', 'scores_interacao', 'sugestao_independentes',
        'sugestao_sinergicas', 'colunas_sem_classificacao', 'matriz_interacao':
        mesmo formato de `sugerir_colunas_sinergicas_shap`, mas calculados
        SÓ sobre as colunas selecionadas.
    """
    matriz_interacao_completa, nome_modelo, n_linhas = _calcular_matriz_interacao_shap(
        df, colunas_candidatas, coluna_alvo
    )

    importancia_total = {
        coluna: float(matriz_interacao_completa.loc[coluna, :].sum())
        for coluna in colunas_candidatas
    }

    colunas_ordenadas = sorted(
        colunas_candidatas, key=lambda coluna: importancia_total[coluna], reverse=True
    )
    colunas_selecionadas_nomes = [
        coluna for coluna in colunas_ordenadas
        if importancia_total[coluna] >= limiar_relevancia_minima_shap
    ]
    if numero_colunas is not None:
        colunas_selecionadas_nomes = colunas_selecionadas_nomes[:numero_colunas]

    if not colunas_selecionadas_nomes:
        raise ValueError(
            f"Nenhuma coluna atingiu o critério de seleção pedido "
            f"(limiar_relevancia_minima_shap={limiar_relevancia_minima_shap}, "
            f"numero_colunas={numero_colunas})."
        )

    submatriz = matriz_interacao_completa.loc[colunas_selecionadas_nomes, colunas_selecionadas_nomes]
    soma_submatriz = float(submatriz.to_numpy().sum())
    matriz_interacao = submatriz / soma_submatriz if soma_submatriz > 0 else submatriz

    (scores_individuais, scores_interacao, sugestao_independentes,
     sugestao_sinergicas, colunas_sem_classificacao) = _classificar_independentes_sinergicas_de_matriz(
        matriz_interacao, colunas_selecionadas_nomes, limiar_sinergia, limiar_independencia, max_tamanho_grupo
    )

    resultado = {
        'colunas_selecionadas': [
            {'coluna': coluna, 'importancia_total': importancia_total[coluna]}
            for coluna in colunas_selecionadas_nomes
        ],
        'scores_individuais': scores_individuais,
        'scores_interacao': scores_interacao,
        'sugestao_independentes': sugestao_independentes,
        'sugestao_sinergicas': sugestao_sinergicas,
        'colunas_sem_classificacao': colunas_sem_classificacao,
        'matriz_interacao': matriz_interacao,
    }

    if imprimir:
        print(f"SELEÇÃO + CLASSIFICAÇÃO VIA SHAP (alvo: '{coluna_alvo}', {n_linhas} linha(s) usadas, "
              f"modelo: {nome_modelo})")
        print()
        print(f"Importância total (efeito principal + interações), das {len(colunas_candidatas)} candidatas:")
        for coluna in colunas_ordenadas:
            marca = "selecionada" if coluna in colunas_selecionadas_nomes else "descartada "
            print(f"  [{marca}] {coluna}: {importancia_total[coluna]:.4f}")

        print()
        print(f"Colunas selecionadas ({len(colunas_selecionadas_nomes)} de {len(colunas_candidatas)}): "
              f"{colunas_selecionadas_nomes}")

        print()
        print(f"Sugestão INDEPENDENTES (dentro das selecionadas, maior interação <= {limiar_independencia}):")
        print(f"  {sugestao_independentes if sugestao_independentes else '(nenhuma)'}")

        print()
        print(f"Sugestão GRUPOS SINÉRGICOS (dentro das selecionadas, interação >= {limiar_sinergia}, "
              f"truncados em {max_tamanho_grupo} coluna(s)):")
        if sugestao_sinergicas:
            for grupo in sugestao_sinergicas:
                print(f"  - {grupo}")
        else:
            print("  (nenhum)")

        if colunas_sem_classificacao:
            print()
            print(f"Colunas SEM classificação clara (revise manualmente): {colunas_sem_classificacao}")

        print()
        print("Lembrete: esta função apenas SUGERE. Copie manualmente o resultado para "
              "COLUNAS_SUBCONJUNTO_POR_PREFETCHER / COLUNAS_INDEPENDENTES_POR_PREFETCHER / "
              "COLUNAS_SINERGICAS_POR_PREFETCHER se fizer sentido — nada é alterado "
              "automaticamente na configuração.")

    return resultado


def _gerar_sugestao_colunas_independentes_sinergicas(df: pd.DataFrame, colunas_subconjunto: list,
                                                       coluna_alvo: str, imprimir: bool) -> dict:
    """
    Escolhe entre `sugerir_colunas_sinergicas_shap` (se USAR_SHAP_PARA_SUGESTAO
    for True) e `sugerir_colunas_sinergicas` (regressão linear), com
    fallback automático para a regressão caso o SHAP não esteja disponível
    (ImportError). Usado por `processar_arquivo_csv`.
    """
    if USAR_SHAP_PARA_SUGESTAO:
        try:
            return sugerir_colunas_sinergicas_shap(
                df, colunas_subconjunto, coluna_alvo=coluna_alvo,
                limiar_sinergia=LIMIAR_SINERGIA_SHAP,
                limiar_independencia=LIMIAR_INDEPENDENCIA_SHAP,
                max_tamanho_grupo=TAMANHO_MAXIMO_GRUPO_SUGESTAO,
                imprimir=imprimir,
            )
        except ImportError as erro:
            print(f"[AVISO] SHAP indisponível ({erro}) — usando a versão por regressão linear "
                  f"como fallback.")

    return sugerir_colunas_sinergicas(
        df, colunas_subconjunto, coluna_alvo=coluna_alvo,
        limiar_sinergia=LIMIAR_SINERGIA_SUGESTAO,
        limiar_independencia=LIMIAR_INDEPENDENCIA_SUGESTAO,
        max_tamanho_grupo=TAMANHO_MAXIMO_GRUPO_SUGESTAO,
        imprimir=imprimir,
    )


# ---------------------------------------------------------------------------
# 3e. Filtra as linhas de um CSV de entrada por trace / resolve o arquivo de execução
# ---------------------------------------------------------------------------

def filtrar_linhas_por_traces(df: pd.DataFrame, caminho_csv: str,
                               traces_selecionados: list = TRACES_SELECIONADOS,
                               usar_todos_os_traces: bool = USAR_TODOS_OS_TRACES,
                               nome_coluna_trace: str = NOME_COLUNA_TRACE) -> pd.DataFrame:
    """
    Filtra as linhas de `df` para manter somente aquelas cujo valor na
    coluna `nome_coluna_trace` (default: NOME_COLUNA_TRACE, ou seja,
    'trace') esteja em `traces_selecionados`. Chamada por
    `processar_arquivo_csv` na etapa de EXECUÇÃO (cálculo de adequabilidade
    linha a linha / CSV de saída), depois que o sistema fuzzy (quartis,
    antecedentes, consequente, regras e medianas dos grupos) já foi
    construído com TODAS as linhas do CSV — ou seja, este filtro NUNCA
    afeta como o sistema fuzzy é calibrado, só quais linhas são avaliadas
    por ele.

    Se `usar_todos_os_traces` for True, o filtro é ignorado e `df` é
    retornado sem alterações (comportamento original do script: todas as
    linhas do CSV, de todos os traces, são avaliadas).

    Se a coluna `nome_coluna_trace` não existir em `df` (CSV sem essa
    coluna), imprime um aviso e retorna `df` sem alterações, em vez de
    interromper o processamento do arquivo.

    Parâmetros
    ----------
    df : pd.DataFrame
        CSV de entrada carregado (o `df` completo, ANTES de virar
        `df_avaliar` em `processar_arquivo_csv` — o sistema fuzzy já foi
        montado com ele por inteiro nesse ponto).
    caminho_csv : str
        Caminho do arquivo, usado só para mensagens de aviso/log.
    traces_selecionados : list
        Valores de `nome_coluna_trace` a manter. Default: TRACES_SELECIONADOS.
    usar_todos_os_traces : bool
        Se True, ignora `traces_selecionados` e retorna `df` inteiro.
        Default: USAR_TODOS_OS_TRACES.
    nome_coluna_trace : str
        Nome da coluna de trace no CSV. Default: NOME_COLUNA_TRACE.

    Retorna
    -------
    pd.DataFrame
        `df` filtrado (nova cópia, índice reiniciado), ou `df` original se
        `usar_todos_os_traces` for True ou a coluna de trace não existir.
    """
    if usar_todos_os_traces:
        return df

    if nome_coluna_trace not in df.columns:
        print(f"[AVISO] Coluna '{nome_coluna_trace}' não encontrada em '{caminho_csv}'; "
              f"USAR_TODOS_OS_TRACES é False, mas não há como filtrar por trace neste "
              f"arquivo. Usando TODAS as linhas dele.")
        return df

    if not traces_selecionados:
        raise ValueError(
            "TRACES_SELECIONADOS está vazia e USAR_TODOS_OS_TRACES é False. "
            "Configure ao menos um trace em TRACES_SELECIONADOS, ou defina "
            "USAR_TODOS_OS_TRACES = True para usar todos os traces disponíveis."
        )

    total_antes = len(df)
    df_filtrado = df[df[nome_coluna_trace].isin(traces_selecionados)].reset_index(drop=True)

    traces_encontrados = set(df_filtrado[nome_coluna_trace].unique())
    traces_nao_encontrados = sorted(set(traces_selecionados) - traces_encontrados)
    if traces_nao_encontrados:
        print(f"[AVISO] Trace(s) de TRACES_SELECIONADOS não encontrado(s) em '{caminho_csv}': "
              f"{traces_nao_encontrados}")

    print(f"Filtro de traces aplicado em '{caminho_csv}': {len(df_filtrado)} de {total_antes} "
          f"linha(s) mantida(s) ({len(traces_encontrados)} de {len(traces_selecionados)} "
          f"trace(s) selecionado(s) encontrado(s) neste arquivo).")

    return df_filtrado


def resolver_caminho_execucao(caminho_entrada: str, prefetcher_atual: str,
                               usar_diretorio_execucao_separado: bool = USAR_DIRETORIO_EXECUCAO_SEPARADO,
                               diretorio_execucao: str = DIRETORIO_CSV_EXECUCAO) -> str:
    """
    Decide de qual ARQUIVO vêm os dados da ETAPA DE EXECUÇÃO (cálculo de
    adequabilidade linha a linha / CSV de saída) para o prefetcher
    `prefetcher_atual`, cujo arquivo de ENTRADA (usado para montar o
    sistema fuzzy) é `caminho_entrada`.

    Se `usar_diretorio_execucao_separado` for False (padrão), retorna o
    PRÓPRIO `caminho_entrada` — comportamento original, um único arquivo
    por prefetcher para tudo (sistema fuzzy + execução).

    Se for True, procura em `diretorio_execucao` o arquivo .csv cuja ÚLTIMA
    coluna (cabeçalho) seja igual a `prefetcher_atual` (mesmo critério de
    `_encontrar_arquivo_entrada_por_prefetcher`) e retorna esse caminho —
    o sistema fuzzy continua montado a partir de `caminho_entrada`, só a
    execução passa a usar o arquivo de `diretorio_execucao`. Levanta
    FileNotFoundError se não achar nenhum correspondente (deixa a exceção
    subir — em `processar_arquivo_csv`, isso interrompe só aquele arquivo).
    """
    if not usar_diretorio_execucao_separado:
        return caminho_entrada

    return _encontrar_arquivo_entrada_por_prefetcher(prefetcher_atual, diretorio_execucao)


# ---------------------------------------------------------------------------
# 4. Processa UM arquivo .csv (pipeline fuzzy completo)
# ---------------------------------------------------------------------------

def processar_arquivo_csv(caminho_csv: str) -> dict:
    """
    Roda o pipeline fuzzy completo para UM único arquivo .csv: carrega o
    arquivo, detecta o prefetcher pela ÚLTIMA coluna do CSV (cujo cabeçalho
    é o nome do prefetcher — ver DIRETORIO_CSV_ENTRADA), monta antecedentes/
    consequente/regras/defuzzificador usando TODAS as linhas do CSV, calcula
    a adequabilidade linha a linha — usando o mesmo arquivo, ou um arquivo
    correspondente de DIRETORIO_CSV_EXECUCAO se USAR_DIRETORIO_EXECUCAO_SEPARADO
    for True (ver `resolver_caminho_execucao`) — apenas para os traces
    selecionados em TRACES_SELECIONADOS (a menos que USAR_TODOS_OS_TRACES
    seja True — ver `filtrar_linhas_por_traces`), e grava o resultado em
    DIRETORIO_CSV_SAIDA.

    Retorna um dicionário com os principais objetos/resultados gerados,
    para uso programático (ex.: inspeção interativa, notebooks).
    """
    df = pd.read_csv(caminho_csv)

    # A última coluna do CSV tem, no cabeçalho, o nome do prefetcher daquele
    # arquivo (ex.: 'bingo' em bancada_bingo.csv) — detectado automaticamente
    # em vez de depender da constante PREFETCHER (que não é mais usada aqui).
    prefetcher_atual = df.columns[-1]

    # Escolhe o COLUNAS_SUBCONJUNTO_<PREFETCHER> correspondente ao prefetcher
    # detectado. Se o prefetcher não estiver em COLUNAS_SUBCONJUNTO_POR_PREFETCHER
    # (nome não reconhecido), a exceção interrompe só este arquivo — o loop em
    # `main()` segue para o próximo CSV do lote.
    if prefetcher_atual not in COLUNAS_SUBCONJUNTO_POR_PREFETCHER:
        raise ValueError(
            f"Prefetcher '{prefetcher_atual}' (detectado em '{caminho_csv}') não tem um "
            f"COLUNAS_SUBCONJUNTO_<PREFETCHER> configurado. "
            f"Prefetchers conhecidos: {sorted(COLUNAS_SUBCONJUNTO_POR_PREFETCHER.keys())}."
        )
    colunas_subconjunto = COLUNAS_SUBCONJUNTO_POR_PREFETCHER[prefetcher_atual]

    # Colunas independentes/sinérgicas configuradas MANUALMENTE para esse
    # prefetcher — usadas como padrão, e também como fallback caso
    # MODO_SELECAO_COLUNAS esteja num modo automático mas a sugestão
    # automática falhe por algum motivo. Se não houver entrada configurada,
    # assume lista vazia (comportamento original: tudo combinado por E/AND).
    colunas_independentes = COLUNAS_INDEPENDENTES_POR_PREFETCHER.get(prefetcher_atual, [])
    colunas_sinergicas = COLUNAS_SINERGICAS_POR_PREFETCHER.get(prefetcher_atual, [])

    print(f"Arquivo: {caminho_csv}")
    print(f"Prefetcher detectado (última coluna do CSV): '{prefetcher_atual}'")
    print(f"Linhas totais no CSV: {len(df)}")
    print()

    modos_validos = {'manual', 'automatico_subconjunto', 'automatico_todas_colunas', 'automatico_shap_unificado'}
    if MODO_SELECAO_COLUNAS not in modos_validos:
        raise ValueError(
            f"MODO_SELECAO_COLUNAS inválido: '{MODO_SELECAO_COLUNAS}'. Use um de: {sorted(modos_validos)}."
        )

    # --- Sugestão automática de colunas independentes/sinérgicas ---
    # Usa a PRÓPRIA coluna de rank do prefetcher (1 a 7, direto do CSV) como
    # alvo — "opção 1" discutida: rank bruto, disponível antes mesmo de
    # montar antecedentes/regras/adequabilidade.
    #
    # As colunas CANDIDATAS à sugestão são COLUNAS_TODAS quando
    # MODO_SELECAO_COLUNAS está em 'automatico_todas_colunas' ou
    # 'automatico_shap_unificado' (deixa o método escolher o próprio
    # subconjunto de variáveis fuzzy a partir de TODAS as colunas
    # disponíveis) — senão, são só o subconjunto manual daquele prefetcher.
    #
    # Roda (calculando, mesmo sem imprimir) se SUGERIR_... estiver ligada OU
    # MODO_SELECAO_COLUNAS não for 'manual': SUGERIR_... só imprime o
    # relatório; o modo automático de fato substitui a config manual pela
    # sugestão (ver bloco abaixo). O método (SHAP ou regressão) é escolhido
    # por USAR_SHAP_PARA_SUGESTAO (exceto em 'automatico_shap_unificado', que
    # sempre usa SHAP), com fallback automático para regressão se o SHAP não
    # estiver disponível (exceto em 'automatico_shap_unificado', que cai
    # direto para a config manual se o SHAP falhar).
    colunas_candidatas_sugestao = (
        COLUNAS_TODAS
        if MODO_SELECAO_COLUNAS in ('automatico_todas_colunas', 'automatico_shap_unificado')
        else colunas_subconjunto
    )
    aplicar_sugestao_automaticamente = MODO_SELECAO_COLUNAS != 'manual'

    resultado_sugestao = None
    if SUGERIR_COLUNAS_SINERGICAS_AUTOMATICAMENTE or aplicar_sugestao_automaticamente:
        if SUGERIR_COLUNAS_SINERGICAS_AUTOMATICAMENTE:
            if MODO_SELECAO_COLUNAS == 'automatico_shap_unificado':
                metodo_sugestao = 'SHAP (seleção + classificação unificada)'
            else:
                metodo_sugestao = 'SHAP' if USAR_SHAP_PARA_SUGESTAO else 'regressão linear'
            origem_candidatas = (
                'COLUNAS_TODAS'
                if MODO_SELECAO_COLUNAS in ('automatico_todas_colunas', 'automatico_shap_unificado')
                else 'COLUNAS_SUBCONJUNTO'
            )
            print(f"SUGESTÃO AUTOMÁTICA DE COLUNAS INDEPENDENTES/SINÉRGICAS "
                  f"(alvo: rank real de '{prefetcher_atual}', candidatas: {origem_candidatas}, "
                  f"método: {metodo_sugestao})")
        try:
            if MODO_SELECAO_COLUNAS == 'automatico_shap_unificado':
                resultado_sugestao = selecionar_e_classificar_colunas_shap(
                    df, colunas_candidatas_sugestao, coluna_alvo=prefetcher_atual,
                    numero_colunas=NUMERO_COLUNAS_SHAP_UNIFICADO,
                    limiar_relevancia_minima_shap=LIMIAR_RELEVANCIA_MINIMA_SHAP_UNIFICADO,
                    limiar_sinergia=LIMIAR_SINERGIA_SHAP,
                    limiar_independencia=LIMIAR_INDEPENDENCIA_SHAP,
                    max_tamanho_grupo=TAMANHO_MAXIMO_GRUPO_SUGESTAO,
                    imprimir=SUGERIR_COLUNAS_SINERGICAS_AUTOMATICAMENTE,
                )
            else:
                resultado_sugestao = _gerar_sugestao_colunas_independentes_sinergicas(
                    df, colunas_candidatas_sugestao, coluna_alvo=prefetcher_atual,
                    imprimir=SUGERIR_COLUNAS_SINERGICAS_AUTOMATICAMENTE,
                )
        except Exception as erro:
            print(f"[AVISO] Não foi possível gerar a sugestão para '{caminho_csv}': {erro}")
        if SUGERIR_COLUNAS_SINERGICAS_AUTOMATICAMENTE:
            print()

    # --- Usa a sugestão automática no lugar da config manual, se pedido ---
    origem_colunas = f"COLUNAS_INDEPENDENTES_{prefetcher_atual.upper()}/COLUNAS_SINERGICAS_{prefetcher_atual.upper()}"
    if aplicar_sugestao_automaticamente:
        if resultado_sugestao is not None:
            # IMPORTANTE: colunas "sem classificação clara" (nem independentes
            # nem sinérgicas na sugestão) são tratadas aqui como INDEPENDENTES
            # em vez de caírem juntas na regra conjunta "restante". Isso evita
            # uma explosão combinatória: a área cega da regra conjunta cresce
            # em 3^n (n = nº de colunas nela), enquanto tratar cada coluna
            # sem classificação como independente cresce apenas linearmente.
            # Sem isso, métodos mais conservadores ao classificar (como o
            # SHAP, que exige conectividade explícita no grafo de interação)
            # podem deixar várias colunas sem classificação, inflando o grupo
            # restante o suficiente para estourar a recursão do scikit-fuzzy
            # mesmo com a REGRA GERAL já dividida em blocos (ver
            # TAMANHO_MAXIMO_CONDICOES_POR_REGRA_GERAL).
            colunas_independentes = (
                resultado_sugestao['sugestao_independentes']
                + resultado_sugestao['colunas_sem_classificacao']
            )
            colunas_sinergicas = resultado_sugestao['sugestao_sinergicas']
            origem_colunas = "sugestão automática"

            if MODO_SELECAO_COLUNAS == 'automatico_todas_colunas':
                # O subconjunto de colunas usado nos antecedentes passa a ser
                # a UNIÃO de tudo que foi classificado (independentes + cada
                # grupo sinérgico) — na prática, o próprio COLUNAS_TODAS, já
                # que as "sem classificação" também foram absorvidas acima.
                colunas_classificadas = set(colunas_independentes)
                for grupo in colunas_sinergicas:
                    colunas_classificadas.update(grupo)
                colunas_subconjunto = [
                    coluna for coluna in colunas_candidatas_sugestao if coluna in colunas_classificadas
                ]
                origem_colunas = "sugestão automática via SHAP (a partir de COLUNAS_TODAS)"

            elif MODO_SELECAO_COLUNAS == 'automatico_shap_unificado':
                # Diferente de 'automatico_todas_colunas': aqui o subconjunto
                # é exatamente o que `selecionar_e_classificar_colunas_shap`
                # SELECIONOU pela importância total — as colunas não
                # selecionadas são DESCARTADAS de verdade (não viram
                # independentes, não entram em regra nenhuma), diferente das
                # "sem classificação clara" (que são as selecionadas que só
                # não formaram grupo nem bateram o corte de independência).
                colunas_subconjunto = [
                    item['coluna'] for item in resultado_sugestao['colunas_selecionadas']
                ]
                colunas_descartadas = [
                    coluna for coluna in colunas_candidatas_sugestao if coluna not in colunas_subconjunto
                ]
                origem_colunas = "seleção+classificação unificada via SHAP (a partir de COLUNAS_TODAS)"
                if colunas_descartadas:
                    print(f"[INFO] {len(colunas_descartadas)} coluna(s) DESCARTADA(S) por baixa "
                          f"importância total no SHAP unificado (nem entram no subconjunto): "
                          f"{colunas_descartadas}")

            if resultado_sugestao['colunas_sem_classificacao']:
                print(f"[INFO] {len(resultado_sugestao['colunas_sem_classificacao'])} coluna(s) sem "
                      f"classificação clara na sugestão foram tratadas como independentes (evita "
                      f"explosão combinatória na regra conjunta restante): "
                      f"{resultado_sugestao['colunas_sem_classificacao']}")
        else:
            print(f"[AVISO] Sugestão automática ligada mas falhou — usando a config manual "
                  f"(COLUNAS_SUBCONJUNTO_{prefetcher_atual.upper()} / {origem_colunas}) como "
                  f"fallback para '{prefetcher_atual}'.")

    print(f"Subconjunto de colunas usado ({origem_colunas}): {colunas_subconjunto}")
    print(f"Colunas independentes usadas ({origem_colunas}): {colunas_independentes}")
    print(f"Grupos sinérgicos usados ({origem_colunas}): {colunas_sinergicas}")

    # --- Aviso defensivo: grupo "restante" grande demais = risco de explosão combinatória ---
    colunas_em_grupo_sinergico = {coluna for grupo in colunas_sinergicas for coluna in grupo}
    colunas_restantes = [
        coluna for coluna in colunas_subconjunto
        if coluna not in colunas_independentes and coluna not in colunas_em_grupo_sinergico
    ]
    if len(colunas_restantes) > LIMITE_VARIAVEIS_REGRA_CONJUNTA_GERAL:
        print(f"[AVISO] O grupo de variáveis restantes (regra conjunta geral) tem "
              f"{len(colunas_restantes)} coluna(s) ({colunas_restantes}) — acima do limite "
              f"recomendado de {LIMITE_VARIAVEIS_REGRA_CONJUNTA_GERAL}. Isso gera até "
              f"3^{len(colunas_restantes)} = {3 ** len(colunas_restantes)} combinações possíveis "
              f"de termos, a maioria virando área cega. Considere mover algumas dessas colunas "
              f"para COLUNAS_INDEPENDENTES_{prefetcher_atual.upper()} ou agrupá-las em "
              f"COLUNAS_SINERGICAS_{prefetcher_atual.upper()}.")
    print()

    # --- Quartis sobre TODAS as linhas, usados para construir os antecedentes ---
    estatisticas = calcular_quartis(df, colunas_subconjunto)
    print("Estatísticas (quartis) do subconjunto, considerando TODAS as linhas:")
    print(estatisticas)
    print()

    antecedentes = criar_antecedentes(estatisticas)

    print("Antecedentes fuzzy criados:")
    for nome, antecedente in antecedentes.items():
        termos = list(antecedente.terms.keys())
        print(f"  - {nome}: termos = {termos}")

    consequente = criar_consequente_adequabilidade()
    print()
    print("Consequente fuzzy criado:")
    print(f"  - {consequente.label}: termos = {list(consequente.terms.keys())} "
          f"(universo de 0 a 10, passo = {PASSO_UNIVERSO})")

    # --- Quartis das linhas em que o prefetcher escolhido foi o vencedor ---
    linhas_vencedoras = selecionar_linhas_por_valores(df, prefetcher_atual, VALORES_VENCEDORES)
    quartis_vencedoras = calcular_quartis(linhas_vencedoras, colunas_subconjunto)
    medianas_vencedoras = quartis_vencedoras.loc['50%']  # mediana extraída dos quartis (usada nas etapas seguintes)

    print()
    print(f"Linhas com '{prefetcher_atual}' em {VALORES_VENCEDORES}: {len(linhas_vencedoras)} de {len(df)}")
    print()
    print("QUARTIS VENCEDORES")
    print(quartis_vencedoras)

    # --- Grau de pertinência das medianas vencedoras em cada antecedente ---
    graus_vencedoras = calcular_graus_pertinencia(antecedentes, medianas_vencedoras)

    print()
    print("GRAUS DE PERTINÊNCIA (medianas vencedoras)")
    for nome, graus in graus_vencedoras.items():
        graus_fmt = ", ".join(f"{termo}={grau:.4f}" for termo, grau in graus.items())
        print(f"  - {nome} (mediana={medianas_vencedoras[nome]:.4f}): {graus_fmt}")

    # --- Quartis das linhas em que o prefetcher escolhido foi intermediário ---
    linhas_intermediarias = selecionar_linhas_por_valores(df, prefetcher_atual, VALORES_INTERMEDIARIOS)
    quartis_intermediarias = calcular_quartis(linhas_intermediarias, colunas_subconjunto)
    medianas_intermediarias = quartis_intermediarias.loc['50%']

    print()
    print(f"Linhas com '{prefetcher_atual}' em {VALORES_INTERMEDIARIOS}: {len(linhas_intermediarias)} de {len(df)}")
    print()
    print("QUARTIS INTERMEDIÁRIOS")
    print(quartis_intermediarias)

    # --- Grau de pertinência das medianas intermediárias em cada antecedente ---
    graus_intermediarias = calcular_graus_pertinencia(antecedentes, medianas_intermediarias)

    print()
    print("GRAUS DE PERTINÊNCIA (medianas intermediárias)")
    for nome, graus in graus_intermediarias.items():
        graus_fmt = ", ".join(f"{termo}={grau:.4f}" for termo, grau in graus.items())
        print(f"  - {nome} (mediana={medianas_intermediarias[nome]:.4f}): {graus_fmt}")

    # --- Quartis das linhas em que o prefetcher escolhido foi perdedor ---
    linhas_perdedoras = selecionar_linhas_por_valores(df, prefetcher_atual, VALORES_PERDEDORES)
    quartis_perdedoras = calcular_quartis(linhas_perdedoras, colunas_subconjunto)
    medianas_perdedoras = quartis_perdedoras.loc['50%']

    print()
    print(f"Linhas com '{prefetcher_atual}' em {VALORES_PERDEDORES}: {len(linhas_perdedoras)} de {len(df)}")
    print()
    print("QUARTIS PERDEDORES")
    print(quartis_perdedoras)

    # --- Grau de pertinência das medianas perdedoras em cada antecedente ---
    graus_perdedoras = calcular_graus_pertinencia(antecedentes, medianas_perdedoras)

    print()
    print("GRAUS DE PERTINÊNCIA (medianas perdedoras)")
    for nome, graus in graus_perdedoras.items():
        graus_fmt = ", ".join(f"{termo}={grau:.4f}" for termo, grau in graus.items())
        print(f"  - {nome} (mediana={medianas_perdedoras[nome]:.4f}): {graus_fmt}")

    # --- Geração automática das regras fuzzy a partir dos graus de pertinência ---
    regras_geradas = gerar_regras_fuzzy(
        antecedentes, consequente, graus_vencedoras, graus_intermediarias, graus_perdedoras,
        colunas_independentes=colunas_independentes,
        colunas_sinergicas=colunas_sinergicas,
    )

    # --- Resolve conflitos entre regras independentes (mesmo termo, consequentes diferentes) ---
    graus_por_grupo = {
        'vencedoras': graus_vencedoras,
        'intermediarias': graus_intermediarias,
        'perdedoras': graus_perdedoras,
    }
    regras_geradas, conflitos_resolvidos = resolver_conflitos_regras_independentes(
        regras_geradas, graus_por_grupo,
        estrategia=ESTRATEGIA_RESOLUCAO_CONFLITOS,
        ordem_prioridade_grupos=ORDEM_PRIORIDADE_GRUPOS,
    )

    print()
    print(f"RESOLUÇÃO DE CONFLITOS (regras independentes) — estratégia: '{ESTRATEGIA_RESOLUCAO_CONFLITOS}'")
    imprimir_relatorio_conflitos_resolvidos(conflitos_resolvidos)

    print()
    print("REGRAS FUZZY GERADAS")
    for nome_grupo, _regra, descricao, _termos, _consequente_termo in regras_geradas:
        print(f"  [{nome_grupo}] {descricao}")

    # --- Verificação de áreas cegas (combinações de termos sem regra) ---
    resumo_areas_cegas = verificar_areas_cegas(
        antecedentes, regras_geradas,
        colunas_independentes=colunas_independentes,
        colunas_sinergicas=colunas_sinergicas,
    )

    print()
    print("VERIFICAÇÃO DE ÁREAS CEGAS")
    imprimir_relatorio_areas_cegas(resumo_areas_cegas)

    # --- Gera uma regra geral (catch-all) para cobrir a área cega, se houver ---
    regra_geral_area_cega = gerar_regra_geral_area_cega(
        antecedentes, consequente, resumo_areas_cegas,
        termo_consequente_padrao=TERMO_PADRAO_AREA_CEGA
    )

    print()
    if regra_geral_area_cega is None:
        print("REGRA GERAL (área cega): não é necessária — todas as combinações já estão cobertas.")
    else:
        _regras_gerais, descricao_geral, _itens_cobertos = regra_geral_area_cega
        print(descricao_geral)

    # --- Cria o defuzzificador a partir das regras (+ regra(s) geral(is) de área cega) ---
    sistema_ctrl, simulador = criar_defuzzificador(regras_geradas, regra_geral_area_cega)

    n_regras_gerais = len(regra_geral_area_cega[0]) if regra_geral_area_cega is not None else 0
    total_regras_sistema = len(regras_geradas) + n_regras_gerais
    print()
    print("DEFUZZIFICADOR CRIADO")
    print(f"  Total de regras no sistema (incluindo a(s) regra(s) geral(is), se houver): {total_regras_sistema}")

    # Teste de sanidade: aplica a mediana de CADA grupo como entrada e verifica
    # se a adequabilidade calculada faz sentido (vencedoras -> alta,
    # intermediárias -> média, perdedoras -> baixa).
    grupos_teste = [
        ('vencedoras', medianas_vencedoras),
        ('intermediarias', medianas_intermediarias),
        ('perdedoras', medianas_perdedoras),
    ]
    for nome_grupo, medianas in grupos_teste:
        valores_entrada = {nome: medianas[nome] for nome in antecedentes.keys()}
        adequabilidade = calcular_adequabilidade(simulador, valores_entrada)
        print(f"  - Mediana do grupo '{nome_grupo}' -> adequabilidade = {adequabilidade:.4f}")

    # --- Aplica o defuzzificador às linhas de EXECUÇÃO, usando as colunas ---
    # --- do subconjunto do prefetcher detectado (mesmas dos antecedentes) ---
    #
    # IMPORTANTE: o sistema fuzzy (quartis, antecedentes, consequente, regras
    # e medianas dos grupos, tudo acima) foi construído com TODAS as linhas
    # de `df` (o arquivo de DIRETORIO_CSV_ENTRADA), independente de
    # TRACES_SELECIONADOS/USAR_TODOS_OS_TRACES e de
    # USAR_DIRETORIO_EXECUCAO_SEPARADO. Essas três configurações só afetam
    # a EXECUÇÃO a partir daqui: de qual arquivo vêm as linhas e quais delas
    # (por trace) recebem o cálculo de adequabilidade.
    caminho_execucao = resolver_caminho_execucao(caminho_csv, prefetcher_atual)
    if caminho_execucao == caminho_csv:
        df_execucao_bruto = df
    else:
        print(f"Execução usará o arquivo '{caminho_execucao}' (DIRETORIO_CSV_EXECUCAO), "
              f"em vez de '{caminho_csv}', para o prefetcher '{prefetcher_atual}'.")
        df_execucao_bruto = pd.read_csv(caminho_execucao)

        # Valida que o arquivo de execução tem as colunas necessárias antes
        # de seguir — evita um KeyError confuso mais adiante em
        # calcular_adequabilidade_dataframe/colunas_para_exibir.
        colunas_necessarias = set(colunas_subconjunto) | {prefetcher_atual}
        colunas_faltando = sorted(colunas_necessarias - set(df_execucao_bruto.columns))
        if colunas_faltando:
            raise ValueError(
                f"Arquivo de execução '{caminho_execucao}' não tem a(s) coluna(s) "
                f"{colunas_faltando}, necessária(s) para avaliar o prefetcher "
                f"'{prefetcher_atual}' (colunas do subconjunto + coluna do prefetcher)."
            )

    df_traces_selecionados = filtrar_linhas_por_traces(df_execucao_bruto, caminho_execucao)

    if PROCESSAR_TODAS_LINHAS:
        df_avaliar = df_traces_selecionados
        descricao_linhas = f"todas as {len(df_avaliar)} linha(s)"
    else:
        df_avaliar = df_traces_selecionados.head(NUMERO_LINHAS_PROCESSAR)
        descricao_linhas = f"as primeiras {len(df_avaliar)} de {len(df_traces_selecionados)} linha(s)"

    print()
    print(f"ADEQUABILIDADE CALCULADA PARA {descricao_linhas.upper()} DE '{caminho_execucao}'")
    df_avaliar = df_avaliar.copy()
    df_avaliar['adequabilidade'] = calcular_adequabilidade_dataframe(
        df_avaliar, simulador, colunas_subconjunto
    )
    colunas_para_exibir = [prefetcher_atual] + colunas_subconjunto + ['adequabilidade']
    print(df_avaliar[colunas_para_exibir].to_string(index=False))

    # --- Grava o resultado (entrada + adequabilidade) em DIRETORIO_CSV_SAIDA ---
    os.makedirs(DIRETORIO_CSV_SAIDA, exist_ok=True)
    nome_base = os.path.splitext(os.path.basename(caminho_csv))[0]
    nome_arquivo_saida = PADRAO_NOME_ARQUIVO_SAIDA.format(nome=nome_base)
    caminho_saida = os.path.join(DIRETORIO_CSV_SAIDA, nome_arquivo_saida)
    df_avaliar[colunas_para_exibir].to_csv(caminho_saida, index=False)
    print()
    print(f"Resultado gravado em: {caminho_saida}")

    # Exemplo de visualização (MOSTRAR_GRAFICOS = True no topo do script para ativar)
    if MOSTRAR_GRAFICOS:
        for antecedente in antecedentes.values():
            antecedente.view()
        consequente.view()
        plt.show()

    return {
        'caminho_csv': caminho_csv,
        'caminho_execucao': caminho_execucao,
        'caminho_saida': caminho_saida,
        'prefetcher': prefetcher_atual,
        'df': df,
        'df_avaliar': df_avaliar,
        'antecedentes': antecedentes,
        'consequente': consequente,
        'quartis_vencedoras': quartis_vencedoras,
        'quartis_intermediarias': quartis_intermediarias,
        'quartis_perdedoras': quartis_perdedoras,
        'medianas_vencedoras': medianas_vencedoras,
        'medianas_intermediarias': medianas_intermediarias,
        'medianas_perdedoras': medianas_perdedoras,
        'graus_vencedoras': graus_vencedoras,
        'graus_intermediarias': graus_intermediarias,
        'graus_perdedoras': graus_perdedoras,
        'regras_geradas': regras_geradas,
        'conflitos_resolvidos': conflitos_resolvidos,
        'resumo_areas_cegas': resumo_areas_cegas,
        'regra_geral_area_cega': regra_geral_area_cega,
        'sistema_ctrl': sistema_ctrl,
        'simulador': simulador,
    }


# ---------------------------------------------------------------------------
# 4.5. Copia a coluna de IPC dos CSVs de entrada para os CSVs de resultado
# ---------------------------------------------------------------------------

def _encontrar_arquivo_entrada_por_prefetcher(prefetcher: str, diretorio_entrada: str = DIRETORIO_CSV_ENTRADA) -> str:
    """
    Varre os .csv de `diretorio_entrada` e retorna o caminho do primeiro
    arquivo cuja ÚLTIMA coluna (cabeçalho) seja igual a `prefetcher` — o
    mesmo critério de detecção usado em `processar_arquivo_csv`.

    Levanta FileNotFoundError se nenhum arquivo corresponder.
    """
    for caminho in sorted(glob.glob(os.path.join(diretorio_entrada, '*.csv'))):
        colunas = pd.read_csv(caminho, nrows=0).columns
        if len(colunas) > 0 and colunas[-1] == prefetcher:
            return caminho

    raise FileNotFoundError(
        f"Nenhum arquivo .csv em '{diretorio_entrada}' tem '{prefetcher}' como última coluna."
    )


def adicionar_coluna_ipc_aos_resultados(diretorio_saida: str = DIRETORIO_CSV_SAIDA,
                                         diretorio_entrada: str = DIRETORIO_CSV_ENTRADA,
                                         diretorio_execucao: str = DIRETORIO_CSV_EXECUCAO,
                                         usar_diretorio_execucao_separado: bool = USAR_DIRETORIO_EXECUCAO_SEPARADO,
                                         nome_coluna_ipc: str = NOME_COLUNA_IPC,
                                         caminho_saida_combinado: str = ARQUIVO_SAIDA_COMBINADO) -> None:
    """
    Depois que TODOS os arquivos de resultado já foram gravados em
    `diretorio_saida` (por `processar_arquivo_csv`), copia a coluna de IPC
    (`nome_coluna_ipc`) do CSV de EXECUÇÃO correspondente de CADA prefetcher
    e a insere no CSV de RESULTADO daquele mesmo prefetcher, logo ANTES da
    coluna 'adequabilidade' — sobrescrevendo o arquivo de resultado no disco.

    "CSV de execução" aqui é o MESMO arquivo que `processar_arquivo_csv`
    usou para calcular a adequabilidade daquele prefetcher: um arquivo de
    `diretorio_entrada` se `usar_diretorio_execucao_separado` for False
    (padrão), ou de `diretorio_execucao` se for True — mesmo critério de
    `resolver_caminho_execucao`.

    Para cada arquivo de resultado encontrado em `diretorio_saida`:
    1. Detecta o prefetcher pela PRIMEIRA coluna (cabeçalho) do resultado —
       mesma convenção usada por `processar_arquivo_csv`/`combinar_saidas_por_coluna`.
    2. Localiza o CSV de execução cuja ÚLTIMA coluna tenha esse mesmo nome
       de prefetcher (`_encontrar_arquivo_entrada_por_prefetcher`, no
       diretório certo conforme `usar_diretorio_execucao_separado`).
    3. Aplica a esse CSV o MESMO filtro de traces e o mesmo corte de linhas
       que `processar_arquivo_csv` aplicou na execução (`filtrar_linhas_por_traces`
       + `PROCESSAR_TODAS_LINHAS`/`NUMERO_LINHAS_PROCESSAR`) — reproduzindo
       exatamente o mesmo subconjunto/ordem de linhas que gerou o resultado,
       para então poder alinhar por POSIÇÃO com segurança (linha N do
       resultado <-> linha N desse subconjunto reproduzido).
    4. Insere a coluna `nome_coluna_ipc` no resultado, imediatamente ANTES
       de 'adequabilidade', e regrava o arquivo de resultado.

    Casos que geram um aviso (impresso) e são pulados, sem interromper os
    demais arquivos: resultado com menos de 2 colunas; resultado sem coluna
    'adequabilidade'; nenhum CSV de execução com aquele prefetcher; CSV de
    execução sem a coluna `nome_coluna_ipc`. `caminho_saida_combinado` (se
    estiver dentro de `diretorio_saida`) é sempre ignorado nessa varredura.
    """
    caminho_saida_combinado_abs = os.path.abspath(caminho_saida_combinado)
    caminhos_resultado = sorted(
        caminho for caminho in glob.glob(os.path.join(diretorio_saida, '*.csv'))
        if os.path.abspath(caminho) != caminho_saida_combinado_abs
    )

    if not caminhos_resultado:
        print(f"Nenhum arquivo .csv de resultado encontrado em '{diretorio_saida}'.")
        return

    diretorio_fonte = diretorio_execucao if usar_diretorio_execucao_separado else diretorio_entrada

    for caminho_resultado in caminhos_resultado:
        df_resultado = pd.read_csv(caminho_resultado)

        if df_resultado.shape[1] < 2:
            print(f"[AVISO] '{caminho_resultado}' tem menos de 2 colunas; pulando.")
            continue

        if 'adequabilidade' not in df_resultado.columns:
            print(f"[AVISO] '{caminho_resultado}' não tem coluna 'adequabilidade'; pulando.")
            continue

        # A primeira coluna do RESULTADO já tem, no cabeçalho, o nome do
        # prefetcher daquele arquivo (ver `processar_arquivo_csv`).
        prefetcher_atual = df_resultado.columns[0]

        try:
            caminho_fonte = _encontrar_arquivo_entrada_por_prefetcher(prefetcher_atual, diretorio_fonte)
        except FileNotFoundError as erro:
            print(f"[AVISO] {erro} — pulando '{caminho_resultado}'.")
            continue

        # Reproduz EXATAMENTE a mesma seleção/ordem de linhas que
        # `processar_arquivo_csv` usou na execução, para o alinhamento por
        # posição (abaixo) fazer sentido mesmo com TRACES_SELECIONADOS/
        # NUMERO_LINHAS_PROCESSAR restringindo as linhas.
        df_fonte = pd.read_csv(caminho_fonte)
        df_fonte_filtrado = filtrar_linhas_por_traces(df_fonte, caminho_fonte)
        if not PROCESSAR_TODAS_LINHAS:
            df_fonte_filtrado = df_fonte_filtrado.head(NUMERO_LINHAS_PROCESSAR)

        if nome_coluna_ipc not in df_fonte_filtrado.columns:
            print(f"[AVISO] Coluna '{nome_coluna_ipc}' não encontrada em '{caminho_fonte}' "
                  f"— '{caminho_resultado}' ficará sem a coluna de IPC.")
            continue

        coluna_ipc = df_fonte_filtrado[nome_coluna_ipc].reset_index(drop=True)

        # Alinha por posição ao número de linhas do resultado (ver docstring).
        if len(coluna_ipc) != len(df_resultado):
            print(f"[AVISO] '{caminho_fonte}' (após filtro de traces/linhas) tem "
                  f"{len(coluna_ipc)} linha(s) mas '{caminho_resultado}' tem "
                  f"{len(df_resultado)} — alinhando por posição (linhas fora do menor "
                  f"dos dois ficam com IPC = NaN). Isso pode indicar que a configuração "
                  f"atual de TRACES_SELECIONADOS/USAR_TODOS_OS_TRACES/"
                  f"USAR_DIRETORIO_EXECUCAO_SEPARADO mudou desde que o resultado foi gerado.")
            coluna_ipc = coluna_ipc.reindex(range(len(df_resultado)))

        # Remove uma coluna de IPC pré-existente (de uma rodada anterior),
        # se houver, antes de inserir a nova — evita duplicar a coluna.
        if nome_coluna_ipc in df_resultado.columns:
            df_resultado = df_resultado.drop(columns=[nome_coluna_ipc])

        indice_adequabilidade = df_resultado.columns.get_loc('adequabilidade')
        df_resultado.insert(indice_adequabilidade, nome_coluna_ipc, coluna_ipc)

        df_resultado.to_csv(caminho_resultado, index=False)
        print(f"  - {caminho_resultado}: coluna '{nome_coluna_ipc}' copiada de '{caminho_fonte}' "
              f"e inserida antes de 'adequabilidade'.")


# ---------------------------------------------------------------------------
# 6. Combina as saídas de todos os prefetchers em um único arquivo .csv
# ---------------------------------------------------------------------------

def adicionar_ranking_adequabilidade(df_combinado: pd.DataFrame, colunas_prefetcher: list = None) -> pd.DataFrame:
    """
    A partir do DataFrame combinado (ver `combinar_saidas_por_coluna`),
    adiciona, SEQUENCIALMENTE, DEPOIS da última coluna preenchida, uma nova
    coluna para cada prefetcher identificado nas primeiras colunas do
    arquivo (cujo cabeçalho — a primeira linha do CSV — já é o nome de cada
    prefetcher).

    Para CADA LINHA, compara os valores de TODAS as colunas
    'adequabilidade_<prefetcher>' daquela linha e preenche a nova coluna de
    cada prefetcher com um número de 1 a N (N = quantidade de prefetchers
    comparados, normalmente 7): 1 para o prefetcher com a MAIOR
    adequabilidade naquela linha, N para o de MENOR adequabilidade. Em caso
    de empate, os empatados recebem o mesmo rank (método 'min' — o próximo
    rank depois de um empate "pula" a quantidade de empatados).

    IMPORTANTE: as novas colunas reaproveitam o MESMO nome das colunas de
    prefetcher já existentes no início do arquivo (ex.: uma nova coluna
    'bingo' aparece de novo, no fim, ao lado de 'adequabilidade_bingo', além
    da 'bingo' original na primeira posição). Isso cria colunas com nome
    DUPLICADO de propósito: a 'bingo' do início é o rank MEDIDO (real, do
    CSV de origem) e a 'bingo' do fim é o rank PREVISTO pela adequabilidade
    fuzzy — lado a lado no mesmo arquivo para facilitar a comparação.

    Parâmetros
    ----------
    df_combinado : pd.DataFrame
        DataFrame no formato retornado por `combinar_saidas_por_coluna`
        (primeiras colunas = nomes dos prefetchers, seguidas pelas colunas
        'adequabilidade_<prefetcher>').
    colunas_prefetcher : list ou None
        Nomes das colunas de prefetcher a usar como base para as novas
        colunas de ranking. Default: None, e nesse caso são detectadas
        automaticamente como as colunas que vêm ANTES da primeira coluna
        'adequabilidade_*' (normalmente as 7 primeiras colunas do arquivo).

    Retorna
    -------
    pd.DataFrame
        Novo DataFrame = `df_combinado` + as colunas de ranking adicionadas
        ao final (mesmo número de linhas).
    """
    if colunas_prefetcher is None:
        colunas_adequabilidade_existentes = [
            coluna for coluna in df_combinado.columns if str(coluna).startswith('adequabilidade_')
        ]
        n_prefetchers = len(colunas_adequabilidade_existentes)
        colunas_prefetcher = list(df_combinado.columns[:n_prefetchers])

    colunas_adequabilidade = [f"adequabilidade_{nome}" for nome in colunas_prefetcher]

    colunas_faltantes = [c for c in colunas_adequabilidade if c not in df_combinado.columns]
    if colunas_faltantes:
        raise ValueError(
            f"Coluna(s) de adequabilidade não encontrada(s) no DataFrame combinado: "
            f"{colunas_faltantes}. Colunas disponíveis: {list(df_combinado.columns)}"
        )

    # Rank por linha entre as colunas de adequabilidade: 1 = maior valor,
    # N = menor valor. method='min' -> empates recebem o mesmo rank.
    ranks_por_linha = df_combinado[colunas_adequabilidade].rank(axis=1, ascending=False, method='min')

    # Monta as novas colunas em um DataFrame separado (mesmo índice de
    # `df_combinado`) e concatena por fora — assim os nomes de coluna
    # reaproveitados (iguais aos das primeiras colunas) ficam de fato
    # DUPLICADOS ao final, em vez de sobrescrever as colunas originais
    # (o que aconteceria com `df_combinado[nome] = ...` direto).
    df_ranking = pd.DataFrame(index=df_combinado.index)
    for nome_prefetcher, coluna_adequabilidade in zip(colunas_prefetcher, colunas_adequabilidade):
        df_ranking[nome_prefetcher] = ranks_por_linha[coluna_adequabilidade].astype('Int64')

    return pd.concat([df_combinado, df_ranking], axis=1)


def adicionar_percentual_diferenca_ipc(df_combinado: pd.DataFrame, colunas_prefetcher: list,
                                        ipc_por_prefetcher: dict,
                                        rank_medido_por_prefetcher: dict) -> pd.DataFrame:
    """
    Calcula, PARA CADA LINHA, o percentual de diferença entre o IPC do
    prefetcher VENCEDOR **medido de verdade** (índice 1 na classificação das
    PRIMEIRAS colunas do arquivo combinado — o rank real, vindo do CSV de
    entrada original) e o IPC de cada um dos DEMAIS prefetchers:

        %dif = (ipc_vencedor - ipc_outro) / ipc_vencedor * 100

    IMPORTANTE: o vencedor usado nesta fórmula vem da classificação MEDIDA
    (`rank_medido_por_prefetcher`, as primeiras colunas do arquivo — ver
    `combinar_saidas_por_coluna`), NÃO da classificação PREVISTA pelo
    sistema fuzzy (o ranking de `adicionar_ranking_adequabilidade`, as
    últimas colunas). O rank previsto só é usado aqui para decidir ONDE
    posicionar cada nova coluna '%dif_<prefetcher>' — imediatamente depois
    da coluna de rank PREVISTO daquele mesmo prefetcher, agora renomeada
    para '<prefetcher>_fuzzy' (em vez de reaproveitar o nome do prefetcher
    duplicado, o que exigia lidar com colunas de mesmo nome):

        ..., <prefetcher_1>_fuzzy, %dif_prefetcher_1, <prefetcher_2>_fuzzy, %dif_prefetcher_2, ...

    Um valor positivo indica que o outro prefetcher teve IPC menor que o
    vencedor medido naquela linha; um valor negativo indica que, apesar de
    não ter vencido de verdade, aquele prefetcher teve IPC MAIOR que o
    vencedor medido naquela linha.

    Na linha em que um prefetcher É o vencedor medido, sua própria coluna
    '%dif_<prefetcher>' fica com NaN naquela linha (não faz sentido comparar
    o vencedor com ele mesmo). Também fica NaN quando falta o IPC do
    vencedor ou do prefetcher daquela coluna, naquela linha (ex.: prefetcher
    sem coluna de IPC disponível — ver `adicionar_coluna_ipc_aos_resultados`).

    Em caso de EMPATE no rank medido = 1 (mais de um prefetcher com valor 1
    na classificação real naquela linha — incomum, mas possível dependendo
    de como VALORES_VENCEDORES foi configurado), TODOS os prefetchers
    empatados em 1º ficam com '%dif' = NaN naquela linha — só o IPC do
    primeiro empatado, na ordem de `colunas_prefetcher`, é usado como
    referência para calcular o '%dif' dos demais.

    Parâmetros
    ----------
    df_combinado : pd.DataFrame
        DataFrame já processado por `adicionar_ranking_adequabilidade`
        (usado aqui só para saber ONDE ficam as colunas de rank previsto).
    colunas_prefetcher : list
        Nomes dos prefetchers, na MESMA ordem usada para gerar as colunas
        de rank previsto (as últimas `len(colunas_prefetcher)` colunas de
        `df_combinado`).
    ipc_por_prefetcher : dict
        {nome_do_prefetcher: pd.Series com o IPC daquele prefetcher}.
        Prefetchers sem IPC disponível podem simplesmente estar ausentes
        deste dicionário.
    rank_medido_por_prefetcher : dict
        {nome_do_prefetcher: pd.Series com o rank MEDIDO/real daquele
        prefetcher} — as primeiras colunas do arquivo combinado (ver
        `combinar_saidas_por_coluna`), onde 1 = vencedor de verdade.

    Ambos os dicionários de série são reindexados internamente para o
    índice de `df_combinado` (posições fora do tamanho original viram NaN),
    evitando desalinhamento ou erro quando os CSVs de entrada têm números
    de linha diferentes entre si.

    Retorna
    -------
    pd.DataFrame
        `df_combinado` com as colunas de rank previsto REORGANIZADAS ao
        final e RENOMEADAS para '<prefetcher>_fuzzy': cada uma seguida
        imediatamente por sua coluna '%dif_<prefetcher>' correspondente
        (calculada com base no rank medido).
    """
    n = len(colunas_prefetcher)
    colunas_rank_previsto = df_combinado.iloc[:, -n:].copy()
    colunas_rank_previsto.columns = colunas_prefetcher  # só usado para POSICIONAR o %dif

    # Reindexa IPC e rank medido para o índice de `df_combinado` — evita
    # IndexError/desalinhamento quando um CSV de entrada tem menos linhas
    # que o arquivo combinado (posições fora do original viram NaN).
    ipc_por_prefetcher = {
        nome: serie.reindex(df_combinado.index)
        for nome, serie in ipc_por_prefetcher.items()
    }
    rank_medido_por_prefetcher = {
        nome: serie.reindex(df_combinado.index)
        for nome, serie in rank_medido_por_prefetcher.items()
    }

    # Matriz do rank MEDIDO (não o previsto!), na ordem de colunas_prefetcher
    # — é ela que decide quem é o vencedor de cada linha.
    valores_rank_medido = np.column_stack([
        rank_medido_por_prefetcher.get(nome, pd.Series(np.nan, index=df_combinado.index)).to_numpy()
        for nome in colunas_prefetcher
    ])

    percentuais = {nome: [] for nome in colunas_prefetcher}

    for indice_linha in range(len(df_combinado)):
        linha_ranks_medidos = valores_rank_medido[indice_linha]

        # TODOS os prefetchers empatados em rank MEDIDO 1 nesta linha (ver docstring).
        posicoes_vencedoras = {
            posicao for posicao, rank in enumerate(linha_ranks_medidos) if pd.notna(rank) and rank == 1
        }
        indice_referencia = min(posicoes_vencedoras) if posicoes_vencedoras else None
        nome_referencia = colunas_prefetcher[indice_referencia] if indice_referencia is not None else None

        ipc_vencedor = float('nan')
        if nome_referencia is not None and nome_referencia in ipc_por_prefetcher:
            ipc_vencedor = ipc_por_prefetcher[nome_referencia].iloc[indice_linha]

        for posicao, nome_prefetcher in enumerate(colunas_prefetcher):
            if posicao in posicoes_vencedoras or pd.isna(ipc_vencedor) or ipc_vencedor == 0:
                percentuais[nome_prefetcher].append(float('nan'))
                continue

            if nome_prefetcher not in ipc_por_prefetcher:
                percentuais[nome_prefetcher].append(float('nan'))
                continue

            ipc_outro = ipc_por_prefetcher[nome_prefetcher].iloc[indice_linha]
            if pd.isna(ipc_outro):
                percentuais[nome_prefetcher].append(float('nan'))
            else:
                percentuais[nome_prefetcher].append((ipc_vencedor - ipc_outro) / ipc_vencedor * 100)

    # Monta o bloco final intercalando rank PREVISTO (renomeado para
    # '<prefetcher>_fuzzy') e %dif (calculado com o rank MEDIDO)
    bloco_final = []
    for nome_prefetcher in colunas_prefetcher:
        bloco_final.append(colunas_rank_previsto[nome_prefetcher].rename(f"{nome_prefetcher}_fuzzy"))
        bloco_final.append(pd.Series(
            percentuais[nome_prefetcher], index=df_combinado.index, name=f"%dif_{nome_prefetcher}"
        ))

    df_sem_rank = df_combinado.iloc[:, :-n]
    return pd.concat([df_sem_rank] + bloco_final, axis=1)


def combinar_saidas_por_coluna(diretorio_saida: str = DIRETORIO_CSV_SAIDA,
                                caminho_saida_combinado: str = ARQUIVO_SAIDA_COMBINADO,
                                nome_coluna_ipc: str = NOME_COLUNA_IPC) -> pd.DataFrame:
    """
    Lê TODOS os arquivos .csv de `diretorio_saida` (os gerados por
    `processar_arquivo_csv` + `adicionar_coluna_ipc_aos_resultados`, cada um
    com as colunas [prefetcher, <colunas do subconjunto...>, 'ipc' (se
    disponível), 'adequabilidade']) e monta um único DataFrame/arquivo .csv
    combinado, na ordem:

    1. Primeiro, um bloco ALTERNADO por prefetcher (na ordem em que os
       arquivos aparecem em `diretorio_saida`): a PRIMEIRA coluna de cada
       arquivo — cujo cabeçalho já é o nome do prefetcher daquele arquivo
       (ver `processar_arquivo_csv`) — seguida IMEDIATAMENTE pela coluna
       `nome_coluna_ipc` DAQUELE MESMO arquivo, se ela existir:

           prefetcher_1, ipc, prefetcher_2, ipc, prefetcher_3, ipc, ...

       Arquivos sem a coluna de IPC (ex.: não tinham no CSV de entrada
       original) entram só com a coluna do prefetcher, sem quebrar a
       sequência dos demais.
    2. Depois, SEQUENCIALMENTE (mesma ordem), a ÚLTIMA coluna de cada
       arquivo ('adequabilidade'), renomeada para 'adequabilidade_<prefetcher>',
       onde <prefetcher> é o nome identificado no cabeçalho da PRIMEIRA
       coluna DAQUELE MESMO arquivo.
    3. Por fim, uma coluna de RANKING previsto por prefetcher, renomeada
       para '<prefetcher>_fuzzy' (para não duplicar o nome da coluna medida
       do passo 1), seguida imediatamente por '%dif_<prefetcher>' — ver
       `adicionar_ranking_adequabilidade` e `adicionar_percentual_diferenca_ipc`.

    Arquivos com números de linha diferentes são alinhados por POSIÇÃO (a
    linha N de um fica ao lado da linha N dos demais); posições que faltarem
    num arquivo mais curto ficam com NaN nas colunas dele.

    Grava o resultado em `caminho_saida_combinado` (criando o diretório se
    necessário) e também o retorna como DataFrame.
    """
    caminho_saida_combinado_abs = os.path.abspath(caminho_saida_combinado)
    caminhos_saida = sorted(
        caminho for caminho in glob.glob(os.path.join(diretorio_saida, '*.csv'))
        if os.path.abspath(caminho) != caminho_saida_combinado_abs
    )

    if not caminhos_saida:
        raise ValueError(f"Nenhum arquivo .csv encontrado em '{diretorio_saida}' para combinar.")

    print(f"{len(caminhos_saida)} arquivo(s) .csv encontrado(s) em '{diretorio_saida}' para combinar:")

    primeiras_colunas = []
    colunas_ipc = []
    ultimas_colunas = []

    for caminho in caminhos_saida:
        df = pd.read_csv(caminho)

        if df.shape[1] < 2:
            print(f"  [AVISO] '{caminho}' tem menos de 2 colunas; pulando.")
            continue

        # A primeira coluna já tem, no cabeçalho, o nome do prefetcher
        # daquele arquivo (ver `processar_arquivo_csv`).
        prefetcher_atual = df.columns[0]

        primeira_coluna = df.iloc[:, 0].reset_index(drop=True)
        primeira_coluna.name = prefetcher_atual
        primeiras_colunas.append(primeira_coluna)

        if nome_coluna_ipc in df.columns:
            coluna_ipc = df[nome_coluna_ipc].reset_index(drop=True)
            coluna_ipc.name = nome_coluna_ipc
        else:
            coluna_ipc = None
        colunas_ipc.append(coluna_ipc)

        ultima_coluna = df.iloc[:, -1].reset_index(drop=True)
        ultima_coluna.name = f"adequabilidade_{prefetcher_atual}"
        ultimas_colunas.append(ultima_coluna)

        descricao_ipc = "" if coluna_ipc is not None else " (sem coluna de ipc)"
        print(f"  - {caminho}: prefetcher '{prefetcher_atual}', {len(df)} linha(s){descricao_ipc}")

    if not primeiras_colunas:
        raise ValueError(f"Nenhum arquivo válido (com 2+ colunas) encontrado em '{diretorio_saida}'.")

    # Monta o bloco inicial alternando prefetcher / ipc (ver docstring)
    bloco_inicial = []
    for primeira_coluna, coluna_ipc in zip(primeiras_colunas, colunas_ipc):
        bloco_inicial.append(primeira_coluna)
        if coluna_ipc is not None:
            bloco_inicial.append(coluna_ipc)

    df_combinado = pd.concat(bloco_inicial + ultimas_colunas, axis=1)

    # --- Adiciona, ao final, o ranking previsto (1 a N) de cada prefetcher,
    # comparando as colunas 'adequabilidade_<prefetcher>' linha a linha ---
    colunas_prefetcher = [coluna.name for coluna in primeiras_colunas]
    df_combinado = adicionar_ranking_adequabilidade(df_combinado, colunas_prefetcher)

    # --- Ao lado de cada rank previsto, adiciona o %dif do IPC de cada ---
    # --- prefetcher em relação ao IPC do vencedor MEDIDO de verdade ---
    # --- (classificação real, das primeiras colunas — não a prevista) ---
    ipc_por_prefetcher = {
        nome: serie for nome, serie in zip(colunas_prefetcher, colunas_ipc) if serie is not None
    }
    rank_medido_por_prefetcher = {
        coluna.name: coluna for coluna in primeiras_colunas
    }
    df_combinado = adicionar_percentual_diferenca_ipc(
        df_combinado, colunas_prefetcher, ipc_por_prefetcher, rank_medido_por_prefetcher
    )

    diretorio_destino = os.path.dirname(caminho_saida_combinado)
    if diretorio_destino:
        os.makedirs(diretorio_destino, exist_ok=True)
    df_combinado.to_csv(caminho_saida_combinado, index=False)

    print()
    print(f"Arquivo combinado (com ipc alternado, ranking previsto e %dif de ipc) gravado em: "
          f"{caminho_saida_combinado} ({df_combinado.shape[0]} linha(s), {df_combinado.shape[1]} coluna(s))")

    return df_combinado


def calcular_percentual_acerto(caminho_saida_combinado: str = ARQUIVO_SAIDA_COMBINADO,
                                colunas_prefetcher: list = None,
                                limiar_diferenca_percentual: float = None,
                                imprimir: bool = True) -> dict:
    """
    Lê o arquivo COMBINADO (`caminho_saida_combinado`, gerado por
    `combinar_saidas_por_coluna`) e calcula cinco coisas:

    1. POR PREFETCHER: o percentual de acerto do sistema fuzzy em prevê-lo
       ESPECIFICAMENTE como vencedor (índice 1):

           acertos = número de linhas em que a coluna de rank PREVISTO
                     daquele prefetcher ('<prefetcher>_fuzzy') é 1 E a
                     coluna de rank MEDIDO/real do MESMO prefetcher (a
                     coluna original, no início do arquivo) também é 1
           percentual_acerto = acertos / total_de_linhas * 100

    2. GERAL EXATO (chave 'geral' no dicionário retornado): o percentual de
       acerto do sistema fuzzy em identificar O 1º COLOCADO de cada linha,
       INDEPENDENTE DE QUAL prefetcher tenha sido — ou seja, conta quantas
       linhas o prefetcher previsto como vencedor (rank previsto = 1) é
       EXATAMENTE o mesmo prefetcher que venceu de verdade (rank medido = 1),
       seja ele qual for, e divide pelo total de linhas. Essa métrica é
       diferente da soma/média das métricas por prefetcher: aqui uma linha
       só conta como acerto se a IDENTIDADE do vencedor bater, não apenas se
       cada prefetcher individualmente "acertou sua própria previsão".

    3. GERAL TOLERANTE TOP-2 (chave 'geral_top2'): uma versão mais permissiva
       do item 2 — conta como acerto também quando o prefetcher previsto
       como vencedor (rank previsto = 1) teve rank MEDIDO 1 OU 2 (ou seja,
       foi realmente o 1º OU o 2º colocado de verdade, não só o 1º). A ideia
       é tratar como "correto por proximidade" uma previsão que errou o
       vencedor exato mas ainda apontou para um dos dois melhores
       prefetchers reais daquela linha, em vez de tratar como erro total.

           percentual_acerto (geral_top2) =
               (linhas em que rank_medido[vencedor_previsto] ∈ {1, 2}) / total_de_linhas * 100

    4. GERAL TOLERANTE TOP-3 (chave 'geral_top3'): mesma ideia do item 3,
       mas ainda mais permissiva — conta como acerto quando o rank MEDIDO
       do prefetcher previsto como vencedor for 1, 2 OU 3:

           percentual_acerto (geral_top3) =
               (linhas em que rank_medido[vencedor_previsto] ∈ {1, 2, 3}) / total_de_linhas * 100

    5. GERAL POR DIFERENÇA PERCENTUAL (chave 'geral_diferenca_percentual'):
       em vez de tolerância por POSIÇÃO no ranking (como os itens 3 e 4),
       tolera por DESEMPENHO — usa as colunas '%dif_<prefetcher>' (a
       diferença percentual de IPC de cada prefetcher em relação ao
       vencedor medido daquela linha — ver `adicionar_percentual_diferenca_ipc`).
       Conta como acerto quando:
       - o prefetcher previsto como vencedor É o vencedor medido de
         verdade (mesmo critério do item 2), OU
       - '%dif_<prefetcher_previsto>' daquela linha é MENOR OU IGUAL a
         `limiar_diferenca_percentual` — ou seja, mesmo não sendo o
         vencedor exato, o IPC do prefetcher previsto ficou "perto o
         suficiente" (dentro do limiar) do IPC do vencedor real.

           percentual_acerto (geral_diferenca_percentual) =
               (linhas em que previsto=medido OU %dif_previsto <= limiar) / total_de_linhas * 100

       Diferente dos itens 3/4 (que toleram por QUANTOS lugares o previsto
       ficou atrás no ranking), este item tolera por QUÃO PERTO o
       desempenho (IPC) ficou — dois prefetchers podem estar em posições de
       rank bem distantes mas com IPCs quase idênticos (ou vice-versa), e
       essa métrica capta isso de um jeito que top2/top3 não captam.

    O arquivo combinado tem, para cada prefetcher, duas colunas com nomes
    distintos (ver `combinar_saidas_por_coluna` /
    `adicionar_percentual_diferenca_ipc`): '<prefetcher>' é o rank MEDIDO/
    real (início do arquivo); '<prefetcher>_fuzzy' é o rank PREVISTO pelo
    sistema fuzzy (ao lado de '%dif_<prefetcher>', mais à direita).

    Em caso de EMPATE (mais de um prefetcher com valor 1 na mesma linha, no
    rank medido e/ou no previsto), o primeiro prefetcher, na ordem de
    `colunas_prefetcher`, é considerado o vencedor daquele lado para efeito
    dos cálculos GERAL, GERAL TOP-2, GERAL TOP-3 e GERAL POR DIFERENÇA
    PERCENTUAL — exceto que, para este último, qualquer prefetcher empatado
    em 1º no rank MEDIDO (não só o primeiro da ordem) conta como acerto
    exato, já que todos eles são igualmente "o vencedor de verdade" naquela
    linha.

    Parâmetros
    ----------
    caminho_saida_combinado : str
        Caminho do arquivo combinado a ler. Default: ARQUIVO_SAIDA_COMBINADO.
    colunas_prefetcher : list ou None
        Nomes dos prefetchers a avaliar. Default: None — nesse caso, são
        detectados automaticamente a partir de todas as colunas do
        cabeçalho que terminam em '_fuzzy' (removendo o sufixo).
    limiar_diferenca_percentual : float ou None
        Limiar (em pontos percentuais) usado pelo item 5. Default: None ->
        usa LIMIAR_DIFERENCA_PERCENTUAL_ACERTO.
    imprimir : bool
        Se True, imprime um relatório legível além de retornar o dicionário.

    Retorna
    -------
    dict
        {nome_do_prefetcher: {'acertos': int, 'total_linhas': int,
        'percentual_acerto': float}, ..., 'geral': {mesmo formato,
        calculado como descrito no item 2}, 'geral_top2': {mesmo formato,
        calculado como descrito no item 3}, 'geral_top3': {mesmo formato,
        calculado como descrito no item 4}, 'geral_diferenca_percentual':
        {mesmo formato, calculado como descrito no item 5}}. Prefetchers
        cujas duas colunas esperadas não forem encontradas ficam de fora do
        dicionário (com um aviso impresso), sem interromper o cálculo dos
        demais nem o cálculo geral. Se as colunas '%dif_<prefetcher>'
        necessárias para o item 5 não existirem no arquivo, essa chave
        simplesmente não aparece no resultado (com um aviso impresso).
    """
    limiar_diferenca_percentual = (
        limiar_diferenca_percentual if limiar_diferenca_percentual is not None
        else LIMIAR_DIFERENCA_PERCENTUAL_ACERTO
    )

    df = pd.read_csv(caminho_saida_combinado)

    if colunas_prefetcher is None:
        colunas_prefetcher = [
            str(coluna)[:-len('_fuzzy')] for coluna in df.columns if str(coluna).endswith('_fuzzy')
        ]

    if not colunas_prefetcher:
        raise ValueError(
            f"Nenhuma coluna '<prefetcher>_fuzzy' encontrada em '{caminho_saida_combinado}'."
        )

    total_linhas = len(df)

    resultado = {}
    colunas_validas = []
    for prefetcher in colunas_prefetcher:
        coluna_medida = prefetcher
        coluna_prevista = f"{prefetcher}_fuzzy"

        if coluna_medida not in df.columns or coluna_prevista not in df.columns:
            print(f"[AVISO] Não encontrei as duas colunas esperadas para '{prefetcher}' "
                  f"('{coluna_medida}' e '{coluna_prevista}') em '{caminho_saida_combinado}'; pulando.")
            continue

        colunas_validas.append(prefetcher)

        acertos = int(((df[coluna_prevista] == 1) & (df[coluna_medida] == 1)).sum())
        percentual_acerto = (acertos / total_linhas * 100) if total_linhas else 0.0

        resultado[prefetcher] = {
            'acertos': acertos,
            'total_linhas': total_linhas,
            'percentual_acerto': percentual_acerto,
        }

    # --- Percentual de acerto GERAL do 1º colocado, independente de qual prefetcher foi ---
    # --- (e também as versões TOLERANTES: consideram correto quando o rank ---
    # --- medido do prefetcher previsto como vencedor é 1 e 2, ou 1, 2 e 3, ---
    # --- ou quando o %dif do previsto em relação ao vencedor medido é pequeno) ---
    if colunas_validas:
        valores_medidos = df[colunas_validas].to_numpy()
        valores_previstos = df[[f"{p}_fuzzy" for p in colunas_validas]].to_numpy()

        colunas_dif_existem = all(f"%dif_{p}" in df.columns for p in colunas_validas)
        if colunas_dif_existem:
            valores_dif = df[[f"%dif_{p}" for p in colunas_validas]].to_numpy()
        else:
            valores_dif = None
            print(f"[AVISO] Nem todas as colunas '%dif_<prefetcher>' esperadas foram encontradas "
                  f"em '{caminho_saida_combinado}' — a métrica 'geral_diferenca_percentual' não "
                  f"será calculada.")

        acertos_geral = 0
        acertos_geral_top2 = 0
        acertos_geral_top3 = 0
        acertos_geral_diferenca_percentual = 0
        for indice_linha in range(total_linhas):
            linha_medida = valores_medidos[indice_linha]
            linha_prevista = valores_previstos[indice_linha]

            posicoes_medido_1 = [
                posicao for posicao, valor in enumerate(linha_medida) if pd.notna(valor) and valor == 1
            ]
            posicoes_previsto_1 = [
                posicao for posicao, valor in enumerate(linha_prevista) if pd.notna(valor) and valor == 1
            ]

            vencedor_medido = colunas_validas[posicoes_medido_1[0]] if posicoes_medido_1 else None
            indice_previsto = posicoes_previsto_1[0] if posicoes_previsto_1 else None
            vencedor_previsto = colunas_validas[indice_previsto] if indice_previsto is not None else None

            # Métrica EXATA: o vencedor previsto precisa ser exatamente o vencedor medido (rank 1).
            if vencedor_medido is not None and vencedor_medido == vencedor_previsto:
                acertos_geral += 1

            # Métricas TOLERANTES POR POSIÇÃO: contam como acerto também
            # quando o prefetcher previsto como vencedor teve rank MEDIDO
            # dentro de um conjunto de colocações próximas (1º/2º, ou
            # 1º/2º/3º) — aceitas como corretas por proximidade no ranking.
            if indice_previsto is not None:
                rank_medido_do_previsto = linha_medida[indice_previsto]
                if pd.notna(rank_medido_do_previsto):
                    if rank_medido_do_previsto in (1, 2):
                        acertos_geral_top2 += 1
                    if rank_medido_do_previsto in (1, 2, 3):
                        acertos_geral_top3 += 1

            # Métrica TOLERANTE POR DIFERENÇA PERCENTUAL: acerto exato (o
            # previsto é QUALQUER um dos empatados em 1º no rank medido) OU
            # o %dif do previsto em relação ao vencedor medido está dentro
            # do limiar — aceito como correto por proximidade de DESEMPENHO
            # (IPC), não de posição no ranking.
            if indice_previsto is not None:
                if indice_previsto in posicoes_medido_1:
                    acertos_geral_diferenca_percentual += 1
                elif valores_dif is not None:
                    diferenca_percentual_do_previsto = valores_dif[indice_linha][indice_previsto]
                    if (pd.notna(diferenca_percentual_do_previsto)
                            and diferenca_percentual_do_previsto <= limiar_diferenca_percentual):
                        acertos_geral_diferenca_percentual += 1

        percentual_acerto_geral = (acertos_geral / total_linhas * 100) if total_linhas else 0.0
        resultado['geral'] = {
            'acertos': acertos_geral,
            'total_linhas': total_linhas,
            'percentual_acerto': percentual_acerto_geral,
        }

        percentual_acerto_geral_top2 = (acertos_geral_top2 / total_linhas * 100) if total_linhas else 0.0
        resultado['geral_top2'] = {
            'acertos': acertos_geral_top2,
            'total_linhas': total_linhas,
            'percentual_acerto': percentual_acerto_geral_top2,
        }

        percentual_acerto_geral_top3 = (acertos_geral_top3 / total_linhas * 100) if total_linhas else 0.0
        resultado['geral_top3'] = {
            'acertos': acertos_geral_top3,
            'total_linhas': total_linhas,
            'percentual_acerto': percentual_acerto_geral_top3,
        }

        if valores_dif is not None:
            percentual_acerto_geral_dif = (
                (acertos_geral_diferenca_percentual / total_linhas * 100) if total_linhas else 0.0
            )
            resultado['geral_diferenca_percentual'] = {
                'acertos': acertos_geral_diferenca_percentual,
                'total_linhas': total_linhas,
                'percentual_acerto': percentual_acerto_geral_dif,
            }

    if imprimir:
        print(f"PERCENTUAL DE ACERTO DO SISTEMA FUZZY (arquivo: '{caminho_saida_combinado}')")
        if not colunas_validas:
            print("  Nenhum prefetcher pôde ser avaliado.")
        for prefetcher in colunas_validas:
            info = resultado[prefetcher]
            print(f"  - {prefetcher}: {info['acertos']}/{info['total_linhas']} linha(s) "
                  f"corretas ({info['percentual_acerto']:.2f}%)")
        if 'geral' in resultado:
            info_geral = resultado['geral']
            print(f"  - GERAL (identidade do 1º colocado, qualquer prefetcher): "
                  f"{info_geral['acertos']}/{info_geral['total_linhas']} linha(s) corretas "
                  f"({info_geral['percentual_acerto']:.2f}%)")
        if 'geral_top2' in resultado:
            info_top2 = resultado['geral_top2']
            print(f"  - GERAL TOLERANTE (previsto = 1º OU 2º colocado real, qualquer prefetcher): "
                  f"{info_top2['acertos']}/{info_top2['total_linhas']} linha(s) corretas "
                  f"({info_top2['percentual_acerto']:.2f}%)")
        if 'geral_top3' in resultado:
            info_top3 = resultado['geral_top3']
            print(f"  - GERAL TOLERANTE (previsto = 1º, 2º OU 3º colocado real, qualquer prefetcher): "
                  f"{info_top3['acertos']}/{info_top3['total_linhas']} linha(s) corretas "
                  f"({info_top3['percentual_acerto']:.2f}%)")
        if 'geral_diferenca_percentual' in resultado:
            info_dif = resultado['geral_diferenca_percentual']
            print(f"  - GERAL POR DIFERENÇA PERCENTUAL (previsto = vencedor real OU %dif <= "
                  f"{limiar_diferenca_percentual:.2f}%, qualquer prefetcher): "
                  f"{info_dif['acertos']}/{info_dif['total_linhas']} linha(s) corretas "
                  f"({info_dif['percentual_acerto']:.2f}%)")

    return resultado


def calcular_percentual_vitorias_reais(caminho_saida_combinado: str = ARQUIVO_SAIDA_COMBINADO,
                                        colunas_prefetcher: list = None,
                                        imprimir: bool = True) -> dict:
    """
    Pós-processa o arquivo COMBINADO (`caminho_saida_combinado`, gerado por
    `combinar_saidas_por_coluna`) e calcula, PARA CADA PREFETCHER, quantas
    vezes ele venceu DE VERDADE (rank MEDIDO/real = 1 — a coluna original de
    cada prefetcher, no início do arquivo, à ESQUERDA; NÃO a coluna
    '<prefetcher>_fuzzy' de rank previsto pelo sistema fuzzy):

        vitorias = número de linhas em que a coluna '<prefetcher>' (rank
                   medido/real) é igual a 1
        percentual_vitorias = vitorias / total_de_linhas * 100

    Diferente de `calcular_percentual_acerto`, esta função não avalia o
    sistema fuzzy nenhuma — é só uma contagem descritiva de quantas vezes
    cada prefetcher REALMENTE venceu no banco de dados original, útil para
    entender o desbalanceamento entre prefetchers (ex.: um prefetcher que
    vence em 60% das linhas domina o dataset, o que pode enviesar a
    avaliação de acerto do sistema fuzzy).

    Como a contagem é feita coluna por coluna, de forma independente, a
    soma dos percentuais de todos os prefetchers pode passar de 100% em
    caso de EMPATE (mais de um prefetcher com rank medido = 1 na mesma
    linha) — cada um dos empatados conta sua própria vitória normalmente,
    sem nenhum desempate entre eles (diferente de `calcular_percentual_acerto`,
    que precisa desempatar para decidir "o" vencedor de cada linha).

    Parâmetros
    ----------
    caminho_saida_combinado : str
        Caminho do arquivo combinado a ler. Default: ARQUIVO_SAIDA_COMBINADO.
    colunas_prefetcher : list ou None
        Nomes dos prefetchers a avaliar. Default: None — nesse caso, são
        detectados automaticamente a partir de todas as colunas do
        cabeçalho que terminam em '_fuzzy' (removendo o sufixo) — mesmo
        critério de `calcular_percentual_acerto`, embora aqui só a coluna
        SEM sufixo (rank medido) seja realmente usada no cálculo.
    imprimir : bool
        Se True, imprime um relatório legível além de retornar o dicionário.

    Retorna
    -------
    dict
        {nome_do_prefetcher: {'vitorias': int, 'total_linhas': int,
        'percentual_vitorias': float}}. Prefetchers cuja coluna de rank
        medido não for encontrada ficam de fora do dicionário (com um aviso
        impresso), sem interromper o cálculo dos demais.
    """
    df = pd.read_csv(caminho_saida_combinado)

    if colunas_prefetcher is None:
        colunas_prefetcher = [
            str(coluna)[:-len('_fuzzy')] for coluna in df.columns if str(coluna).endswith('_fuzzy')
        ]

    if not colunas_prefetcher:
        raise ValueError(
            f"Nenhuma coluna '<prefetcher>_fuzzy' encontrada em '{caminho_saida_combinado}' "
            f"(usada para detectar automaticamente os prefetchers)."
        )

    total_linhas = len(df)

    resultado = {}
    for prefetcher in colunas_prefetcher:
        if prefetcher not in df.columns:
            print(f"[AVISO] Não encontrei a coluna de rank medido '{prefetcher}' em "
                  f"'{caminho_saida_combinado}'; pulando.")
            continue

        vitorias = int((df[prefetcher] == 1).sum())
        percentual_vitorias = (vitorias / total_linhas * 100) if total_linhas else 0.0

        resultado[prefetcher] = {
            'vitorias': vitorias,
            'total_linhas': total_linhas,
            'percentual_vitorias': percentual_vitorias,
        }

    if imprimir:
        print(f"PERCENTUAL DE VITÓRIAS REAIS POR PREFETCHER (arquivo: '{caminho_saida_combinado}')")
        if not resultado:
            print("  Nenhum prefetcher pôde ser avaliado.")
        for prefetcher, info in sorted(resultado.items(), key=lambda item: item[1]['vitorias'], reverse=True):
            print(f"  - {prefetcher}: {info['vitorias']}/{info['total_linhas']} linha(s) "
                  f"({info['percentual_vitorias']:.2f}%)")

    return resultado


# ---------------------------------------------------------------------------
# 7. Execução principal
# ---------------------------------------------------------------------------

def main():
    """
    Localiza todos os arquivos .csv em DIRETORIO_CSV_ENTRADA e roda o
    pipeline fuzzy completo (`processar_arquivo_csv`) para cada um deles,
    gravando um arquivo de saída por CSV em DIRETORIO_CSV_SAIDA. Falhas em
    um arquivo são reportadas mas não interrompem o processamento dos demais.

    Ao final, se COMBINAR_SAIDAS_AUTOMATICAMENTE for True, chama
    `combinar_saidas_por_coluna()` para juntar todas as saídas geradas em um
    único arquivo (ARQUIVO_SAIDA_COMBINADO).
    """
    caminhos_csv = sorted(glob.glob(os.path.join(DIRETORIO_CSV_ENTRADA, '*.csv')))

    if not caminhos_csv:
        print(f"Nenhum arquivo .csv encontrado em '{DIRETORIO_CSV_ENTRADA}'.")
        return {}

    print(f"{len(caminhos_csv)} arquivo(s) .csv encontrado(s) em '{DIRETORIO_CSV_ENTRADA}':")
    for caminho in caminhos_csv:
        print(f"  - {caminho}")

    resultados_por_arquivo = {}
    for caminho_csv in caminhos_csv:
        print()
        print("=" * 80)
        print(f"PROCESSANDO ARQUIVO: {caminho_csv}")
        print("=" * 80)
        try:
            resultados_por_arquivo[caminho_csv] = processar_arquivo_csv(caminho_csv)
        except Exception as erro:
            print(f"[ERRO] Falha ao processar '{caminho_csv}': {erro}")

    print()
    print("=" * 80)
    print(f"PROCESSAMENTO CONCLUÍDO: {len(resultados_por_arquivo)} de {len(caminhos_csv)} "
          f"arquivo(s) processado(s) com sucesso.")
    print("=" * 80)

    if ADICIONAR_COLUNA_IPC_AUTOMATICAMENTE:
        print()
        print("=" * 80)
        print("ADICIONANDO COLUNA DE IPC AOS ARQUIVOS DE RESULTADO")
        print("=" * 80)
        try:
            adicionar_coluna_ipc_aos_resultados()
        except Exception as erro:
            print(f"[ERRO] Falha ao adicionar a coluna de IPC aos resultados: {erro}")

    df_combinado = None
    if COMBINAR_SAIDAS_AUTOMATICAMENTE:
        print()
        print("=" * 80)
        print("COMBINANDO SAÍDAS EM UM ÚNICO ARQUIVO")
        print("=" * 80)
        try:
            df_combinado = combinar_saidas_por_coluna()
        except Exception as erro:
            print(f"[ERRO] Falha ao combinar as saídas: {erro}")

    percentual_acerto = None
    if CALCULAR_PERCENTUAL_ACERTO_AUTOMATICAMENTE and df_combinado is not None:
        print()
        print("=" * 80)
        print("CALCULANDO PERCENTUAL DE ACERTO DO SISTEMA FUZZY")
        print("=" * 80)
        try:
            percentual_acerto = calcular_percentual_acerto()
        except Exception as erro:
            print(f"[ERRO] Falha ao calcular o percentual de acerto: {erro}")

    percentual_vitorias_reais = None
    if CALCULAR_PERCENTUAL_VITORIAS_REAIS_AUTOMATICAMENTE and df_combinado is not None:
        print()
        print("=" * 80)
        print("CALCULANDO PERCENTUAL DE VITÓRIAS REAIS POR PREFETCHER")
        print("=" * 80)
        try:
            percentual_vitorias_reais = calcular_percentual_vitorias_reais()
        except Exception as erro:
            print(f"[ERRO] Falha ao calcular o percentual de vitórias reais: {erro}")

    return {
        'resultados_por_arquivo': resultados_por_arquivo,
        'df_combinado': df_combinado,
        'percentual_acerto': percentual_acerto,
        'percentual_vitorias_reais': percentual_vitorias_reais,
    }


if __name__ == '__main__':
    main()
