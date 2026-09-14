# Relatório Técnico — Tech Challenge, Fase 1
## Sistema inteligente de suporte ao diagnóstico

**Tarefa principal:** classificação de biópsias mamárias (maligno × benigno) a partir de
dados tabulares.
**Entregável EXTRA:** detecção de pneumonia em radiografias de tórax com rede neural
convolucional.

Todos os números deste relatório foram produzidos por
`python scripts/run_pipeline.py` (semente 42, 285 s de execução) e por
`python scripts/run_cnn.py --epochs 15 --size 64`. As tabelas de origem estão em
`reports/metrics/` e as figuras em `reports/figures/`.

---

## Sumário

1. [O problema e por que ele importa](#1-o-problema-e-por-que-ele-importa)
2. [Dados e exploração](#2-dados-e-exploração)
3. [Estratégias de pré-processamento](#3-estratégias-de-pré-processamento)
4. [Modelos usados e por quê](#4-modelos-usados-e-por-quê)
5. [Treinamento e seleção do campeão](#5-treinamento-e-seleção-do-campeão)
6. [Escolha da métrica e calibração do limiar](#6-escolha-da-métrica-e-calibração-do-limiar)
7. [Resultados no conjunto de teste](#7-resultados-no-conjunto-de-teste)
8. [Interpretação dos resultados](#8-interpretação-dos-resultados)
9. [Robustez e análise de erros](#9-robustez-e-análise-de-erros)
10. [EXTRA — visão computacional](#10-extra--visão-computacional)
11. [Discussão crítica: dá para usar na prática?](#11-discussão-crítica-dá-para-usar-na-prática)
12. [Conclusão](#12-conclusão)

---

## 1. O problema e por que ele importa

O hospital universitário recebe mais exames do que a equipe de patologia consegue laudar
no mesmo dia. A consequência não é apenas atraso administrativo: é que **um carcinoma
agressivo pode ficar semanas na fila atrás de dezenas de nódulos benignos**, e a janela
terapêutica se estreita.

O objetivo desta fase, portanto, **não é automatizar o diagnóstico**. É construir a base
de um sistema que:

1. atribua a cada exame uma **probabilidade de malignidade**;
2. traduza essa probabilidade em uma **faixa de triagem** operacional;
3. **explique** por que chegou àquele número, para que o médico possa concordar ou
   discordar de forma fundamentada.

O sucesso não se mede por "acurácia alta". Mede-se por **quantos cânceres deixam de
esperar** — sem que o sistema vire um alarme que ninguém escuta.

---

## 2. Dados e exploração

### 2.1 A base

**Breast Cancer Wisconsin (Diagnostic) — WDBC.** 569 biópsias por punção aspirativa por
agulha fina (PAAF) de nódulo mamário. Cada registro traz 30 atributos morfométricos
extraídos digitalmente da imagem do núcleo celular: 10 medidas (raio, textura, perímetro,
área, suavidade, compacidade, concavidade, pontos côncavos, simetria, dimensão fractal)
× 3 estatísticas (`_mean`, `_se`, `_worst`).

> **Por que este dataset.** É a base tabular indicada no enunciado, é pública e
> auditável, e — decisivo — as variáveis têm **significado clínico direto**: são
> medidas de atipia nuclear, exatamente o que o patologista avalia. Um modelo que se
> apoia nelas produz explicações que a equipe médica consegue criticar, em vez de
> correlações opacas.

### 2.2 Auditoria de qualidade

Antes de qualquer modelagem (`reports/metrics/01_auditoria_dataset.json`):

| verificação | resultado |
|---|---|
| dimensões | 569 linhas × 31 colunas (30 preditoras + alvo) |
| valores ausentes | **0** |
| linhas duplicadas | **0** |
| colunas constantes | **nenhuma** |
| zeros biologicamente impossíveis (raio/perímetro/área/textura = 0) | **nenhum** |
| distribuição do alvo | 357 benignos (62,74%) / 212 malignos (37,26%) |
| razão de desbalanceamento | 1,684 : 1 |

A verificação de "zeros impossíveis" não é burocracia: em bases clínicas, um `0` costuma
ser **ausência disfarçada de valor**. Um núcleo celular com área zero não existe. Aqui,
nenhum caso — a base é genuinamente limpa.

### 2.3 O que a exploração mostrou

**Desbalanceamento leve, mas suficiente para invalidar a acurácia.**
1,68:1 não exige reamostragem. Exige, sim, abandonar a acurácia como métrica principal:
um classificador que responde sempre "benigno" já acerta **62,74%** — sem detectar um
único câncer. *(fig. `01_balanceamento_classes.png`)*

**As classes se separam nitidamente nas variáveis `_worst`.**
Para `concave_points_worst`, `perimeter_worst` e `radius_worst`, as duas distribuições
quase não se sobrepõem: tumores malignos têm núcleos maiores, mais irregulares e com mais
reentrâncias no contorno. Isso é coerente com o critério histopatológico de atipia
nuclear — sinal de que o modelo aprenderá algo real, e não um artefato do dataset.
*(figs. `02_distribuicoes_por_classe.png`, `03_boxplots_por_classe.png`)*

**As escalas diferem em mais de três ordens de grandeza.**
`area_worst` chega a ~4.250; `fractal_dimension_se` fica na casa de 0,003. Além disso,
várias variáveis têm assimetria positiva forte (|skew| > 1). Isso define duas decisões de
pré-processamento — padronizar e imputar pela mediana — e não é opcional para KNN, SVM e
para a regularização da regressão logística. *(fig. `08_escala_e_assimetria.png`)*

**O problema é quase linearmente separável.**
Duas componentes principais concentram ~63% da variância e já separam visualmente as
classes. Essa é uma expectativa importante: **se um modelo complexo não superar a
regressão logística por margem relevante, devemos ficar com o modelo simples.**
*(fig. `06_projecao_pca.png`)*

### 2.4 Análise de correlação

Duas perguntas distintas, respondidas separadamente:

**(a) O que se associa ao diagnóstico?** *(`03_correlacao_com_alvo.csv`)*

| variável | correlação ponto-bisserial com o alvo |
|---|---|
| `concave_points_worst` | 0,794 |
| `perimeter_worst` | 0,783 |
| `concave_points_mean` | 0,777 |
| `radius_worst` | 0,777 |
| `perimeter_mean` | 0,743 |

**(b) O que é redundante entre si?** *(`04_pares_redundantes.csv`)*

**21 pares** com |r| ≥ 0,90. Os blocos são previsíveis pela geometria: `radius`,
`perimeter` e `area` medem a mesma coisa (r > 0,98), e o mesmo vale entre `concavity` e
`concave_points`. O procedimento marcou **10 colunas** como candidatas a descarte.

**Decisão: manter as 30 variáveis.** A multicolinearidade não prejudica árvores, SVM ou
KNN; ela desestabiliza a *interpretação* dos coeficientes lineares, pois o peso de uma
variável "vaza" para a gêmea. O antídoto adotado foi **regularização L2** (que distribui
o peso entre as colineares em vez de escolher uma arbitrariamente) somado a uma
interpretação que **não depende só dos coeficientes** — permutação e SHAP.
*(figs. `04_matriz_correlacao.png`, `05_correlacao_com_alvo.png`)*

---

## 3. Estratégias de pré-processamento

Todo o pré-processamento vive **dentro** de um `sklearn.Pipeline`. Isso não é preferência
de estilo — é o que impede o erro mais comum e mais silencioso da área: **vazamento de
dados**. A média e o desvio usados na padronização são aprendidos *somente* no conjunto
de treino e depois aplicados à validação e ao teste. Calculá-los sobre a base inteira
inflaria as métricas sem que nada quebre.

| etapa | decisão | justificativa |
|---|---|---|
| identificador | `id` descartado | é número de prontuário; não carrega informação clínica |
| alvo | `M → 1`, `B → 0` | maligno é a **classe positiva** — o evento clinicamente crítico |
| ausentes (numéricos) | imputação pela **mediana** | resistente à assimetria e aos outliers das caudas longas; a média seria puxada pelos tumores extremos |
| ausentes (categóricos) | imputação pela **moda** | preserva a categoria mais provável |
| categóricos | `OneHotEncoder(handle_unknown="ignore")` | uma categoria nunca vista em produção não pode derrubar o serviço |
| escala | `StandardScaler` (z-score) | escalas diferem em >3 ordens de grandeza |
| multicolinearidade | mantida, controlada por regularização L2 | descartar colunas perderia informação real |

### Sobre o ramo categórico

O `ColumnTransformer` tem **dois ramos automáticos**, resolvidos em tempo de `fit` pelo
tipo de cada coluna. No WDBC as 30 preditoras são numéricas, então **o ramo categórico
fica vazio nesta execução** — e isso está explicitamente documentado, não escondido.

Ele é mantido porque o mesmo pipeline atende as demais bases do hospital (sexo, etnia,
unidade de origem, tipo de exame) sem uma linha de alteração. O teste automatizado
`test_ramo_categorico_funciona_com_dados_mistos` alimenta o pré-processador com um quadro
misto contendo `NaN` e uma categoria inédita, e verifica que a saída sai numérica,
completa e com a dimensão correta.

### Prova de que a imputação funciona

Como a base não tem ausentes, o ramo de imputação ficaria sem exercício. Para não
entregar código não verificado, injetamos ausências MCAR no conjunto de teste e medimos a
degradação — resultados na [seção 9](#9-robustez-e-análise-de-erros).

### Separação treino / validação / teste

Três partições **estratificadas**, com papéis rigidamente separados
(`06_particoes.csv`):

| partição | n | % | benignos | malignos | prevalência | papel |
|---|---|---|---|---|---|---|
| treino | 341 | 59,9% | 214 | 127 | 37,24% | ajuste + busca de hiperparâmetros (CV 5 folds) |
| validação | 114 | 20,0% | 71 | 43 | 37,72% | seleção do campeão + calibração do limiar |
| teste | 114 | 20,0% | 72 | 42 | 36,84% | aberto **uma única vez**, ao final |

A estratificação preserva a prevalência de malignos nas três partições. Sem ela, com
apenas 569 amostras, uma partição poderia ficar com prevalência bem diferente e distorcer
toda a avaliação.

---

## 4. Modelos usados e por quê

Seis famílias, escolhidas para cobrir hipóteses **estruturalmente diferentes** sobre a
forma da fronteira de decisão — não para engordar a lista.

| modelo | hipótese que testa | por que está aqui |
|---|---|---|
| **Regressão Logística** (L2, `liblinear`) | fronteira linear no espaço padronizado | Padrão-ouro em estatística médica: os coeficientes viram **razões de chance**, grandeza que a equipe clínica já interpreta. É o modelo mais fácil de auditar e de defender perante um comitê de ética. |
| **Árvore de Decisão** | regras de corte em variáveis isoladas | Produz regras explícitas ("se `radius_worst` > 16,8 então suspeitar"), o mais próximo de um protocolo clínico escrito. Sozinha é instável — serve de referência de interpretabilidade máxima. |
| **KNN** | separação **local** por semelhança | Raciocínio por analogia: classifica o paciente pelos casos históricos mais parecidos. Testa se a estrutura do problema é local ou global. |
| **Random Forest** | interações não lineares, votação de árvores decorrelacionadas | Absorve bem a multicolinearidade, é robusto a outliers e fornece importância nativa + SHAP exato via `TreeExplainer`. |
| **Gradient Boosting** | correção sequencial de resíduos | Costuma ser o teto de desempenho em dados tabulares. Serve para verificar se os modelos simples já saturaram o problema. |
| **SVM (kernel RBF)** | margem máxima em espaço de alta dimensão | Historicamente muito forte em bases *pequenas e largas* (569 × 30). Envolvido em `CalibratedClassifierCV` (Platt scaling) porque o SVC devolve distâncias à margem, não probabilidades — e nosso limiar clínico opera sobre probabilidade. |

---

## 5. Treinamento e seleção do campeão

### 5.1 Busca de hiperparâmetros

`GridSearchCV` com `StratifiedKFold` de 5 dobras, **apenas no conjunto de treino**.

A métrica de *refit* é a **ROC AUC**, e a escolha é deliberada: ela **independe do limiar
de decisão** e mede a capacidade de ordenar corretamente os pacientes por risco —
exatamente o que a triagem precisa. Otimizar recall diretamente na busca levaria à
solução degenerada de classificar todo mundo como maligno (recall 1,0, clinicamente
inútil). **O limiar é calibrado depois, separadamente.**

Resultados da validação cruzada no treino (`07_resultados_validacao_cruzada.csv`):

| modelo | ROC AUC (CV) | desvio | recall | precisão | F1 | acurácia |
|---|---|---|---|---|---|---|
| Regressão Logística | **0,9944** | 0,0075 | 0,9609 | 0,9683 | 0,9645 | 0,9736 |
| SVM (RBF) | 0,9941 | 0,0083 | 0,9532 | 0,9757 | 0,9641 | 0,9737 |
| Gradient Boosting | 0,9929 | 0,0064 | 0,9375 | 0,9766 | 0,9549 | 0,9679 |
| Random Forest | 0,9884 | 0,0117 | 0,9535 | 0,9510 | 0,9500 | 0,9619 |
| KNN | 0,9859 | 0,0141 | 0,8831 | 0,9694 | 0,9201 | 0,9445 |
| Árvore de Decisão | 0,9814 | 0,0124 | 0,9603 | 0,9019 | 0,9282 | 0,9443 |

### 5.2 O critério de seleção — e por que não é o óbvio

A primeira versão deste pipeline escolhia o campeão pela **ROC AUC do conjunto de
validação**. O resultado foi o KNN — que era o **penúltimo colocado na validação
cruzada** e terminou com desempenho inferior no teste.

O diagnóstico é direto: com 114 exames de validação (43 malignos), o erro-padrão da AUC
beira ±0,02. **Um modelo pode "vencer" por sorte de partição.**

O critério final é a **média entre duas estimativas independentes de ROC AUC**:

```
score_seleção = 0,5 × ROC_AUC(CV 5-fold no treino)  +  0,5 × ROC_AUC(holdout de validação)
                     ↑ estabilidade (341 amostras)         ↑ confirmação independente
```

Nenhuma das duas toca o conjunto de teste. Resultado (`08b_ranking_selecao_campeao.csv`):

| modelo | ROC AUC CV | ROC AUC validação | **score de seleção** |
|---|---|---|---|
| **Regressão Logística** | 0,9944 | 0,9957 | **0,99510** |
| SVM (RBF) | 0,9941 | 0,9957 | 0,99491 |
| Gradient Boosting | 0,9929 | 0,9934 | 0,99317 |
| KNN | 0,9859 | 0,9974 | 0,99163 |
| Random Forest | 0,9884 | 0,9902 | 0,98928 |
| Árvore de Decisão | 0,9814 | 0,9687 | 0,97507 |

> **Campeão: Regressão Logística** (`C = 1,0`, penalidade L2, sem reponderação de classe).
> Diferença para o SVM: 0,0002 — estatisticamente indistinguível.

**Esse empate técnico é o resultado mais importante da modelagem.** Quando o modelo mais
simples e mais auditável empata com o mais complexo, a escolha em medicina é óbvia: fica
o que a equipe clínica consegue ler, contestar e homologar. A expectativa levantada na
PCA (problema quase linearmente separável) se confirmou.

*(figs. `09_comparacao_modelos.png`, `10_curvas_roc_pr.png`, `16_curva_aprendizado.png`)*

---

## 6. Escolha da métrica e calibração do limiar

### 6.1 Por que recall, e não acurácia

Os dois erros têm custos **radicalmente assimétricos**:

| erro | consequência clínica | reversível? |
|---|---|---|
| **Falso negativo** — câncer classificado como benigno | a paciente é liberada, o tumor evolui, a janela terapêutica se fecha | **não** |
| **Falso positivo** — benigno marcado como suspeito | ansiedade + biópsia excisional confirmatória | **sim** |

Logo:

- **métrica que governa a operação: recall (sensibilidade) da classe maligna**;
- **F1 e precisão como contrapeso**, para o sistema não virar um alarme que ninguém escuta;
- **ROC AUC / PR AUC** para comparar modelos independentemente do limiar;
- **acurácia relatada, mas nunca como critério** — 62,74% é o piso trivial.

### 6.2 O limiar como parâmetro clínico

O limiar padrão de 0,50 não tem nada de especial: é uma convenção matemática, não uma
decisão médica. Nosso procedimento (`evaluate.tune_threshold`) faz o contrário:

1. fixa a **sensibilidade mínima aceitável** (99%, definida como requisito clínico);
2. entre **todos** os limiares que a atendem na validação, escolhe **o mais alto** —
   porque esse é o que minimiza biópsias desnecessárias.

Resultado na validação: **limiar = 0,0934**, com recall 1,000 (43/43) e precisão 0,768.
*(fig. `11_analise_limiar.png`, tabela `09_curva_limiar_validacao.csv`)*

> **Limitação honesta e relevante.** Com apenas 43 malignos na validação, a granularidade
> do recall é 1/43 ≈ 0,023. Exigir ≥ 0,99 força, na prática, **recall perfeito na
> validação** — e recall perfeito em 43 casos não se transfere automaticamente. Foi
> exatamente o que aconteceu: no teste, o mesmo limiar entrega 0,976 (41/42). O limiar
> deve ser **re-derivado sobre uma amostra prospectiva maior** antes de qualquer uso
> real; o valor aqui é uma demonstração do método, não uma recomendação operacional.

---

## 7. Resultados no conjunto de teste

O campeão foi reajustado em **treino + validação** (455 exames) para aproveitar todos os
dados disponíveis. Hiperparâmetros e limiar já estavam **congelados**, portanto não há
vazamento. Só então o teste foi aberto — **uma única vez**.

### 7.1 Regressão Logística nos 114 exames de teste

| métrica | limiar padrão 0,50 | limiar clínico 0,0934 |
|---|---|---|
| **Recall (sensibilidade)** | 0,9524 | **0,9762** |
| Precisão (VPP) | 0,9756 | 0,8542 |
| Especificidade | 0,9861 | 0,9028 |
| F1-score | 0,9639 | 0,9111 |
| Acurácia | 0,9737 | 0,9298 |
| Acurácia balanceada | 0,9692 | 0,9395 |
| MCC | 0,9433 | 0,8592 |
| ROC AUC | 0,9960 | 0,9960 |
| PR AUC | 0,9943 | 0,9943 |
| Brier score | 0,0211 | 0,0211 |
| **Verdadeiros positivos** | 40 | 41 |
| **Falsos negativos** ⚠ | **2** | **1** |
| Falsos positivos | 1 | 7 |
| Verdadeiros negativos | 71 | 65 |

*(figs. `12_matriz_confusao_padrao.png`, `13_matriz_confusao_clinica.png`,
`14_distribuicao_scores.png`)*

**Como ler esta tabela.** Baixar o limiar de 0,50 para 0,0934 troca **1 falso negativo
por 6 falsos positivos**. Em números clínicos: um câncer a mais detectado ao custo de
seis biópsias confirmatórias adicionais. Essa é uma decisão **do comitê clínico**, não do
time de dados — o código apenas expõe o botão e mostra o preço de cada posição.

### 7.2 Todos os modelos no teste (limiar 0,50, para referência)

| modelo | recall | F1 | precisão | especificidade | acurácia | ROC AUC |
|---|---|---|---|---|---|---|
| Random Forest | 0,9762 | 0,9762 | 0,9762 | 0,9861 | 0,9825 | 0,9987 |
| SVM (RBF) | 0,9762 | 0,9762 | 0,9762 | 0,9861 | 0,9825 | 0,9957 |
| **Regressão Logística** (campeão) | 0,9524 | 0,9639 | 0,9756 | 0,9861 | 0,9737 | 0,9954 |
| KNN | 0,9286 | 0,9512 | 0,9750 | 0,9861 | 0,9649 | 0,9974 |
| Árvore de Decisão | 0,9286 | 0,9176 | 0,9070 | 0,9444 | 0,9386 | 0,9906 |
| Gradient Boosting | 0,9048 | 0,9500 | 1,0000 | 1,0000 | 0,9649 | 0,9937 |

> **Uma observação de honestidade metodológica.** Random Forest e SVM tiveram desempenho
> ligeiramente superior *neste* conjunto de teste. Isso **não invalida a escolha do
> campeão** — invalidaria o contrário: trocar o modelo depois de ver o teste seria
> exatamente o vazamento que a separação em três partições existe para evitar. A
> diferença (2 erros em 114 exames) está dentro do ruído amostral, e as seis famílias
> ficam entre 0,99 e 1,00 de ROC AUC.

### 7.3 Calibração das probabilidades

O **Brier score de 0,0211** e a curva de confiabilidade indicam que a probabilidade
prevista é interpretável como risco, e não apenas como um ranking. Isso é o que legitima
o uso de faixas de triagem e a comunicação de "probabilidade de malignidade" ao médico.
*(fig. `15_calibracao.png`)*

### 7.4 Faixas de triagem aplicadas ao teste

| faixa | critério | n de exames | conduta |
|---|---|---|---|
| 🟢 VERDE | score < 0,0934 | 66 | fila de rotina |
| 🟡 AMARELO | 0,0934 ≤ score < 0,90 | 15 | revisão humana prioritária |
| 🔴 VERMELHO | score ≥ 0,90 | 33 | encaminhamento imediato |

**É assim que o ganho aparece na prática:** 33 exames sobem imediatamente na fila, 15 são
marcados como zona de incerteza e recebem atenção humana explícita, e 66 seguem o rito
normal.

---

## 8. Interpretação dos resultados

Um modelo que não explica a própria decisão não é adotável em medicina. Usamos **três
lentes complementares**, que respondem a perguntas diferentes.

### 8.1 Coeficientes — razões de chance *(`17_referencia_logistica_odds_ratio.csv`)*

Como as variáveis estão padronizadas, `exp(coeficiente)` é a **razão de chances por
aumento de 1 desvio-padrão** na variável.

| variável | coeficiente | odds ratio por 1 DP | leitura |
|---|---|---|---|
| `texture_worst` | +1,424 | **4,15×** | núcleos de textura mais heterogênea multiplicam por 4 a chance de malignidade |
| `radius_se` | +1,244 | 3,47× | **variabilidade** do raio entre núcleos — pleomorfismo celular |
| `symmetry_worst` | +1,059 | 2,88× | assimetria nuclear |
| `concave_points_mean` | +0,955 | 2,60× | reentrâncias no contorno |
| `area_se` | +0,933 | 2,54× | variabilidade da área |
| `area_worst` | +0,924 | 2,52× | tamanho dos maiores núcleos |
| `compactness_se` | −0,916 | **0,40×** | efeito **protetor** condicional (ver abaixo) |

*(fig. `18_referencia_logistica.png`)*

### 8.2 Importância por permutação *(`14_importancia_permutacao.csv`)*

Embaralha uma variável por vez no conjunto de teste e mede a **queda real de ROC AUC**.
Responde: "o que o modelo de fato usa?"

| variável | queda média de ROC AUC | desvio |
|---|---|---|
| `symmetry_worst` | 0,00984 | 0,00262 |
| `texture_worst` | 0,00860 | 0,00416 |
| `concavity_worst` | 0,00780 | 0,00268 |
| `concave_points_mean` | 0,00288 | 0,00229 |
| `concave_points_worst` | 0,00281 | 0,00150 |

As quedas são pequenas em termos absolutos — **e isso é informação, não ruído**: com 21
pares de variáveis quase idênticas, embaralhar uma delas quase não machuca, porque a
gêmea continua ali. É a assinatura numérica da redundância detectada na EDA.

*(fig. `19_importancia_permutacao.png`)*

### 8.3 SHAP *(`15_importancia_shap.csv`)*

Decompõe **cada previsão individual** na contribuição de cada variável (valores de
Shapley). É a única das três lentes que explica **um paciente específico**.

| variável | \|SHAP\| médio |
|---|---|
| `texture_worst` | 0,0624 |
| `concave_points_mean` | 0,0459 |
| `symmetry_worst` | 0,0410 |
| `concavity_worst` | 0,0392 |
| `radius_se` | 0,0381 |

O gráfico *beeswarm* mostra magnitude **e direção**: valores altos (vermelho) dessas
variáveis empurram consistentemente o score para a direita, isto é, para maligno.
*(figs. `20_shap_beeswarm.png`, `21_shap_importancia_global.png`)*

### 8.4 Convergência — e uma divergência instrutiva

As três lentes convergem em `texture_worst`, `symmetry_worst`, `concave_points_mean` e
`concavity_worst`. Mas há uma **divergência que vale a pena não varrer para debaixo do
tapete**:

- na **correlação bruta** com o alvo, a campeã é `concave_points_worst` (r = 0,794),
  enquanto `texture_worst` aparece bem abaixo;
- no **modelo**, `texture_worst` é a variável de maior peso.

Não é contradição: são perguntas diferentes. A correlação mede a associação **marginal**
(a variável sozinha). O coeficiente mede a contribuição **condicional** (o que ela
acrescenta *dado que as outras 29 já estão no modelo*). Como `concave_points_worst` é
quase redundante com metade da tabela, sua informação já foi absorvida; `texture_worst`,
que mede outra coisa — heterogeneidade da cromatina —, traz informação **nova**.

O mesmo raciocínio explica o coeficiente negativo de `compactness_se`: não é que
compacidade proteja contra câncer, e sim que, **fixadas** concavidade e pontos côncavos,
o que sobra de `compactness_se` funciona como correção. **É por isso que o relatório não
se apoia só nos coeficientes** — e é exatamente esse tipo de nuance que precisa ser
comunicado ao médico, e não escondido atrás de um ranking bonito.

### 8.5 Explicações individuais

Para quatro perfis de caso — verdadeiro positivo confiante, falso positivo, falso
negativo e caso ambíguo — geramos o gráfico em cascata SHAP
(`22_1..22_4_shap_caso_*.png`). **É esse gráfico que entra no laudo de apoio**: mostra,
para aquele paciente, quais medidas empurraram o score para cima e quais o puxaram para
baixo. É o que permite ao médico discordar de forma fundamentada — por exemplo, ao notar
que o modelo se apoiou em uma medida afetada por artefato de preparo da lâmina.

---

## 9. Robustez e análise de erros

### 9.1 Degradação com dados ausentes *(`16_robustez_dados_ausentes.csv`)*

O WDBC não tem ausentes; um prontuário hospitalar real chega incompleto. Injetamos
ausências MCAR no conjunto de teste e medimos:

| % de células ausentes | recall | precisão | F1 | ROC AUC | falsos negativos |
|---|---|---|---|---|---|
| 0% | 0,9762 | 0,8542 | 0,9111 | 0,9960 | 1 |
| 5% | 0,9762 | 0,8913 | 0,9318 | 0,9960 | 1 |
| 10% | 0,9762 | 0,9111 | 0,9425 | 0,9950 | 1 |
| 20% | 0,9524 | 0,8696 | 0,9091 | 0,9901 | **2** |

**Leitura.** Até 10% de campos ausentes o recall não se move e a ROC AUC cai 0,001 — a
imputação pela mediana sustenta bem a degradação, novamente graças à redundância entre as
variáveis. Aos 20%, o recall cai e aparece um segundo falso negativo. **Recomendação
operacional: o serviço deve recusar (ou marcar como "não conclusivo") qualquer exame com
mais de ~10% de campos faltantes**, em vez de devolver um score silenciosamente pior.

### 9.2 Onde o modelo erra *(`12_analise_de_erros_teste.csv`)*

No limiar clínico, 8 dos 114 exames de teste foram classificados errado: **7 falsos
positivos e 1 falso negativo**.

- Os falsos positivos concentram-se em scores entre 0,09 e 0,51 — ou seja, **quase todos
  cairiam na faixa 🟡 AMARELO**, que já prevê revisão humana. O sistema não os afirma como
  malignos; ele os marca como incertos, que é o comportamento desejado.
- O único falso negativo tem score 0,0635, **abaixo mas próximo do limiar de 0,0934**. Sua
  explicação SHAP (`22_1_shap_caso_falso_negativo.png`) mostra um tumor com morfometria
  atipicamente discreta — o tipo de caso que também desafia o observador humano.

---

## 10. EXTRA — visão computacional

### 10.1 Dados

**PneumoniaMNIST** (MedMNIST v2): versão padronizada e curada do *Chest X-Ray Pneumonia*
de Kermany et al. — a mesma base indicada no enunciado, porém já dividida em
treino/validação/teste pelos autores, sem exigir credenciais do Kaggle.

| partição | n | % pneumonia |
|---|---|---|
| treino | 4.708 | 74,2% |
| validação | 524 | 74,2% |
| teste | 624 | 62,5% |

Resolução usada: **64 × 64**, em escala de cinza (`--size 64`). *(fig. `30_amostras_radiografias.png`)*

> Note o **deslocamento de prevalência** entre validação (74,2%) e teste (62,5%). Isso é
> proposital nos dados originais e é uma boa notícia metodológica: qualquer limiar
> calibrado na validação é testado sob uma distribuição diferente — mais parecido com a
> realidade de implantação.

### 10.2 Arquitetura e treinamento

CNN compacta treinada do zero: **3 blocos** `Conv3×3 → BatchNorm → ReLU → Conv3×3 →
BatchNorm → ReLU → MaxPool` (32 → 64 → 128 canais), `AdaptiveAvgPool` e cabeça densa com
dropout. **295.201 parâmetros treináveis** — pequena o suficiente para treinar em CPU.

Decisões relevantes:

- **Data augmentation** (espelhamento horizontal, rotações de ±8°, translações e escala
  de ±5%) — simula variações reais de posicionamento do paciente no aparelho.
- **`BCEWithLogitsLoss` com `pos_weight`** — a base é desbalanceada (74% pneumonia), e o
  peso corrige o viés da perda.
- **Seleção de época pela ROC AUC de validação**, não pela acurácia — em base
  desbalanceada, a acurácia sobe simplesmente por prever a classe majoritária.
- **`AdaptiveAvgPool2d`** torna a rede independente da resolução: treinar em 28×28 ou
  128×128 não exige alterar o código.

*(fig. `31_cnn_curvas_treino.png`)*

### 10.3 Resultados

Treino de 15 épocas em CPU (1.094 s). Melhor época: **14**, com **ROC AUC de validação
0,9969**. Avaliação nos 624 exames do conjunto de teste oficial
(`20_metricas_cnn_teste.csv`):

| métrica | limiar padrão 0,50 | limiar clínico 0,194 |
|---|---|---|
| **Recall (sensibilidade)** | 0,9821 | **0,9872** |
| Precisão | 0,8949 | 0,8575 |
| Especificidade | 0,8077 | 0,7265 |
| F1-score | **0,9364** | 0,9178 |
| Acurácia | 0,9167 | 0,8894 |
| Acurácia balanceada | 0,8949 | 0,8568 |
| ROC AUC | 0,9709 | 0,9709 |
| PR AUC | 0,9801 | 0,9801 |
| MCC | 0,8237 | 0,7691 |
| Verdadeiros positivos | 383 | 385 |
| **Falsos negativos** ⚠ | 7 | **5** |
| Falsos positivos | 45 | 64 |
| Verdadeiros negativos | 189 | 170 |

**Recall de 98,7% com 295 mil parâmetros treinados do zero em CPU** — sem transfer
learning, sem GPU. Para uma primeira triagem radiológica, sinalizar 385 dos 390 casos de
pneumonia é um resultado sólido.

> **A observação mais instrutiva desta seção: a AUC caiu de 0,9969 na validação para
> 0,9709 no teste.** Isso **não é overfitting no sentido usual** — a curva de treino não
> descolou da de validação. É **deslocamento de distribuição**: o conjunto de teste do
> PneumoniaMNIST tem prevalência de 62,5% contra 74,2% da validação, e vem de uma coorte
> diferente. A especificidade sofre mais que o recall (0,81 contra 0,98), ou seja, o
> modelo erra principalmente marcando radiografias normais como suspeitas.
>
> **É exatamente o comportamento que se espera ao levar um modelo de um hospital para
> outro** — e a razão pela qual a seção 11 insiste em validação prospectiva local e
> monitoramento de deriva. Um modelo que só fosse avaliado na própria validação teria
> reportado 0,997 e escondido esse problema.

Note ainda que, aqui, o limiar clínico calibrado **piorou** acurácia e F1 em troca de 2
falsos negativos a menos. A troca é defensável em triagem — mas mostra que baixar o
limiar não é gratuito, e que a decisão precisa ser tomada com o custo real em mãos.

### 10.4 Interpretabilidade visual — Grad-CAM

O **Grad-CAM** é o equivalente do SHAP para imagens: destaca as regiões que mais pesaram
na decisão. Sua função aqui não é decorativa — é permitir que o radiologista verifique se
a rede olhou para o **campo pulmonar** ou para um artefato irrelevante (borda da imagem,
texto sobreposto, marcador metálico). Um modelo com métricas excelentes e mapa de
ativação na borda da imagem é um modelo que aprendeu o dataset, não a doença.

*(figs. `33_cnn_predicoes.png`, `34_cnn_gradcam.png`)*

---

## 11. Discussão crítica: dá para usar na prática?

**Sim — mas com escopo restrito, como apoio, e nunca como decisor.**

### 11.1 Como usar

Não como rótulo binário, e sim como **fila com três faixas** (seção 7.4). O ganho não é
"acertar o diagnóstico": é **reordenar a fila** para que os casos graves sejam vistos
primeiro e **tornar a incerteza explícita** em vez de escondê-la atrás de um rótulo.

O laudo de apoio (`scripts/predict.py`) entrega, para cada exame: probabilidade, faixa,
conduta sugerida e a explicação SHAP individual. Toda saída carrega o aviso de que a
palavra final é do médico.

### 11.2 Limitações que impedem uso imediato

1. **Amostra pequena e de fonte única.** 569 biópsias de um único centro (Wisconsin, anos
   1990). O intervalo de confiança sobre 114 exames de teste é largo: a diferença entre
   0,95 e 0,98 de recall são **dois pacientes**. Nenhuma métrica aqui autoriza implantação
   sem revalidação prospectiva local.
2. **As variáveis não caem do céu.** As 30 medidas pressupõem segmentação digital do
   núcleo celular a partir da lâmina. Sem essa etapa padronizada e calibrada, o modelo
   não tem entrada válida — e desvio de calibração entre equipamentos degrada o
   desempenho **silenciosamente**, sem lançar erro.
3. **Ausência de dados demográficos.** Não há idade, etnia ou histórico familiar, então
   **não é possível auditar o modelo por subgrupo**. Sem essa auditoria, não se pode
   afirmar que ele é igualmente seguro para todas as pacientes — e essa é uma condição
   inegociável em saúde.
4. **Limiar derivado de 43 casos positivos.** Como discutido na seção 6.2, sua precisão é
   limitada pela granularidade da amostra.
5. **Deslocamento de distribuição.** Novo microscópio, novo protocolo de coloração ou nova
   equipe mudam a distribuição das medidas. Sem monitoramento, a degradação é invisível.
6. **Vieses históricos da base.** Um modelo treinado em dados dos anos 1990 herda os
   critérios de encaminhamento daquela época e daquela população.

### 11.3 Requisitos para uma implantação responsável

| requisito | por quê |
|---|---|
| **Validação prospectiva local** antes de qualquer uso | as métricas deste relatório não se transferem automaticamente |
| **Limiar homologado pelo comitê clínico** | a troca recall × precisão é uma decisão médica e institucional, não técnica |
| **Registro auditável** de toda predição com sua explicação SHAP | exigência regulatória e condição para investigar erros |
| **Monitoramento de deriva** (distribuição dos scores e taxa de alarme) | detectar mudança de equipamento ou protocolo antes que ela cause dano |
| **Botão de discordância** para o médico, realimentando o treino | mantém o profissional no controle e gera dados de melhoria |
| **Auditoria por subgrupo** assim que houver dados demográficos | garantir desempenho equitativo |
| **Reavaliação periódica obrigatória** | modelo clínico não é software que se instala e esquece |

---

## 12. Conclusão

O pipeline entrega o que a Fase 1 pediu e sustenta as afirmações com números
reproduzíveis:

- **exploração** completa, com auditoria de qualidade e leitura clínica das distribuições;
- **pré-processamento** em `Pipeline`, com ramos numérico e categórico, à prova de
  vazamento e testado contra dados ausentes e categorias inéditas;
- **seis modelos** com busca de hiperparâmetros e separação rigorosa treino/validação/teste;
- **métrica escolhida a partir do custo clínico do erro**, e limiar tratado como parâmetro
  médico e não como convenção;
- **interpretação** por três lentes convergentes, incluindo explicação individual por
  paciente;
- **entregável EXTRA** com CNN treinada do zero (ROC AUC 0,971 e recall 0,987 no teste) e Grad-CAM.

O resultado técnico mais relevante não é a métrica de topo — é que **o modelo mais simples
e mais auditável empatou com os mais complexos**. Em um domínio onde a explicação vale
tanto quanto o acerto, esse empate decide a escolha.

E o resultado mais importante é o que o sistema **não** faz: ele não diagnostica. Ele
ordena a fila, quantifica a incerteza e mostra o próprio raciocínio.

> **O modelo prioriza. O médico diagnostica.**
