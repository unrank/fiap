# Tech Challenge — Fase 1
## Sistema inteligente de suporte ao diagnóstico

Base de um sistema de IA para apoio à triagem de exames em um hospital universitário.
A entrega principal é um pipeline completo de **Machine Learning sobre dados tabulares**
que classifica biópsias mamárias em **maligno** ou **benigno**; o entregável **EXTRA** é
uma **CNN** que detecta pneumonia em radiografias de tórax.

> **O modelo prioriza. O médico diagnostica.**
> Toda a saída do sistema é apoio à decisão. A palavra final sobre o diagnóstico é,
> sempre, do médico responsável.

---

## Sumário

- [1. O problema](#1-o-problema)
- [2. Como executar](#2-como-executar)
- [3. Estrutura do projeto](#3-estrutura-do-projeto)
- [4. Dados](#4-dados)
- [5. Metodologia](#5-metodologia)
- [6. Resultados](#6-resultados)
- [7. Entregável EXTRA — visão computacional](#7-entregável-extra--visão-computacional)
- [8. Testes](#8-testes)
- [9. Limitações e uso responsável](#9-limitações-e-uso-responsável)
- [10. Referências](#10-referências)

---

## 1. O problema

Um hospital universitário recebe mais exames do que a equipe de patologia consegue
laudar no mesmo dia. O objetivo **não é substituir o patologista**: é **reordenar a
fila** para que os casos com alta suspeita de malignidade sejam vistos primeiro, e
tornar explícitos os casos de incerteza em vez de escondê-los atrás de um rótulo binário.

O sistema entrega, para cada exame, uma **probabilidade de malignidade**, uma **faixa de
triagem** e a **explicação individual** de como chegou àquele número.

| faixa | critério | conduta sugerida |
|---|---|---|
| 🟢 **VERDE** | score abaixo do limiar clínico | segue a fila de rotina |
| 🟡 **AMARELO** | entre o limiar clínico e 0,90 | revisão humana prioritária |
| 🔴 **VERMELHO** | score ≥ 0,90 | encaminhamento imediato ao mastologista |

---

## 2. Como executar

### 2.1 Docker (recomendado — reprodutível, sem instalar nada)

```bash
docker build -t medai-fase1 .
```

```bash
docker run --rm -v "$(pwd)/reports:/app/reports" -v "$(pwd)/models:/app/models" medai-fase1
```

O container roda o pipeline completo e grava figuras em `reports/figures/`, métricas em
`reports/metrics/` e o modelo serializado em `models/`.
**Não é necessário acesso à internet:** o dataset vem embarcado no scikit-learn.

Atalhos com Docker Compose:

```bash
docker compose run --rm pipeline
```

```bash
docker compose run --rm predict
```

```bash
docker compose run --rm tests
```

### 2.2 Ambiente local (Python 3.10+)

```bash
python -m venv .venv && source .venv/bin/activate
```

No Windows (PowerShell), ative com `.venv\Scripts\Activate.ps1`.

```bash
pip install -r requirements.txt
```

```bash
python scripts/run_pipeline.py
```

### 2.3 Comandos disponíveis

| comando | o que faz | tempo aprox. (CPU) |
|---|---|---|
| `python scripts/run_pipeline.py` | pipeline completo: EDA, 6 modelos, avaliação, SHAP | ~3–4 min |
| `python scripts/run_pipeline.py --quick` | grade de hiperparâmetros reduzida, para demonstração | ~1 min |
| `python scripts/run_pipeline.py --no-shap --no-figures` | só as métricas | ~2 min |
| `python scripts/predict.py --demo` | laudo de triagem para 10 exames de exemplo | segundos |
| `python scripts/predict.py --input exames.csv --output laudo.csv` | inferência em lote | segundos |
| `python scripts/build_notebook.py --execute` | gera e executa o notebook de demonstração | ~5 min |
| `pytest -q` | suíte de testes automatizados | ~10 s |
| `python scripts/run_cnn.py` | **EXTRA**: CNN para pneumonia (28×28, 12 épocas) | ~5 min |
| `python scripts/run_cnn.py --epochs 15 --size 64` | **EXTRA**: versão usada nos resultados deste README | ~18 min |

Flags úteis: `--seed`, `--cv-folds`, `--target-recall`, `--source {auto,csv,sklearn,uci}`.

### 2.4 Ambiente em que os resultados deste README foram produzidos

Todos os números reportados vêm de execuções reais neste ambiente (semente 42):

| pacote | versão | | pacote | versão |
|---|---|---|---|---|
| Python | 3.12.10 | | scikit-learn | 1.9.0 |
| numpy | 2.5.2 | | shap | 0.52.0 |
| pandas | 3.0.5 | | matplotlib | 3.11.1 |
| scipy | 1.18.1 | | seaborn | 0.13.2 |
| torch (EXTRA) | 2.14.0+cpu | | medmnist (EXTRA) | 3.0.2 |

O `requirements.txt` usa versões **mínimas** (`>=`) para não travar a instalação em
ambientes diferentes; se precisar reproduzir bit a bit, fixe as versões acima.

---

## 3. Estrutura do projeto

```
.
├── src/medai/                    # pacote da aplicação
│   ├── config.py                 # caminhos, semente, constantes do problema
│   ├── data.py                   # carga, normalização e auditoria do dataset
│   ├── eda.py                    # análise exploratória e figuras
│   ├── preprocessing.py          # ColumnTransformer, split, correlação
│   ├── models.py                 # catálogo de 6 modelos + busca de hiperparâmetros
│   ├── evaluate.py               # métricas, curvas, calibração do limiar
│   ├── interpret.py              # feature importance, permutação e SHAP
│   ├── cnn.py                    # EXTRA - CNN + Grad-CAM
│   └── utils.py                  # logging, estilo das figuras, persistência
├── scripts/
│   ├── run_pipeline.py           # tarefa principal, ponta a ponta
│   ├── run_cnn.py                # EXTRA - visão computacional
│   ├── predict.py                # inferência / laudo de triagem
│   └── build_notebook.py         # gera e executa o notebook
├── notebooks/01_analise_completa.ipynb
├── tests/test_pipeline.py        # 15 testes automatizados
├── data/raw/                     # cópia do dataset (CSV)
├── reports/
│   ├── figures/                  # todas as figuras geradas
│   ├── metrics/                  # todas as tabelas e métricas (CSV/JSON)
│   └── RELATORIO_TECNICO.md      # relatório técnico completo
├── docs/
│   ├── Tech_Challenge_Fase1.pdf  # PDF de entrega (capa + relatório + figuras)
│   └── ROTEIRO_VIDEO.md          # roteiro minutado do vídeo de demonstração
├── models/                       # artefatos serializados (gerados, não versionados)
├── Dockerfile                    # imagem base + estágio "extra" (CNN)
├── docker-compose.yml
├── requirements.txt
└── requirements-extra.txt        # dependências só do EXTRA (PyTorch)
```

---

## 4. Dados

### Tarefa principal — Breast Cancer Wisconsin (Diagnostic)

| item | valor |
|---|---|
| amostras | 569 biópsias (PAAF de nódulo mamário) |
| variáveis | 30 numéricas + alvo (`M` = maligno, `B` = benigno) |
| distribuição | 357 benignos (62,7%) / 212 malignos (37,3%) |
| ausentes / duplicatas | nenhum / nenhuma |
| fonte | [UCI ML Repository](https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic) · [Kaggle](https://www.kaggle.com/datasets/uciml/breast-cancer-wisconsin-data) |

As 30 variáveis são **10 medidas morfométricas do núcleo celular** (raio, textura,
perímetro, área, suavidade, compacidade, concavidade, pontos côncavos, simetria e
dimensão fractal) × **3 estatísticas** (`_mean`, `_se`, `_worst`).

**Como o dado é obtido.** O `data.py` tenta, nesta ordem: (1) o CSV local em
`data/raw/`; (2) a cópia oficial embarcada no scikit-learn — idêntica à do UCI e
funcional **offline**; (3) download direto do UCI. Em qualquer caso, uma cópia é gravada
em `data/raw/breast_cancer_wisconsin.csv` para tornar a execução reproduzível.

### EXTRA — PneumoniaMNIST

Versão padronizada e curada do *Chest X-Ray Pneumonia* (Kermany et al.), distribuída
pela coleção **MedMNIST v2**: 5.856 radiografias pediátricas de tórax já divididas em
treino/validação/teste pelos autores, sem necessidade de credenciais do Kaggle. É a
mesma base indicada no enunciado, em formato reproduzível. O download (poucos MB) é
automático na primeira execução.

---

## 5. Metodologia

### 5.1 Pré-processamento

Todo o pré-processamento vive **dentro** de um `sklearn.Pipeline`. Isso não é
preferência de estilo — é o que impede **vazamento de dados**: a média e o desvio usados
na padronização são aprendidos *somente* no conjunto de treino e depois aplicados à
validação e ao teste.

| etapa | decisão | por quê |
|---|---|---|
| identificador | `id` é descartado | é número de prontuário: não carrega informação clínica |
| alvo | `M → 1`, `B → 0` | maligno é a **classe positiva**, o evento clinicamente crítico |
| ausentes (numéricos) | imputação pela **mediana** | resistente à assimetria e aos outliers das caudas longas |
| ausentes (categóricos) | imputação pela **moda** | preserva a categoria mais provável |
| categóricos | `OneHotEncoder(handle_unknown="ignore")` | uma categoria nunca vista não pode derrubar o serviço |
| escala | `StandardScaler` (z-score) | escalas diferem em >3 ordens de grandeza; KNN/SVM/regularização exigem |
| multicolinearidade | mantida, controlada por regularização | 21 pares com \|r\| ≥ 0,90; descartar perderia informação |

O `ColumnTransformer` tem dois ramos automáticos (numérico e categórico). No WDBC as 30
preditoras são numéricas, então o ramo categórico fica vazio nesta execução — ele é
mantido porque o mesmo pipeline atende outras bases clínicas (sexo, unidade, tipo de
exame) sem alteração de código. O teste
`test_ramo_categorico_funciona_com_dados_mistos` comprova que esse ramo funciona.

### 5.2 Separação treino / validação / teste

Três partições **estratificadas** (60/20/20), com papéis rigidamente separados:

| partição | n | papel |
|---|---|---|
| treino | 341 | ajuste dos modelos e busca de hiperparâmetros por validação cruzada (5 folds) |
| validação | 114 | seleção do campeão e calibração do limiar clínico |
| teste | 114 | aberto **uma única vez**, ao final |

### 5.3 Modelos

Seis famílias, cada uma com grade de hiperparâmetros buscada por `GridSearchCV` com
`StratifiedKFold` **apenas no treino**:

| modelo | por que está no catálogo |
|---|---|
| **Regressão Logística** | padrão-ouro em estatística médica; coeficientes viram *odds ratio* |
| **Árvore de Decisão** | produz regras explícitas, próximas de um protocolo clínico escrito |
| **KNN** | raciocínio por analogia com casos históricos semelhantes |
| **Random Forest** | absorve multicolinearidade, captura interações, importância nativa |
| **Gradient Boosting** | teto de desempenho típico em dados tabulares |
| **SVM (RBF)** | forte em bases pequenas e largas; calibrado via Platt scaling |

A métrica de *refit* da busca é a **ROC AUC**, que independe do limiar. Otimizar recall
diretamente levaria à solução degenerada de classificar todos como malignos
(recall 1,0 e clinicamente inútil).

### 5.4 Seleção do campeão

Escolher pela ROC AUC da validação sozinha seria frágil: com 114 exames (43 malignos), o
erro-padrão da AUC beira ±0,02 e o ranking muda com a semente. O critério adotado é a
**média entre a ROC AUC da validação cruzada no treino** (estabilidade, 341 amostras em
5 folds) **e a ROC AUC do holdout de validação** (confirmação independente). O conjunto
de teste permanece intocado durante toda a seleção.

### 5.5 Métrica e limiar de decisão

Os dois erros têm custos **assimétricos**:

- **Falso negativo** (câncer classificado como benigno) — o paciente é liberado, o tumor
  evolui e a janela terapêutica se fecha. Custo potencial: a vida.
- **Falso positivo** (benigno marcado como suspeito) — gera ansiedade e uma biópsia
  confirmatória. É caro e desagradável, mas **reversível**.

Logo a métrica que governa a operação é o **recall da classe maligna**, com F1 e
precisão de contrapeso para o sistema não virar um alarme permanente. **A acurácia
isolada é enganosa**: como 62,7% da base é benigna, responder sempre "benigno" já acerta
62,7% sem detectar um único câncer.

O limiar deixa de ser o 0,50 padrão e passa a ser um **parâmetro clínico**: fixamos a
sensibilidade mínima aceitável (99%) e, entre todos os limiares que a atendem, ficamos
com o mais alto — o que minimiza biópsias desnecessárias.

---

## 6. Resultados

### Modelo campeão: **Regressão Logística** (L2, `C = 1,0`)

Escolhido pelo critério da seção 5.4, com **score de seleção 0,99510** contra 0,99491 do
SVM — um empate técnico. Quando o modelo mais simples e mais auditável empata com o mais
complexo, a escolha em medicina é o que a equipe clínica consegue ler e contestar.

**Desempenho nos 114 exames de teste** (abertos uma única vez):

| métrica | limiar padrão 0,50 | limiar clínico 0,0934 |
|---|---|---|
| **Recall (sensibilidade)** | 0,9524 | **0,9762** |
| Precisão | 0,9756 | 0,8542 |
| Especificidade | 0,9861 | 0,9028 |
| F1-score | 0,9639 | 0,9111 |
| Acurácia | 0,9737 | 0,9298 |
| ROC AUC | 0,9960 | 0,9960 |
| **Falsos negativos** ⚠ | 2 | **1** |
| Falsos positivos | 1 | 7 |

Baixar o limiar troca **1 falso negativo por 6 falsos positivos** — um câncer a mais
detectado ao custo de seis biópsias confirmatórias. Essa é uma decisão do comitê clínico;
o código apenas expõe o botão e mostra o preço de cada posição.

**Comparação entre os seis modelos** (teste, limiar 0,50):

| modelo | recall | F1 | acurácia | ROC AUC |
|---|---|---|---|---|
| Random Forest | 0,9762 | 0,9762 | 0,9825 | 0,9987 |
| SVM (RBF) | 0,9762 | 0,9762 | 0,9825 | 0,9957 |
| **Regressão Logística** (campeão) | 0,9524 | 0,9639 | 0,9737 | 0,9954 |
| KNN | 0,9286 | 0,9512 | 0,9649 | 0,9974 |
| Árvore de Decisão | 0,9286 | 0,9176 | 0,9386 | 0,9906 |
| Gradient Boosting | 0,9048 | 0,9500 | 0,9649 | 0,9937 |

> RF e SVM foram ligeiramente melhores *neste* teste. Trocar o campeão depois de ver o
> conjunto de teste seria exatamente o vazamento que a separação em três partições existe
> para evitar — a diferença (2 erros em 114) está dentro do ruído amostral.

**Variáveis mais influentes** (convergência entre permutação, SHAP e coeficientes):
`texture_worst` (OR 4,15× por desvio-padrão), `radius_se` (3,47×), `symmetry_worst`
(2,88×), `concave_points_mean` (2,60×) e `concavity_worst`.

**Robustez:** até 10% de campos ausentes, o recall não se move e a ROC AUC cai 0,001.

Os números completos e reproduzíveis ficam em `reports/metrics/` (21 tabelas CSV/JSON) e
as figuras em `reports/figures/` (30 imagens). A análise detalhada, com a discussão
crítica, está no **[relatório técnico](reports/RELATORIO_TECNICO.md)**.

Um resumo dos artefatos gerados:

| arquivo | conteúdo |
|---|---|
| `00_manifesto_execucao.json` | tudo que a execução produziu, em um único JSON |
| `01_auditoria_dataset.json` | auditoria de qualidade da base |
| `02_estatisticas_descritivas.csv` | descritivas estendidas (skew, curtose, CV) |
| `03..05_correlacao*` | correlação com o alvo e pares redundantes |
| `06_particoes.csv` | tamanho e prevalência em cada partição |
| `07_resultados_validacao_cruzada.csv` | CV dos 6 modelos no treino |
| `08_metricas_validacao.csv`, `08b_ranking_selecao_campeao.csv` | seleção do campeão |
| `09_curva_limiar_validacao.csv` | trade-off recall × precisão por limiar |
| `10..11_metricas_teste*` | resultado final no teste |
| `12_analise_de_erros_teste.csv` | cada exame classificado errado |
| `13..15, 17` | importância nativa, permutação, SHAP e *odds ratio* de referência |
| `16_robustez_dados_ausentes.csv` | degradação com 5%, 10% e 20% de células ausentes |

---

## 7. Entregável EXTRA — visão computacional

```bash
pip install -r requirements-extra.txt
```

```bash
python scripts/run_cnn.py --epochs 15 --size 64
```

Ou via Docker (estágio dedicado, imagem maior):

```bash
docker build --target extra -t medai-fase1-cnn . && docker run --rm -v "$(pwd)/reports:/app/reports" medai-fase1-cnn
```

CNN compacta treinada do zero (3 blocos `Conv-BN-ReLU` + cabeça densa, **295.201
parâmetros**), com *data augmentation* que simula variações reais de posicionamento do
paciente, perda ponderada para o desbalanceamento (~74% de pneumonia no treino) e seleção
de época pela ROC AUC de validação. A interpretabilidade visual usa **Grad-CAM** — o
equivalente do SHAP para imagens: permite ao radiologista conferir se a rede olhou para o
campo pulmonar ou para um artefato irrelevante da imagem.

**Resultado** (15 épocas, 64×64, ~18 min em CPU; 624 radiografias de teste):

| métrica | limiar 0,50 | limiar clínico 0,194 |
|---|---|---|
| Recall | 0,9821 | **0,9872** |
| F1-score | 0,9364 | 0,9178 |
| Acurácia | 0,9167 | 0,8894 |
| ROC AUC | 0,9709 | 0,9709 |
| Falsos negativos | 7 | **5** |

> A ROC AUC cai de 0,9969 na validação para 0,9709 no teste. Não é overfitting: é
> **deslocamento de distribuição** — o teste oficial do PneumoniaMNIST tem prevalência de
> 62,5% contra 74,2% da validação. É exatamente o que acontece ao levar um modelo de um
> hospital para outro, e a razão pela qual a seção 9 insiste em validação prospectiva
> local.

---

## 8. Testes

```bash
pytest -q
```

15 testes cobrem o que, se quebrar, **invalida silenciosamente o resultado científico**:

- as partições são disjuntas, completas e preservam a prevalência;
- a escala é aprendida **somente no treino** (guarda explícita contra vazamento);
- o ramo categórico do pré-processador funciona e sobrevive a categorias nunca vistas;
- o pipeline continua produzindo probabilidades finitas com 15% de valores ausentes;
- as métricas batem com um cálculo manual de matriz de confusão;
- o limiar calibrado é de fato o **maior** que atende o recall exigido;
- o artefato serializado reproduz exatamente as previsões do modelo em memória.

---

## 9. Limitações e uso responsável

1. **Amostra pequena, de uma única fonte.** 569 biópsias de um único centro (Wisconsin,
   anos 1990). O intervalo de confiança sobre 114 exames de teste é largo: qualquer
   métrica aqui precisa ser revalidada **prospectivamente** na população do hospital.
2. **As variáveis não caem do céu.** As 30 medidas pressupõem segmentação digital do
   núcleo celular a partir da lâmina. Sem essa etapa padronizada e calibrada, o modelo
   não tem entrada válida — e desvio de calibração entre equipamentos degrada o
   desempenho silenciosamente.
3. **Ausência de dados demográficos.** Não há idade, etnia ou histórico familiar, então
   **não é possível auditar o modelo por subgrupo**. Sem essa auditoria, não se pode
   afirmar que ele é igualmente seguro para todas as pacientes.
4. **Deslocamento de distribuição.** Novo microscópio, novo protocolo de coloração ou
   nova equipe mudam a distribuição das medidas. Exige monitoramento contínuo.

**Requisitos para uma implantação responsável:** validação prospectiva local; limiar
homologado pelo comitê clínico (e não pelo time de dados); registro auditável de toda
predição com sua explicação SHAP; monitoramento de deriva; e um botão explícito de
discordância para o médico, cujo uso realimenta o próximo ciclo de treino.

---

## 10. Referências

- Street, W. N., Wolberg, W. H., Mangasarian, O. L. (1993). *Nuclear feature extraction
  for breast tumor diagnosis*. IS&T/SPIE International Symposium on Electronic Imaging.
- Dua, D., Graff, C. (2019). *UCI Machine Learning Repository* — Breast Cancer Wisconsin
  (Diagnostic) Data Set.
- Yang, J. et al. (2023). *MedMNIST v2 — A large-scale lightweight benchmark for 2D and
  3D biomedical image classification*. Scientific Data 10, 41.
- Kermany, D. S. et al. (2018). *Identifying medical diagnoses and treatable diseases by
  image-based deep learning*. Cell 172(5).
- Lundberg, S. M., Lee, S.-I. (2017). *A unified approach to interpreting model
  predictions* (SHAP). NeurIPS.
- Selvaraju, R. R. et al. (2017). *Grad-CAM: Visual explanations from deep networks via
  gradient-based localization*. ICCV.
