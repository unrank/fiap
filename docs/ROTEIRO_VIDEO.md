# Roteiro do vídeo de demonstração — Tech Challenge Fase 1

**Limite:** 15 minutos. **Onde publicar:** YouTube ou Vimeo, público ou **não listado**.
**O que o avaliador precisa ver:** o sistema em execução + explicação breve do fluxo.

Sugestão de divisão (13 min de conteúdo + folga):

---

## 0 · Abertura — 0:00 a 0:45

Mostre a tela do repositório aberto.

> "Fase 1 do Tech Challenge: a base de um sistema de IA de apoio à triagem de exames
> para um hospital universitário. A tarefa principal é classificar biópsias mamárias em
> maligno ou benigno a partir de dados tabulares; o entregável extra é uma CNN que detecta
> pneumonia em radiografias de tórax.
>
> A frase que organiza o projeto inteiro é: **o modelo prioriza, o médico diagnostica.**
> Nada aqui substitui o laudo — o sistema ordena a fila e explica o próprio raciocínio."

---

## 1 · O problema e o dataset — 0:45 a 2:30

Abra `README.md` na seção **4. Dados**.

- 569 biópsias, 30 medidas morfométricas do núcleo celular, alvo M/B.
- **Por que este dataset:** as variáveis têm significado clínico direto — são medidas de
  atipia nuclear, exatamente o que o patologista avalia. Isso torna as explicações
  criticáveis pela equipe médica.
- Mostre `reports/metrics/01_auditoria_dataset.json`: 0 ausentes, 0 duplicadas, 0 zeros
  impossíveis, desbalanceamento 1,68:1.

> "Esse desbalanceamento já derruba a acurácia como métrica: responder sempre 'benigno'
> acerta 62,7% sem detectar um único câncer."

---

## 2 · Estrutura do projeto — 2:30 a 3:30

Percorra a árvore de pastas rapidamente (`src/medai/`, `scripts/`, `tests/`, `reports/`).

- `src/medai/` é o pacote: dados, EDA, pré-processamento, modelos, avaliação,
  interpretabilidade e CNN.
- `scripts/run_pipeline.py` roda tudo ponta a ponta.
- `tests/` tem 15 testes que protegem contra vazamento de dados.

---

## 3 · Execução ao vivo do pipeline — 3:30 a 6:00

```bash
python scripts/run_pipeline.py
```

Enquanto roda, narre as 11 etapas que aparecem no log. Se preferir não esperar os ~4
minutos, use `--quick` **e diga em voz alta que é o modo rápido**, mostrando em seguida os
resultados da execução completa já salvos em `reports/`.

Aponte no log:
- a separação estratificada 341 / 114 / 114;
- a busca de hiperparâmetros dos 6 modelos;
- a linha do **MODELO CAMPEÃO**;
- o **limiar calibrado**;
- o resumo final com recall, F1 e falsos negativos.

---

## 4 · Exploração e pré-processamento — 6:00 a 8:00

Abra as figuras em `reports/figures/`:

| figura | o que dizer |
|---|---|
| `01_balanceamento_classes.png` | o desbalanceamento e por que a acurácia engana |
| `02_distribuicoes_por_classe.png` | as classes se separam nas variáveis `_worst` |
| `08_escala_e_assimetria.png` | escalas diferem em 3 ordens de grandeza → padronizar |
| `04_matriz_correlacao.png` | 21 pares com \|r\| ≥ 0,90 — raio, perímetro e área medem a mesma geometria |

> "Todo o pré-processamento vive dentro de um `Pipeline` do scikit-learn. Isso não é
> estilo: é o que impede vazamento de dados. A média e o desvio da padronização são
> aprendidos só no treino."

Mostre o teste `test_escala_e_aprendida_somente_no_treino` em `tests/test_pipeline.py`.

---

## 5 · Modelagem e escolha do campeão — 8:00 a 10:00

Abra `reports/metrics/08b_ranking_selecao_campeao.csv`.

- Seis modelos, cada um testando uma hipótese diferente sobre a fronteira de decisão.
- **O ponto forte para destacar:** escolher pela validação sozinha é frágil — com 114
  exames o erro-padrão da AUC beira ±0,02. O critério usa a média entre a validação
  cruzada no treino e o holdout de validação.
- Regressão Logística venceu por 0,0002 sobre o SVM: **empate técnico**. Ganha o modelo
  que a equipe clínica consegue auditar.

Mostre `09_comparacao_modelos.png` e `10_curvas_roc_pr.png`.

---

## 6 · Métrica e limiar clínico — 10:00 a 11:30

Abra `11_analise_limiar.png`.

> "Falso negativo é câncer não detectado — irreversível. Falso positivo é uma biópsia a
> mais — desagradável, mas reversível. Por isso a métrica que governa é o recall.
>
> E o limiar de 0,50 não tem nada de especial: é convenção matemática, não decisão médica.
> Nós fixamos a sensibilidade mínima exigida e escolhemos o maior limiar que a atende."

Mostre as duas matrizes de confusão lado a lado (`12_` e `13_`): **1 falso negativo a
menos ao custo de 6 falsos positivos**. Diga que essa troca é decisão do comitê clínico.

---

## 7 · Interpretabilidade — 11:30 a 13:00

- `20_shap_beeswarm.png` — magnitude e direção de cada variável.
- `22_1_shap_caso_falso_negativo.png` — a explicação individual de um paciente.

> "É esse gráfico que entra no laudo de apoio. Ele permite ao médico discordar de forma
> fundamentada — por exemplo, notando que o modelo se apoiou numa medida afetada por
> artefato de preparo da lâmina."

Rode a inferência ao vivo:

```bash
python scripts/predict.py --demo
```

Mostre as faixas VERDE / AMARELO / VERMELHO e o rodapé do laudo.

---

## 8 · EXTRA: CNN + Grad-CAM — 13:00 a 14:15

Mostre `31_cnn_curvas_treino.png`, `32_cnn_matriz_confusao.png` e `34_cnn_gradcam.png`.

- CNN de 295 mil parâmetros, treinada do zero em CPU.
- Recall 0,987 no teste, ROC AUC 0,971.
- **Grad-CAM é o SHAP das imagens:** confirma que a rede olhou o campo pulmonar, e não uma
  borda ou um marcador metálico.
- Cite a queda de AUC de 0,997 (validação) para 0,971 (teste): deslocamento de
  distribuição, não overfitting.

---

## 9 · Fechamento crítico — 14:15 a 15:00

> "Pode ser usado na prática? Sim, como apoio e com escopo restrito. As limitações são
> reais: 569 biópsias de um único centro dos anos 1990, sem dados demográficos — o que
> impede auditar o modelo por subgrupo. Antes de qualquer uso: validação prospectiva
> local, limiar homologado pelo comitê clínico, registro auditável de cada predição e
> monitoramento de deriva.
>
> O modelo prioriza. O médico diagnostica."

---

## Checklist antes de gravar

- [ ] `python scripts/run_pipeline.py` executado, com `reports/` populado
- [ ] `python scripts/run_cnn.py --epochs 15 --size 64` executado
- [ ] `pytest -q` passando (15 testes)
- [ ] Notebook `notebooks/01_analise_completa.ipynb` com as saídas salvas
- [ ] Terminal com fonte grande e tema legível
- [ ] Repositório publicado e o link colado na capa do PDF
- [ ] PDF gerado com `python scripts/build_pdf.py --repo <URL> --video <URL> --autores "..."`
