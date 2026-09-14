"""
Gera (e opcionalmente executa) o notebook de demonstracao do projeto.

O notebook e construido a partir deste script para que fique versionado como
codigo, sempre reproduzivel e livre de saidas obsoletas coladas a mao.

Uso
---
    python scripts/build_notebook.py            # gera o .ipynb vazio
    python scripts/build_notebook.py --execute  # gera e executa (preenche as saidas)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_PATH = ROOT / "notebooks" / "01_analise_completa.ipynb"

MD = "markdown"
CODE = "code"

CELLS: list[tuple[str, str]] = [
    (MD, """\
# Tech Challenge - Fase 1
## Sistema inteligente de suporte ao diagnostico

**Tarefa principal:** classificacao de biopsias mamarias em **maligno** ou **benigno**
a partir de dados tabulares, usando algoritmos de aprendizado de maquina.

**Dataset:** Breast Cancer Wisconsin (Diagnostic) - 569 biopsias por puncao aspirativa
por agulha fina (PAAF), 30 atributos morfometricos extraidos digitalmente da imagem do
nucleo celular.

**Contexto clinico.** Um hospital universitario recebe mais exames do que a equipe de
patologia consegue laudar no mesmo dia. A proposta nao e substituir o patologista: e
**ordenar a fila** de forma que os casos com alta suspeita de malignidade sejam vistos
primeiro, e sinalizar explicitamente os casos de incerteza. A palavra final sobre o
diagnostico continua sendo, sempre, do medico responsavel.

> Este notebook e uma demonstracao narrada. O pipeline completo, automatizado e
> testado, vive em `scripts/run_pipeline.py` e no pacote `src/medai/`.
"""),
    (MD, "## 0. Preparacao do ambiente"),
    (CODE, """\
import sys, warnings
from pathlib import Path

ROOT = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
sys.path.insert(0, str(ROOT / "src"))
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from IPython.display import Image, display

from medai import data as data_mod
from medai import eda, evaluate, interpret, models
from medai.config import ensure_dirs, set_global_seed, RANDOM_STATE
from medai.preprocessing import (
    build_preprocessor, correlation_analysis, inject_missing, split_train_val_test,
)

ensure_dirs()
set_global_seed(RANDOM_STATE)
pd.set_option("display.width", 180)
pd.set_option("display.max_columns", 40)
print("ambiente pronto | semente =", RANDOM_STATE)"""),

    (MD, """\
## 1. Carga e auditoria da base

Antes de qualquer modelagem, auditamos a base: dimensoes, tipos, valores ausentes,
duplicatas, colunas constantes e **zeros biologicamente impossiveis** (raio ou area
iguais a zero seriam ausencia disfarcada de valor, um problema comum em bases medicas)."""),
    (CODE, """\
df = data_mod.load_raw()
print("dimensoes:", df.shape)
df.head()"""),
    (CODE, """\
auditoria = data_mod.audit(df)
for chave, valor in auditoria.items():
    print(f"{chave:.<28} {valor}")"""),
    (MD, """\
**Leitura.** A base esta limpa: nenhum valor ausente, nenhuma linha duplicada, nenhuma
coluna constante e nenhum zero impossivel. O desbalanceamento e leve (1,68:1 a favor
dos benignos) - nao chega a exigir reamostragem, mas **e suficiente para tornar a
acuracia uma metrica enganosa**: um modelo que responde sempre "benigno" ja acerta
62,7% dos casos sem detectar um unico cancer."""),
    (CODE, """\
descritivas = data_mod.describe_numeric(df)
descritivas[["mean", "std", "min", "max", "skew", "cv"]].head(12)"""),
    (MD, """\
**Leitura.** As escalas sao radicalmente diferentes: `area_worst` chega a ~4.250
enquanto `fractal_dimension_se` fica na casa de 0,003 - mais de tres ordens de grandeza.
Varias variaveis tambem tem assimetria positiva forte (`skew` > 1). Isso ja define duas
decisoes de pre-processamento: **padronizar a escala** (obrigatorio para KNN, SVM e para
a regularizacao da regressao logistica) e **imputar pela mediana**, nao pela media."""),

    (MD, "## 2. Analise exploratoria"),
    (CODE, """\
X, y = data_mod.split_features_target(df)
top_feats = eda.top_discriminative_features(df, y, k=9)
print("variaveis mais discriminantes:", top_feats)

display(Image(eda.plot_class_balance(df)))"""),
    (CODE, "display(Image(eda.plot_distributions(df, top_feats)))"),
    (MD, """\
**Leitura.** Para as variaveis do grupo `worst` (media dos 3 maiores nucleos de cada
lamina) as duas distribuicoes praticamente nao se sobrepoem: tumores malignos tem
nucleos maiores, mais irregulares e com mais pontos concavos no contorno. Isso e
coerente com o criterio histopatologico de atipia nuclear - o modelo vai aprender algo
que o patologista ja usa, e nao um artefato do dataset."""),
    (CODE, "display(Image(eda.plot_boxplots(df, top_feats)))"),
    (CODE, "display(Image(eda.plot_scale_and_skew(df)))"),
    (CODE, "display(Image(eda.plot_pca_projection(df)))"),
    (MD, """\
**Leitura.** Duas componentes principais ja concentram ~63% da variancia e separam
visualmente as classes. Isso sinaliza que o problema e **quase linearmente separavel** -
uma expectativa importante: se um modelo complexo nao superar a regressao logistica por
uma margem relevante, devemos ficar com o modelo simples."""),
    (CODE, "display(Image(eda.plot_pairwise(df, top_feats[:4])))"),

    (MD, """\
## 3. Analise de correlacao

Duas perguntas distintas: (a) quais variaveis se associam ao diagnostico e (b) quais
variaveis sao redundantes entre si (multicolinearidade)."""),
    (CODE, """\
corr = correlation_analysis(X, y, threshold=0.90)
display(Image(eda.plot_correlation_heatmap(corr.matrix)))"""),
    (CODE, """\
print(f"pares com |r| >= 0,90: {len(corr.redundant_pairs)}")
display(corr.redundant_pairs.head(10))
print("\\ncolunas redundantes sugeridas para descarte:")
print(corr.suggested_drop)"""),
    (CODE, "display(Image(eda.plot_target_correlation(corr.with_target)))"),
    (MD, """\
**Leitura.** Ha blocos de redundancia quase perfeita - `radius`, `perimeter` e `area`
medem a mesma geometria do nucleo (r > 0,98). Isso **nao prejudica** arvores, SVM ou
KNN, mas desestabiliza a interpretacao dos coeficientes da regressao logistica: o peso
de uma variavel pode "vazar" para a gemea. Por isso a interpretacao final se apoia em
permutacao e SHAP, e nao apenas nos coeficientes. Optamos por **manter as 30 variaveis**
e controlar a colinearidade via regularizacao L2, preservando a informacao."""),

    (MD, """\
## 4. Separacao treino / validacao / teste

Tres particoes estratificadas (60/20/20), com papeis rigidamente separados:

| particao | papel |
|---|---|
| **treino** (60%) | ajuste dos modelos e busca de hiperparametros por validacao cruzada |
| **validacao** (20%) | escolha do campeao e calibracao do limiar clinico |
| **teste** (20%) | aberto **uma unica vez**, ao final, para a estimativa honesta |"""),
    (CODE, """\
split = split_train_val_test(X, y)
split.summary()"""),

    (MD, """\
## 5. Pipeline de pre-processamento

Todo o pre-processamento vive dentro de um `sklearn.Pipeline`. Isso e o que impede
**vazamento de dados**: a media e o desvio usados na padronizacao sao aprendidos so no
treino e depois aplicados a validacao e ao teste.

O `ColumnTransformer` tem dois ramos automaticos: numerico (mediana + z-score) e
categorico (moda + one-hot com `handle_unknown="ignore"`). No WDBC as 30 preditoras sao
numericas, entao o ramo categorico fica vazio - mas ele e mantido porque o mesmo
pipeline atende outras bases clinicas (sexo, unidade, tipo de exame) sem alteracao de
codigo."""),
    (CODE, """\
pre = build_preprocessor(split.X_train)
pre.fit(split.X_train)
X_train_proc = pre.transform(split.X_train)
print("saida do pre-processador:", np.asarray(X_train_proc).shape)
pd.DataFrame(np.asarray(X_train_proc), columns=pre.get_feature_names_out()).iloc[:4, :6].round(3)"""),
    (CODE, """\
# Prova de que o ramo categorico funciona e de que a imputacao cobre valores ausentes.
demo = pd.DataFrame({
    "idade":   [45.0, 62.0, np.nan, 51.0],
    "sexo":    ["F", "F", "M", None],
    "unidade": ["HC", "HC", "UPA", "UPA"],
})
pre_demo = build_preprocessor(demo)
saida = pd.DataFrame(np.asarray(pre_demo.fit_transform(demo)),
                     columns=pre_demo.get_feature_names_out())
print("entrada com ausentes e categoricas -> saida numerica sem NaN:")
saida.round(3)"""),

    (MD, """\
## 6. Modelagem

Seis familias de modelos, cada uma com uma grade de hiperparametros buscada por
`GridSearchCV` com `StratifiedKFold` de 5 dobras **apenas no conjunto de treino**.

A metrica de *refit* da busca e a **ROC AUC**, que independe do limiar de decisao.
Otimizar recall diretamente levaria a solucao degenerada de classificar todos como
malignos (recall 1,0, clinicamente inutil). O limiar e calibrado depois, a parte."""),
    (CODE, """\
for spec in models.model_catalog():
    print(f"{spec.label}\\n  {spec.rationale}\\n")"""),
    (CODE, """\
%%time
treinados = models.train_all(split.X_train, split.y_train)"""),
    (CODE, """\
cv = models.cv_summary(treinados)
cv[["modelo", "roc_auc", "roc_auc_std", "recall", "f1", "accuracy", "overfit_gap_roc_auc"]]"""),

    (MD, """\
## 7. Selecao do campeao

Escolher pela ROC AUC da validacao sozinha seria fragil: com 114 exames (43 malignos) o
erro-padrao da AUC beira +/- 0,02 e o ranking muda com a semente. Por isso o criterio e a
**media entre a ROC AUC da validacao cruzada no treino (estabilidade) e a ROC AUC do
holdout de validacao (confirmacao independente)**. O teste continua intocado."""),
    (CODE, """\
val_scores, val_results = {}, {}
for chave, tm in treinados.items():
    s = models.predict_scores(tm.pipeline, split.X_val)
    val_scores[chave] = s
    val_results[tm.label] = evaluate.compute_metrics(split.y_val, s)

tabela_val = evaluate.metrics_table(val_results)
tabela_val"""),
    (CODE, """\
ranking = pd.DataFrame([
    {
        "modelo": tm.label,
        "chave": chave,
        "roc_auc_cv_treino": round(tm.cv_scores["roc_auc"], 4),
        "roc_auc_validacao": round(val_results[tm.label]["roc_auc"], 4),
        "score_selecao": round(0.5 * tm.cv_scores["roc_auc"]
                               + 0.5 * val_results[tm.label]["roc_auc"], 5),
    }
    for chave, tm in treinados.items()
]).sort_values("score_selecao", ascending=False).set_index("chave")

campeao_chave = ranking.index[0]
campeao = treinados[campeao_chave]
print("MODELO CAMPEAO:", campeao.label, "| melhores parametros:", campeao.best_params)
ranking"""),
    (CODE, """\
display(Image(evaluate.plot_model_comparison(tabela_val)))
display(Image(evaluate.plot_roc_pr(
    {tm.label: (split.y_val, val_scores[k]) for k, tm in treinados.items()}
)))"""),

    (MD, """\
## 8. Escolha da metrica e calibracao do limiar

Os dois erros tem custos assimetricos:

- **Falso negativo** (cancer classificado como benigno): o paciente e liberado, o tumor
  evolui e a janela terapeutica se fecha. Custo potencial: a vida.
- **Falso positivo** (benigno marcado como suspeito): gera ansiedade e uma biopsia
  confirmatoria. E caro e desagradavel, mas **reversivel**.

Logo a metrica que governa a operacao e o **recall da classe maligna**, com F1 e
precisao de contrapeso para o sistema nao virar um alarme permanente. O limiar deixa de
ser o 0,50 default e passa a ser um **parametro clinico**: fixamos a sensibilidade
minima aceitavel e, entre todos os limiares que a atendem, ficamos com o mais alto -
o que minimiza biopsias desnecessarias."""),
    (CODE, """\
escolha = evaluate.tune_threshold(split.y_val, val_scores[campeao_chave], target_recall=0.99)
print(f"limiar clinico = {escolha.threshold:.4f}")
print(f"  recall  na validacao = {escolha.achieved_recall:.4f}")
print(f"  precisao na validacao = {escolha.achieved_precision:.4f}")
display(Image(evaluate.plot_threshold_analysis(escolha)))"""),

    (MD, """\
## 9. Avaliacao final no conjunto de teste

O campeao e reajustado em treino + validacao (para aproveitar todos os dados
disponiveis) com os hiperparametros e o limiar **ja congelados** - portanto sem
vazamento - e so entao o teste e aberto, uma unica vez."""),
    (CODE, """\
pipeline_final = models.build_pipeline(campeao.spec, split.X_trainval)
pipeline_final.set_params(**{f"model__{k}": v for k, v in campeao.best_params.items()})
pipeline_final.fit(split.X_trainval, split.y_trainval)

scores_teste = models.predict_scores(pipeline_final, split.X_test)
evaluate.metrics_table({
    f"{campeao.label} @ limiar padrao 0.50": evaluate.compute_metrics(split.y_test, scores_teste, 0.50),
    f"{campeao.label} @ limiar clinico {escolha.threshold:.3f}":
        evaluate.compute_metrics(split.y_test, scores_teste, escolha.threshold),
})"""),
    (CODE, """\
display(Image(evaluate.plot_confusion(split.y_test, scores_teste, 0.50,
        f"Matriz de confusao - {campeao.label} (teste)", "nb_confusao_padrao")))
display(Image(evaluate.plot_confusion(split.y_test, scores_teste, escolha.threshold,
        f"Matriz de confusao - {campeao.label} (teste, limiar clinico)", "nb_confusao_clinica")))"""),
    (CODE, """\
display(Image(evaluate.plot_score_distribution(split.y_test, scores_teste, escolha.threshold)))
display(Image(evaluate.plot_calibration(split.y_test, scores_teste, campeao.label)))"""),
    (MD, """\
**Leitura.** Baixar o limiar de 0,50 para o valor calibrado troca falsos negativos por
falsos positivos - exatamente a troca que a clinica quer fazer. Cada falso negativo
evitado custa algumas biopsias confirmatorias a mais, e essa e uma decisao **do comite
clinico**, nao do cientista de dados: o codigo apenas expoe o botao."""),
    (CODE, """\
erros = evaluate.error_analysis(split.X_test, split.y_test, scores_teste,
                                escolha.threshold, top_feats)
print(f"exames classificados incorretamente no teste: {len(erros)}")
erros"""),

    (MD, """\
## 10. Interpretabilidade

Um modelo que nao explica a propria decisao nao e adotavel em medicina. Tres lentes
complementares: **importancia por permutacao** (o que o modelo realmente usa),
**SHAP global** (magnitude e direcao) e **SHAP individual** (por que ESTE paciente
foi sinalizado)."""),
    (CODE, """\
perm = interpret.permutation_report(pipeline_final, split.X_test, split.y_test, n_repeats=20)
display(Image(interpret.plot_permutation_importance(perm, campeao.label)))
perm.head(10)"""),
    (CODE, """\
shap_res = interpret.compute_shap(pipeline_final, split.X_trainval, split.X_test,
                                  tree_based=campeao.spec.tree_based)
display(Image(interpret.plot_shap_bar(shap_res, campeao.label)))
display(Image(interpret.plot_shap_summary(shap_res, campeao.label)))"""),
    (CODE, """\
casos = interpret.pick_cases(split.y_test, scores_teste, escolha.threshold)
print("casos selecionados para explicacao individual:", casos)

for nome, pos in casos.items():
    display(Image(interpret.plot_shap_waterfall(
        shap_res, pos,
        f"{nome} | probabilidade prevista = {scores_teste[pos]:.3f}",
        f"nb_shap_{nome}",
    )))"""),
    (MD, """\
**Leitura.** O grafico em cascata e o que efetivamente entra no laudo de apoio: mostra,
para um paciente especifico, quais medidas empurraram o score para cima e quais o
puxaram para baixo. E isso que permite ao medico **discordar de forma fundamentada** -
por exemplo, ao notar que o modelo se apoiou em uma medida afetada por um artefato de
preparo da lamina."""),
    (CODE, """\
# Modelo de referencia "caixa de vidro": razoes de chance sao a linguagem clinica.
ref_spec = next(s for s in models.model_catalog() if s.key == "logistic_regression")
ref = models.build_pipeline(ref_spec, split.X_trainval)
ref.set_params(model__C=1.0)
ref.fit(split.X_trainval, split.y_trainval)
interpret.native_importance(ref).head(10)"""),

    (MD, """\
## 11. Robustez a dados ausentes

O WDBC nao tem valores faltantes, o que deixaria o ramo de imputacao sem exercicio. Um
prontuario hospitalar real, porem, chega incompleto. Injetamos ausencias MCAR para medir
a degradacao."""),
    (CODE, """\
linhas = []
for frac in (0.0, 0.05, 0.10, 0.20):
    Xm = split.X_test if frac == 0 else inject_missing(split.X_test, frac)
    m = evaluate.compute_metrics(split.y_test, models.predict_scores(pipeline_final, Xm),
                                 escolha.threshold)
    linhas.append({"ausentes": f"{frac:.0%}", "recall": round(m["recall"], 4),
                   "precisao": round(m["precision"], 4), "f1": round(m["f1"], 4),
                   "roc_auc": round(m["roc_auc"], 4), "falsos_negativos": m["fn"]})
pd.DataFrame(linhas).set_index("ausentes")"""),

    (MD, """\
## 12. Discussao critica - o modelo pode ser usado na pratica?

**Sim, mas com escopo restrito e como apoio, nunca como decisor.**

**Como usar.** Um fluxo de triagem em tres faixas, nao um rotulo binario:

| faixa | criterio | conduta |
|---|---|---|
| VERDE | score abaixo do limiar clinico | segue a fila de rotina |
| AMARELO | entre o limiar e 0,90 | revisao humana prioritaria (zona de incerteza) |
| VERMELHO | score >= 0,90 | encaminhamento imediato ao mastologista |

O ganho nao e "acertar o diagnostico" - e **reordenar a fila** para que os casos graves
sejam vistos primeiro e para que a incerteza seja explicitada em vez de escondida atras
de um rotulo.

**Limitacoes honestas.**

1. **Amostra pequena e de uma unica fonte.** 569 biopsias de um unico centro (Wisconsin,
   anos 1990). Um intervalo de confianca sobre 114 exames de teste e largo; qualquer
   metrica aqui precisa ser revalidada prospectivamente na populacao do hospital.
2. **As variaveis nao caem do ceu.** As 30 medidas pressupoem segmentacao digital do
   nucleo celular a partir da lamina. Sem essa etapa padronizada e calibrada, o modelo
   nao tem entrada valida - e desvio de calibracao entre equipamentos degrada o
   desempenho silenciosamente.
3. **Ausencia de dados demograficos.** Nao ha idade, etnia ou historico familiar, entao
   nao e possivel auditar o modelo por subgrupo. Sem essa auditoria nao se pode afirmar
   que ele e igualmente seguro para todas as pacientes.
4. **Deslocamento de distribuicao.** Novo microscopio, novo protocolo de coloracao ou
   nova equipe mudam a distribuicao das medidas. Exige monitoramento continuo do score e
   das taxas de alarme.

**Requisitos para uma implantacao responsavel:** validacao prospectiva local; limiar
homologado pelo comite clinico (e nao pelo time de dados); registro auditavel de toda
predicao com sua explicacao SHAP; monitoramento de deriva; e um botao explicito de
discordancia para o medico, cujo uso realimenta o proximo ciclo de treino.

> **O modelo prioriza. O medico diagnostica.**"""),
]


def build():
    import nbformat as nbf

    nb = nbf.v4.new_notebook()
    nb.cells = [
        nbf.v4.new_markdown_cell(src) if kind == MD else nbf.v4.new_code_cell(src)
        for kind, src in CELLS
    ]
    nb.metadata.update(
        {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {"name": "python", "version": sys.version.split()[0]},
        }
    )
    NOTEBOOK_PATH.parent.mkdir(parents=True, exist_ok=True)
    nbf.write(nb, NOTEBOOK_PATH)
    print(f"notebook gerado: {NOTEBOOK_PATH} ({len(nb.cells)} celulas)")
    return nb


def execute(nb):
    from nbclient import NotebookClient
    import nbformat as nbf

    client = NotebookClient(
        nb,
        timeout=2400,
        kernel_name="python3",
        resources={"metadata": {"path": str(NOTEBOOK_PATH.parent)}},
    )
    client.execute()
    nbf.write(nb, NOTEBOOK_PATH)
    print(f"notebook executado e salvo: {NOTEBOOK_PATH}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--execute", action="store_true")
    ns = parser.parse_args()
    notebook = build()
    if ns.execute:
        execute(notebook)
