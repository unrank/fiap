"""
Interpretabilidade do modelo: importancia nativa, por permutacao e SHAP.

Em medicina, um modelo que nao explica a propria decisao nao e adotavel. Um
laudo de apoio precisa dizer *por que* aquele exame foi marcado como suspeito
para que o medico possa concordar, discordar ou pedir um exame adicional.
Usamos tres lentes complementares:

* **Importancia nativa**   - coeficientes (regressao logistica) ou reducao de
  impureza (arvores). Barata, mas enviesada por multicolinearidade.
* **Importancia por permutacao** - embaralha uma variavel por vez e mede a
  queda real de desempenho no conjunto de validacao. Independe do modelo e
  responde "o que o modelo de fato usa".
* **SHAP** - decompoe cada previsao individual na contribuicao de cada
  variavel (valores de Shapley). E a unica das tres que explica **um paciente
  especifico**, que e o que o medico tem a frente.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.pipeline import Pipeline

from .config import CLASS_NAMES, PLOT, RANDOM_STATE
from .utils import apply_plot_style, get_logger, save_fig

log = get_logger()


# ---------------------------------------------------------------------------
# Importancia nativa do estimador
# ---------------------------------------------------------------------------
def native_importance(pipeline: Pipeline) -> pd.DataFrame | None:
    """
    Extrai coeficientes ou ``feature_importances_`` do estimador final.

    Devolve ``None`` para modelos sem importancia nativa (KNN, SVM-RBF).
    """
    model = pipeline.named_steps["model"]
    names = list(pipeline.named_steps["preprocessor"].get_feature_names_out())

    if hasattr(model, "coef_"):
        coefs = np.ravel(model.coef_)
        df = pd.DataFrame({"feature": names, "coeficiente": coefs})
        # exp(coef) sobre variaveis padronizadas = razao de chances por
        # aumento de 1 desvio-padrao na variavel.
        df["odds_ratio_por_dp"] = np.exp(coefs)
        df["importancia"] = np.abs(coefs)
        return df.sort_values("importancia", ascending=False).reset_index(drop=True)

    if hasattr(model, "feature_importances_"):
        df = pd.DataFrame(
            {"feature": names, "importancia": model.feature_importances_}
        )
        return df.sort_values("importancia", ascending=False).reset_index(drop=True)

    return None


# ---------------------------------------------------------------------------
# Importancia por permutacao
# ---------------------------------------------------------------------------
def permutation_report(
    pipeline: Pipeline,
    X: pd.DataFrame,
    y: pd.Series,
    scoring: str = "roc_auc",
    n_repeats: int = 30,
    random_state: int = RANDOM_STATE,
) -> pd.DataFrame:
    """Importancia por permutacao calculada sobre as features originais."""
    result = permutation_importance(
        pipeline, X, y, scoring=scoring, n_repeats=n_repeats,
        random_state=random_state, n_jobs=-1,
    )
    df = pd.DataFrame(
        {
            "feature": X.columns,
            "queda_media": result.importances_mean,
            "desvio": result.importances_std,
        }
    )
    return df.sort_values("queda_media", ascending=False).reset_index(drop=True)


def plot_permutation_importance(
    df: pd.DataFrame,
    model_name: str,
    top_n: int = 15,
    filename: str = "19_importancia_permutacao",
):
    """Barras horizontais da queda de ROC AUC ao embaralhar cada variavel."""
    import matplotlib.pyplot as plt

    apply_plot_style()
    data = df.head(top_n).iloc[::-1]

    fig, ax = plt.subplots(figsize=(10, 0.42 * len(data) + 2.2))
    bars = ax.barh(
        data["feature"], data["queda_media"], xerr=data["desvio"],
        color=PLOT.benign_color, error_kw={"ecolor": "#555555", "elinewidth": 1},
    )
    ax.bar_label(bars, fmt="%.4f", fontsize=8, padding=8)
    ax.set(
        xlabel="queda media da ROC AUC ao embaralhar a variavel",
        title=f"Importancia por permutacao - {model_name}\n"
              f"(quanto o modelo realmente perde sem cada variavel)",
    )
    ax.tick_params(labelsize=8.5)
    return save_fig(fig, filename)


def plot_native_importance(
    df: pd.DataFrame,
    model_name: str,
    top_n: int = 15,
    filename: str = "17_importancia_nativa",
):
    """Importancia nativa (coeficientes ou ganho de impureza)."""
    import matplotlib.pyplot as plt

    apply_plot_style()
    data = df.head(top_n).iloc[::-1]
    has_sign = "coeficiente" in data.columns
    values = data["coeficiente"] if has_sign else data["importancia"]
    colors = (
        [PLOT.malignant_color if v > 0 else PLOT.benign_color for v in values]
        if has_sign
        else PLOT.benign_color
    )

    fig, ax = plt.subplots(figsize=(10, 0.42 * len(data) + 2.2))
    bars = ax.barh(data["feature"], values, color=colors)
    ax.bar_label(bars, fmt="%.3f", fontsize=8, padding=3)
    if has_sign:
        ax.axvline(0, color="black", lw=1)
        xlabel = "coeficiente (positivo -> empurra para maligno)"
    else:
        xlabel = "importancia (reducao media de impureza)"
    ax.set(xlabel=xlabel, title=f"Importancia nativa - {model_name}")
    ax.tick_params(labelsize=8.5)
    return save_fig(fig, filename)


# ---------------------------------------------------------------------------
# SHAP
# ---------------------------------------------------------------------------
@dataclass
class ShapResult:
    """Valores SHAP ja normalizados para a classe positiva (maligno)."""

    values: np.ndarray            # (n_amostras, n_features)
    base_value: float
    data: pd.DataFrame            # features ja pre-processadas
    feature_names: list[str]

    def importance(self) -> pd.DataFrame:
        """Importancia global = media do |SHAP| por variavel."""
        return (
            pd.DataFrame(
                {
                    "feature": self.feature_names,
                    "shap_medio_abs": np.abs(self.values).mean(axis=0),
                }
            )
            .sort_values("shap_medio_abs", ascending=False)
            .reset_index(drop=True)
        )


def compute_shap(
    pipeline: Pipeline,
    X_background: pd.DataFrame,
    X_explain: pd.DataFrame,
    tree_based: bool,
    max_background: int = 100,
    random_state: int = RANDOM_STATE,
) -> ShapResult:
    """
    Calcula os valores SHAP do modelo para ``X_explain``.

    Modelos de arvore usam o ``TreeExplainer`` (exato e rapido); os demais
    caem no explicador por permutacao, alimentado com uma amostra reduzida
    do treino como distribuicao de referencia.
    """
    import shap

    pre = pipeline.named_steps["preprocessor"]
    model = pipeline.named_steps["model"]

    bg = pre.transform(X_background)
    Xt = pre.transform(X_explain)
    names = list(pre.get_feature_names_out())
    bg = pd.DataFrame(np.asarray(bg), columns=names)
    Xt = pd.DataFrame(np.asarray(Xt), columns=names, index=X_explain.index)

    if tree_based:
        explainer = shap.TreeExplainer(model, feature_perturbation="tree_path_dependent")
        explanation = explainer(Xt, check_additivity=False)
    else:
        sample = bg.sample(min(max_background, len(bg)), random_state=random_state)
        explainer = shap.Explainer(
            lambda d: model.predict_proba(d)[:, 1], sample, algorithm="permutation"
        )
        explanation = explainer(Xt, max_evals=2 * len(names) + 1)

    values = np.asarray(explanation.values)
    base = np.asarray(explanation.base_values)

    # Classificadores podem devolver (n, features, n_classes): pegamos a
    # classe positiva (maligno).
    if values.ndim == 3:
        values = values[:, :, 1]
        base = base[:, 1] if base.ndim == 2 else base
    base_value = float(np.ravel(base)[0]) if np.size(base) else 0.0

    log.info("SHAP calculado para %d exames x %d variaveis", *values.shape)
    return ShapResult(values=values, base_value=base_value, data=Xt, feature_names=names)


def plot_shap_summary(
    result: ShapResult,
    model_name: str,
    max_display: int = 15,
    filename: str = "20_shap_beeswarm",
):
    """Beeswarm: direcao e magnitude do efeito de cada variavel."""
    import warnings

    import matplotlib.pyplot as plt
    import shap

    apply_plot_style()
    plt.figure(figsize=(10, 0.42 * max_display + 2.4))
    # O beeswarm do shap usa o RNG global do numpy para espalhar os pontos;
    # em numpy >= 2.5 isso emite um FutureWarning que nao afeta o resultado.
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=FutureWarning)
        shap.summary_plot(
            result.values, result.data, feature_names=result.feature_names,
            max_display=max_display, show=False, plot_size=None,
        )
    fig = plt.gcf()
    fig.suptitle(
        f"SHAP - contribuicao individual de cada variavel ({model_name})\n"
        "vermelho = valor alto da variavel | direita = empurra para MALIGNO",
        fontsize=12, fontweight="bold", y=1.02,
    )
    return save_fig(fig, filename)


def plot_shap_bar(
    result: ShapResult,
    model_name: str,
    max_display: int = 15,
    filename: str = "21_shap_importancia_global",
):
    """Importancia global SHAP (media do valor absoluto)."""
    import matplotlib.pyplot as plt

    apply_plot_style()
    data = result.importance().head(max_display).iloc[::-1]

    fig, ax = plt.subplots(figsize=(10, 0.42 * len(data) + 2.2))
    bars = ax.barh(data["feature"], data["shap_medio_abs"], color=PLOT.malignant_color)
    ax.bar_label(bars, fmt="%.4f", fontsize=8, padding=3)
    ax.set(
        xlabel="media de |valor SHAP| (impacto medio na probabilidade prevista)",
        title=f"Importancia global por SHAP - {model_name}",
    )
    ax.tick_params(labelsize=8.5)
    return save_fig(fig, filename)


def plot_shap_waterfall(
    result: ShapResult,
    row_position: int,
    title: str,
    filename: str,
    max_display: int = 12,
):
    """Explicacao individual: como o modelo chegou ao laudo de UM paciente."""
    import matplotlib.pyplot as plt
    import shap

    apply_plot_style()
    explanation = shap.Explanation(
        values=result.values[row_position],
        base_values=result.base_value,
        data=result.data.iloc[row_position].to_numpy(),
        feature_names=result.feature_names,
    )
    plt.figure(figsize=(10, 0.45 * max_display + 2.2))
    shap.plots.waterfall(explanation, max_display=max_display, show=False)
    fig = plt.gcf()
    fig.suptitle(title, fontsize=12, fontweight="bold", y=1.02)
    return save_fig(fig, filename)


def pick_cases(
    y_true: pd.Series,
    y_scores: np.ndarray,
    threshold: float,
) -> dict[str, int]:
    """
    Seleciona posicoes representativas para as explicacoes individuais.

    Escolhe o falso negativo mais grave (se houver), um falso positivo, o
    verdadeiro positivo mais confiante e o caso mais ambiguo (score proximo
    do limiar) - os quatro perfis que o medico precisa entender.
    """
    y = np.asarray(y_true).astype(int)
    pred = (y_scores >= threshold).astype(int)
    cases: dict[str, int] = {}

    fn = np.where((y == 1) & (pred == 0))[0]
    if fn.size:
        cases["falso_negativo"] = int(fn[np.argmin(y_scores[fn])])

    fp = np.where((y == 0) & (pred == 1))[0]
    if fp.size:
        cases["falso_positivo"] = int(fp[np.argmax(y_scores[fp])])

    tp = np.where((y == 1) & (pred == 1))[0]
    if tp.size:
        cases["verdadeiro_positivo"] = int(tp[np.argmax(y_scores[tp])])

    cases["caso_ambiguo"] = int(np.argmin(np.abs(y_scores - threshold)))
    return cases
