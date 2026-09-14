"""
Avaliacao dos modelos: metricas, curvas, matriz de confusao e limiar de decisao.

Escolha da metrica principal
----------------------------
Em triagem oncologica os dois erros tem custos radicalmente diferentes:

* **Falso negativo** (maligno classificado como benigno) - o pior desfecho
  possivel. O paciente e liberado, o tumor evolui e a janela terapeutica se
  fecha. O custo e a propria vida.
* **Falso positivo** (benigno classificado como maligno) - gera ansiedade e
  um exame confirmatorio (biopsia excisional). E caro e desagradavel, mas
  reversivel.

Por isso a metrica que governa a decisao e o **recall (sensibilidade) da
classe maligna**, com o F1-score e a precisao servindo de contrapeso para que
o modelo nao vire um alarme permanente. A **acuracia isolada e enganosa**:
como 62,7% da base e benigna, um classificador que sempre responde "benigno"
ja acerta 62,7% - e nao detecta nenhum cancer.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

from .config import CLASS_NAMES, PLOT, TARGET_RECALL
from .utils import apply_plot_style, get_logger, save_fig

log = get_logger()

DEFAULT_THRESHOLD = 0.50


# ---------------------------------------------------------------------------
# Metricas
# ---------------------------------------------------------------------------
def compute_metrics(
    y_true: np.ndarray | pd.Series,
    y_scores: np.ndarray,
    threshold: float = DEFAULT_THRESHOLD,
) -> dict[str, float]:
    """Calcula o conjunto completo de metricas para um limiar dado."""
    y_true = np.asarray(y_true).astype(int)
    y_pred = (y_scores >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    return {
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "specificity": float(tn / (tn + fp)) if (tn + fp) else 0.0,
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "mcc": float(matthews_corrcoef(y_true, y_pred)),
        "roc_auc": float(roc_auc_score(y_true, y_scores)),
        "pr_auc": float(average_precision_score(y_true, y_scores)),
        "brier": float(brier_score_loss(y_true, y_scores)),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
        "n": int(len(y_true)),
    }


def metrics_table(results: dict[str, dict[str, float]]) -> pd.DataFrame:
    """Converte ``{modelo: metricas}`` em uma tabela ordenada por recall/F1."""
    df = pd.DataFrame(results).T
    cols = [
        "recall",
        "f1",
        "precision",
        "specificity",
        "accuracy",
        "balanced_accuracy",
        "roc_auc",
        "pr_auc",
        "mcc",
        "brier",
        "threshold",
        "tp",
        "fn",
        "fp",
        "tn",
    ]
    cols = [c for c in cols if c in df.columns]
    df = df[cols]
    numeric = df.columns.difference(["tp", "fn", "fp", "tn", "n"])
    df[numeric] = df[numeric].astype(float).round(4)
    for c in ("tp", "fn", "fp", "tn"):
        if c in df.columns:
            df[c] = df[c].astype(int)
    return df.sort_values(["recall", "f1"], ascending=False)


# ---------------------------------------------------------------------------
# Calibracao do limiar de decisao
# ---------------------------------------------------------------------------
@dataclass
class ThresholdChoice:
    """Limiar escolhido e o que ele custa em termos de precisao."""

    threshold: float
    achieved_recall: float
    achieved_precision: float
    achieved_f1: float
    target_recall: float
    curve: pd.DataFrame = field(repr=False, default_factory=pd.DataFrame)


def tune_threshold(
    y_true: np.ndarray | pd.Series,
    y_scores: np.ndarray,
    target_recall: float = TARGET_RECALL,
) -> ThresholdChoice:
    """
    Escolhe o **maior** limiar que ainda garante ``target_recall``.

    Racional clinico: fixamos primeiro a sensibilidade minima aceitavel para
    nao perder casos malignos; entre todos os limiares que a atendem, ficamos
    com o mais alto, porque ele minimiza os falsos positivos (menos biopsias
    desnecessarias). Isso e o oposto de "chutar 0,5": o limiar vira um
    parametro clinico negociado com a equipe medica.
    """
    y_true = np.asarray(y_true).astype(int)
    grid = np.unique(np.round(np.concatenate([y_scores, [0.0, 1.0]]), 6))

    rows = []
    for thr in grid:
        y_pred = (y_scores >= thr).astype(int)
        rows.append(
            {
                "threshold": float(thr),
                "recall": recall_score(y_true, y_pred, zero_division=0),
                "precision": precision_score(y_true, y_pred, zero_division=0),
                "f1": f1_score(y_true, y_pred, zero_division=0),
                "specificity": (
                    ((y_pred == 0) & (y_true == 0)).sum() / max((y_true == 0).sum(), 1)
                ),
                "alarmes_%": 100.0 * y_pred.mean(),
            }
        )
    curve = pd.DataFrame(rows)

    eligible = curve[curve["recall"] >= target_recall]
    if eligible.empty:  # nenhum limiar atinge a meta -> usamos o mais sensivel
        log.warning(
            "nenhum limiar atinge recall >= %.2f; usando o limiar de recall maximo",
            target_recall,
        )
        best = curve.loc[curve["recall"].idxmax()]
    else:
        best = eligible.loc[eligible["threshold"].idxmax()]

    log.info(
        "limiar calibrado = %.4f | recall = %.4f | precisao = %.4f (meta de recall %.2f)",
        best["threshold"],
        best["recall"],
        best["precision"],
        target_recall,
    )
    return ThresholdChoice(
        threshold=float(best["threshold"]),
        achieved_recall=float(best["recall"]),
        achieved_precision=float(best["precision"]),
        achieved_f1=float(best["f1"]),
        target_recall=target_recall,
        curve=curve,
    )


def triage_bands(
    y_scores: np.ndarray,
    low: float,
    high: float,
) -> pd.Series:
    """
    Classifica cada exame em uma das tres faixas do fluxo de triagem.

    * ``verde``    - risco baixo, entra na fila de rotina;
    * ``amarelo``  - zona cinzenta, revisao humana prioritaria;
    * ``vermelho`` - alta suspeita, encaminhamento imediato.
    """
    bands = np.where(y_scores >= high, "vermelho", np.where(y_scores >= low, "amarelo", "verde"))
    return pd.Series(bands, name="faixa_triagem")


# ---------------------------------------------------------------------------
# Figuras
# ---------------------------------------------------------------------------
def plot_confusion(
    y_true: np.ndarray | pd.Series,
    y_scores: np.ndarray,
    threshold: float,
    title: str,
    filename: str,
):
    """Matriz de confusao com contagens absolutas e percentual por linha."""
    import matplotlib.pyplot as plt

    apply_plot_style()
    y_true = np.asarray(y_true).astype(int)
    y_pred = (y_scores >= threshold).astype(int)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    pct = cm / cm.sum(axis=1, keepdims=True) * 100

    fig, ax = plt.subplots(figsize=(6.2, 5.2))
    im = ax.imshow(pct, cmap="Blues", vmin=0, vmax=100)

    labels = [
        ["Verdadeiro negativo", "Falso positivo\n(biopsia desnecessaria)"],
        ["FALSO NEGATIVO\n(cancer nao detectado)", "Verdadeiro positivo"],
    ]
    for i in range(2):
        for j in range(2):
            color = "white" if pct[i, j] > 55 else "#222222"
            ax.text(
                j, i - 0.12, f"{cm[i, j]}",
                ha="center", va="center", fontsize=22, fontweight="bold", color=color,
            )
            ax.text(
                j, i + 0.20, f"{pct[i, j]:.1f}%\n{labels[i][j]}",
                ha="center", va="center", fontsize=8, color=color,
            )

    ax.set_xticks([0, 1], [f"Predito: {CLASS_NAMES[0]}", f"Predito: {CLASS_NAMES[1]}"])
    ax.set_yticks([0, 1], [f"Real: {CLASS_NAMES[0]}", f"Real: {CLASS_NAMES[1]}"])
    ax.set_title(f"{title}\n(limiar = {threshold:.3f})")
    ax.grid(False)
    fig.colorbar(im, ax=ax, label="% da classe real", fraction=0.046)
    return save_fig(fig, filename)


def plot_roc_pr(
    curves: dict[str, tuple[np.ndarray, np.ndarray]],
    filename: str = "10_curvas_roc_pr",
):
    """Curvas ROC e Precision-Recall de todos os modelos, lado a lado."""
    import matplotlib.pyplot as plt

    apply_plot_style()
    fig, (ax_roc, ax_pr) = plt.subplots(1, 2, figsize=(14, 6))

    for (name, (y_true, y_scores)), color in zip(curves.items(), PLOT.palette * 3):
        y_true = np.asarray(y_true).astype(int)
        fpr, tpr, _ = roc_curve(y_true, y_scores)
        auc = roc_auc_score(y_true, y_scores)
        ax_roc.plot(fpr, tpr, label=f"{name} (AUC={auc:.4f})", color=color, lw=2)

        prec, rec, _ = precision_recall_curve(y_true, y_scores)
        ap = average_precision_score(y_true, y_scores)
        ax_pr.plot(rec, prec, label=f"{name} (AP={ap:.4f})", color=color, lw=2)

    ax_roc.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.5, label="Aleatorio")
    ax_roc.set(
        xlabel="Taxa de falsos positivos (1 - especificidade)",
        ylabel="Taxa de verdadeiros positivos (recall)",
        title="Curva ROC",
    )
    ax_roc.legend(loc="lower right", fontsize=8)

    baseline = float(np.mean(np.asarray(list(curves.values())[0][0]).astype(int)))
    ax_pr.axhline(baseline, ls="--", c="k", lw=1, alpha=0.5, label=f"Base ({baseline:.2f})")
    ax_pr.set(
        xlabel="Recall (sensibilidade)",
        ylabel="Precisao (valor preditivo positivo)",
        title="Curva Precisao x Recall",
    )
    ax_pr.legend(loc="lower left", fontsize=8)

    fig.suptitle("Capacidade discriminativa dos modelos", fontsize=14, fontweight="bold")
    return save_fig(fig, filename)


def plot_model_comparison(table: pd.DataFrame, filename: str = "09_comparacao_modelos"):
    """Barras horizontais comparando recall, F1, precisao e ROC AUC."""
    import matplotlib.pyplot as plt

    apply_plot_style()
    metrics = [m for m in ("recall", "f1", "precision", "roc_auc") if m in table.columns]
    data = table[metrics].sort_values("recall")

    fig, ax = plt.subplots(figsize=(11, 0.85 * len(data) + 2.5))
    y = np.arange(len(data))
    height = 0.8 / len(metrics)

    for i, metric in enumerate(metrics):
        offset = (i - (len(metrics) - 1) / 2) * height
        bars = ax.barh(
            y + offset, data[metric], height=height,
            label=metric, color=PLOT.palette[i % len(PLOT.palette)],
        )
        ax.bar_label(bars, fmt="%.3f", fontsize=7, padding=2)

    ax.set_yticks(y, data.index)
    ax.set_xlim(0, 1.10)
    ax.set_xlabel("valor da metrica")
    ax.set_title("Comparacao dos modelos no conjunto de validacao", pad=26)
    # Legenda acima do eixo para nao cobrir as barras do ultimo modelo.
    ax.legend(
        ncols=len(metrics), fontsize=9, frameon=False,
        loc="lower center", bbox_to_anchor=(0.5, 1.005),
    )
    return save_fig(fig, filename)


def plot_threshold_analysis(
    choice: ThresholdChoice,
    filename: str = "11_analise_limiar",
):
    """Mostra o trade-off recall x precisao e onde o limiar foi fixado."""
    import matplotlib.pyplot as plt

    apply_plot_style()
    curve = choice.curve.sort_values("threshold")

    fig, ax = plt.subplots(figsize=(11, 6))
    ax.plot(curve["threshold"], curve["recall"], label="Recall (sensibilidade)",
            color=PLOT.malignant_color, lw=2.2)
    ax.plot(curve["threshold"], curve["precision"], label="Precisao (VPP)",
            color=PLOT.benign_color, lw=2.2)
    ax.plot(curve["threshold"], curve["f1"], label="F1-score", color="#EDAE49", lw=2)
    ax.plot(curve["threshold"], curve["specificity"], label="Especificidade",
            color="#5C9E31", lw=1.6, ls="--")

    ax.axhline(choice.target_recall, color="grey", ls=":", lw=1.4,
               label=f"Recall minimo exigido ({choice.target_recall:.2f})")
    ax.axvline(choice.threshold, color="black", ls="-.", lw=1.6,
               label=f"Limiar escolhido = {choice.threshold:.3f}")
    ax.axvline(DEFAULT_THRESHOLD, color="grey", ls="--", lw=1.2, alpha=0.7,
               label="Limiar padrao = 0,50")

    ax.set(
        xlabel="Limiar de decisao aplicado a probabilidade de malignidade",
        ylabel="valor da metrica",
        title="Calibracao do limiar de decisao (conjunto de validacao)",
        xlim=(0, 1), ylim=(0, 1.05),
    )
    ax.legend(loc="lower left", fontsize=9)
    return save_fig(fig, filename)


def plot_calibration(
    y_true: np.ndarray | pd.Series,
    y_scores: np.ndarray,
    model_name: str,
    filename: str = "15_calibracao",
):
    """Curva de confiabilidade: a probabilidade prevista corresponde a realidade?"""
    import matplotlib.pyplot as plt

    apply_plot_style()
    y_true = np.asarray(y_true).astype(int)
    n_bins = min(10, max(3, len(y_true) // 12))
    prob_true, prob_pred = calibration_curve(y_true, y_scores, n_bins=n_bins, strategy="quantile")

    fig, (ax, ax_hist) = plt.subplots(
        2, 1, figsize=(8, 8), sharex=True,
        gridspec_kw={"height_ratios": [3, 1], "hspace": 0.08},
    )
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="Calibracao perfeita")
    ax.plot(prob_pred, prob_true, "o-", color=PLOT.malignant_color, lw=2, ms=7,
            label=model_name)
    ax.set(
        ylabel="Fracao observada de malignos",
        title=f"Curva de calibracao - {model_name}\n"
              f"Brier score = {brier_score_loss(y_true, y_scores):.4f} (quanto menor, melhor)",
    )
    ax.legend(loc="upper left")

    ax_hist.hist(y_scores, bins=25, color=PLOT.benign_color, alpha=0.85)
    ax_hist.set(xlabel="Probabilidade prevista de malignidade", ylabel="n exames")
    return save_fig(fig, filename)


def plot_score_distribution(
    y_true: np.ndarray | pd.Series,
    y_scores: np.ndarray,
    threshold: float,
    filename: str = "14_distribuicao_scores",
):
    """Distribuicao das probabilidades previstas por classe real."""
    import matplotlib.pyplot as plt

    apply_plot_style()
    y_true = np.asarray(y_true).astype(int)

    fig, ax = plt.subplots(figsize=(11, 5.5))
    bins = np.linspace(0, 1, 41)
    ax.hist(y_scores[y_true == 0], bins=bins, alpha=0.75,
            label=f"{CLASS_NAMES[0]} (real)", color=PLOT.benign_color)
    ax.hist(y_scores[y_true == 1], bins=bins, alpha=0.75,
            label=f"{CLASS_NAMES[1]} (real)", color=PLOT.malignant_color)
    ax.axvline(threshold, color="black", ls="-.", lw=2,
               label=f"Limiar clinico = {threshold:.3f}")
    ax.axvline(DEFAULT_THRESHOLD, color="grey", ls="--", lw=1.2,
               label="Limiar padrao = 0,50")
    ax.set(
        xlabel="Probabilidade prevista de malignidade",
        ylabel="n exames",
        title="Separacao das classes pelo modelo campeao",
    )
    ax.legend()
    return save_fig(fig, filename)


def plot_learning_curve(
    pipeline,
    X: pd.DataFrame,
    y: pd.Series,
    cv,
    model_name: str,
    filename: str = "16_curva_aprendizado",
):
    """Curva de aprendizado: o modelo se beneficiaria de mais pacientes?"""
    import matplotlib.pyplot as plt
    from sklearn.model_selection import learning_curve

    apply_plot_style()
    sizes, train_scores, val_scores = learning_curve(
        pipeline, X, y, cv=cv, scoring="roc_auc", n_jobs=-1,
        train_sizes=np.linspace(0.15, 1.0, 8), shuffle=True, random_state=42,
    )

    fig, ax = plt.subplots(figsize=(10, 6))
    for scores, label, color in (
        (train_scores, "Treino", PLOT.benign_color),
        (val_scores, "Validacao cruzada", PLOT.malignant_color),
    ):
        mean, std = scores.mean(axis=1), scores.std(axis=1)
        ax.plot(sizes, mean, "o-", color=color, label=label, lw=2)
        ax.fill_between(sizes, mean - std, mean + std, color=color, alpha=0.15)

    ax.set(
        xlabel="n de pacientes no treino",
        ylabel="ROC AUC",
        title=f"Curva de aprendizado - {model_name}",
    )
    ax.legend(loc="lower right")
    return save_fig(fig, filename)


def error_analysis(
    X: pd.DataFrame,
    y_true: pd.Series,
    y_scores: np.ndarray,
    threshold: float,
    top_features: list[str] | None = None,
) -> pd.DataFrame:
    """Lista os exames classificados errado, para revisao clinica dirigida."""
    y_pred = (y_scores >= threshold).astype(int)
    y_arr = np.asarray(y_true).astype(int)
    wrong = y_pred != y_arr

    out = pd.DataFrame(
        {
            "indice_original": X.index[wrong],
            "classe_real": np.where(y_arr[wrong] == 1, "Maligno", "Benigno"),
            "classe_prevista": np.where(y_pred[wrong] == 1, "Maligno", "Benigno"),
            "prob_malignidade": np.round(y_scores[wrong], 4),
            "tipo_erro": np.where(y_arr[wrong] == 1, "FALSO NEGATIVO", "falso positivo"),
        }
    )
    if top_features:
        cols = [c for c in top_features if c in X.columns][:5]
        extra = X.loc[X.index[wrong], cols].round(3).reset_index(drop=True)
        out = pd.concat([out.reset_index(drop=True), extra], axis=1)
    return out.sort_values("prob_malignidade", ascending=False).reset_index(drop=True)
