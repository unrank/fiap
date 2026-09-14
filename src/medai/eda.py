"""
Analise exploratoria de dados (EDA) e geracao das figuras do relatorio.

Todas as funcoes recebem o ``DataFrame`` bruto (antes de qualquer
transformacao) e devolvem o caminho da figura gerada em ``reports/figures``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .config import CLASS_NAMES, ID_COL, LABEL_MAP, PLOT, TARGET_COL
from .utils import apply_plot_style, get_logger, save_fig

log = get_logger()


def _feature_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Somente as colunas preditoras numericas."""
    drop = [c for c in (ID_COL, TARGET_COL) if c in df.columns]
    return df.drop(columns=drop).select_dtypes(include=np.number)


def _pretty(name: str) -> str:
    """Converte ``concave_points_worst`` em ``concave points (worst)``."""
    for suffix, tag in (("_mean", "media"), ("_se", "erro-padrao"), ("_worst", "pior")):
        if name.endswith(suffix):
            return f"{name[: -len(suffix)].replace('_', ' ')} ({tag})"
    return name.replace("_", " ")


# ---------------------------------------------------------------------------
# 1. Balanceamento das classes
# ---------------------------------------------------------------------------
def plot_class_balance(df: pd.DataFrame, filename: str = "01_balanceamento_classes"):
    """Contagem e proporcao de diagnosticos benignos x malignos."""
    import matplotlib.pyplot as plt

    apply_plot_style()
    counts = df[TARGET_COL].value_counts().reindex(["B", "M"])
    labels = [CLASS_NAMES[0], CLASS_NAMES[1]]
    colors = [PLOT.benign_color, PLOT.malignant_color]

    fig, (ax_bar, ax_pie) = plt.subplots(1, 2, figsize=(12, 5))

    bars = ax_bar.bar(labels, counts.values, color=colors, width=0.6)
    ax_bar.bar_label(bars, fmt="%d", fontsize=13, fontweight="bold", padding=3)
    ax_bar.set(ylabel="n de biopsias", title="Contagem por diagnostico")
    ax_bar.set_ylim(0, counts.max() * 1.18)

    ax_pie.pie(
        counts.values, labels=labels, colors=colors, autopct="%1.1f%%",
        startangle=90, wedgeprops={"edgecolor": "white", "linewidth": 2},
        textprops={"fontsize": 12},
    )
    ax_pie.set_title(
        f"Proporcao das classes\n(razao de desbalanceamento = "
        f"{counts.max() / counts.min():.2f}:1)"
    )

    fig.suptitle(
        "Distribuicao do alvo - Breast Cancer Wisconsin (n = %d)" % len(df),
        fontsize=14, fontweight="bold",
    )
    return save_fig(fig, filename)


# ---------------------------------------------------------------------------
# 2. Distribuicoes por classe
# ---------------------------------------------------------------------------
def plot_distributions(
    df: pd.DataFrame,
    features: list[str],
    filename: str = "02_distribuicoes_por_classe",
    ncols: int = 3,
):
    """Histograma + KDE de cada feature, separado por diagnostico."""
    import matplotlib.pyplot as plt
    import seaborn as sns

    apply_plot_style()
    nrows = int(np.ceil(len(features) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(5.2 * ncols, 3.6 * nrows))
    axes = np.atleast_1d(axes).ravel()

    for ax, feat in zip(axes, features):
        for label, color in (("B", PLOT.benign_color), ("M", PLOT.malignant_color)):
            sns.histplot(
                df.loc[df[TARGET_COL] == label, feat],
                ax=ax, color=color, kde=True, stat="density",
                alpha=0.5, bins=28, edgecolor=None,
                label=CLASS_NAMES[LABEL_MAP[label]],
            )
        ax.set(title=_pretty(feat), xlabel="", ylabel="densidade")
        ax.legend(fontsize=8)

    for ax in axes[len(features):]:
        ax.axis("off")

    fig.suptitle(
        "Distribuicao das variaveis mais discriminantes por diagnostico",
        fontsize=15, fontweight="bold",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    return save_fig(fig, filename)


# ---------------------------------------------------------------------------
# 3. Boxplots
# ---------------------------------------------------------------------------
def plot_boxplots(
    df: pd.DataFrame,
    features: list[str],
    filename: str = "03_boxplots_por_classe",
    ncols: int = 5,
):
    """Boxplots por classe - evidenciam separacao de medianas e outliers."""
    import matplotlib.pyplot as plt
    import seaborn as sns

    apply_plot_style()
    nrows = int(np.ceil(len(features) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.1 * ncols, 3.4 * nrows))
    axes = np.atleast_1d(axes).ravel()

    plot_df = df.copy()
    plot_df["Diagnostico"] = plot_df[TARGET_COL].map(
        {"B": CLASS_NAMES[0], "M": CLASS_NAMES[1]}
    )

    for ax, feat in zip(axes, features):
        sns.boxplot(
            data=plot_df, x="Diagnostico", y=feat, ax=ax,
            hue="Diagnostico", legend=False,
            palette={CLASS_NAMES[0]: PLOT.benign_color, CLASS_NAMES[1]: PLOT.malignant_color},
            fliersize=2.5, width=0.62,
        )
        ax.set(title=_pretty(feat), xlabel="", ylabel="")
        ax.tick_params(labelsize=8)

    for ax in axes[len(features):]:
        ax.axis("off")

    fig.suptitle(
        "Dispersao e outliers por diagnostico", fontsize=15, fontweight="bold"
    )
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    return save_fig(fig, filename)


# ---------------------------------------------------------------------------
# 4. Correlacao
# ---------------------------------------------------------------------------
def plot_correlation_heatmap(
    corr: pd.DataFrame,
    filename: str = "04_matriz_correlacao",
):
    """Mapa de calor da correlacao entre as 30 preditoras."""
    import matplotlib.pyplot as plt
    import seaborn as sns

    apply_plot_style()
    fig, ax = plt.subplots(figsize=(13.5, 11))
    mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
    sns.heatmap(
        corr, mask=mask, cmap="RdBu_r", center=0, vmin=-1, vmax=1,
        square=True, linewidths=0.35, linecolor="white",
        cbar_kws={"shrink": 0.62, "label": "correlacao de Pearson"}, ax=ax,
    )
    ax.set_title(
        "Matriz de correlacao entre as variaveis preditoras\n"
        "(blocos vermelhos = grupos altamente redundantes)",
        fontsize=14, fontweight="bold", pad=14,
    )
    ax.tick_params(labelsize=8)
    return save_fig(fig, filename)


def plot_target_correlation(
    with_target: pd.DataFrame,
    filename: str = "05_correlacao_com_alvo",
    top_n: int = 30,
):
    """Ranking das variaveis por correlacao com o diagnostico."""
    import matplotlib.pyplot as plt

    apply_plot_style()
    data = with_target.head(top_n).iloc[::-1]
    values = data["corr_com_alvo"].to_numpy()
    colors = [PLOT.malignant_color if v > 0 else PLOT.benign_color for v in values]

    fig, ax = plt.subplots(figsize=(10, 0.34 * len(data) + 2.2))
    bars = ax.barh([_pretty(i) for i in data.index], values, color=colors)
    ax.bar_label(bars, fmt="%.3f", fontsize=7.5, padding=2)
    ax.axvline(0, color="black", lw=1)
    ax.set(
        xlabel="correlacao ponto-bisserial com o alvo (1 = maligno)",
        title="Quanto cada variavel se associa ao diagnostico maligno",
        xlim=(min(values.min(), 0) - 0.12, values.max() + 0.14),
    )
    ax.tick_params(labelsize=8)
    return save_fig(fig, filename)


# ---------------------------------------------------------------------------
# 5. Estrutura multivariada
# ---------------------------------------------------------------------------
def plot_pca_projection(
    df: pd.DataFrame,
    filename: str = "06_projecao_pca",
):
    """Projeta as 30 variaveis padronizadas em 2 componentes principais."""
    import matplotlib.pyplot as plt
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler

    apply_plot_style()
    X = _feature_frame(df)
    y = df[TARGET_COL]

    Z = StandardScaler().fit_transform(X)
    pca = PCA(n_components=min(10, X.shape[1]), random_state=42)
    comps = pca.fit_transform(Z)
    var = pca.explained_variance_ratio_

    fig, (ax, ax_var) = plt.subplots(1, 2, figsize=(14, 6),
                                     gridspec_kw={"width_ratios": [1.35, 1]})

    for label, color in (("B", PLOT.benign_color), ("M", PLOT.malignant_color)):
        m = (y == label).to_numpy()
        ax.scatter(comps[m, 0], comps[m, 1], s=26, alpha=0.72, color=color,
                   edgecolor="white", linewidth=0.4,
                   label=f"{CLASS_NAMES[LABEL_MAP[label]]} (n={m.sum()})")
    ax.set(
        xlabel=f"PC1 ({var[0] * 100:.1f}% da variancia)",
        ylabel=f"PC2 ({var[1] * 100:.1f}% da variancia)",
        title="Projecao PCA - as classes ja se separam em 2 dimensoes",
    )
    ax.legend()

    cum = np.cumsum(var) * 100
    ax_var.bar(range(1, len(var) + 1), var * 100, color=PLOT.benign_color,
               label="variancia individual")
    ax_var.plot(range(1, len(var) + 1), cum, "o-", color=PLOT.malignant_color,
                label="variancia acumulada")
    ax_var.axhline(95, ls="--", color="grey", lw=1, label="95%")
    ax_var.set(xlabel="componente principal", ylabel="% da variancia explicada",
               title="Variancia explicada por componente")
    ax_var.legend(fontsize=9)

    fig.suptitle("Estrutura multivariada dos dados", fontsize=14, fontweight="bold")
    return save_fig(fig, filename)


def plot_pairwise(
    df: pd.DataFrame,
    features: list[str],
    filename: str = "07_dispersao_pareada",
):
    """Matriz de dispersao das variaveis mais discriminantes."""
    import matplotlib.pyplot as plt
    import seaborn as sns

    apply_plot_style()
    plot_df = df[list(features) + [TARGET_COL]].copy()
    plot_df["Diagnostico"] = plot_df[TARGET_COL].map(
        {"B": CLASS_NAMES[0], "M": CLASS_NAMES[1]}
    )
    grid = sns.pairplot(
        plot_df.drop(columns=[TARGET_COL]),
        hue="Diagnostico",
        palette={CLASS_NAMES[0]: PLOT.benign_color, CLASS_NAMES[1]: PLOT.malignant_color},
        diag_kind="kde", plot_kws={"s": 18, "alpha": 0.6, "edgecolor": "none"},
        height=2.3, corner=True,
    )
    grid.figure.suptitle(
        "Dispersao pareada das variaveis mais discriminantes",
        fontsize=14, fontweight="bold", y=1.01,
    )
    return save_fig(grid.figure, filename)


# ---------------------------------------------------------------------------
# 6. Escala e assimetria (justificam o pre-processamento)
# ---------------------------------------------------------------------------
def plot_scale_and_skew(df: pd.DataFrame, filename: str = "08_escala_e_assimetria"):
    """
    Mostra por que padronizar e obrigatorio: as variaveis vivem em escalas
    que diferem por mais de tres ordens de grandeza.
    """
    import matplotlib.pyplot as plt

    apply_plot_style()
    X = _feature_frame(df)
    stats = pd.DataFrame({"media": X.mean(), "desvio": X.std(), "assimetria": X.skew()})
    stats = stats.sort_values("media")

    fig, (ax_scale, ax_skew) = plt.subplots(1, 2, figsize=(15, 8))

    ax_scale.barh([_pretty(i) for i in stats.index], stats["media"],
                  xerr=stats["desvio"], color=PLOT.benign_color,
                  error_kw={"ecolor": "#555555", "elinewidth": 1})
    ax_scale.set_xscale("log")
    ax_scale.set(xlabel="media (escala logaritmica) +/- 1 desvio-padrao",
                 title="Escalas muito diferentes\n-> padronizacao e obrigatoria")
    ax_scale.tick_params(labelsize=7.5)

    skew_sorted = stats["assimetria"].sort_values()
    colors = [PLOT.malignant_color if abs(v) > 1 else PLOT.benign_color
              for v in skew_sorted]
    ax_skew.barh([_pretty(i) for i in skew_sorted.index], skew_sorted.values, color=colors)
    ax_skew.axvline(1, ls="--", color="grey", lw=1)
    ax_skew.axvline(-1, ls="--", color="grey", lw=1)
    ax_skew.set(xlabel="coeficiente de assimetria (skewness)",
                title="Assimetria por variavel\n(vermelho: |skew| > 1)")
    ax_skew.tick_params(labelsize=7.5)

    fig.suptitle("Diagnostico da escala e da forma das distribuicoes",
                 fontsize=14, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    return save_fig(fig, filename)


# ---------------------------------------------------------------------------
# Selecao automatica das features para as figuras
# ---------------------------------------------------------------------------
def top_discriminative_features(
    df: pd.DataFrame, y: pd.Series, k: int = 9
) -> list[str]:
    """As ``k`` variaveis com maior correlacao absoluta com o diagnostico."""
    X = _feature_frame(df)
    corr = X.apply(lambda c: c.corr(y.astype(float))).abs().sort_values(ascending=False)
    return corr.head(k).index.tolist()
