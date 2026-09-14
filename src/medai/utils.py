"""Utilitarios transversais: logging, estilo de figuras e serializacao."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")  # backend sem interface grafica (funciona em Docker/CI)
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from .config import FIGURES_DIR, METRICS_DIR, PLOT

_LOG_FORMAT = "%(asctime)s | %(levelname)-7s | %(message)s"


def get_logger(name: str = "medai") -> logging.Logger:
    """Devolve um logger configurado (idempotente)."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter(_LOG_FORMAT, datefmt="%H:%M:%S"))
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
        logger.propagate = False
    return logger


def apply_plot_style() -> None:
    """Aplica o estilo visual unico usado em todo o relatorio."""
    sns.set_theme(style="whitegrid", context="notebook")
    plt.rcParams.update(
        {
            "figure.dpi": PLOT.dpi,
            "savefig.dpi": PLOT.dpi,
            "figure.figsize": PLOT.figsize,
            "axes.titlesize": 13,
            "axes.titleweight": "bold",
            "axes.labelsize": 11,
            "axes.edgecolor": "#444444",
            "axes.prop_cycle": plt.cycler(color=list(PLOT.palette)),
            "grid.alpha": 0.30,
            "savefig.bbox": "tight",
            "savefig.facecolor": "white",
            "font.size": 10,
        }
    )


def save_fig(fig: plt.Figure, name: str, subdir: str | None = None) -> Path:
    """Salva a figura em ``reports/figures`` e a fecha, devolvendo o caminho."""
    out_dir = FIGURES_DIR if subdir is None else FIGURES_DIR / subdir
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{name}.png"
    fig.savefig(path)
    plt.close(fig)
    get_logger().info("figura salva -> %s", path.relative_to(path.parents[3]))
    return path


class _NumpyEncoder(json.JSONEncoder):
    """Permite serializar tipos numpy dentro de JSON."""

    def default(self, o: Any) -> Any:  # noqa: D102
        if isinstance(o, np.integer):
            return int(o)
        if isinstance(o, np.floating):
            return float(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, Path):
            return str(o)
        return super().default(o)


def save_json(payload: dict[str, Any], name: str) -> Path:
    """Persiste um dicionario de metricas em ``reports/metrics/<name>.json``."""
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    path = METRICS_DIR / f"{name}.json"
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, cls=_NumpyEncoder),
        encoding="utf-8",
    )
    return path


def save_table(df, name: str) -> Path:
    """Persiste um DataFrame em CSV dentro de ``reports/metrics``."""
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    path = METRICS_DIR / f"{name}.csv"
    df.to_csv(path, index=True, encoding="utf-8")
    return path


def section(title: str, logger: logging.Logger | None = None) -> None:
    """Imprime um cabecalho de secao legivel no log do pipeline."""
    log = logger or get_logger()
    log.info("")
    log.info("=" * 78)
    log.info(title.upper())
    log.info("=" * 78)
