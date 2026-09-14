"""Configuracao central do projeto: caminhos, semente e constantes."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

# --------------------------------------------------------------------------
# Caminhos
# --------------------------------------------------------------------------
# Raiz = dois niveis acima deste arquivo (src/medai/config.py -> raiz).
ROOT_DIR: Path = Path(__file__).resolve().parents[2]

DATA_DIR: Path = ROOT_DIR / "data"
RAW_DATA_DIR: Path = DATA_DIR / "raw"
MEDMNIST_DIR: Path = DATA_DIR / "medmnist"

REPORTS_DIR: Path = ROOT_DIR / "reports"
FIGURES_DIR: Path = REPORTS_DIR / "figures"
METRICS_DIR: Path = REPORTS_DIR / "metrics"
MODELS_DIR: Path = ROOT_DIR / "models"

RAW_CSV_PATH: Path = RAW_DATA_DIR / "breast_cancer_wisconsin.csv"


def ensure_dirs() -> None:
    """Cria todos os diretorios de saida usados pelo pipeline."""
    for path in (RAW_DATA_DIR, MEDMNIST_DIR, FIGURES_DIR, METRICS_DIR, MODELS_DIR):
        path.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------------------------------
# Reprodutibilidade
# --------------------------------------------------------------------------
RANDOM_STATE: int = 42


def set_global_seed(seed: int = RANDOM_STATE) -> None:
    """Fixa a semente das bibliotecas usadas, para resultados reproduziveis."""
    import random

    import numpy as np

    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:  # torch e opcional (so o entregavel EXTRA usa)
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


# --------------------------------------------------------------------------
# Definicao do problema
# --------------------------------------------------------------------------
TARGET_COL: str = "diagnosis"
ID_COL: str = "id"

#: Mapeamento do rotulo original para o alvo binario do modelo.
#: 1 = maligno (classe positiva, o evento clinicamente critico)
#: 0 = benigno
LABEL_MAP: dict[str, int] = {"M": 1, "B": 0}
CLASS_NAMES: tuple[str, str] = ("Benigno", "Maligno")

#: Proporcoes da separacao treino / validacao / teste.
TEST_SIZE: float = 0.20
VAL_SIZE: float = 0.20  # fracao do total (aplicada sobre o que sobra do teste)
CV_FOLDS: int = 5

#: Recall minimo exigido do modelo em triagem clinica.
#: Falso negativo = cancer maligno classificado como benigno -> risco de morte.
TARGET_RECALL: float = 0.99


@dataclass(frozen=True)
class PlotStyle:
    """Padroniza a aparencia de todas as figuras do relatorio."""

    dpi: int = 130
    figsize: tuple[float, float] = (10.0, 6.0)
    palette: tuple[str, ...] = field(
        default=("#2E7D9A", "#D1495B", "#EDAE49", "#5C9E31", "#8367C7", "#00798C")
    )
    benign_color: str = "#2E7D9A"
    malignant_color: str = "#D1495B"


PLOT: PlotStyle = PlotStyle()
