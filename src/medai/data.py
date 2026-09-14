"""
Carga, auditoria e persistencia do dataset clinico tabular.

Dataset: **Breast Cancer Wisconsin (Diagnostic) - WDBC**
    569 biopsias por puncao aspirativa por agulha fina (PAAF) de nodulo mamario.
    30 atributos morfometricos extraidos digitalmente da imagem do nucleo
    celular (10 medidas x 3 estatisticas: media, erro-padrao e "pior" valor).
    Alvo: M = maligno, B = benigno.

Fontes (nesta ordem de preferencia):
    1. CSV local em ``data/raw/breast_cancer_wisconsin.csv``;
    2. copia oficial embarcada no scikit-learn (identica a do UCI, funciona
       offline - usada tambem dentro do container Docker);
    3. download direto do repositorio UCI.
"""

from __future__ import annotations

import io
from typing import Literal

import numpy as np
import pandas as pd

from .config import ID_COL, LABEL_MAP, RAW_CSV_PATH, RAW_DATA_DIR, TARGET_COL
from .utils import get_logger

log = get_logger()

UCI_URL = (
    "https://archive.ics.uci.edu/ml/machine-learning-databases/"
    "breast-cancer-wisconsin/wdbc.data"
)

#: As 10 caracteristicas morfometricas medidas em cada nucleo celular.
BASE_FEATURES: tuple[str, ...] = (
    "radius",             # raio medio (centro -> perimetro)
    "texture",            # desvio-padrao dos niveis de cinza
    "perimeter",          # perimetro do nucleo
    "area",               # area do nucleo
    "smoothness",         # variacao local dos comprimentos do raio
    "compactness",        # perimetro^2 / area - 1.0
    "concavity",          # severidade das porcoes concavas do contorno
    "concave_points",     # numero de porcoes concavas do contorno
    "symmetry",           # simetria do nucleo
    "fractal_dimension",  # "aproximacao da linha de costa" - 1
)

#: Sufixos: media, erro-padrao e media dos 3 maiores valores ("worst").
STAT_SUFFIXES: tuple[str, ...] = ("mean", "se", "worst")

#: Nomes finais das 30 colunas preditoras, na ordem original do WDBC.
FEATURE_COLUMNS: tuple[str, ...] = tuple(
    f"{base}_{suffix}" for suffix in STAT_SUFFIXES for base in BASE_FEATURES
)


# ---------------------------------------------------------------------------
# Carga
# ---------------------------------------------------------------------------
def _from_sklearn() -> pd.DataFrame:
    """Le a copia do WDBC embarcada no scikit-learn e a normaliza."""
    from sklearn.datasets import load_breast_cancer

    bunch = load_breast_cancer(as_frame=True)
    df = bunch.data.copy()
    df.columns = list(FEATURE_COLUMNS)

    # Atencao: no scikit-learn 0 = malignant e 1 = benign (invertido em
    # relacao a convencao do arquivo original). Voltamos aos rotulos M/B.
    df.insert(0, TARGET_COL, np.where(np.asarray(bunch.target) == 0, "M", "B"))
    return df


def _from_uci() -> pd.DataFrame:
    """Baixa o arquivo bruto ``wdbc.data`` do UCI e o normaliza."""
    import urllib.request

    log.info("baixando WDBC do UCI: %s", UCI_URL)
    with urllib.request.urlopen(UCI_URL, timeout=60) as resp:  # noqa: S310
        raw = resp.read().decode("utf-8")

    return pd.read_csv(
        io.StringIO(raw),
        header=None,
        names=[ID_COL, TARGET_COL, *FEATURE_COLUMNS],
    )


def load_raw(
    source: Literal["auto", "csv", "sklearn", "uci"] = "auto",
    persist: bool = True,
) -> pd.DataFrame:
    """
    Carrega o dataset bruto como ``DataFrame``.

    Parameters
    ----------
    source
        ``auto`` tenta CSV local -> scikit-learn -> UCI. As demais opcoes
        forcam uma fonte especifica.
    persist
        Se ``True``, grava uma copia em ``data/raw/`` para tornar a execucao
        reproduzivel e permitir rodar totalmente offline.
    """
    df: pd.DataFrame | None = None

    if source in ("auto", "csv") and RAW_CSV_PATH.exists():
        log.info("lendo dataset local: %s", RAW_CSV_PATH)
        df = pd.read_csv(RAW_CSV_PATH)
    elif source == "csv":
        raise FileNotFoundError(f"CSV nao encontrado em {RAW_CSV_PATH}")

    if df is None and source in ("auto", "sklearn"):
        log.info("carregando WDBC embarcado no scikit-learn (offline)")
        df = _from_sklearn()

    if df is None:
        df = _from_uci()

    df = normalize(df)

    if persist and not RAW_CSV_PATH.exists():
        RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
        df.to_csv(RAW_CSV_PATH, index=False)
        log.info("copia do dataset gravada em %s", RAW_CSV_PATH)

    log.info("dataset carregado: %d linhas x %d colunas", df.shape[0], df.shape[1])
    return df


def normalize(df: pd.DataFrame) -> pd.DataFrame:
    """Padroniza nomes de coluna, tipos e remove colunas vazias do CSV Kaggle."""
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]

    # O CSV distribuido no Kaggle traz uma coluna final vazia ("Unnamed: 32").
    unnamed = [c for c in df.columns if c.lower().startswith("unnamed")]
    if unnamed:
        log.info("removendo coluna(s) vazia(s) do CSV original: %s", unnamed)
        df = df.drop(columns=unnamed)

    # O Kaggle usa "concave points_mean"; padronizamos para snake_case.
    df.columns = [c.replace("concave points", "concave_points") for c in df.columns]

    if TARGET_COL in df.columns:
        df[TARGET_COL] = df[TARGET_COL].astype(str).str.strip().str.upper()
    return df


# ---------------------------------------------------------------------------
# Separacao X / y
# ---------------------------------------------------------------------------
def split_features_target(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """
    Devolve ``(X, y)`` com o alvo ja codificado (1 = maligno, 0 = benigno).

    A coluna identificadora (``id``) e descartada: e um numero de prontuario,
    nao carrega informacao clinica e so introduziria ruido no modelo.
    """
    drop_cols = [c for c in (ID_COL, TARGET_COL) if c in df.columns]
    X = df.drop(columns=drop_cols)
    y = df[TARGET_COL].map(LABEL_MAP)

    if y.isna().any():
        invalid = df.loc[y.isna(), TARGET_COL].unique().tolist()
        raise ValueError(f"rotulos nao reconhecidos em '{TARGET_COL}': {invalid}")

    y = y.astype("int8")
    y.name = "target"
    return X, y


# ---------------------------------------------------------------------------
# Auditoria de qualidade
# ---------------------------------------------------------------------------
def audit(df: pd.DataFrame) -> dict:
    """
    Audita a qualidade do dataset antes de qualquer transformacao.

    Verifica dimensoes, tipos, valores ausentes, duplicatas, colunas
    constantes, zeros biologicamente impossiveis (que em bases medicas
    costumam ser codificacao disfarcada de "ausente") e balanceamento
    das classes.
    """
    numeric = df.select_dtypes(include="number")
    categorical = df.select_dtypes(exclude="number")

    missing = df.isna().sum()
    constant = [c for c in numeric.columns if numeric[c].nunique(dropna=True) <= 1]

    # Raio, perimetro, area e textura nunca podem valer zero em um nucleo real.
    impossible_zero_cols = [
        c
        for c in numeric.columns
        if any(c.startswith(p) for p in ("radius", "perimeter", "area", "texture"))
    ]
    zeros = {c: int((numeric[c] == 0).sum()) for c in impossible_zero_cols}

    report: dict = {
        "n_rows": int(df.shape[0]),
        "n_cols": int(df.shape[1]),
        "n_numeric": int(numeric.shape[1]),
        "n_categorical": int(categorical.shape[1]),
        "categorical_columns": categorical.columns.tolist(),
        "missing_total": int(missing.sum()),
        "missing_by_column": {k: int(v) for k, v in missing[missing > 0].items()},
        "duplicated_rows": int(df.duplicated().sum()),
        "duplicated_ids": (
            int(df[ID_COL].duplicated().sum()) if ID_COL in df.columns else None
        ),
        "constant_columns": constant,
        "impossible_zeros": {k: v for k, v in zeros.items() if v > 0},
    }

    if TARGET_COL in df.columns:
        counts = df[TARGET_COL].value_counts()
        report["class_counts"] = {str(k): int(v) for k, v in counts.items()}
        report["class_balance_pct"] = {
            str(k): round(float(v), 2) for k, v in (counts / counts.sum() * 100).items()
        }
        report["imbalance_ratio"] = round(float(counts.max() / counts.min()), 3)

    return report


def describe_numeric(df: pd.DataFrame) -> pd.DataFrame:
    """Estatisticas descritivas estendidas (inclui assimetria e curtose)."""
    numeric = df.select_dtypes(include="number")
    if ID_COL in numeric.columns:
        numeric = numeric.drop(columns=[ID_COL])

    desc = numeric.describe().T
    desc["skew"] = numeric.skew()
    desc["kurtosis"] = numeric.kurtosis()
    desc["cv"] = desc["std"] / desc["mean"]  # coeficiente de variacao
    desc["missing"] = numeric.isna().sum()
    return desc.round(4)
