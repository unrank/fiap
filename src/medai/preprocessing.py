"""
Pipeline de pre-processamento e separacao treino / validacao / teste.

Todo o pre-processamento vive dentro de um ``sklearn.Pipeline``. Isso e
essencial em um contexto clinico por tres motivos:

1. **Sem vazamento de dados** - a media/desvio usados na padronizacao sao
   aprendidos *somente* no conjunto de treino e depois aplicados a validacao
   e teste. Calcular isso no dataset inteiro inflaria artificialmente as
   metricas.
2. **Reprodutibilidade** - o objeto serializado (``joblib``) contem dados +
   transformacoes + modelo. O que roda em producao e exatamente o que foi
   validado.
3. **Robustez em producao** - exames novos podem chegar com campos ausentes
   ou com uma categoria nunca vista; os imputadores e o ``OneHotEncoder``
   com ``handle_unknown="ignore"`` evitam que o servico quebre.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    OneHotEncoder,
    PowerTransformer,
    RobustScaler,
    StandardScaler,
)
from sklearn.model_selection import train_test_split

from .config import RANDOM_STATE, TEST_SIZE, VAL_SIZE
from .utils import get_logger

log = get_logger()

ScalerName = Literal["standard", "robust", "power", "none"]


# ---------------------------------------------------------------------------
# Separacao treino / validacao / teste
# ---------------------------------------------------------------------------
@dataclass
class DataSplit:
    """Guarda as tres particoes do dataset, sempre estratificadas pelo alvo."""

    X_train: pd.DataFrame
    X_val: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_val: pd.Series
    y_test: pd.Series

    @property
    def X_trainval(self) -> pd.DataFrame:
        """Treino + validacao - usado no reajuste final do modelo campeao."""
        return pd.concat([self.X_train, self.X_val], axis=0)

    @property
    def y_trainval(self) -> pd.Series:
        return pd.concat([self.y_train, self.y_val], axis=0)

    def summary(self) -> pd.DataFrame:
        """Tabela com tamanho e prevalencia de malignos em cada particao."""
        rows = []
        for name, y in (
            ("treino", self.y_train),
            ("validacao", self.y_val),
            ("teste", self.y_test),
        ):
            rows.append(
                {
                    "particao": name,
                    "n": int(len(y)),
                    "pct_do_total": round(
                        100 * len(y) / (len(self.y_train) + len(self.y_val) + len(self.y_test)),
                        1,
                    ),
                    "benignos": int((y == 0).sum()),
                    "malignos": int((y == 1).sum()),
                    "prevalencia_maligno_%": round(100 * float(y.mean()), 2),
                }
            )
        return pd.DataFrame(rows).set_index("particao")


def split_train_val_test(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = TEST_SIZE,
    val_size: float = VAL_SIZE,
    random_state: int = RANDOM_STATE,
) -> DataSplit:
    """
    Divide os dados em treino / validacao / teste de forma **estratificada**.

    A estratificacao preserva a prevalencia de casos malignos (~37%) nas tres
    particoes; sem ela, com apenas 569 amostras, uma particao poderia ficar
    com prevalencia muito diferente e distorcer a avaliacao.

    O conjunto de **teste e aberto uma unica vez**, ao final, para estimar o
    desempenho em dados nunca vistos. A **validacao** e quem escolhe o modelo
    e calibra o limiar de decisao.
    """
    # 1o corte: separa o teste do restante.
    X_rest, X_test, y_rest, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )
    # 2o corte: separa a validacao de dentro do restante, de modo que a
    # validacao represente `val_size` do total original.
    val_fraction_of_rest = val_size / (1.0 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_rest,
        y_rest,
        test_size=val_fraction_of_rest,
        stratify=y_rest,
        random_state=random_state,
    )

    split = DataSplit(X_train, X_val, X_test, y_train, y_val, y_test)
    log.info(
        "split estratificado -> treino=%d | validacao=%d | teste=%d",
        len(y_train),
        len(y_val),
        len(y_test),
    )
    return split


# ---------------------------------------------------------------------------
# Pipeline de transformacao
# ---------------------------------------------------------------------------
def _make_scaler(kind: ScalerName):
    """Devolve o transformador de escala escolhido."""
    if kind == "standard":
        # z-score: media 0, desvio 1. Indispensavel para KNN, SVM e para a
        # regularizacao da regressao logistica.
        return StandardScaler()
    if kind == "robust":
        # Usa mediana e IQR - menos sensivel aos outliers das caudas longas.
        return RobustScaler()
    if kind == "power":
        # Yeo-Johnson: reduz a forte assimetria positiva de area/concavity.
        return PowerTransformer(method="yeo-johnson", standardize=True)
    return "passthrough"


def infer_column_types(X: pd.DataFrame) -> tuple[list[str], list[str]]:
    """Separa as colunas de ``X`` em (numericas, categoricas)."""
    numeric = X.select_dtypes(include=np.number).columns.tolist()
    categorical = [c for c in X.columns if c not in numeric]
    return numeric, categorical


def build_preprocessor(
    X: pd.DataFrame,
    scaler: ScalerName = "standard",
) -> ColumnTransformer:
    """
    Constroi o ``ColumnTransformer`` de pre-processamento.

    Dois ramos independentes, aplicados automaticamente conforme o tipo de
    cada coluna:

    * **numerico**   - imputacao pela *mediana* (resistente a outliers) +
      padronizacao de escala;
    * **categorico** - imputacao pela *moda* + ``OneHotEncoder``
      (``handle_unknown="ignore"``, para nao quebrar com categorias novas).

    No WDBC as 30 preditoras sao todas numericas, logo o ramo categorico fica
    vazio nesta execucao. Ele e mantido porque o mesmo pipeline e reaproveitado
    para outras bases clinicas (sexo, etnia, tipo de exame, unidade
    hospitalar...) e porque o ``ColumnTransformer`` resolve os ramos em tempo
    de ``fit`` - o codigo nao precisa mudar.
    """
    numeric_cols, categorical_cols = infer_column_types(X)

    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", _make_scaler(scaler)),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "onehot",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False, drop=None),
            ),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, numeric_cols),
            ("cat", categorical_pipeline, categorical_cols),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )
    preprocessor.set_output(transform="pandas")

    log.info(
        "preprocessador: %d coluna(s) numerica(s) [escala=%s], %d categorica(s)",
        len(numeric_cols),
        scaler,
        len(categorical_cols),
    )
    return preprocessor


def get_feature_names(fitted_preprocessor: ColumnTransformer) -> list[str]:
    """Nomes das colunas produzidas pelo pre-processador ja ajustado."""
    return list(fitted_preprocessor.get_feature_names_out())


# ---------------------------------------------------------------------------
# Analise de correlacao
# ---------------------------------------------------------------------------
@dataclass
class CorrelationReport:
    """Resultado da analise de correlacao."""

    matrix: pd.DataFrame                 # matriz de correlacao entre preditoras
    with_target: pd.DataFrame            # correlacao de cada preditora com o alvo
    redundant_pairs: pd.DataFrame        # pares acima do limiar de multicolinearidade
    suggested_drop: list[str]            # colunas redundantes candidatas a remocao


def correlation_analysis(
    X: pd.DataFrame,
    y: pd.Series,
    threshold: float = 0.90,
    method: str = "pearson",
) -> CorrelationReport:
    """
    Analisa (a) a correlacao entre preditoras e (b) a correlacao com o alvo.

    Parameters
    ----------
    threshold
        Acima deste valor absoluto o par e considerado redundante
        (multicolinearidade). Para cada par redundante sugerimos remover a
        variavel com menor correlacao com o alvo.
    """
    numeric = X.select_dtypes(include=np.number)
    corr = numeric.corr(method=method)

    # Correlacao ponto-bisserial (Pearson entre continua e alvo binario).
    with_target = (
        numeric.apply(lambda col: col.corr(y.astype(float), method=method))
        .rename("corr_com_alvo")
        .to_frame()
    )
    with_target["abs"] = with_target["corr_com_alvo"].abs()
    with_target = with_target.sort_values("abs", ascending=False).drop(columns="abs")

    # Pares redundantes (triangulo superior da matriz).
    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
    stacked = upper.stack()
    pairs = stacked[stacked.abs() >= threshold].reset_index()
    pairs.columns = ["feature_a", "feature_b", "correlacao"]
    pairs["correlacao"] = pairs["correlacao"].round(4)
    pairs = pairs.sort_values("correlacao", key=np.abs, ascending=False)

    # Para cada par, marcamos como descartavel a variavel menos ligada ao alvo.
    strength = with_target["corr_com_alvo"].abs()
    drop: set[str] = set()
    for _, row in pairs.iterrows():
        a, b = row["feature_a"], row["feature_b"]
        if a in drop or b in drop:
            continue
        drop.add(a if strength.get(a, 0) < strength.get(b, 0) else b)

    log.info(
        "correlacao: %d par(es) com |r| >= %.2f | %d coluna(s) redundante(s)",
        len(pairs),
        threshold,
        len(drop),
    )
    return CorrelationReport(
        matrix=corr,
        with_target=with_target.round(4),
        redundant_pairs=pairs.reset_index(drop=True),
        suggested_drop=sorted(drop),
    )


# ---------------------------------------------------------------------------
# Teste de robustez a dados ausentes
# ---------------------------------------------------------------------------
def inject_missing(
    X: pd.DataFrame,
    fraction: float = 0.05,
    random_state: int = RANDOM_STATE,
) -> pd.DataFrame:
    """
    Insere valores ausentes MCAR em ``fraction`` das celulas.

    O WDBC nao possui valores ausentes, o que deixaria o ramo de imputacao do
    pipeline sem exercicio. Esta funcao simula a realidade de um prontuario
    hospitalar (campos nao preenchidos) para **comprovar** que o pipeline
    continua funcionando e medir a degradacao do desempenho.
    """
    rng = np.random.default_rng(random_state)
    X_missing = X.copy()
    mask = rng.random(X.shape) < fraction
    X_missing = X_missing.mask(pd.DataFrame(mask, index=X.index, columns=X.columns))
    log.info(
        "injetados %d valores ausentes (%.1f%% das celulas)",
        int(mask.sum()),
        100 * mask.mean(),
    )
    return X_missing
