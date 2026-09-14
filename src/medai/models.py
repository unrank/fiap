"""
Catalogo de modelos, grades de hiperparametros e rotina de treinamento.

Estrategia de selecao
---------------------
1. Cada candidato e um ``Pipeline`` completo: pre-processamento + estimador.
2. Os hiperparametros sao buscados por ``GridSearchCV`` com
   ``StratifiedKFold`` **apenas no conjunto de treino**.
3. A metrica de *refit* da busca e a **ROC AUC**, que independe do limiar de
   decisao e mede a capacidade de ordenar corretamente os pacientes por risco.
   Otimizar recall diretamente na busca levaria a solucoes degeneradas
   (classificar todo mundo como maligno tem recall 1,0 e e clinicamente
   inutil).
4. Escolhido o modelo, o **limiar** e calibrado a parte no conjunto de
   validacao para atingir o recall minimo exigido pela triagem
   (ver ``evaluate.tune_threshold``).
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from .config import CV_FOLDS, RANDOM_STATE
from .preprocessing import ScalerName, build_preprocessor
from .utils import get_logger

log = get_logger()


@dataclass(frozen=True)
class ModelSpec:
    """Descreve um candidato: estimador, grade de busca e justificativa."""

    key: str
    label: str
    estimator: BaseEstimator
    param_grid: dict[str, list[Any]]
    rationale: str
    scaler: ScalerName = "standard"
    #: True quando o modelo expoe importancia/coeficientes de forma nativa.
    interpretable: bool = False
    #: True para modelos baseados em arvore (SHAP usa o TreeExplainer).
    tree_based: bool = False


def model_catalog(random_state: int = RANDOM_STATE) -> list[ModelSpec]:
    """Devolve os seis modelos candidatos avaliados no desafio."""
    return [
        ModelSpec(
            key="logistic_regression",
            label="Regressao Logistica",
            estimator=LogisticRegression(
                max_iter=5000, solver="liblinear", random_state=random_state
            ),
            param_grid={
                # C controla a forca da regularizacao L2, que e o antidoto
                # contra a forte multicolinearidade das 30 preditoras.
                "model__C": [0.005, 0.01, 0.05, 0.1, 0.5, 1.0, 10.0, 100.0],
                "model__class_weight": [None, "balanced"],
            },
            rationale=(
                "Padrao-ouro em estatistica medica: os coeficientes viram "
                "razoes de chance (odds ratio), grandeza que a equipe clinica "
                "ja interpreta. Serve de linha de base honesta e e o modelo "
                "mais facil de auditar e defender perante um comite de etica."
            ),
            interpretable=True,
        ),
        ModelSpec(
            key="decision_tree",
            label="Arvore de Decisao",
            estimator=DecisionTreeClassifier(random_state=random_state),
            param_grid={
                "model__max_depth": [3, 4, 5, 7, None],
                "model__min_samples_leaf": [1, 3, 5, 10],
                "model__criterion": ["gini", "entropy"],
                "model__class_weight": [None, "balanced"],
            },
            rationale=(
                "Produz regras explicitas do tipo 'se raio_worst > 16,8 entao "
                "suspeitar de malignidade'. E o modelo mais proximo de um "
                "protocolo clinico escrito, embora sozinho seja instavel."
            ),
            interpretable=True,
            tree_based=True,
        ),
        ModelSpec(
            key="knn",
            label="KNN (k-vizinhos mais proximos)",
            estimator=KNeighborsClassifier(),
            param_grid={
                "model__n_neighbors": [3, 5, 7, 9, 11, 15],
                "model__weights": ["uniform", "distance"],
                "model__p": [1, 2],
            },
            rationale=(
                "Raciocinio por analogia - classifica o paciente pelos casos "
                "historicos mais parecidos, logica proxima da consulta a "
                "casos anteriores. Exige padronizacao de escala e serve para "
                "testar se a separacao entre classes e local ou global."
            ),
        ),
        ModelSpec(
            key="random_forest",
            label="Random Forest",
            estimator=RandomForestClassifier(
                random_state=random_state, n_jobs=-1
            ),
            param_grid={
                "model__n_estimators": [300, 600],
                "model__max_depth": [None, 6, 10],
                "model__min_samples_leaf": [1, 2, 4],
                "model__max_features": ["sqrt", 0.5],
                "model__class_weight": [None, "balanced"],
            },
            rationale=(
                "Conjunto de arvores decorrelacionadas: absorve bem a forte "
                "multicolinearidade do WDBC, captura interacoes nao lineares e "
                "e robusto a outliers. Fornece feature importance nativa e "
                "aceita SHAP exato via TreeExplainer."
            ),
            interpretable=True,
            tree_based=True,
        ),
        ModelSpec(
            key="gradient_boosting",
            label="Gradient Boosting",
            estimator=GradientBoostingClassifier(random_state=random_state),
            param_grid={
                "model__n_estimators": [150, 300],
                "model__learning_rate": [0.05, 0.1],
                "model__max_depth": [2, 3],
                "model__subsample": [0.8, 1.0],
            },
            rationale=(
                "Boosting sequencial: cada arvore corrige o erro da anterior. "
                "Costuma ser o teto de desempenho em dados tabulares e serve "
                "para verificar se os modelos simples ja saturaram o problema."
            ),
            interpretable=True,
            tree_based=True,
        ),
        ModelSpec(
            key="svm_rbf",
            label="SVM (kernel RBF)",
            # O SVC nao produz probabilidades por natureza (so distancias ate a
            # margem). Envolve-lo em CalibratedClassifierCV aplica escalonamento
            # de Platt e devolve probabilidades calibradas - requisito do nosso
            # limiar clinico, que opera sobre a probabilidade e nao sobre o
            # sinal da decisao.
            estimator=CalibratedClassifierCV(
                SVC(random_state=random_state), method="sigmoid", ensemble=False, cv=5
            ),
            param_grid={
                "model__estimator__C": [0.1, 1.0, 10.0, 100.0],
                "model__estimator__gamma": ["scale", 0.01, 0.1],
                "model__estimator__class_weight": [None, "balanced"],
            },
            rationale=(
                "Maximiza a margem entre as classes em um espaco de alta "
                "dimensao - historicamente muito forte em bases pequenas e "
                "largas (569 x 30) como esta. Envolvido em CalibratedClassifierCV "
                "para produzir probabilidades calibradas. Custo: e uma caixa-preta e "
                "exige SHAP por permutacao (mais lento)."
            ),
        ),
    ]


def build_pipeline(spec: ModelSpec, X: pd.DataFrame) -> Pipeline:
    """Monta ``pre-processamento -> estimador`` como um unico objeto."""
    return Pipeline(
        steps=[
            ("preprocessor", build_preprocessor(X, scaler=spec.scaler)),
            ("model", spec.estimator),
        ]
    )


@dataclass
class TrainedModel:
    """Resultado do treinamento e da busca de hiperparametros de um candidato."""

    spec: ModelSpec
    pipeline: Pipeline
    best_params: dict[str, Any]
    cv_results: pd.DataFrame
    cv_best_score: float
    cv_scores: dict[str, float] = field(default_factory=dict)
    fit_seconds: float = 0.0

    @property
    def key(self) -> str:
        return self.spec.key

    @property
    def label(self) -> str:
        return self.spec.label


def train_candidate(
    spec: ModelSpec,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    cv_folds: int = CV_FOLDS,
    refit_metric: str = "roc_auc",
    random_state: int = RANDOM_STATE,
) -> TrainedModel:
    """Executa a busca em grade de um candidato e devolve o melhor pipeline."""
    pipeline = build_pipeline(spec, X_train)
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)

    scoring = {
        "roc_auc": "roc_auc",
        "average_precision": "average_precision",
        "recall": "recall",
        "precision": "precision",
        "f1": "f1",
        "accuracy": "accuracy",
    }

    search = GridSearchCV(
        estimator=pipeline,
        param_grid=spec.param_grid,
        scoring=scoring,
        refit=refit_metric,
        cv=cv,
        n_jobs=-1,
        return_train_score=True,
        error_score="raise",
    )

    t0 = time.perf_counter()
    search.fit(X_train, y_train)
    elapsed = time.perf_counter() - t0

    idx = int(search.best_index_)
    cv_scores = {
        name: float(search.cv_results_[f"mean_test_{name}"][idx]) for name in scoring
    }
    cv_scores["roc_auc_std"] = float(search.cv_results_["std_test_roc_auc"][idx])
    # Diferenca treino - validacao na CV: diagnostico rapido de overfitting.
    cv_scores["overfit_gap_roc_auc"] = float(
        search.cv_results_["mean_train_roc_auc"][idx] - cv_scores["roc_auc"]
    )

    log.info(
        "%-22s | CV %s = %.4f (+/- %.4f) | recall = %.4f | %d combinacoes em %.1fs",
        spec.key,
        refit_metric,
        search.best_score_,
        cv_scores["roc_auc_std"],
        cv_scores["recall"],
        len(search.cv_results_["params"]),
        elapsed,
    )

    return TrainedModel(
        spec=spec,
        pipeline=search.best_estimator_,
        best_params={k.replace("model__", ""): v for k, v in search.best_params_.items()},
        cv_results=pd.DataFrame(search.cv_results_),
        cv_best_score=float(search.best_score_),
        cv_scores=cv_scores,
        fit_seconds=elapsed,
    )


def train_all(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    specs: list[ModelSpec] | None = None,
    cv_folds: int = CV_FOLDS,
    refit_metric: str = "roc_auc",
) -> dict[str, TrainedModel]:
    """Treina todos os candidatos do catalogo e devolve um dicionario por chave."""
    specs = specs or model_catalog()
    trained: dict[str, TrainedModel] = {}
    for spec in specs:
        trained[spec.key] = train_candidate(
            spec, X_train, y_train, cv_folds=cv_folds, refit_metric=refit_metric
        )
    return trained


def cv_summary(trained: dict[str, TrainedModel]) -> pd.DataFrame:
    """Tabela comparativa dos resultados de validacao cruzada no treino."""
    rows = []
    for tm in trained.values():
        row = {"modelo": tm.label, "chave": tm.key}
        row.update({k: round(v, 4) for k, v in tm.cv_scores.items()})
        row["tempo_busca_s"] = round(tm.fit_seconds, 1)
        row["melhores_params"] = str(tm.best_params)
        rows.append(row)
    df = pd.DataFrame(rows).set_index("chave")
    return df.sort_values("roc_auc", ascending=False)


def predict_scores(pipeline: Pipeline, X: pd.DataFrame) -> np.ndarray:
    """
    Devolve a probabilidade estimada de malignidade (classe 1).

    Usa ``predict_proba`` quando disponivel e cai para ``decision_function``
    normalizada nos estimadores que nao expoem probabilidade.
    """
    if hasattr(pipeline, "predict_proba"):
        return pipeline.predict_proba(X)[:, 1]
    scores = pipeline.decision_function(X)
    return 1.0 / (1.0 + np.exp(-scores))
