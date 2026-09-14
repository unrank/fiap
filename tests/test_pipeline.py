"""
Testes automatizados das partes criticas do pipeline.

Rodar com:  pytest -q

Os testes cobrem o que, se quebrar, invalida silenciosamente o resultado
cientifico: vazamento de dados entre particoes, tratamento de valores
ausentes, ramo categorico do pre-processador e o calculo das metricas.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from medai import data as data_mod
from medai import evaluate, models
from medai.config import LABEL_MAP
from medai.preprocessing import (
    build_preprocessor,
    correlation_analysis,
    inject_missing,
    split_train_val_test,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def raw_df() -> pd.DataFrame:
    return data_mod.load_raw()


@pytest.fixture(scope="module")
def xy(raw_df):
    return data_mod.split_features_target(raw_df)


# ---------------------------------------------------------------------------
# Dados
# ---------------------------------------------------------------------------
def test_dataset_tem_a_forma_esperada(raw_df):
    assert len(raw_df) == 569
    assert set(raw_df["diagnosis"].unique()) == {"B", "M"}
    for col in ("radius_mean", "concave_points_worst", "fractal_dimension_se"):
        assert col in raw_df.columns


def test_auditoria_nao_encontra_problemas(raw_df):
    report = data_mod.audit(raw_df)
    assert report["missing_total"] == 0
    assert report["duplicated_rows"] == 0
    assert report["constant_columns"] == []
    assert report["impossible_zeros"] == {}


def test_alvo_codificado_com_maligno_como_classe_positiva(xy):
    _, y = xy
    assert set(np.unique(y)) == {0, 1}
    assert LABEL_MAP["M"] == 1
    assert int((y == 1).sum()) == 212  # malignos no WDBC


def test_identificador_nao_entra_como_preditora(raw_df):
    df = raw_df.copy()
    df.insert(0, "id", range(len(df)))
    X, _ = data_mod.split_features_target(df)
    assert "id" not in X.columns
    assert "diagnosis" not in X.columns


# ---------------------------------------------------------------------------
# Separacao das particoes
# ---------------------------------------------------------------------------
def test_particoes_sao_disjuntas_e_completas(xy):
    X, y = xy
    s = split_train_val_test(X, y)
    idx_train, idx_val, idx_test = (
        set(s.X_train.index), set(s.X_val.index), set(s.X_test.index)
    )
    assert not (idx_train & idx_val)
    assert not (idx_train & idx_test)
    assert not (idx_val & idx_test)
    assert len(idx_train | idx_val | idx_test) == len(X)


def test_particoes_preservam_a_prevalencia(xy):
    X, y = xy
    s = split_train_val_test(X, y)
    base = float(y.mean())
    for part in (s.y_train, s.y_val, s.y_test):
        assert abs(float(part.mean()) - base) < 0.02


# ---------------------------------------------------------------------------
# Pre-processamento
# ---------------------------------------------------------------------------
def test_escala_e_aprendida_somente_no_treino(xy):
    """Guarda contra vazamento: o scaler nao pode ver validacao nem teste."""
    X, y = xy
    s = split_train_val_test(X, y)
    pre = build_preprocessor(s.X_train)
    pre.fit(s.X_train)

    scaler = pre.named_transformers_["num"].named_steps["scaler"]
    esperado = s.X_train.mean().to_numpy()
    assert np.allclose(scaler.mean_, esperado, atol=1e-9)

    # A media do treino transformado deve ser ~0; a do teste, nao exatamente.
    assert abs(float(np.asarray(pre.transform(s.X_train)).mean())) < 1e-9


def test_ramo_categorico_funciona_com_dados_mistos():
    """O ColumnTransformer precisa lidar com colunas categoricas e faltantes."""
    df = pd.DataFrame(
        {
            "idade": [45.0, 62.0, np.nan, 51.0, 70.0],
            "sexo": ["F", "F", "M", None, "F"],
            "unidade": ["HC", "HC", "UPA", "UPA", "HC"],
        }
    )
    pre = build_preprocessor(df)
    out = np.asarray(pre.fit_transform(df))

    assert not np.isnan(out).any(), "imputacao deixou NaN passar"
    # 1 numerica + one-hot de sexo (F/M) + one-hot de unidade (HC/UPA) = 5
    assert out.shape == (5, 5)

    # Categoria nunca vista nao pode quebrar a inferencia.
    novo = pd.DataFrame({"idade": [30.0], "sexo": ["X"], "unidade": ["Pronto-socorro"]})
    assert np.asarray(pre.transform(novo)).shape == (1, 5)


def test_pipeline_sobrevive_a_valores_ausentes(xy):
    X, y = xy
    s = split_train_val_test(X, y)
    spec = next(m for m in models.model_catalog() if m.key == "logistic_regression")
    pipe = models.build_pipeline(spec, s.X_train)
    pipe.fit(s.X_train, s.y_train)

    X_missing = inject_missing(s.X_test, fraction=0.15)
    assert X_missing.isna().to_numpy().sum() > 0

    proba = pipe.predict_proba(X_missing)[:, 1]
    assert proba.shape == (len(s.X_test),)
    assert np.isfinite(proba).all()


def test_correlacao_detecta_multicolinearidade(xy):
    X, y = xy
    report = correlation_analysis(X, y, threshold=0.90)
    assert not report.redundant_pairs.empty
    # raio e perimetro descrevem a mesma geometria: tem de aparecer.
    pares = set(
        zip(report.redundant_pairs["feature_a"], report.redundant_pairs["feature_b"])
    )
    assert ("radius_mean", "perimeter_mean") in pares or (
        "perimeter_mean", "radius_mean"
    ) in pares
    assert report.with_target.index[0].endswith(("_worst", "_mean"))


# ---------------------------------------------------------------------------
# Metricas e limiar
# ---------------------------------------------------------------------------
def test_metricas_batem_com_calculo_manual():
    y_true = np.array([1, 1, 1, 0, 0, 0, 0, 0])
    y_scores = np.array([0.9, 0.8, 0.2, 0.7, 0.1, 0.1, 0.3, 0.05])
    m = evaluate.compute_metrics(y_true, y_scores, threshold=0.5)

    assert (m["tp"], m["fn"], m["fp"], m["tn"]) == (2, 1, 1, 4)
    assert m["recall"] == pytest.approx(2 / 3)
    assert m["precision"] == pytest.approx(2 / 3)
    assert m["specificity"] == pytest.approx(4 / 5)
    assert m["accuracy"] == pytest.approx(6 / 8)
    assert m["f1"] == pytest.approx(2 / 3)


def test_limiar_calibrado_atinge_o_recall_exigido():
    rng = np.random.default_rng(0)
    y_true = np.r_[np.ones(40, dtype=int), np.zeros(60, dtype=int)]
    y_scores = np.r_[
        rng.beta(6, 2, 40),  # malignos concentrados em scores altos
        rng.beta(2, 6, 60),  # benignos em scores baixos
    ]
    choice = evaluate.tune_threshold(y_true, y_scores, target_recall=0.95)

    assert choice.achieved_recall >= 0.95
    # e o limiar tem de ser o MAIOR que ainda atende a meta
    acima = (y_scores >= choice.threshold + 1e-6)
    recall_acima = ((acima) & (y_true == 1)).sum() / (y_true == 1).sum()
    assert recall_acima < 0.95


def test_faixas_de_triagem_sao_monotonicas():
    scores = np.array([0.05, 0.40, 0.60, 0.95])
    bands = evaluate.triage_bands(scores, low=0.35, high=0.90).tolist()
    assert bands == ["verde", "amarelo", "amarelo", "vermelho"]


# ---------------------------------------------------------------------------
# Modelo treinado ponta a ponta
# ---------------------------------------------------------------------------
def test_modelo_simples_atinge_desempenho_clinico_minimo(xy):
    """Regressao logistica basica precisa superar um patamar minimo no teste."""
    X, y = xy
    s = split_train_val_test(X, y)
    spec = next(m for m in models.model_catalog() if m.key == "logistic_regression")
    pipe = models.build_pipeline(spec, s.X_train)
    pipe.fit(s.X_train, s.y_train)

    scores = models.predict_scores(pipe, s.X_test)
    m = evaluate.compute_metrics(s.y_test, scores, 0.5)
    assert m["roc_auc"] > 0.97
    assert m["recall"] > 0.85


def test_artefato_serializado_reproduz_as_previsoes(tmp_path, xy):
    import joblib

    X, y = xy
    s = split_train_val_test(X, y)
    spec = next(m for m in models.model_catalog() if m.key == "decision_tree")
    pipe = models.build_pipeline(spec, s.X_train)
    pipe.fit(s.X_train, s.y_train)

    antes = pipe.predict_proba(s.X_test)[:, 1]
    path = tmp_path / "modelo.joblib"
    joblib.dump({"pipeline": pipe, "feature_order": list(X.columns)}, path)
    depois = joblib.load(path)["pipeline"].predict_proba(s.X_test)[:, 1]

    assert np.allclose(antes, depois)
