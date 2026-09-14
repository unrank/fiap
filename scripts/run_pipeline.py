"""
Pipeline completo do Tech Challenge - Fase 1 (tarefa principal).

Executa, de ponta a ponta:

    1. carga e auditoria de qualidade do dataset;
    2. analise exploratoria + figuras;
    3. analise de correlacao e multicolinearidade;
    4. separacao estratificada treino / validacao / teste;
    5. treino de 6 familias de modelos com busca de hiperparametros (CV);
    6. selecao do campeao (CV + validacao) e calibracao do limiar clinico;
    7. avaliacao final, uma unica vez, no conjunto de teste;
    8. interpretabilidade (importancia nativa, permutacao e SHAP);
    9. teste de robustez a dados ausentes;
   10. serializacao do modelo e de todas as metricas.

Uso
---
    python scripts/run_pipeline.py                # execucao padrao
    python scripts/run_pipeline.py --quick        # grade reduzida (~30 s)
    python scripts/run_pipeline.py --no-shap      # pula a etapa SHAP
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import warnings
from datetime import datetime, timezone
from pathlib import Path

# Ruido interno do scikit-learn ao paralelizar estimadores aninhados
# (CalibratedClassifierCV dentro do GridSearchCV). Nao afeta os resultados e
# poluiria o log da demonstracao; silenciamos apenas esta mensagem especifica.
warnings.filterwarnings(
    "ignore", message=".*sklearn.utils.parallel.delayed.*", category=UserWarning
)

import joblib
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from medai import data as data_mod  # noqa: E402
from medai import eda, evaluate, interpret, models  # noqa: E402
from medai.config import (  # noqa: E402
    CV_FOLDS,
    METRICS_DIR,
    MODELS_DIR,
    RANDOM_STATE,
    TARGET_RECALL,
    ensure_dirs,
    set_global_seed,
)
from medai.preprocessing import (  # noqa: E402
    correlation_analysis,
    inject_missing,
    split_train_val_test,
)
from medai.utils import get_logger, save_json, save_table, section  # noqa: E402

log = get_logger()


# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Pipeline de diagnostico - Fase 1")
    p.add_argument(
        "--source", default="auto", choices=["auto", "csv", "sklearn", "uci"],
        help="origem do dataset (padrao: auto)",
    )
    p.add_argument("--quick", action="store_true",
                   help="grade de hiperparametros reduzida, para demonstracao rapida")
    p.add_argument("--no-shap", action="store_true", help="pula a analise SHAP")
    p.add_argument("--no-figures", action="store_true", help="nao gera figuras")
    p.add_argument("--target-recall", type=float, default=TARGET_RECALL,
                   help=f"recall minimo exigido na triagem (padrao: {TARGET_RECALL})")
    p.add_argument("--cv-folds", type=int, default=CV_FOLDS)
    p.add_argument("--seed", type=int, default=RANDOM_STATE)
    return p.parse_args()


def shrink_grids(specs: list[models.ModelSpec]) -> list[models.ModelSpec]:
    """Reduz cada grade ao primeiro valor de cada hiperparametro (modo --quick)."""
    from dataclasses import replace

    return [
        replace(s, param_grid={k: v[:1] for k, v in s.param_grid.items()})
        for s in specs
    ]


# ---------------------------------------------------------------------------
def main() -> int:
    args = parse_args()
    ensure_dirs()
    set_global_seed(args.seed)
    started = time.perf_counter()

    manifest: dict = {
        "executado_em": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "semente": args.seed,
        "recall_alvo": args.target_recall,
        "modo_rapido": args.quick,
    }

    # ---------------------------------------------------------------- 1. dados
    section("1. carga e auditoria do dataset")
    df = data_mod.load_raw(source=args.source)
    audit = data_mod.audit(df)
    log.info("auditoria: %s", json.dumps(audit, ensure_ascii=False)[:400])
    save_json(audit, "01_auditoria_dataset")

    desc = data_mod.describe_numeric(df)
    save_table(desc, "02_estatisticas_descritivas")
    log.info("estatisticas descritivas salvas (%d variaveis)", len(desc))

    X, y = data_mod.split_features_target(df)
    manifest["dataset"] = {
        "nome": "Breast Cancer Wisconsin (Diagnostic) - WDBC",
        "n_amostras": int(len(df)),
        "n_features": int(X.shape[1]),
        "prevalencia_maligno": round(float(y.mean()), 4),
        "valores_ausentes": audit["missing_total"],
        "linhas_duplicadas": audit["duplicated_rows"],
    }

    # ------------------------------------------------------------------ 2. EDA
    section("2. analise exploratoria")
    top_feats = eda.top_discriminative_features(df, y, k=9)
    log.info("variaveis mais discriminantes: %s", ", ".join(top_feats[:5]))

    if not args.no_figures:
        eda.plot_class_balance(df)
        eda.plot_distributions(df, top_feats)
        eda.plot_boxplots(df, top_feats[:10])
        eda.plot_scale_and_skew(df)
        eda.plot_pca_projection(df)
        eda.plot_pairwise(df, top_feats[:4])

    # ---------------------------------------------------------- 3. correlacao
    section("3. analise de correlacao")
    corr = correlation_analysis(X, y, threshold=0.90)
    save_table(corr.with_target, "03_correlacao_com_alvo")
    save_table(corr.redundant_pairs, "04_pares_redundantes")
    save_json(
        {
            "limiar_multicolinearidade": 0.90,
            "n_pares_redundantes": int(len(corr.redundant_pairs)),
            "colunas_redundantes_sugeridas": corr.suggested_drop,
            "top10_correlacao_com_alvo": corr.with_target.head(10)[
                "corr_com_alvo"
            ].to_dict(),
        },
        "05_resumo_correlacao",
    )
    if not args.no_figures:
        eda.plot_correlation_heatmap(corr.matrix)
        eda.plot_target_correlation(corr.with_target)

    manifest["correlacao"] = {
        "pares_acima_de_0.90": int(len(corr.redundant_pairs)),
        "colunas_redundantes": corr.suggested_drop,
    }

    # -------------------------------------------------------------- 4. split
    section("4. separacao treino / validacao / teste")
    split = split_train_val_test(X, y, random_state=args.seed)
    summary = split.summary()
    save_table(summary, "06_particoes")
    log.info("\n%s", summary.to_string())
    manifest["particoes"] = summary.reset_index().to_dict(orient="records")

    # -------------------------------------------------------- 5. treinamento
    section("5. treinamento e busca de hiperparametros (CV no treino)")
    specs = models.model_catalog(random_state=args.seed)
    if args.quick:
        specs = shrink_grids(specs)
        log.warning("modo --quick: grades reduzidas a 1 combinacao por modelo")

    trained = models.train_all(
        split.X_train, split.y_train, specs=specs, cv_folds=args.cv_folds
    )
    cv_table = models.cv_summary(trained)
    save_table(cv_table, "07_resultados_validacao_cruzada")
    log.info("\n%s", cv_table.drop(columns=["melhores_params"]).to_string())

    # --------------------------------------------- 6. selecao na validacao
    section("6. avaliacao na validacao e escolha do campeao")
    val_results: dict[str, dict[str, float]] = {}
    val_scores: dict[str, np.ndarray] = {}
    for key, tm in trained.items():
        scores = models.predict_scores(tm.pipeline, split.X_val)
        val_scores[key] = scores
        val_results[tm.label] = evaluate.compute_metrics(split.y_val, scores)

    val_table = evaluate.metrics_table(val_results)
    save_table(val_table, "08_metricas_validacao")
    log.info("\n%s", val_table.to_string())

    # ------------------------------------------------------------------
    # Criterio de selecao do campeao
    # ------------------------------------------------------------------
    # Usar SO a ROC AUC da validacao seria fragil: com apenas 114 exames
    # (43 malignos) o erro-padrao da AUC beira +/- 0,02, e o ranking muda
    # conforme a semente - um modelo pode "vencer" por sorte de particao.
    # Usar SO a validacao cruzada ignoraria uma confirmacao independente.
    #
    # Adotamos a media das duas estimativas de ROC AUC:
    #   * CV 5-fold no treino (341 exames) -> estabilidade;
    #   * holdout de validacao (114 exames) -> confirmacao independente.
    # Ambas sao calculadas sem tocar no conjunto de teste. A metrica e a ROC
    # AUC porque independe do limiar, que so sera fixado na etapa seguinte.
    ranking = pd.DataFrame(
        [
            {
                "modelo": tm.label,
                "chave": key,
                "roc_auc_cv_treino": round(tm.cv_scores["roc_auc"], 4),
                "desvio_cv": round(tm.cv_scores["roc_auc_std"], 4),
                "roc_auc_validacao": round(val_results[tm.label]["roc_auc"], 4),
                "recall_validacao": round(val_results[tm.label]["recall"], 4),
                "score_selecao": round(
                    0.5 * tm.cv_scores["roc_auc"]
                    + 0.5 * val_results[tm.label]["roc_auc"],
                    5,
                ),
            }
            for key, tm in trained.items()
        ]
    ).sort_values("score_selecao", ascending=False).set_index("chave")

    save_table(ranking, "08b_ranking_selecao_campeao")
    log.info("\n%s", ranking.to_string())

    champion_key = str(ranking.index[0])
    champion = trained[champion_key]
    champion_label = champion.label
    log.info(
        "MODELO CAMPEAO: %s | score de selecao = %.4f "
        "(ROC AUC: CV %.4f | validacao %.4f)",
        champion_label,
        ranking.iloc[0]["score_selecao"],
        ranking.iloc[0]["roc_auc_cv_treino"],
        ranking.iloc[0]["roc_auc_validacao"],
    )
    manifest["selecao_campeao"] = {
        "criterio": "media da ROC AUC de CV(treino) e da ROC AUC de validacao",
        "campeao": champion_label,
        "ranking": ranking.reset_index().to_dict(orient="records"),
    }

    if not args.no_figures:
        evaluate.plot_model_comparison(val_table)
        evaluate.plot_roc_pr(
            {tm.label: (split.y_val, val_scores[k]) for k, tm in trained.items()}
        )

    # -------------------------------------------- 7. calibracao do limiar
    section("7. calibracao do limiar de decisao (na validacao)")
    choice = evaluate.tune_threshold(
        split.y_val, val_scores[champion_key], target_recall=args.target_recall
    )
    save_table(choice.curve.round(4), "09_curva_limiar_validacao")
    if not args.no_figures:
        evaluate.plot_threshold_analysis(choice)

    manifest["limiar"] = {
        "valor": round(choice.threshold, 4),
        "recall_validacao": round(choice.achieved_recall, 4),
        "precisao_validacao": round(choice.achieved_precision, 4),
        "recall_alvo": choice.target_recall,
    }

    # --------------------------------------------------- 8. teste final
    section("8. avaliacao final no conjunto de teste (aberto uma unica vez)")

    # O campeao e reajustado em treino + validacao para aproveitar todos os
    # dados disponiveis antes da medicao final. Os hiperparametros e o limiar
    # ja estao congelados, portanto nao ha vazamento.
    final_pipeline = models.build_pipeline(champion.spec, split.X_trainval)
    final_pipeline.set_params(
        **{f"model__{k}": v for k, v in champion.best_params.items()}
    )
    final_pipeline.fit(split.X_trainval, split.y_trainval)

    test_scores = models.predict_scores(final_pipeline, split.X_test)
    test_default = evaluate.compute_metrics(split.y_test, test_scores, 0.50)
    test_clinical = evaluate.compute_metrics(split.y_test, test_scores, choice.threshold)

    test_table = evaluate.metrics_table(
        {
            f"{champion_label} @ limiar padrao 0.50": test_default,
            f"{champion_label} @ limiar clinico {choice.threshold:.3f}": test_clinical,
        }
    )
    save_table(test_table, "10_metricas_teste")
    log.info("\n%s", test_table.to_string())

    # Todos os modelos no teste, para a tabela comparativa do relatorio.
    test_all: dict[str, dict[str, float]] = {}
    for key, tm in trained.items():
        s = models.predict_scores(tm.pipeline, split.X_test)
        test_all[tm.label] = evaluate.compute_metrics(split.y_test, s, 0.50)
    save_table(evaluate.metrics_table(test_all), "11_metricas_teste_todos_modelos")

    if not args.no_figures:
        evaluate.plot_confusion(
            split.y_test, test_scores, 0.50,
            f"Matriz de confusao - {champion_label} (teste)",
            "12_matriz_confusao_padrao",
        )
        evaluate.plot_confusion(
            split.y_test, test_scores, choice.threshold,
            f"Matriz de confusao - {champion_label} (teste, limiar clinico)",
            "13_matriz_confusao_clinica",
        )
        evaluate.plot_score_distribution(split.y_test, test_scores, choice.threshold)
        evaluate.plot_calibration(split.y_test, test_scores, champion_label)
        from sklearn.model_selection import StratifiedKFold

        evaluate.plot_learning_curve(
            models.build_pipeline(champion.spec, X), X, y,
            StratifiedKFold(n_splits=args.cv_folds, shuffle=True, random_state=args.seed),
            champion_label,
        )

    # Analise de erros + faixas de triagem
    errors = evaluate.error_analysis(
        split.X_test, split.y_test, test_scores, choice.threshold, top_feats
    )
    save_table(errors, "12_analise_de_erros_teste")
    log.info("erros no teste (limiar clinico): %d", len(errors))

    bands = evaluate.triage_bands(test_scores, low=choice.threshold, high=0.90)
    band_counts = bands.value_counts().to_dict()
    log.info("faixas de triagem no teste: %s", band_counts)

    manifest["teste"] = {
        "campeao": champion_label,
        "limiar_padrao_0.50": {k: round(v, 4) for k, v in test_default.items()},
        "limiar_clinico": {k: round(v, 4) for k, v in test_clinical.items()},
        "faixas_triagem": {str(k): int(v) for k, v in band_counts.items()},
        "n_erros_limiar_clinico": int(len(errors)),
    }

    # ------------------------------------------- 9. interpretabilidade
    section("9. interpretabilidade")
    native = interpret.native_importance(final_pipeline)
    if native is not None:
        save_table(native, "13_importancia_nativa")
        log.info("top-5 importancia nativa: %s", native["feature"].head(5).tolist())
        if not args.no_figures:
            interpret.plot_native_importance(native, champion_label)
    else:
        log.info("%s nao expoe importancia nativa (modelo caixa-preta)", champion_label)

    # Modelo de referencia "caixa de vidro". Independentemente de quem vence,
    # sempre publicamos os coeficientes de uma regressao logistica ajustada aos
    # mesmos dados: em medicina, razoes de chance (odds ratio) sao a linguagem
    # com que a equipe clinica audita e contesta um modelo.
    ref_spec = next(s for s in models.model_catalog(args.seed)
                    if s.key == "logistic_regression")
    ref_pipeline = models.build_pipeline(ref_spec, split.X_trainval)
    ref_pipeline.set_params(model__C=1.0)
    ref_pipeline.fit(split.X_trainval, split.y_trainval)
    ref_coefs = interpret.native_importance(ref_pipeline)
    save_table(ref_coefs, "17_referencia_logistica_odds_ratio")
    log.info(
        "referencia logistica - maiores odds ratio: %s",
        ref_coefs.head(3)[["feature", "odds_ratio_por_dp"]].round(2).to_dict("records"),
    )
    if not args.no_figures:
        interpret.plot_native_importance(
            ref_coefs, "Regressao Logistica (referencia interpretavel)",
            filename="18_referencia_logistica",
        )
    manifest["referencia_logistica"] = ref_coefs.head(10).round(4).to_dict("records")

    perm = interpret.permutation_report(
        final_pipeline, split.X_test, split.y_test, n_repeats=20, random_state=args.seed
    )
    save_table(perm, "14_importancia_permutacao")
    log.info("top-5 por permutacao: %s", perm["feature"].head(5).tolist())
    if not args.no_figures:
        interpret.plot_permutation_importance(perm, champion_label)

    manifest["interpretabilidade"] = {
        "top10_permutacao": perm.head(10)[["feature", "queda_media"]].round(5).to_dict(
            orient="records"
        )
    }

    if not args.no_shap:
        try:
            shap_res = interpret.compute_shap(
                final_pipeline, split.X_trainval, split.X_test,
                tree_based=champion.spec.tree_based,
            )
            shap_imp = shap_res.importance()
            save_table(shap_imp, "15_importancia_shap")
            log.info("top-5 SHAP: %s", shap_imp["feature"].head(5).tolist())
            manifest["interpretabilidade"]["top10_shap"] = (
                shap_imp.head(10).round(5).to_dict(orient="records")
            )

            if not args.no_figures:
                interpret.plot_shap_summary(shap_res, champion_label)
                interpret.plot_shap_bar(shap_res, champion_label)

                cases = interpret.pick_cases(
                    split.y_test, test_scores, choice.threshold
                )
                titles = {
                    "verdadeiro_positivo": "Caso corretamente sinalizado como MALIGNO",
                    "falso_positivo": "Falso positivo - benigno sinalizado como suspeito",
                    "falso_negativo": "FALSO NEGATIVO - maligno nao sinalizado",
                    "caso_ambiguo": "Caso ambiguo - score proximo do limiar",
                }
                for i, (case, pos) in enumerate(cases.items(), start=1):
                    interpret.plot_shap_waterfall(
                        shap_res, pos,
                        f"{titles[case]}\n"
                        f"probabilidade prevista = {test_scores[pos]:.3f}",
                        f"22_{i}_shap_caso_{case}",
                    )
                log.info("explicacoes individuais geradas: %s", list(cases))
        except Exception as exc:  # pragma: no cover - SHAP e opcional
            log.error("falha ao calcular SHAP (%s); seguindo sem esta etapa", exc)

    # ------------------------------------- 10. robustez a dados ausentes
    section("10. teste de robustez a valores ausentes")
    robustness = []
    for frac in (0.0, 0.05, 0.10, 0.20):
        X_test_missing = (
            split.X_test if frac == 0 else inject_missing(split.X_test, frac, args.seed)
        )
        s = models.predict_scores(final_pipeline, X_test_missing)
        m = evaluate.compute_metrics(split.y_test, s, choice.threshold)
        robustness.append(
            {
                "pct_celulas_ausentes": f"{frac:.0%}",
                "recall": round(m["recall"], 4),
                "precisao": round(m["precision"], 4),
                "f1": round(m["f1"], 4),
                "roc_auc": round(m["roc_auc"], 4),
                "falsos_negativos": m["fn"],
            }
        )
    robustness_df = pd.DataFrame(robustness).set_index("pct_celulas_ausentes")
    save_table(robustness_df, "16_robustez_dados_ausentes")
    log.info("\n%s", robustness_df.to_string())
    manifest["robustez_ausentes"] = robustness_df.reset_index().to_dict(orient="records")

    # ---------------------------------------------- 11. serializacao
    section("11. serializacao do modelo")
    artifact = {
        "pipeline": final_pipeline,
        "threshold": choice.threshold,
        "model_label": champion_label,
        "model_key": champion_key,
        "best_params": champion.best_params,
        "feature_order": list(X.columns),
        "class_names": ["Benigno", "Maligno"],
        "trained_at": manifest["executado_em"],
        "test_metrics": test_clinical,
        "medai_version": "1.0.0",
    }
    model_path = MODELS_DIR / "modelo_diagnostico.joblib"
    joblib.dump(artifact, model_path)
    log.info("modelo salvo em %s (%.1f KB)", model_path, model_path.stat().st_size / 1024)

    manifest["modelo_salvo"] = str(model_path.relative_to(model_path.parents[1]))
    manifest["melhores_hiperparametros"] = champion.best_params
    manifest["duracao_segundos"] = round(time.perf_counter() - started, 1)
    save_json(manifest, "00_manifesto_execucao")

    section("pipeline concluido")
    log.info("duracao total: %.1f s", manifest["duracao_segundos"])
    log.info("metricas -> %s", METRICS_DIR)
    log.info("figuras  -> %s", METRICS_DIR.parent / "figures")
    log.info(
        "RESUMO | campeao=%s | teste: recall=%.4f f1=%.4f acuracia=%.4f FN=%d",
        champion_label,
        test_clinical["recall"],
        test_clinical["f1"],
        test_clinical["accuracy"],
        test_clinical["fn"],
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
