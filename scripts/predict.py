"""
Inferencia: aplica o modelo treinado a novos exames e emite o laudo de triagem.

Este script representa o ponto de contato do sistema com o hospital: recebe um
CSV de exames (mesmo formato do dataset, sem a coluna ``diagnosis``) e devolve,
para cada paciente, a probabilidade de malignidade e a faixa de triagem.

**A saida e um apoio a decisao, nunca um diagnostico.** O laudo final e sempre
do medico responsavel.

Uso
---
    python scripts/predict.py --input exames.csv --output laudo.csv
    python scripts/predict.py --demo            # usa 10 casos do proprio dataset
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from medai import data as data_mod  # noqa: E402
from medai.config import MODELS_DIR, RANDOM_STATE  # noqa: E402
from medai.utils import get_logger  # noqa: E402

log = get_logger()

DEFAULT_MODEL = MODELS_DIR / "modelo_diagnostico.joblib"

#: Limite superior da zona cinzenta. Acima disso o encaminhamento e imediato.
HIGH_RISK = 0.90


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Triagem assistida por IA - Fase 1")
    p.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    p.add_argument("--input", type=Path, help="CSV com os exames a serem triados")
    p.add_argument("--output", type=Path, help="CSV de saida com o laudo")
    p.add_argument("--demo", action="store_true",
                   help="roda uma demonstracao com 10 casos do proprio dataset")
    p.add_argument("--n-demo", type=int, default=10)
    return p.parse_args()


def load_artifact(path: Path) -> dict:
    if not path.exists():
        raise SystemExit(
            f"modelo nao encontrado em {path}\n"
            "execute primeiro:  python scripts/run_pipeline.py"
        )
    artifact = joblib.load(path)
    log.info(
        "modelo carregado: %s | limiar clinico = %.4f | treinado em %s",
        artifact["model_label"], artifact["threshold"], artifact["trained_at"],
    )
    return artifact


def build_report(
    artifact: dict,
    X: pd.DataFrame,
    y_true: pd.Series | None = None,
) -> pd.DataFrame:
    """Aplica o modelo e monta o laudo de triagem."""
    pipeline = artifact["pipeline"]
    threshold = float(artifact["threshold"])

    missing = [c for c in artifact["feature_order"] if c not in X.columns]
    if missing:
        raise SystemExit(
            f"faltam {len(missing)} coluna(s) no arquivo de entrada: {missing[:6]}..."
        )
    X = X[artifact["feature_order"]]

    proba = pipeline.predict_proba(X)[:, 1]
    band = np.where(
        proba >= HIGH_RISK, "VERMELHO",
        np.where(proba >= threshold, "AMARELO", "VERDE"),
    )
    action = {
        "VERMELHO": "Encaminhamento imediato ao mastologista",
        "AMARELO": "Revisao humana prioritaria (zona de incerteza)",
        "VERDE": "Seguir fila de rotina",
    }

    report = pd.DataFrame(
        {
            "exame": X.index,
            "prob_malignidade": np.round(proba, 4),
            "sugestao_ia": np.where(proba >= threshold, "SUSPEITO", "sem sinais"),
            "faixa_triagem": band,
            "conduta_sugerida": [action[b] for b in band],
        }
    )
    if y_true is not None:
        report["diagnostico_real"] = np.where(
            np.asarray(y_true) == 1, "Maligno", "Benigno"
        )
        report["acertou"] = (
            (proba >= threshold).astype(int) == np.asarray(y_true)
        )
    return report


def main() -> int:
    args = parse_args()
    artifact = load_artifact(args.model)

    if args.demo or args.input is None:
        log.info("modo demonstracao: amostrando %d exames do dataset", args.n_demo)
        df = data_mod.load_raw()
        X_all, y_all = data_mod.split_features_target(df)
        idx = (
            X_all.sample(args.n_demo, random_state=RANDOM_STATE).index
            if args.n_demo < len(X_all)
            else X_all.index
        )
        report = build_report(artifact, X_all.loc[idx], y_all.loc[idx])
    else:
        # normalize() aceita tanto o CSV do Kaggle quanto o gerado pelo pipeline.
        raw = data_mod.normalize(pd.read_csv(args.input))
        y_true = None
        if "diagnosis" in raw.columns:
            X, y_true = data_mod.split_features_target(raw)
        else:
            X = raw.drop(columns=[c for c in ("id",) if c in raw.columns])
        report = build_report(artifact, X, y_true)

    pd.set_option("display.width", 200)
    print()
    print("=" * 100)
    print("  LAUDO DE TRIAGEM ASSISTIDA POR IA - APOIO A DECISAO, NAO DIAGNOSTICO")
    print(f"  modelo: {artifact['model_label']} | limiar clinico: {artifact['threshold']:.4f}")
    print("=" * 100)
    print(report.to_string(index=False))
    print("=" * 100)
    print("  A palavra final sobre o diagnostico e sempre do medico responsavel.")
    print("=" * 100)
    print()

    if "acertou" in report.columns:
        log.info("acertos nesta amostra: %d de %d",
                 int(report["acertou"].sum()), len(report))
        if args.demo or args.input is None:
            log.warning(
                "amostra de demonstracao retirada do dataset completo - parte dela foi "
                "vista no treino. NAO use este numero como estimativa de desempenho; "
                "as metricas validas estao em reports/metrics/10_metricas_teste.csv"
            )
    log.info("distribuicao das faixas: %s", report["faixa_triagem"].value_counts().to_dict())

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        report.to_csv(args.output, index=False, encoding="utf-8")
        log.info("laudo gravado em %s", args.output)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
