"""
EXTRA - Deteccao de pneumonia em radiografias de torax com CNN.

Executa o pipeline de visao computacional de ponta a ponta:

    1. download e inspecao do PneumoniaMNIST (Chest X-Ray de Kermany et al.);
    2. treino de uma CNN compacta com data augmentation e perda ponderada;
    3. selecao de epoca pela ROC AUC de validacao;
    4. calibracao do limiar clinico na validacao;
    5. avaliacao final no conjunto de teste oficial;
    6. interpretabilidade visual com Grad-CAM.

Uso
---
    python scripts/run_cnn.py                       # 12 epocas, 28x28
    python scripts/run_cnn.py --epochs 20 --size 64 # mais fiel, mais lento
    python scripts/run_cnn.py --epochs 3            # demonstracao rapida
"""

from __future__ import annotations

import argparse
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from medai import cnn as cnn_mod  # noqa: E402
from medai import evaluate  # noqa: E402
from medai.config import MODELS_DIR, RANDOM_STATE, ensure_dirs, set_global_seed  # noqa: E402
from medai.utils import get_logger, save_json, save_table, section  # noqa: E402

log = get_logger()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="EXTRA - CNN para pneumonia (Fase 1)")
    p.add_argument("--epochs", type=int, default=12)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--size", type=int, default=28, choices=[28, 64, 128, 224],
                   help="resolucao das imagens (28 e a padrao do MedMNIST)")
    p.add_argument("--target-recall", type=float, default=0.98,
                   help="sensibilidade minima exigida para pneumonia")
    p.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    p.add_argument("--no-figures", action="store_true")
    p.add_argument("--seed", type=int, default=RANDOM_STATE)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    ensure_dirs()
    set_global_seed(args.seed)
    started = time.perf_counter()

    import torch

    torch.set_num_threads(max(1, (torch.get_num_threads() or 4)))

    # ------------------------------------------------------------ 1. dados
    section("EXTRA 1. carga do PneumoniaMNIST")
    loaders, datasets, info = cnn_mod.load_pneumonia(
        image_size=args.size, batch_size=args.batch_size
    )
    log.info("descricao oficial: %s", info["description"][:160])
    if not args.no_figures:
        cnn_mod.plot_samples(datasets["train"])

    # ------------------------------------------------------------ 2. modelo
    section("EXTRA 2. treinamento da CNN")
    model = cnn_mod.build_cnn(image_size=args.size)
    n_params = cnn_mod.count_parameters(model)
    log.info("CNN com %s parametros treinaveis", f"{n_params:,}".replace(",", "."))

    model, history = cnn_mod.train_cnn(
        model, loaders, epochs=args.epochs, lr=args.lr, device=args.device
    )
    if not args.no_figures:
        cnn_mod.plot_history(history)

    # ------------------------------------------------- 3. limiar na validacao
    section("EXTRA 3. calibracao do limiar na validacao")
    y_val, s_val = cnn_mod.evaluate_cnn(model, loaders["val"], args.device)
    choice = evaluate.tune_threshold(y_val, s_val, target_recall=args.target_recall)

    # ------------------------------------------------------- 4. teste final
    section("EXTRA 4. avaliacao no conjunto de teste oficial")
    y_test, s_test = cnn_mod.evaluate_cnn(model, loaders["test"], args.device)

    m_default = evaluate.compute_metrics(y_test, s_test, 0.50)
    m_clinical = evaluate.compute_metrics(y_test, s_test, choice.threshold)
    table = evaluate.metrics_table(
        {
            "CNN @ limiar padrao 0.50": m_default,
            f"CNN @ limiar clinico {choice.threshold:.3f}": m_clinical,
        }
    )
    save_table(table, "20_metricas_cnn_teste")
    log.info("\n%s", table.to_string())

    if not args.no_figures:
        evaluate.plot_confusion(
            y_test, s_test, choice.threshold,
            "CNN - deteccao de pneumonia (teste)", "32_cnn_matriz_confusao",
        )
        cnn_mod.plot_predictions_grid(model, datasets["test"], s_test, choice.threshold)

        # Grad-CAM: 2 acertos confiantes + 2 erros, para inspecao clinica.
        y_pred = (s_test >= choice.threshold).astype(int)
        wrong = np.where(y_pred != y_test)[0]
        confident = np.argsort(-np.abs(s_test - 0.5))[:2]
        picks = list(confident) + list(wrong[:2])
        cnn_mod.plot_gradcam(model, datasets["test"], [int(i) for i in picks],
                             device=args.device)

    # ------------------------------------------------------ 5. serializacao
    section("EXTRA 5. serializacao")
    ckpt_path = MODELS_DIR / "cnn_pneumonia.pt"
    torch.save(
        {
            "state_dict": model.state_dict(),
            "image_size": args.size,
            "threshold": choice.threshold,
            "class_labels": list(cnn_mod.CLASS_LABELS),
            "n_params": n_params,
            "test_metrics": m_clinical,
            "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        },
        ckpt_path,
    )
    log.info("checkpoint salvo em %s (%.1f MB)",
             ckpt_path, ckpt_path.stat().st_size / 1024**2)

    save_json(
        {
            "dataset": "PneumoniaMNIST (MedMNIST v2 / Chest X-Ray - Kermany et al.)",
            "resolucao": f"{args.size}x{args.size}",
            "n_treino": len(datasets["train"]),
            "n_validacao": len(datasets["val"]),
            "n_teste": len(datasets["test"]),
            "epocas": args.epochs,
            "melhor_epoca": history.best_epoch,
            "auc_validacao": round(history.best_val_auc, 4),
            "parametros_treinaveis": n_params,
            "limiar_clinico": round(choice.threshold, 4),
            "teste_limiar_padrao": {k: round(v, 4) for k, v in m_default.items()},
            "teste_limiar_clinico": {k: round(v, 4) for k, v in m_clinical.items()},
            "duracao_segundos": round(time.perf_counter() - started, 1),
        },
        "21_manifesto_cnn",
    )

    section("EXTRA concluido")
    log.info(
        "RESUMO CNN | teste: recall=%.4f f1=%.4f acuracia=%.4f AUC=%.4f | %.1fs",
        m_clinical["recall"], m_clinical["f1"], m_clinical["accuracy"],
        m_clinical["roc_auc"], time.perf_counter() - started,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
