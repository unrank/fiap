"""
EXTRA - Visao computacional: deteccao de pneumonia em radiografias de torax.

Dataset
-------
**PneumoniaMNIST** (colecao MedMNIST v2), que e a versao padronizada e
oficialmente curada do *Chest X-Ray Pneumonia* de Kermany et al. - a mesma
base indicada no enunciado do desafio, porem ja dividida em treino/validacao/
teste pelos autores, sem necessidade de credenciais do Kaggle e com download
de poucos MB. Sao 5.856 radiografias pediatricas rotuladas em
``0 = normal`` e ``1 = pneumonia``.

    Yang et al. (2023). *MedMNIST v2 - A large-scale lightweight benchmark
    for 2D and 3D biomedical image classification*. Scientific Data 10, 41.

Modelo
------
CNN compacta treinada do zero (3 blocos convolucionais + cabeca densa),
projetada para convergir em CPU em poucos minutos. Como a base e desbalanceada
(~74% de pneumonia no treino), a funcao de perda usa ``pos_weight`` e a
selecao de epoca e feita pela **ROC AUC de validacao**, nao pela acuracia.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .config import MEDMNIST_DIR, PLOT, RANDOM_STATE
from .utils import apply_plot_style, get_logger, save_fig

log = get_logger()

DATA_FLAG = "pneumoniamnist"
CLASS_LABELS: tuple[str, str] = ("Normal", "Pneumonia")


# ---------------------------------------------------------------------------
# Dados
# ---------------------------------------------------------------------------
def load_pneumonia(image_size: int = 28, batch_size: int = 128, num_workers: int = 0):
    """
    Baixa (uma unica vez) e devolve os ``DataLoader`` de treino/validacao/teste.

    O *data augmentation* aplicado no treino (pequenas rotacoes, translacoes e
    espelhamento horizontal) simula variacoes reais de posicionamento do
    paciente no aparelho de raio-X e reduz o sobreajuste.
    """
    import medmnist
    import torch
    from medmnist import INFO
    from torch.utils.data import DataLoader
    from torchvision import transforms

    MEDMNIST_DIR.mkdir(parents=True, exist_ok=True)
    info = INFO[DATA_FLAG]
    DataClass = getattr(medmnist, info["python_class"])

    normalize = transforms.Normalize(mean=[0.5], std=[0.5])
    train_tf = transforms.Compose(
        [
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomAffine(degrees=8, translate=(0.05, 0.05), scale=(0.95, 1.05)),
            transforms.ToTensor(),
            normalize,
        ]
    )
    eval_tf = transforms.Compose([transforms.ToTensor(), normalize])

    kwargs: dict = {"root": str(MEDMNIST_DIR), "download": True}
    if image_size != 28:
        kwargs["size"] = image_size

    datasets = {
        "train": DataClass(split="train", transform=train_tf, **kwargs),
        "val": DataClass(split="val", transform=eval_tf, **kwargs),
        "test": DataClass(split="test", transform=eval_tf, **kwargs),
    }

    generator = torch.Generator().manual_seed(RANDOM_STATE)
    loaders = {
        name: DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=(name == "train"),
            num_workers=num_workers,
            generator=generator if name == "train" else None,
        )
        for name, ds in datasets.items()
    }

    for name, ds in datasets.items():
        labels = np.asarray(ds.labels).ravel()
        log.info(
            "%-5s: %4d imagens %s | pneumonia = %.1f%%",
            name, len(ds), tuple(ds.imgs.shape[1:]), 100 * labels.mean(),
        )
    return loaders, datasets, info


# ---------------------------------------------------------------------------
# Arquitetura
# ---------------------------------------------------------------------------
def build_cnn(image_size: int = 28, in_channels: int = 1, dropout: float = 0.35):
    """
    CNN compacta: 3 blocos ``Conv -> BatchNorm -> ReLU -> Conv -> ... -> MaxPool``.

    ``AdaptiveAvgPool2d`` no final torna a rede independente da resolucao de
    entrada, permitindo treinar em 28x28 ou 64x64 sem alterar o codigo.
    """
    import torch.nn as nn

    def block(cin: int, cout: int) -> nn.Sequential:
        return nn.Sequential(
            nn.Conv2d(cin, cout, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(cout),
            nn.ReLU(inplace=True),
            nn.Conv2d(cout, cout, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(cout),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )

    return nn.Sequential(
        block(in_channels, 32),
        block(32, 64),
        block(64, 128),
        nn.AdaptiveAvgPool2d(1),
        nn.Flatten(),
        nn.Dropout(dropout),
        nn.Linear(128, 64),
        nn.ReLU(inplace=True),
        nn.Dropout(dropout / 2),
        nn.Linear(64, 1),  # logit unico -> BCEWithLogitsLoss
    )


# ---------------------------------------------------------------------------
# Treinamento
# ---------------------------------------------------------------------------
@dataclass
class CNNHistory:
    """Historico de treino, epoca a epoca."""

    train_loss: list[float] = field(default_factory=list)
    val_loss: list[float] = field(default_factory=list)
    train_auc: list[float] = field(default_factory=list)
    val_auc: list[float] = field(default_factory=list)
    best_epoch: int = 0
    best_val_auc: float = 0.0


def _epoch_scores(model, loader, device) -> tuple[np.ndarray, np.ndarray, float]:
    """Roda o modelo em modo avaliacao e devolve (y_true, probabilidades, loss)."""
    import torch
    import torch.nn as nn

    model.eval()
    criterion = nn.BCEWithLogitsLoss()
    scores, targets, losses = [], [], []
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            y = labels.float().view(-1, 1).to(device)
            logits = model(images)
            losses.append(criterion(logits, y).item() * len(y))
            scores.append(torch.sigmoid(logits).cpu().numpy().ravel())
            targets.append(labels.numpy().ravel())
    y_true = np.concatenate(targets)
    y_score = np.concatenate(scores)
    return y_true, y_score, float(np.sum(losses) / len(y_true))


def train_cnn(
    model,
    loaders,
    epochs: int = 12,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    device: str = "cpu",
):
    """
    Treina a CNN e devolve ``(modelo_com_melhores_pesos, historico)``.

    Selecao de epoca pela **ROC AUC de validacao** - em base desbalanceada a
    acuracia sobe simplesmente por prever sempre a classe majoritaria.
    """
    import copy

    import torch
    import torch.nn as nn
    from sklearn.metrics import roc_auc_score

    model = model.to(device)

    # Peso da classe positiva = n_negativos / n_positivos no treino.
    labels = np.asarray(loaders["train"].dataset.labels).ravel()
    pos_weight = torch.tensor(
        [(labels == 0).sum() / max((labels == 1).sum(), 1)], dtype=torch.float32
    ).to(device)
    log.info("pos_weight aplicado a perda = %.3f", float(pos_weight))

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    history = CNNHistory()
    best_state = copy.deepcopy(model.state_dict())

    for epoch in range(1, epochs + 1):
        model.train()
        running, seen = 0.0, 0
        for images, batch_labels in loaders["train"]:
            images = images.to(device)
            y = batch_labels.float().view(-1, 1).to(device)

            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(images), y)
            loss.backward()
            optimizer.step()

            running += loss.item() * len(y)
            seen += len(y)
        scheduler.step()

        y_tr, s_tr, _ = _epoch_scores(model, loaders["train"], device)
        y_va, s_va, val_loss = _epoch_scores(model, loaders["val"], device)
        train_auc = roc_auc_score(y_tr, s_tr)
        val_auc = roc_auc_score(y_va, s_va)

        history.train_loss.append(running / seen)
        history.val_loss.append(val_loss)
        history.train_auc.append(float(train_auc))
        history.val_auc.append(float(val_auc))

        marker = ""
        if val_auc > history.best_val_auc:
            history.best_val_auc = float(val_auc)
            history.best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            marker = "  <- melhor"

        log.info(
            "epoca %2d/%d | perda treino %.4f | perda val %.4f | AUC treino %.4f | AUC val %.4f%s",
            epoch, epochs, history.train_loss[-1], val_loss, train_auc, val_auc, marker,
        )

    model.load_state_dict(best_state)
    log.info("melhor epoca: %d (AUC de validacao = %.4f)",
             history.best_epoch, history.best_val_auc)
    return model, history


def evaluate_cnn(model, loader, device: str = "cpu") -> tuple[np.ndarray, np.ndarray]:
    """Devolve ``(y_true, probabilidade_de_pneumonia)`` para um ``DataLoader``."""
    y_true, y_score, _ = _epoch_scores(model, loader, device)
    return y_true, y_score


# ---------------------------------------------------------------------------
# Figuras
# ---------------------------------------------------------------------------
def plot_samples(dataset, filename: str = "30_amostras_radiografias", n: int = 12):
    """Grade com exemplos de radiografias de cada classe."""
    import matplotlib.pyplot as plt

    apply_plot_style()
    imgs = np.asarray(dataset.imgs)
    labels = np.asarray(dataset.labels).ravel()

    rng = np.random.default_rng(RANDOM_STATE)
    chosen = np.concatenate(
        [rng.choice(np.where(labels == c)[0], n // 2, replace=False) for c in (0, 1)]
    )

    ncols = 6
    nrows = int(np.ceil(len(chosen) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(2.0 * ncols, 2.25 * nrows))
    for ax, idx in zip(np.atleast_1d(axes).ravel(), chosen):
        ax.imshow(imgs[idx], cmap="gray")
        ax.set_title(CLASS_LABELS[labels[idx]], fontsize=10,
                     color=PLOT.benign_color if labels[idx] == 0 else PLOT.malignant_color)
        ax.axis("off")
    fig.suptitle("PneumoniaMNIST - exemplos de radiografias de torax",
                 fontsize=14, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.subplots_adjust(hspace=0.42)
    return save_fig(fig, filename)


def plot_history(history: CNNHistory, filename: str = "31_cnn_curvas_treino"):
    """Curvas de perda e de ROC AUC ao longo das epocas."""
    import matplotlib.pyplot as plt

    apply_plot_style()
    epochs = range(1, len(history.train_loss) + 1)
    fig, (ax_loss, ax_auc) = plt.subplots(1, 2, figsize=(13, 5))

    ax_loss.plot(epochs, history.train_loss, "o-", label="treino", color=PLOT.benign_color)
    ax_loss.plot(epochs, history.val_loss, "o-", label="validacao", color=PLOT.malignant_color)
    ax_loss.set(xlabel="epoca", ylabel="perda (BCE)", title="Evolucao da perda")
    ax_loss.legend()

    ax_auc.plot(epochs, history.train_auc, "o-", label="treino", color=PLOT.benign_color)
    ax_auc.plot(epochs, history.val_auc, "o-", label="validacao", color=PLOT.malignant_color)
    ax_auc.axvline(history.best_epoch, ls="--", color="black", lw=1.4,
                   label=f"melhor epoca ({history.best_epoch})")
    ax_auc.set(xlabel="epoca", ylabel="ROC AUC", title="Evolucao da ROC AUC")
    ax_auc.legend()

    fig.suptitle("Treinamento da CNN - PneumoniaMNIST", fontsize=14, fontweight="bold")
    return save_fig(fig, filename)


def plot_predictions_grid(
    model,
    dataset,
    y_score: np.ndarray,
    threshold: float,
    filename: str = "33_cnn_predicoes",
    n: int = 12,
):
    """
    Mostra acertos e erros do modelo no teste, com a probabilidade prevista.

    Prioriza os erros - sao eles que a equipe clinica precisa inspecionar.
    """
    import matplotlib.pyplot as plt

    apply_plot_style()
    imgs = np.asarray(dataset.imgs)
    y_true = np.asarray(dataset.labels).ravel()
    y_pred = (y_score >= threshold).astype(int)

    wrong = np.where(y_pred != y_true)[0]
    right = np.where(y_pred == y_true)[0]
    rng = np.random.default_rng(RANDOM_STATE)
    n_wrong = min(len(wrong), n // 2)
    chosen = np.concatenate(
        [
            wrong[np.argsort(-np.abs(y_score[wrong] - threshold))][:n_wrong],
            rng.choice(right, n - n_wrong, replace=False),
        ]
    )

    ncols = 6
    nrows = int(np.ceil(len(chosen) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(2.1 * ncols, 2.5 * nrows))
    for ax, idx in zip(np.atleast_1d(axes).ravel(), chosen):
        ok = y_pred[idx] == y_true[idx]
        ax.imshow(imgs[idx], cmap="gray")
        ax.set_title(
            f"real: {CLASS_LABELS[y_true[idx]]}\n"
            f"prev: {CLASS_LABELS[y_pred[idx]]} ({y_score[idx]:.2f})",
            fontsize=8.5, color="#2E7D32" if ok else "#C62828",
            fontweight="normal" if ok else "bold",
        )
        ax.axis("off")
        for spine in ax.spines.values():
            spine.set_visible(False)

    fig.suptitle(
        "Predicoes da CNN no conjunto de teste\n"
        "(vermelho = erro; erros mais confiantes aparecem primeiro)",
        fontsize=13, fontweight="bold",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.subplots_adjust(hspace=0.62)
    return save_fig(fig, filename)


def plot_gradcam(
    model,
    dataset,
    indices: list[int],
    device: str = "cpu",
    filename: str = "34_cnn_gradcam",
):
    """
    Grad-CAM: destaca as regioes do pulmao que mais pesaram na decisao.

    E o equivalente do SHAP para imagens - permite ao radiologista conferir se
    a rede olhou para o campo pulmonar ou para um artefato irrelevante da
    imagem (borda, texto, marcador metalico).
    """
    import matplotlib.pyplot as plt
    import torch
    import torch.nn.functional as F

    apply_plot_style()
    model = model.to(device).eval()

    # Ultima camada convolucional da rede (o bloco mais profundo).
    target_layer = None
    for module in model.modules():
        if isinstance(module, torch.nn.Conv2d):
            target_layer = module
    if target_layer is None:  # pragma: no cover
        raise RuntimeError("nenhuma camada convolucional encontrada")

    activations: dict[str, torch.Tensor] = {}
    gradients: dict[str, torch.Tensor] = {}
    h1 = target_layer.register_forward_hook(
        lambda m, i, o: activations.__setitem__("value", o.detach())
    )
    h2 = target_layer.register_full_backward_hook(
        lambda m, gi, go: gradients.__setitem__("value", go[0].detach())
    )

    imgs = np.asarray(dataset.imgs)
    labels = np.asarray(dataset.labels).ravel()

    ncols = len(indices)
    fig, axes = plt.subplots(2, ncols, figsize=(2.6 * ncols, 6.4))
    axes = np.atleast_2d(axes)

    try:
        for col, idx in enumerate(indices):
            image, _ = dataset[idx]
            x = image.unsqueeze(0).to(device).requires_grad_(True)

            logit = model(x)
            model.zero_grad(set_to_none=True)
            logit.backward()

            weights = gradients["value"].mean(dim=(2, 3), keepdim=True)
            cam = F.relu((weights * activations["value"]).sum(dim=1, keepdim=True))
            cam = F.interpolate(cam, size=imgs.shape[1:3], mode="bilinear",
                                align_corners=False)
            cam = cam.squeeze().cpu().numpy()
            if cam.max() > cam.min():
                cam = (cam - cam.min()) / (cam.max() - cam.min())

            prob = float(torch.sigmoid(logit).item())
            axes[0, col].imshow(imgs[idx], cmap="gray")
            axes[0, col].set_title(
                f"{CLASS_LABELS[labels[idx]]}\np(pneumonia) = {prob:.2f}", fontsize=9
            )
            axes[1, col].imshow(imgs[idx], cmap="gray")
            axes[1, col].imshow(cam, cmap="jet", alpha=0.45)
            axes[1, col].set_title("Grad-CAM", fontsize=9)
            for row in (0, 1):
                axes[row, col].axis("off")
    finally:
        h1.remove()
        h2.remove()

    fig.suptitle(
        "Onde a CNN olhou para decidir\n"
        "(linha de cima: radiografia original | linha de baixo: mapa de ativacao)",
        fontsize=13, fontweight="bold",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.88))
    fig.subplots_adjust(hspace=0.30)
    return save_fig(fig, filename)


def count_parameters(model) -> int:
    """Numero de parametros treinaveis da rede."""
    return int(sum(p.numel() for p in model.parameters() if p.requires_grad))
