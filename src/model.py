"""
src/model.py
============
v8 — Dual-Stream 구조물 안정성 분류 모델

변경점:
  1. Triple-Stream(front/top/diff) → Dual-Stream(front/top)
  2. Diff 스트림 완전 제거
  3. Shared Backbone 기반 EfficientNet-B0
  4. CPU 학습 시간 단축을 위해 경량화
  5. train.py / predict.py 호환을 위한 loss/util 유지
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


# ──────────────────────────────────────────────────────────────────
#  Backbone Model
# ──────────────────────────────────────────────────────────────────

class DualStreamEfficientNet(nn.Module):
    """Dual-Stream EfficientNet-B0 분류 모델.

    front / top 두 이미지를 동일 backbone(shared weights)에 통과시켜
    특징 벡터를 추출한 뒤 concatenate하여 분류.

    Args:
        num_classes: 출력 클래스 수
        pretrained:  ImageNet pretrained 사용 여부
        dropout:     classifier dropout
    """

    FEAT_DIM = 1280  # EfficientNet-B0 classifier 제거 후 출력 차원

    def __init__(
        self,
        num_classes: int = 2,
        pretrained: bool = True,
        dropout: float = 0.3,
        stoch_depth_p: float = 0.0,  # train.py 호환용 인자 유지
    ):
        super().__init__()
        weights = models.EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
        self.backbone = models.efficientnet_b0(weights=weights)
        self.backbone.classifier = nn.Identity()

        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(self.FEAT_DIM * 2, 512),
            nn.BatchNorm1d(512),
            nn.SiLU(),
            nn.Dropout(p=dropout * 0.5),
            nn.Linear(512, 128),
            nn.BatchNorm1d(128),
            nn.SiLU(),
            nn.Linear(128, num_classes),
        )

    def forward(self, front: torch.Tensor, top: torch.Tensor) -> torch.Tensor:
        f = self.backbone(front)  # (B, 1280)
        t = self.backbone(top)    # (B, 1280)
        x = torch.cat([f, t], dim=1)
        return self.classifier(x)


# 기존 코드 호환 별칭
TripleStreamEfficientNet = DualStreamEfficientNet


# ──────────────────────────────────────────────────────────────────
#  Temperature Scaler
# ──────────────────────────────────────────────────────────────────

class TemperatureScaler(nn.Module):
    """Val set 기반 temperature scaling."""

    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model
        self.temperature = nn.Parameter(torch.ones(1) * 1.0)

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        return logits / self.temperature.clamp(min=0.1)

    @torch.enable_grad()
    def fit(self, val_loader, device, max_iter: int = 50, lr: float = 0.01) -> float:
        self.model.eval()
        self.to(device)

        all_logits, all_labels = [], []
        with torch.no_grad():
            for front, top, labels in val_loader:
                logits = self.model(front.to(device), top.to(device))
                all_logits.append(logits.cpu())
                all_labels.append(labels)

        all_logits = torch.cat(all_logits).to(device)
        all_labels = torch.cat(all_labels).to(device)

        # soft label 방어
        if all_labels.ndim > 1:
            all_labels = all_labels.argmax(dim=1)

        optimizer = torch.optim.LBFGS([self.temperature], lr=lr, max_iter=max_iter)
        nll_fn = nn.CrossEntropyLoss()

        def _eval():
            optimizer.zero_grad()
            loss = nll_fn(self(all_logits), all_labels)
            loss.backward()
            return loss

        optimizer.step(_eval)
        t_val = self.temperature.item()
        print(f"   🌡️  Temperature Scaling: T = {t_val:.4f}")
        return t_val


# ──────────────────────────────────────────────────────────────────
#  SAM Optimizer
# ──────────────────────────────────────────────────────────────────

class SAM(torch.optim.Optimizer):
    """Sharpness-Aware Minimization."""

    def __init__(self, params, base_optimizer, rho: float = 0.05, **kwargs):
        defaults = dict(rho=rho, **kwargs)
        super().__init__(params, defaults)
        self.base_optimizer = base_optimizer
        self.param_groups = self.base_optimizer.param_groups
        for group in self.param_groups:
            group.setdefault("rho", rho)

    @torch.no_grad()
    def first_step(self, zero_grad: bool = False):
        grad_norm = self._grad_norm()
        for group in self.param_groups:
            scale = group["rho"] / (grad_norm + 1e-12)
            for p in group["params"]:
                if p.grad is None:
                    continue
                e_w = p.grad * scale
                p.add_(e_w)
                self.state[p]["e_w"] = e_w
        if zero_grad:
            self.zero_grad()

    @torch.no_grad()
    def second_step(self, zero_grad: bool = False):
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                p.sub_(self.state[p]["e_w"])
        self.base_optimizer.step()
        if zero_grad:
            self.zero_grad()

    def _grad_norm(self) -> torch.Tensor:
        shared_device = self.param_groups[0]["params"][0].device
        norms = [
            p.grad.norm(p=2).to(shared_device)
            for group in self.param_groups
            for p in group["params"]
            if p.grad is not None
        ]
        return torch.stack(norms).norm(p=2)

    def step(self, closure=None):
        self.base_optimizer.step(closure)

    def state_dict(self):
        return self.base_optimizer.state_dict()

    def load_state_dict(self, state_dict):
        self.base_optimizer.load_state_dict(state_dict)


# ──────────────────────────────────────────────────────────────────
#  Loss Functions
# ──────────────────────────────────────────────────────────────────

class FocalLoss(nn.Module):
    """Focal Loss + Label Smoothing + Soft Label 지원."""

    def __init__(
        self,
        alpha: float = 1.0,
        gamma: float = 2.0,
        label_smoothing: float = 0.1,
        reduction: str = "mean",
    ):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self._ce = nn.CrossEntropyLoss(
            reduction="none", label_smoothing=label_smoothing
        )

    @property
    def label_smoothing(self):
        return self._ce.label_smoothing

    @label_smoothing.setter
    def label_smoothing(self, value):
        self._ce.label_smoothing = float(value)

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        if targets.ndim > 1:
            log_probs = F.log_softmax(inputs, dim=1)
            probs = torch.exp(log_probs)
            pt = (probs * targets).sum(dim=1)
            focal_weight = (1.0 - pt) ** self.gamma

            if self.label_smoothing > 0:
                num_classes = targets.size(1)
                targets = targets * (1.0 - self.label_smoothing) + self.label_smoothing / num_classes

            loss = -(targets * log_probs).sum(dim=1) * focal_weight
        else:
            ce = self._ce(inputs, targets)
            pt = torch.exp(-ce)
            loss = self.alpha * (1.0 - pt) ** self.gamma * ce

        return loss.mean() if self.reduction == "mean" else loss.sum()


class PhysicsConsistencyLoss(nn.Module):
    """좌우 반전 일관성 regularization."""

    def __init__(self, temperature: float = 2.0):
        super().__init__()
        self.T = temperature

    def forward(self, model, front, top, logits_orig):
        with torch.enable_grad():
            lf = torch.flip(front, [3])
            lt = torch.flip(top, [3])
            logits_flip = model(lf, lt)

        p = F.log_softmax(logits_orig / self.T, dim=1)
        q = F.softmax(logits_flip / self.T, dim=1)
        q_log = F.log_softmax(logits_flip / self.T, dim=1)
        p_soft = F.softmax(logits_orig / self.T, dim=1)

        return (
            F.kl_div(p, q, reduction="batchmean")
            + F.kl_div(q_log, p_soft, reduction="batchmean")
        ) * 0.5


# ──────────────────────────────────────────────────────────────────
#  Explainability
# ──────────────────────────────────────────────────────────────────

class GradCAM:
    """Dual-Stream backbone 마지막 conv에 대한 Grad-CAM."""

    def __init__(self, model: DualStreamEfficientNet):
        self.model = model
        target_layer = model.backbone.features[-1]
        self._feat = None
        self._grad = None
        self._hooks = [
            target_layer.register_forward_hook(
                lambda m, i, o: setattr(self, "_feat", o.detach())
            ),
            target_layer.register_full_backward_hook(
                lambda m, gi, go: setattr(self, "_grad", go[0].detach())
            ),
        ]

    def remove_hooks(self):
        for h in self._hooks:
            h.remove()

    def __call__(self, front, top, class_idx: int = 1) -> np.ndarray:
        self.model.eval()
        if front.dim() == 3:
            front = front.unsqueeze(0)
        if top.dim() == 3:
            top = top.unsqueeze(0)

        out = self.model(front, top)
        self.model.zero_grad()
        out[0, class_idx].backward()

        w = self._grad.mean(dim=(2, 3), keepdim=True)
        cam = F.relu((w * self._feat).sum(dim=1))[0].cpu().numpy()
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        return cam


def compute_gradcam_consistency(model, front, top, device, class_idx: int = 1) -> float:
    front, top = front.to(device), top.to(device)
    cam_fn = GradCAM(model)
    try:
        co = cam_fn(front.clone(), top.clone(), class_idx)
        cf = cam_fn(
            torch.flip(front, [3]).clone(),
            torch.flip(top, [3]).clone(),
            class_idx,
        )
    finally:
        cam_fn.remove_hooks()

    cf_aligned = cf[:, ::-1]
    a, b = co.ravel(), cf_aligned.ravel()
    if a.std() < 1e-8 or b.std() < 1e-8:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


def compute_ece(probs: np.ndarray, labels: np.ndarray, n_bins: int = 10) -> float:
    """Expected Calibration Error."""
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece, n = 0.0, len(probs)
    for lo, hi in zip(bins[:-1], bins[1:]):
        m = (probs >= lo) & (probs < hi)
        if m.sum() == 0:
            continue
        ece += m.sum() * abs(labels[m].mean() - probs[m].mean())
    return ece / n


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    m = DualStreamEfficientNet(pretrained=False)
    x = torch.randn(2, 3, 224, 224)
    print(f"Output : {m(x, x).shape}")
    print(f"Params : {count_parameters(m):,}")
