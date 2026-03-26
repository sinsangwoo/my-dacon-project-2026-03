"""
Dacon 구조 안정성 분류 — Model (v6)
=====================================
변경점:
  - Backbone: ConvNeXt-Tiny → EfficientNet-B1 (경량화, CPU 15분/epoch 목표)
  - StochasticDepth 드롭 래퍼 추가 (암기 방지)
  - FocalLoss: gamma=2, label_smoothing=0.1 (요구사항 반영)
  - TemperatureScaler: val 세트 기반 post-hoc ECE 교정
  - SAMOptimizer: Sharpness-Aware Minimization (Flat Minima)
  - GradCAM: EfficientNet-B1 마지막 Conv 블록 기준으로 수정
  - compute_ece, PhysicsConsistencyLoss 유지
"""

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


# ──────────────────────────────────────────────────────────────────
#  Stochastic Depth (레이어 무작위 생략)
# ──────────────────────────────────────────────────────────────────

class StochasticDepth(nn.Module):
    """학습 시 확률 drop_prob 로 전체 레지듀얼 브랜치를 생략.
    torchvision ≥0.11 에 내장 버전이 있으나 직접 구현해 의존성 제거.
    """
    def __init__(self, drop_prob: float = 0.1):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.training or self.drop_prob == 0.0:
            return x
        keep = 1.0 - self.drop_prob
        # 배치 차원만 랜덤 마스크 — 나머지는 브로드캐스트
        shape = (x.size(0),) + (1,) * (x.ndim - 1)
        mask  = torch.empty(shape, dtype=x.dtype, device=x.device)
        mask  = mask.bernoulli_(keep).div_(keep)
        return x * mask


# ──────────────────────────────────────────────────────────────────
#  Triple-Stream EfficientNet-B1
# ──────────────────────────────────────────────────────────────────

class TripleStreamEfficientNet(nn.Module):
    """Triple-Stream EfficientNet-B1 분류 모델.

    front / top / diff 세 스트림을 각각 독립 EfficientNet-B1 backbone
    에 통과시켜 1280-d 특징 벡터를 추출, concatenate 후 FC 분류기.

    Stochastic Depth 를 각 스트림 출력에 적용하여 소규모 데이터셋의
    암기(memorization)를 억제.

    Args:
        num_classes:   출력 클래스 수 (default: 2)
        pretrained:    ImageNet 사전학습 가중치 사용 (default: True)
        dropout:       Classifier dropout (default: 0.4)
        stoch_depth_p: Stochastic Depth drop 확률 (default: 0.1)
    """

    FEAT_DIM = 1280   # EfficientNet-B1 의 AdaptiveAvgPool 이후 채널 수

    def __init__(
        self,
        num_classes:   int   = 2,
        pretrained:    bool  = True,
        dropout:       float = 0.4,
        stoch_depth_p: float = 0.1,
    ):
        super().__init__()
        weights = (
            models.EfficientNet_B1_Weights.IMAGENET1K_V2
            if pretrained else None
        )
        self.stream_front = self._make_backbone(weights)
        self.stream_top   = self._make_backbone(weights)
        self.stream_diff  = self._make_backbone(weights)

        self.sd_front = StochasticDepth(stoch_depth_p)
        self.sd_top   = StochasticDepth(stoch_depth_p)
        self.sd_diff  = StochasticDepth(stoch_depth_p)

        concat_dim = self.FEAT_DIM * 3   # 3840
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(concat_dim, 512),
            nn.BatchNorm1d(512),
            nn.SiLU(),
            nn.Dropout(p=dropout / 2),
            nn.Linear(512, 128),
            nn.BatchNorm1d(128),
            nn.SiLU(),
            nn.Linear(128, num_classes),
        )

    @staticmethod
    def _make_backbone(weights):
        """EfficientNet-B1 backbone — classifier 제거, 1280-d 특징 출력."""
        net = models.efficientnet_b1(weights=weights)
        # net.classifier = [Dropout, Linear]  →  Identity 로 교체
        net.classifier = nn.Identity()
        return net

    def forward(
        self,
        front: torch.Tensor,
        top:   torch.Tensor,
        diff:  torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            front/top/diff: (B, 3, H, W)
        Returns:
            logits: (B, num_classes)
        """
        f = self.sd_front(self.stream_front(front))   # (B, 1280)
        t = self.sd_top  (self.stream_top(top))
        d = self.sd_diff (self.stream_diff(diff))
        return self.classifier(torch.cat([f, t, d], dim=1))   # (B, 3840) → (B, C)


# ──────────────────────────────────────────────────────────────────
#  Temperature Scaler (post-hoc ECE 교정)
# ──────────────────────────────────────────────────────────────────

class TemperatureScaler(nn.Module):
    """학습 완료 후 Val set 으로 최적 Temperature T 를 탐색.

    사용법:
        scaler = TemperatureScaler(model).to(device)
        scaler.fit(val_loader, device)        # NLL 최소화로 T 탐색
        calibrated_logits = scaler(logits)    # logits / T
        T_val = scaler.temperature.item()
    """

    def __init__(self, model: nn.Module):
        super().__init__()
        self.model       = model
        self.temperature = nn.Parameter(torch.ones(1) * 1.5)

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        return logits / self.temperature.clamp(min=0.1)

    @torch.enable_grad()
    def fit(
        self,
        val_loader,
        device,
        max_iter: int = 50,
        lr: float = 0.01,
    ) -> float:
        """Val NLL 을 최소화하는 T 를 LBFGS 로 탐색.

        Returns:
            최적 temperature 값 (float)
        """
        self.model.eval()
        self.to(device)

        # Val 전체 logits / labels 수집
        all_logits, all_labels = [], []
        with torch.no_grad():
            for front, top, diff, labels in val_loader:
                logits = self.model(
                    front.to(device), top.to(device), diff.to(device)
                )
                all_logits.append(logits.cpu())
                all_labels.append(labels)

        all_logits = torch.cat(all_logits).to(device)
        all_labels = torch.cat(all_labels).to(device)

        optimizer = torch.optim.LBFGS(
            [self.temperature], lr=lr, max_iter=max_iter
        )
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
#  SAM Optimizer (Sharpness-Aware Minimization)
# ──────────────────────────────────────────────────────────────────

class SAM(torch.optim.Optimizer):
    """Sharpness-Aware Minimization.

    기본 optimizer 를 내부에 감싸는 래퍼. 2-step update:
        1) 파라미터를 ε-perturbation 방향으로 이동 (ascent)
        2) perturbed 파라미터에서 gradient 계산 후 실제 업데이트 (descent)

    사용법:
        base_opt = torch.optim.AdamW(model.parameters(), lr=3e-4)
        optimizer = SAM(model.parameters(), base_opt)
        ...
        loss.backward()
        optimizer.first_step(zero_grad=True)
        criterion(model(x), y).backward()
        optimizer.second_step(zero_grad=True)
    """

    def __init__(self, params, base_optimizer, rho: float = 0.05, **kwargs):
        defaults = dict(rho=rho, **kwargs)
        super().__init__(params, defaults)
        self.base_optimizer = base_optimizer
        self.param_groups   = self.base_optimizer.param_groups
        # 초기 defaults 를 base optimizer 와 동기화
        for group in self.param_groups:
            group.setdefault("rho", rho)

    @torch.no_grad()
    def first_step(self, zero_grad: bool = False):
        """ε-perturbation: 파라미터를 gradient 방향의 sharp 지점으로 이동."""
        grad_norm = self._grad_norm()
        for group in self.param_groups:
            scale = group["rho"] / (grad_norm + 1e-12)
            for p in group["params"]:
                if p.grad is None:
                    continue
                e_w = p.grad * scale
                p.add_(e_w)                    # ascent
                self.state[p]["e_w"] = e_w    # 나중에 복구용
        if zero_grad:
            self.zero_grad()

    @torch.no_grad()
    def second_step(self, zero_grad: bool = False):
        """perturbed gradient 로 실제 파라미터 업데이트."""
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                p.sub_(self.state[p]["e_w"])   # 원래 위치로 복구
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
        # SAM 은 first_step / second_step 으로 사용하므로
        # 단순 step() 호출 시에는 base optimizer 만 실행
        self.base_optimizer.step(closure)

    # state_dict / load_state_dict 위임
    def state_dict(self):
        return self.base_optimizer.state_dict()

    def load_state_dict(self, state_dict):
        self.base_optimizer.load_state_dict(state_dict)


# ──────────────────────────────────────────────────────────────────
#  Loss Functions
# ──────────────────────────────────────────────────────────────────

class FocalLoss(nn.Module):
    """Focal Loss + Label Smoothing.
    동적 Annealing 지원 (v6.1+):
      OneCycleLR 진척도에 따라 label_smoothing=0.0, gamma=0.0 으로 선형 수렴 지원.
    """

    def __init__(
        self,
        alpha:           float = 1.0,
        gamma:           float = 2.0,
        label_smoothing: float = 0.1,
        reduction:       str   = "mean",
    ):
        super().__init__()
        self.alpha           = alpha
        self.gamma           = gamma
        self.reduction       = reduction
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
            # Soft Label (Distillation) support
            log_probs = F.log_softmax(inputs, dim=1)
            probs = torch.exp(log_probs)
            # Focal weight: (1-pt)^gamma
            pt = (probs * targets).sum(dim=1)
            focal_weight = (1.0 - pt) ** self.gamma
            
            # Label Smoothing for Soft Targets (KL-Divergence based implementation if needed)
            # 여기서는 targets 가 이미 soft 이므로 LS 적용 시 targets 를 smoothing
            if self.label_smoothing > 0:
                num_classes = targets.size(1)
                targets = targets * (1.0 - self.label_smoothing) + self.label_smoothing / num_classes
            
            # Soft Cross Entropy
            loss = -(targets * log_probs).sum(dim=1) * focal_weight
        else:
            # Hard Label (Standard)
            ce   = self._ce(inputs, targets)
            pt   = torch.exp(-ce)
            loss = self.alpha * (1.0 - pt) ** self.gamma * ce
        return loss.mean() if self.reduction == "mean" else loss.sum()


class PhysicsConsistencyLoss(nn.Module):
    """좌우 반전 입력에 대한 예측 분포 일관성 — Symmetric KL."""

    def __init__(self, temperature: float = 2.0):
        super().__init__()
        self.T = temperature

    def forward(self, model, front, top, diff, logits_orig):
        with torch.enable_grad():
            lf = torch.flip(front, [3])
            lt = torch.flip(top,   [3])
            ld = torch.flip(diff,  [3])
            logits_flip = model(lf, lt, ld)

        p      = F.log_softmax(logits_orig  / self.T, dim=1)
        q      = F.softmax(logits_flip      / self.T, dim=1)
        q_log  = F.log_softmax(logits_flip  / self.T, dim=1)
        p_soft = F.softmax(logits_orig      / self.T, dim=1)

        return (F.kl_div(p, q, reduction="batchmean")
                + F.kl_div(q_log, p_soft, reduction="batchmean")) * 0.5


# ──────────────────────────────────────────────────────────────────
#  Explainability Utilities
# ──────────────────────────────────────────────────────────────────

class GradCAM:
    """EfficientNet-B1 의 마지막 Conv 블록에 대한 Grad-CAM.

    사용법:
        cam = GradCAM(model, target_stream='stream_front')
        saliency = cam(front, top, diff, class_idx=1)
        cam.remove_hooks()
    """

    def __init__(self, model: TripleStreamEfficientNet,
                 target_stream: str = "stream_front"):
        self.model  = model
        stream      = getattr(model, target_stream)
        # EfficientNet-B1: stream.features[-1] 이 마지막 Conv 블록
        target_layer = stream.features[-1]
        self._feat = self._grad = None
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

    def __call__(self, front, top, diff, class_idx: int = 1) -> np.ndarray:
        self.model.eval()
        for t in (front, top, diff):
            if t.dim() == 3:
                t.unsqueeze_(0)
        out = self.model(front, top, diff)
        self.model.zero_grad()
        out[0, class_idx].backward()

        w   = self._grad.mean(dim=(2, 3), keepdim=True)
        cam = F.relu((w * self._feat).sum(dim=1))[0].cpu().numpy()
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        return cam


def compute_gradcam_consistency(
    model, front, top, diff, device, class_idx: int = 1
) -> float:
    """원본 vs 좌우반전 GradCAM Pearson r (0~1, 높을수록 물리적 일관성)."""
    front, top, diff = front.to(device), top.to(device), diff.to(device)
    cam_fn = GradCAM(model, target_stream="stream_front")
    try:
        co = cam_fn(front.clone(), top.clone(), diff.clone(), class_idx)
        cf = cam_fn(
            torch.flip(front, [3]).clone(),
            torch.flip(top,   [3]).clone(),
            torch.flip(diff,  [3]).clone(),
            class_idx,
        )
    finally:
        cam_fn.remove_hooks()

    cf_aligned = cf[:, ::-1]
    a, b = co.ravel(), cf_aligned.ravel()
    if a.std() < 1e-8 or b.std() < 1e-8:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


def compute_ece(
    probs: np.ndarray, labels: np.ndarray, n_bins: int = 10
) -> float:
    """Expected Calibration Error."""
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece, n = 0.0, len(probs)
    for lo, hi in zip(bins[:-1], bins[1:]):
        m = (probs >= lo) & (probs < hi)
        if m.sum() == 0:
            continue
        ece += m.sum() * abs(labels[m].mean() - probs[m].mean())
    return ece / n


# ──────────────────────────────────────────────────────────────────
#  Utilities
# ──────────────────────────────────────────────────────────────────

def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    m = TripleStreamEfficientNet(pretrained=False)
    x = torch.randn(2, 3, 240, 240)
    print(f"Output : {m(x, x, x).shape}")
    print(f"Params : {count_parameters(m):,}")
