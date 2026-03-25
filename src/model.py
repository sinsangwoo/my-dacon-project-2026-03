"""
Dacon 구조 안정성 분류 - Model (v5)
=====================================
변경:
  - FocalLoss gamma 기본값 3.0 (확신도 부스팅)
  - PhysicsConsistencyLoss 유지
  - GradCAM 유틸리티 추가 (Explainability)
  - ECE (Expected Calibration Error) 계산 함수 추가
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


# ──────────────────────────────────────────────────────────────────
#  Triple-Stream ConvNeXt-Tiny
# ──────────────────────────────────────────────────────────────────

class TripleStreamConvNeXt(nn.Module):
    """Triple-Stream ConvNeXt-Tiny 분류 모델.

    front / top / diff(Temporal Difference Map) 세 스트림을
    각각 독립적인 ConvNeXt-Tiny backbone 에 통과시켜
    768-d 특징 벡터를 추출, concatenate 후 FC 분류기로 예측.
    """

    def __init__(self, num_classes: int = 2, pretrained: bool = True, dropout: float = 0.3):
        super().__init__()
        weights = models.ConvNeXt_Tiny_Weights.IMAGENET1K_V1 if pretrained else None
        self.stream_front = self._make_backbone(weights)
        self.stream_top   = self._make_backbone(weights)
        self.stream_diff  = self._make_backbone(weights)

        concat_dim = 768 * 3   # 2304
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(concat_dim, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Dropout(p=dropout),
            nn.Linear(512, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Linear(128, num_classes),
        )

    @staticmethod
    def _make_backbone(weights):
        backbone = models.convnext_tiny(weights=weights)
        # classifier: [LayerNorm2d, Flatten, Linear] → Linear 제거 → 768-d 출력
        backbone.classifier = nn.Sequential(
            backbone.classifier[0],   # LayerNorm2d(768)
            backbone.classifier[1],   # Flatten
        )
        return backbone

    def forward(self, front, top, diff):
        f = self.stream_front(front)
        t = self.stream_top(top)
        d = self.stream_diff(diff)
        return self.classifier(torch.cat([f, t, d], dim=1))


# ──────────────────────────────────────────────────────────────────
#  Loss Functions
# ──────────────────────────────────────────────────────────────────

class FocalLoss(nn.Module):
    """Focal Loss. gamma=3.0 으로 결정 경계를 더 날카롭게."""

    def __init__(self, alpha: float = 1.0, gamma: float = 3.0,
                 label_smoothing: float = 0.05, reduction: str = "mean"):
        super().__init__()
        self.alpha           = alpha
        self.gamma           = gamma
        self.label_smoothing = label_smoothing
        self.reduction       = reduction
        # label_smoothing 은 CrossEntropyLoss 에 내장 지원
        self._ce = nn.CrossEntropyLoss(reduction="none",
                                       label_smoothing=label_smoothing)

    def forward(self, inputs, targets):
        ce   = self._ce(inputs, targets)
        pt   = torch.exp(-ce)
        loss = self.alpha * (1.0 - pt) ** self.gamma * ce
        return loss.mean() if self.reduction == "mean" else loss.sum()


class PhysicsConsistencyLoss(nn.Module):
    """좌우 반전 입력에 대한 예측 분포 일관성 Symmetric KL Loss."""

    def __init__(self, temperature: float = 2.0):
        super().__init__()
        self.T = temperature

    def forward(self, model, front, top, diff, logits_orig):
        lf = torch.flip(front, [3])
        lt = torch.flip(top,   [3])
        ld = torch.flip(diff,  [3])
        logits_flip = model(lf, lt, ld)

        p      = F.log_softmax(logits_orig / self.T, dim=1)
        q      = F.softmax(logits_flip    / self.T, dim=1)
        q_log  = F.log_softmax(logits_flip / self.T, dim=1)
        p_soft = F.softmax(logits_orig    / self.T, dim=1)

        kl_fwd = F.kl_div(p,     q,      reduction="batchmean")
        kl_bwd = F.kl_div(q_log, p_soft, reduction="batchmean")
        return (kl_fwd + kl_bwd) * 0.5


# ──────────────────────────────────────────────────────────────────
#  Explainability Utilities
# ──────────────────────────────────────────────────────────────────

class GradCAM:
    """ConvNeXt-Tiny 의 마지막 stage 에 대한 Grad-CAM.

    사용법:
        cam = GradCAM(model, target_stream='stream_front')
        saliency = cam(front, top, diff, class_idx=1)  # (H, W) numpy array
        cam.remove_hooks()
    """

    def __init__(self, model: TripleStreamConvNeXt, target_stream: str = "stream_front"):
        self.model   = model
        self.stream  = getattr(model, target_stream)
        self._feat   = None
        self._grad   = None
        self._hooks  = []
        # ConvNeXt-Tiny 의 마지막 stage = features[-1]
        target_layer = self.stream.features[-1]
        self._hooks.append(
            target_layer.register_forward_hook(self._save_feat)
        )
        self._hooks.append(
            target_layer.register_full_backward_hook(self._save_grad)
        )

    def _save_feat(self, _, __, output):
        self._feat = output.detach()

    def _save_grad(self, _, __, grad_output):
        self._grad = grad_output[0].detach()

    def remove_hooks(self):
        for h in self._hooks:
            h.remove()

    def __call__(self, front, top, diff, class_idx: int = 1) -> np.ndarray:
        self.model.eval()
        front = front.unsqueeze(0) if front.dim() == 3 else front
        top   = top.unsqueeze(0)   if top.dim()   == 3 else top
        diff  = diff.unsqueeze(0)  if diff.dim()  == 3 else diff

        out = self.model(front, top, diff)       # forward → hook 실행
        self.model.zero_grad()
        out[0, class_idx].backward()             # backward → grad hook

        weights = self._grad.mean(dim=(2, 3), keepdim=True)   # (B, C, 1, 1)
        cam     = (weights * self._feat).sum(dim=1)            # (B, H, W)
        cam     = F.relu(cam)[0].cpu().numpy()                 # (H, W)
        cam     = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        return cam


def compute_gradcam_consistency(model, front, top, diff, device, class_idx=1) -> float:
    """원본 vs 좌우반전 GradCAM 의 공간적 일치도 (Pearson r).

    Returns:
        0 ~ 1 float. 1 에 가까울수록 모델이 좌우 대칭적으로
        동일한 물리적 영역을 주목함.
    """
    front = front.to(device)
    top   = top.to(device)
    diff  = diff.to(device)

    cam_fn = GradCAM(model, target_stream="stream_front")
    try:
        cam_orig = cam_fn(front.clone(), top.clone(), diff.clone(), class_idx)
        cam_flip = cam_fn(
            torch.flip(front, [3]).clone(),
            torch.flip(top,   [3]).clone(),
            torch.flip(diff,  [3]).clone(),
            class_idx,
        )
    finally:
        cam_fn.remove_hooks()

    # 반전 이미지의 CAM 도 좌우 flip 해서 원본 좌표계와 맞춤
    cam_flip_aligned = cam_flip[:, ::-1]

    # Pearson 상관계수
    a = cam_orig.ravel()
    b = cam_flip_aligned.ravel()
    if a.std() < 1e-8 or b.std() < 1e-8:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


def compute_ece(probs: np.ndarray, labels: np.ndarray, n_bins: int = 10) -> float:
    """Expected Calibration Error (ECE) 계산.

    Args:
        probs:   (N,) 배열, unstable 클래스에 대한 예측 확률
        labels:  (N,) 정수 배열 (0 or 1)
        n_bins:  calibration 구간 수

    Returns:
        ECE 값 (낮을수록 잘 교정된 모델)
    """
    bins    = np.linspace(0.0, 1.0, n_bins + 1)
    ece     = 0.0
    n       = len(probs)
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (probs >= lo) & (probs < hi)
        if mask.sum() == 0:
            continue
        acc  = labels[mask].mean()
        conf = probs[mask].mean()
        ece += mask.sum() * abs(acc - conf)
    return ece / n


# ──────────────────────────────────────────────────────────────────
#  Utilities
# ──────────────────────────────────────────────────────────────────

def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    _m = TripleStreamConvNeXt(pretrained=False)
    _x = torch.randn(2, 3, 268, 268)
    out = _m(_x, _x, _x)
    print(f"Output: {out.shape}  Params: {count_parameters(_m):,}")
