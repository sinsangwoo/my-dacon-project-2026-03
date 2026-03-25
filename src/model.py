"""
Dacon 구조 안정성 분류 - Triple-Stream ConvNeXt-Tiny (Physical Intelligence v4)
================================================================================
Front / Top / Diff(Temporal Difference Map) 이미지를 각각 별도
ConvNeXt-Tiny backbone에 통과 → 특징 벡터 Concatenate → FC 분류기

Diff Stream: 영상의 첫 프레임 - 끝 프레임 차이를 시각화한 물리 변형 지도.
물리적으로 불안정한 구조물일수록 Diff 값이 크게 나타나는 핵심 피처.

PhysicsConsistencyLoss: 좌우 반전된 입력에 대해 예측이 일관되도록 강제하는
정규화 손실. 모델이 배경 노이즈가 아닌 구조적 형상(물리 법칙)을 학습하게 함.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


class TripleStreamConvNeXt(nn.Module):
    """Triple-Stream ConvNeXt-Tiny 분류 모델.

    구조:
        front_img → ConvNeXt-Tiny (stream_front) → feat_front (768,)
        top_img   → ConvNeXt-Tiny (stream_top)   → feat_top   (768,)
        diff_img  → ConvNeXt-Tiny (stream_diff)  → feat_diff  (768,)
        concat(feat_front, feat_top, feat_diff) → FC → num_classes

    Args:
        num_classes: 출력 클래스 수 (default: 2)
        pretrained:  ImageNet 사전학습 가중치 사용 여부 (default: True)
        dropout:     Classifier dropout 비율 (default: 0.3)
    """

    def __init__(
        self,
        num_classes: int = 2,
        pretrained: bool = True,
        dropout: float = 0.3,
    ):
        super().__init__()

        weights = models.ConvNeXt_Tiny_Weights.IMAGENET1K_V1 if pretrained else None

        self.stream_front = self._make_backbone(weights)
        self.stream_top   = self._make_backbone(weights)
        self.stream_diff  = self._make_backbone(weights)

        # ConvNeXt-Tiny feature dim = 768
        feat_dim   = 768
        concat_dim = feat_dim * 3   # front + top + diff = 2304

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
        """ConvNeXt-Tiny backbone 생성 (Linear head 제거, 768-d 벡터 출력)."""
        backbone = models.convnext_tiny(weights=weights)
        # classifier 원본: [LayerNorm2d, Flatten, Linear]
        # Linear 층만 제거하고 LayerNorm + Flatten 유지 → 768-d 출력
        backbone.classifier = nn.Sequential(
            backbone.classifier[0],  # LayerNorm2d(768)
            backbone.classifier[1],  # Flatten
        )
        return backbone

    def forward(
        self,
        front: torch.Tensor,
        top:   torch.Tensor,
        diff:  torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            front: (B, 3, H, W)  front 이미지 텐서
            top:   (B, 3, H, W)  top 이미지 텐서
            diff:  (B, 3, H, W)  temporal difference map 텐서
        Returns:
            logits: (B, num_classes)
        """
        feat_front = self.stream_front(front)   # (B, 768)
        feat_top   = self.stream_top(top)        # (B, 768)
        feat_diff  = self.stream_diff(diff)      # (B, 768)

        combined = torch.cat([feat_front, feat_top, feat_diff], dim=1)  # (B, 2304)
        return self.classifier(combined)


# ─────────────────────────────────────────────────────────────────────────────
#  Loss Functions
# ─────────────────────────────────────────────────────────────────────────────

class FocalLoss(nn.Module):
    """Focal Loss: 어려운 샘플에 더 높은 가중치를 부여하는 손실 함수.

    Args:
        alpha:     전체 스케일 계수 (default: 1.0)
        gamma:     focusing 파라미터 (default: 2.0, 클수록 어려운 샘플 집중)
        reduction: 'mean' | 'sum' | 'none'
    """

    def __init__(self, alpha: float = 1.0, gamma: float = 2.0, reduction: str = "mean"):
        super().__init__()
        self.alpha     = alpha
        self.gamma     = gamma
        self.reduction = reduction
        self._ce       = nn.CrossEntropyLoss(reduction="none")

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce_loss   = self._ce(inputs, targets)
        pt        = torch.exp(-ce_loss)
        focal     = self.alpha * (1.0 - pt) ** self.gamma * ce_loss

        if self.reduction == "mean":
            return focal.mean()
        if self.reduction == "sum":
            return focal.sum()
        return focal


class PhysicsConsistencyLoss(nn.Module):
    """Physics Consistency Regularization Loss.

    좌우 반전(horizontal flip)된 입력에 대해 모델의 출력 확률 분포가
    원본과 동일하도록 KL divergence로 강제한다.

    - 배경 텍스처, 조명 노이즈 등 방향에 의존적인 spurious feature 억제
    - 구조적 형상(좌우 대칭성) 기반의 물리적으로 일관된 표현 학습 유도

    Loss = KL( softmax(logits_orig) || softmax(logits_flip) )
           + KL( softmax(logits_flip) || softmax(logits_orig) )   [symmetric]
    """

    def __init__(self, temperature: float = 2.0):
        """
        Args:
            temperature: softmax 온도. 클수록 분포를 부드럽게 만들어
                         soft-target KL이 더 안정적으로 계산됨.
        """
        super().__init__()
        self.T = temperature

    def forward(
        self,
        model: nn.Module,
        front: torch.Tensor,
        top:   torch.Tensor,
        diff:  torch.Tensor,
        logits_orig: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            model:       TripleStreamConvNeXt 인스턴스
            front/top/diff: 원본 배치 텐서  (B, 3, H, W)
            logits_orig: 원본 입력에 대한 로짓  (B, C)  — 재계산 방지용 캐시
        Returns:
            scalar 손실값
        """
        # 좌우 반전
        front_flip = torch.flip(front, dims=[3])
        top_flip   = torch.flip(top,   dims=[3])
        diff_flip  = torch.flip(diff,  dims=[3])

        logits_flip = model(front_flip, top_flip, diff_flip)

        # Symmetric KL divergence (온도 스케일링 적용)
        p = F.log_softmax(logits_orig / self.T, dim=1)
        q = F.softmax(logits_flip    / self.T, dim=1)
        q_log = F.log_softmax(logits_flip  / self.T, dim=1)
        p_soft = F.softmax(logits_orig     / self.T, dim=1)

        kl_fwd = F.kl_div(p,     q,      reduction="batchmean")  # KL(orig || flip)
        kl_bwd = F.kl_div(q_log, p_soft, reduction="batchmean")  # KL(flip || orig)

        return (kl_fwd + kl_bwd) * 0.5


# ─────────────────────────────────────────────────────────────────────────────
#  Utilities
# ─────────────────────────────────────────────────────────────────────────────

def count_parameters(model: nn.Module) -> int:
    """모델의 학습 가능한 파라미터 수를 반환."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    # 빠른 smoke test
    _model = TripleStreamConvNeXt(pretrained=False)
    _front = torch.randn(2, 3, 224, 224)
    _top   = torch.randn(2, 3, 224, 224)
    _diff  = torch.randn(2, 3, 224, 224)

    _out = _model(_front, _top, _diff)
    print(f"Input  shape : {_front.shape}")
    print(f"Output shape : {_out.shape}")         # (2, 2)
    print(f"Parameters   : {count_parameters(_model):,}")

    # PhysicsConsistencyLoss smoke test
    _pcs_loss_fn = PhysicsConsistencyLoss(temperature=2.0)
    _pcs_val = _pcs_loss_fn(_model, _front, _top, _diff, _out)
    print(f"PCS Loss (untrained): {_pcs_val.item():.6f}")
