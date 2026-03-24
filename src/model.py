"""
Dacon 구조 안정성 분류 - Dual-Stream EfficientNet-B0
=====================================================
Front / Top 이미지를 각각 별도 EfficientNet-B0 backbone에 통과 →
특징 벡터 Concatenate → FC 분류기
"""

import torch
import torch.nn as nn
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
        pretrained: ImageNet 사전학습 가중치 사용 여부 (default: True)
        dropout: Classifier dropout 비율 (default: 0.3)
    """

    def __init__(
        self,
        num_classes: int = 2,
        pretrained: bool = True,
        dropout: float = 0.3,
    ):
        super().__init__()

        # ── Backbone 생성 ─────────────────────────────────────────
        weights = models.ConvNeXt_Tiny_Weights.IMAGENET1K_V1 if pretrained else None
        
        self.stream_front = self._make_backbone(weights)
        self.stream_top = self._make_backbone(weights)
        self.stream_diff = self._make_backbone(weights)

        # ConvNeXt-Tiny feature dim = 768
        feat_dim = 768
        concat_dim = feat_dim * 3  # front + top + diff

        # ── Classifier ────────────────────────────────────────────
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
        """ConvNeXt-Tiny backbone 생성 (classifier 제거)."""
        backbone = models.convnext_tiny(weights=weights)
        # 원래 classifier: (0): LayerNorm2d, (1): Flatten, (2): Linear
        # 우리는 특징 벡터 추출을 위해 classifier를 Identity로 교체하거나
        # 마지막 Linear만 Identity로 교체할 수 있습니다.
        # 여기서는 LayerNorm과 Flatten까지만 사용하도록 설정합니다.
        backbone.classifier = nn.Sequential(
            backbone.classifier[0],
            backbone.classifier[1]
        )
        return backbone

    def forward(self, front: torch.Tensor, top: torch.Tensor, diff: torch.Tensor) -> torch.Tensor:
        """
        Args:
            front: (B, 3, H, W) front 이미지 텐서
            top:   (B, 3, H, W) top 이미지 텐서
            diff:  (B, 3, H, W) difference map 이미지 텐서
        Returns:
            logits: (B, num_classes) 로짓
        """
        feat_front = self.stream_front(front)  # (B, 768)
        feat_top = self.stream_top(top)        # (B, 768)
        feat_diff = self.stream_diff(diff)     # (B, 768)

        combined = torch.cat([feat_front, feat_top, feat_diff], dim=1)  # (B, 2304)
        logits = self.classifier(combined)
        return logits


class FocalLoss(nn.Module):
    """
    Focal Loss: 정답 클래스에 대한 예측 확률이 높을수록 가중치를 낮추고, 
    확률이 낮을수록(어려운 샘플) 가중치를 높이는 손실 함수.
    """
    def __init__(self, alpha: float = 1, gamma: float = 2, reduction: str = 'mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.ce = nn.CrossEntropyLoss(reduction='none')

    def forward(self, inputs, targets):
        ce_loss = self.ce(inputs, targets)
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


def count_parameters(model: nn.Module) -> int:
    """모델의 학습 가능한 파라미터 수를 반환."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    # 빠른 검증용
    model = TripleStreamConvNeXt(pretrained=False)
    front_dummy = torch.randn(2, 3, 224, 224)
    top_dummy = torch.randn(2, 3, 224, 224)
    diff_dummy = torch.randn(2, 3, 224, 224)
    out = model(front_dummy, top_dummy, diff_dummy)
    print(f"Front/Top/Diff shape: {front_dummy.shape}")
    print(f"Output shape:         {out.shape}")
    print(f"Parameters:           {count_parameters(model):,}")
