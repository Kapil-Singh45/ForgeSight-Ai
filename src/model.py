from typing import List, Dict
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


class PatchFeatureExtractor(nn.Module):
    """
    Extracts multi-scale patch embeddings from a pre-trained CNN backbone (e.g. ResNet-18).
    
    As shown in PatchCore (Roth et al., 2022), extracting intermediate activations from
    layer2 and layer3, aggregating neighborhood context via average pooling, and fusing
    the features provides rich visual representations for anomaly detection.
    """

    def __init__(
        self,
        backbone_name: str = "resnet18",
        layers: List[str] = ("layer2", "layer3"),
        pretrained: bool = True,
    ):
        super().__init__()
        self.backbone_name = backbone_name
        self.layers = list(layers)

        # Load standard torchvision backbone
        if backbone_name == "resnet18":
            weights = models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
            self.backbone = models.resnet18(weights=weights)
        elif backbone_name == "resnet34":
            weights = models.ResNet34_Weights.IMAGENET1K_V1 if pretrained else None
            self.backbone = models.resnet34(weights=weights)
        elif backbone_name == "resnet50":
            weights = models.ResNet50_Weights.IMAGENET1K_V1 if pretrained else None
            self.backbone = models.resnet50(weights=weights)
        elif backbone_name == "wide_resnet50_2":
            weights = models.Wide_ResNet50_2_Weights.IMAGENET1K_V1 if pretrained else None
            self.backbone = models.wide_resnet50_2(weights=weights)
        else:
            raise ValueError(f"Unsupported backbone: {backbone_name}")

        # Freeze all weights - no backpropagation needed
        for param in self.backbone.parameters():
            param.requires_grad = False
        self.backbone.eval()

        # Storage for forward hooks
        self.features: Dict[str, torch.Tensor] = {}
        self.hooks = []
        self._register_hooks()

        # Patch neighborhood aggregator (avg pool with stride 1 to maintain spatial resolution)
        self.avg_pool = nn.AvgPool2d(kernel_size=3, stride=1, padding=1)

    def _register_hooks(self):
        """Attaches forward hooks to extract activations from designated layers."""
        for layer_name in self.layers:
            layer = getattr(self.backbone, layer_name)
            hook = layer.register_forward_hook(self._hook_fn(layer_name))
            self.hooks.append(hook)

    def _hook_fn(self, layer_name: str):
        def hook(module, input, output):
            self.features[layer_name] = output
        return hook

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass extracting and concatenating multi-scale patch embeddings.

        Args:
            x: Input image tensor of shape (B, C, H, W).

        Returns:
            Tensor of patch embeddings of shape (B, N_patches, Feature_dim).
        """
        self.features.clear()
        _ = self.backbone(x)

        # Retrieve hook outputs
        extracted = [self.features[l] for l in self.layers]

        # Use layer2 spatial resolution as reference target
        ref_h, ref_w = extracted[0].shape[-2:]

        # Resize all features to matching spatial resolution and apply patch context pooling
        pooled_features = []
        for feat in extracted:
            # Aggregate spatial neighborhood context
            pooled = self.avg_pool(feat)
            if pooled.shape[-2:] != (ref_h, ref_w):
                # Bilinear resize to match resolution
                pooled = F.interpolate(
                    pooled, size=(ref_h, ref_w), mode="bilinear", align_corners=False
                )
            pooled_features.append(pooled)

        # Concatenate along channel dimension: Shape (B, Total_Channels, ref_h, ref_w)
        combined = torch.cat(pooled_features, dim=1)
        B, C, H, W = combined.shape

        # Reshape to (B, H*W, C) where each row is a patch embedding vector
        # Transpose to (B, H, W, C) -> (B, H*W, C)
        patch_tokens = combined.permute(0, 2, 3, 1).reshape(B, H * W, C)
        return patch_tokens

    def close(self):
        """Remove PyTorch hooks to avoid memory leaks."""
        for hook in self.hooks:
            hook.remove()
        self.hooks.clear()
