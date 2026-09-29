"""
1D Vision Transformer for 12-lead ECG (500 Hz).

  12 leads x 5000 samples
  -> overlapping patch embedding: Conv1d(12 -> D, kernel=100, stride=50) = 99 tokens,
     so every token mixes all 12 leads over a 0.2 s window
  -> [CLS] token + learnable positional embeddings
  -> Pre-LN transformer encoder
  -> LayerNorm([CLS]) -> Linear -> 5 superclass logits (multi-label)
"""

import torch
import torch.nn as nn


class PatchEmbedding1D(nn.Module):
    def __init__(self, in_channels, patch_size, stride, embed_dim):
        super().__init__()
        self.proj = nn.Conv1d(in_channels, embed_dim, kernel_size=patch_size, stride=stride, bias=False)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        return self.norm(self.proj(x).transpose(1, 2))  # (B, num_patches, D)


class ECGViT1D(nn.Module):
    def __init__(
        self,
        in_channels=12,
        seq_len=5000,
        patch_size=100,
        patch_stride=50,
        embed_dim=192,
        depth=6,
        num_heads=6,
        mlp_ratio=2.0,
        dropout=0.1,
        num_classes=5,
    ):
        super().__init__()
        self.num_patches = (seq_len - patch_size) // patch_stride + 1

        self.patch_embed = PatchEmbedding1D(in_channels, patch_size, patch_stride, embed_dim)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches + 1, embed_dim))
        self.pos_drop = nn.Dropout(dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=int(embed_dim * mlp_ratio),
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=depth, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Sequential(nn.Dropout(0.2), nn.Linear(embed_dim, num_classes))

        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.LayerNorm):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)

    @classmethod
    def from_config(cls, config, num_classes):
        m = config["model"]
        return cls(
            in_channels=len(config["data"]["leads"]),
            seq_len=config["data"]["seq_len"],
            patch_size=m["patch_size"],
            patch_stride=m["patch_stride"],
            embed_dim=m["embed_dim"],
            depth=m["depth"],
            num_heads=m["num_heads"],
            mlp_ratio=m["mlp_ratio"],
            dropout=m["dropout"],
            num_classes=num_classes,
        )

    def extract_features(self, x):
        x = self.patch_embed(x)
        x = torch.cat([self.cls_token.expand(x.shape[0], -1, -1), x], dim=1)
        x = self.pos_drop(x + self.pos_embed)
        x = self.transformer(x)
        return self.norm(x[:, 0])

    def forward(self, x):
        return self.head(self.extract_features(x))
