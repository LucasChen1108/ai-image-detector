"""
CLIP visual encoder as the high-level "semantic" branch.

Uses open_clip so the backbone is verifiably open-source/public-weights
(competition requirement). Default ViT-B-32 (~151M params); ViT-L-14 (~428M)
is a drop-in swap in configs/baseline_clip.yaml if you have the compute — both
are comfortably under the 2B-parameter cap.

We freeze the backbone by default and only train a small projection head,
which (a) trains fast on limited hackathon compute/time, and (b) keeps CLIP's
broad semantic generalization intact rather than overfitting it to the
training generators — directly serving the "generalization vs specialization"
trade-off called out in the brief. Unfreezing the last N transformer blocks
is exposed as an option for a later ablation if time allows.
"""
import open_clip
import torch
import torch.nn as nn


class ClipSemanticBranch(nn.Module):
    def __init__(self, model_name: str = "ViT-B-32", pretrained: str = "openai",
                 unfreeze_last_n_blocks: int = 0, out_dim: int = 256):
        super().__init__()
        # OpenAI's original CLIP checkpoints were trained with QuickGELU
        # (x * sigmoid(1.702x)), not the standard GELU open_clip defaults to.
        # Loading 'openai' weights without forcing this produces a silent
        # activation-function mismatch (open_clip warns "QuickGELU mismatch")
        # that subtly degrades the visual encoder relative to how it was
        # actually trained. LAION-trained checkpoints (laion2b/laion400m)
        # use standard GELU and should NOT force this.
        force_quick_gelu = "openai" in pretrained.lower()
        model, _, preprocess = open_clip.create_model_and_transforms(
            model_name, pretrained=pretrained, force_quick_gelu=force_quick_gelu
        )
        self.visual = model.visual
        self.preprocess = preprocess  # hand this to the Dataset

        for p in self.visual.parameters():
            p.requires_grad = False

        if unfreeze_last_n_blocks > 0:
            blocks = getattr(self.visual.transformer, "resblocks", None)
            if blocks is not None:
                for block in list(blocks)[-unfreeze_last_n_blocks:]:
                    for p in block.parameters():
                        p.requires_grad = True

        clip_dim = self.visual.output_dim
        self.project = nn.Sequential(
            nn.Linear(clip_dim, out_dim),
            nn.ReLU(inplace=True),
            nn.LayerNorm(out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        with torch.set_grad_enabled(any(p.requires_grad for p in self.visual.parameters())):
            feats = self.visual(x)
        return self.project(feats.float())
