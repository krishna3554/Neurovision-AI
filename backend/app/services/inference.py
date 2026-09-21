"""Inference (FR-5): sliding-window prediction + uncertainty entropy."""
import numpy as np
import torch
from monai.inferers import sliding_window_inference

from backend.app.core.config import settings
from backend.app.models.net import TriViewResMambaUNet


class SegmentationModel:
    def __init__(self, checkpoint=None):
        self.dev = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = TriViewResMambaUNet().to(self.dev).eval()
        self.load_weights(checkpoint or settings.checkpoint)

    def load_weights(self, path):
        import os

        if not os.path.exists(path):
            return  # allow init without weights (tests / cold start)
        ck = torch.load(path, map_location=self.dev)
        state = ck.get("model", ck) if isinstance(ck, dict) else ck
        self.model.load_state_dict(state)

    @torch.no_grad()
    def predict(self, image):  # image: [3,X,Y,Z] tensor
        self.model.eval()
        x = image[None].to(self.dev)
        p = sliding_window_inference(
            x, settings.patch_size, 2, self.model, overlap=0.5
        )
        prob = torch.sigmoid(p)[0, 0].cpu().numpy()
        ent = -(prob * np.log(prob + 1e-8) + (1 - prob) * np.log(1 - prob + 1e-8))
        return prob, ent  # probability map + uncertainty entropy
