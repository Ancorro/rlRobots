"""ACT (Action Chunking Transformer, Zhao et al. 2023) from our LeRobot stack (vla/requirements.txt, LeRobot 0.6.1) as a
candidate on OpenHLM-data, wrapped to the chunk-policy interface of openhlm/policy.py: forward(hist, hold, frames,
target) -> hold + ACT's normalised action chunk. LeRobot's ACTPolicy model is used as is (ResNet18 backbone per camera,
transformer encoder/decoder, CVAE); only its inputs are ours:
  observation.state              the current normalised state (the last history frame)
  observation.environment_state  the whole normalised history, flattened (ACT has no history of its own), and the task one-hot
  observation.images.<camera>    the cached frames (openhlm/frames.py), ImageNet-normalised as ACT's preprocessor would
  action                         the normalised residual chunk (training only, for the CVAE encoder)
Imports lerobot lazily, so the rest of the pipeline runs without it. Without a camera ACT runs on the states alone
(its environment_state input), which LeRobot allows.
"""
import torch
import torch.nn as nn

IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


class ACTChunk(nn.Module):
    def __init__(self, history, state_dim, horizon, action_dim, cameras=0, image_size=96, dim_model=256, n_heads=8,
                 dim_feedforward=1024, n_encoder_layers=2, n_decoder_layers=1, use_vae=True, kl_weight=10.0, latent_dim=32,
                 dropout=0.1, pretrained=True, ctx_dim=1, **_):
        super().__init__()
        from lerobot.configs.types import FeatureType, PolicyFeature
        from lerobot.policies.act.configuration_act import ACTConfig
        from lerobot.policies.act.modeling_act import ACTPolicy
        self.H, self.K, self.Da, self.cameras, self.ctx_dim = history, horizon, action_dim, cameras, ctx_dim
        inputs = {"observation.state": PolicyFeature(FeatureType.STATE, (state_dim,)),
                  "observation.environment_state": PolicyFeature(FeatureType.ENV, (history * state_dim + ctx_dim,))}
        self.image_keys = [f"observation.images.cam{i}" for i in range(cameras)]
        for k in self.image_keys:
            inputs[k] = PolicyFeature(FeatureType.VISUAL, (3, image_size, image_size))
        outputs = {"action": PolicyFeature(FeatureType.ACTION, (action_dim,))}
        weights = "ResNet18_Weights.IMAGENET1K_V1" if (pretrained and cameras) else None
        cfg = dict(input_features=inputs, output_features=outputs, chunk_size=horizon, n_action_steps=horizon, dim_model=dim_model,
                   n_heads=n_heads, dim_feedforward=dim_feedforward, n_encoder_layers=n_encoder_layers, n_decoder_layers=n_decoder_layers,
                   use_vae=use_vae, latent_dim=latent_dim, kl_weight=kl_weight, dropout=dropout, pretrained_backbone_weights=weights)
        try:
            self.policy = ACTPolicy(ACTConfig(**cfg))
        except Exception as e:  # noqa: BLE001  (no network for the ImageNet weights: train the backbone from scratch)
            if weights is None:
                raise
            print(f"ACT: pretrained backbone not available ({type(e).__name__}: {str(e)[:80]}); backbone from scratch", flush=True)
            cfg["pretrained_backbone_weights"] = None
            self.policy = ACTPolicy(ACTConfig(**cfg))
        self.kl_weight = kl_weight if use_vae else 0.0
        self.aux_loss = None

    def batch(self, hist, frames, ctx):
        if ctx is None:
            ctx = torch.zeros(hist.shape[0], self.ctx_dim, device=hist.device)
            ctx[:, 0] = 1.0
        b = {"observation.state": hist[:, -1], "observation.environment_state": torch.cat([hist.flatten(1), ctx], 1)}
        if self.cameras:
            mean, std = IMAGENET_MEAN.to(frames.device), IMAGENET_STD.to(frames.device)
            for i, k in enumerate(self.image_keys):
                b[k] = (frames[:, i] - mean) / std
            b["observation.images"] = [b[k] for k in self.image_keys]
        return b

    def forward(self, hist, hold, frames=None, target=None, ctx=None):
        b = self.batch(hist, frames, ctx)
        if self.training and target is not None:
            b["action"] = target - hold[:, None, :]
            b["action_is_pad"] = torch.zeros(target.shape[:2], dtype=torch.bool, device=target.device)
        out, (mu, log_sigma_x2) = self.policy.model(b)
        self.aux_loss = None
        if mu is not None and self.kl_weight:
            self.aux_loss = self.kl_weight * (-0.5 * (1 + log_sigma_x2 - mu.pow(2) - log_sigma_x2.exp())).sum(-1).mean()
        return hold[:, None, :] + out
