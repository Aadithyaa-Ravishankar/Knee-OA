"""
ordinal_net.py
==============
Pipeline configurable KL-grade model.

backbone : resnet50 | convnext_tiny | efficientnet_v2_s   (torchvision, ImageNet or random init)
dual     : full image + center zoom through the same trunk (as Stage 5/6), or global only
head     :
  cascade : binary OA gate + 3 cumulative thresholds over KL1..4 (Stage 6 head)
  cumbce  : 4 independent cumulative thresholds (Stage 4/5 head)
  coral   : 4 cumulative thresholds sharing one weight vector (rank-consistent)
  sord    : 5-way softmax trained on soft ordinal targets exp(-alpha * (k - y)^2)

All heads expose to_probs(raw) -> (B, 5) and decode(raw) -> (B,) hard grade.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

from src.evaluation.ensemble import s6_class_probs, s6_predict_with_threshold
from src.losses.ordinal_loss import corn_predict, corn_probs
from src.models.dual_stream import center_zoom, medlat_crops

HEADS = ("cascade", "cumbce", "coral", "sord")
BACKBONES = ("resnet50", "convnext_tiny", "efficientnet_v2_s")
STREAM_MODES = ("center", "medlat")
FUSION_MODES = ("concat", "scaled", "gated")


def periphery_mask(
    x: torch.Tensor,
    roi_frac: float = 0.85,
    suppress: float = 1.0,
) -> torch.Tensor:
    """Attenuate pixels outside a center ROI (joint box) on the global stream input.

    suppress=0 → identity; suppress=1 → zero outside the box; in between → soft fade.
    """
    if suppress <= 0.0:
        return x
    suppress = float(min(max(suppress, 0.0), 1.0))
    roi_frac = float(min(max(roi_frac, 0.4), 0.95))
    squeeze = False
    if x.dim() == 3:
        x = x.unsqueeze(0)
        squeeze = True
    _b, _c, h, w = x.shape
    ch, cw = int(round(h * roi_frac)), int(round(w * roi_frac))
    top, left = (h - ch) // 2, (w - cw) // 2
    inside = x.new_zeros(1, 1, h, w)
    inside[:, :, top : top + ch, left : left + cw] = 1.0
    weight = 1.0 - suppress * (1.0 - inside)
    out = x * weight
    return out.squeeze(0) if squeeze else out


def _build_backbone(name: str, pretrained: bool) -> tuple[nn.Module, nn.Module, int]:
    """Returns (body -> spatial map, pool -> (B, C) vector, C)."""
    if name == "resnet50":
        m = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None)
        body = nn.Sequential(m.conv1, m.bn1, m.relu, m.maxpool, m.layer1, m.layer2, m.layer3, m.layer4)
        pool = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(1))
        return body, pool, 2048
    if name == "convnext_tiny":
        m = models.convnext_tiny(weights=models.ConvNeXt_Tiny_Weights.IMAGENET1K_V1 if pretrained else None)
        pool = nn.Sequential(nn.AdaptiveAvgPool2d(1), m.classifier[0], nn.Flatten(1))
        return m.features, pool, 768
    if name == "efficientnet_v2_s":
        m = models.efficientnet_v2_s(weights=models.EfficientNet_V2_S_Weights.IMAGENET1K_V1 if pretrained else None)
        pool = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(1))
        return m.features, pool, 1280
    raise ValueError(f"unknown backbone {name}; choose from {BACKBONES}")


class CoralHead(nn.Module):
    def __init__(self, d: int, n_thr: int = 4):
        super().__init__()
        self.fc = nn.Linear(d, 1, bias=False)
        self.bias = nn.Parameter(torch.linspace(1.0, -1.0, n_thr))

    def forward(self, z):
        return self.fc(z) + self.bias


class OrdinalNet(nn.Module):
    def __init__(
        self,
        backbone: str = "resnet50",
        head: str = "cascade",
        dual: bool = True,
        local_frac: float = 0.72,
        dropout: float = 0.3,
        pretrained: bool = True,
        sord_alpha: float = 1.5,
        oa_threshold: float = 0.5,
        n_aux: int = 0,
        stream_mode: str = "center",
        medial_is_image_left_for_r: bool = True,
        fusion: str = "concat",
        periphery_suppress: float = 0.0,
        global_roi_frac: float = 0.85,
        spatial_aux: bool = False,
        spatial_aux_global: bool = False,
    ) -> None:
        super().__init__()
        if head not in HEADS:
            raise ValueError(f"unknown head {head}; choose from {HEADS}")
        if stream_mode not in STREAM_MODES:
            raise ValueError(f"unknown stream_mode {stream_mode}; choose from {STREAM_MODES}")
        if fusion not in FUSION_MODES:
            raise ValueError(f"unknown fusion {fusion}; choose from {FUSION_MODES}")
        if stream_mode == "medlat" and not dual:
            raise ValueError("stream_mode=medlat requires dual=True")
        if spatial_aux and (not dual or stream_mode != "center" or n_aux != 4):
            raise ValueError("spatial_aux requires dual center stream and n_aux=4 (jsn/ost med+lat)")
        if spatial_aux_global and not spatial_aux:
            raise ValueError("spatial_aux_global requires spatial_aux=True")
        self.backbone_name = backbone
        self.head_name = head
        self.dual = dual
        self.local_frac = local_frac
        self.sord_alpha = sord_alpha
        self.oa_threshold = oa_threshold
        self.n_aux = n_aux
        self.stream_mode = stream_mode
        self.medial_is_image_left_for_r = medial_is_image_left_for_r
        self.fusion = fusion
        self.periphery_suppress = float(min(max(periphery_suppress, 0.0), 1.0))
        self.global_roi_frac = float(min(max(global_roi_frac, 0.4), 0.95))
        self.spatial_aux = bool(spatial_aux)
        self.spatial_aux_global = bool(spatial_aux_global)

        self.body, self.pool, c = _build_backbone(backbone, pretrained)
        n_streams = 1 if not dual else (3 if stream_mode == "medlat" else 2)
        d = c * n_streams
        self.dropout = nn.Dropout(dropout)
        if head == "cascade":
            self.fc = nn.Linear(d, 4)  # [bin, oa_t0, oa_t1, oa_t2]
        elif head == "cumbce":
            self.fc = nn.Linear(d, 4)
        elif head == "coral":
            self.fc = CoralHead(d, 4)
        else:
            self.fc = nn.Linear(d, 5)
        # Four-level OARSI grades (0-3) as 3 cumulative thresholds each. Training targets only.
        if n_aux and spatial_aux:
            # Med / lat strip → (jsn_*, ost_*) each 3 thresholds → 6 logits per side.
            self.aux_fc = None
            self.aux_med = nn.Linear(c, 6)
            self.aux_lat = nn.Linear(c, 6)
        elif n_aux:
            self.aux_fc = nn.Linear(d, n_aux * 3)
            self.aux_med = None
            self.aux_lat = None
        else:
            self.aux_fc = None
            self.aux_med = None
            self.aux_lat = None
        # Local-heavy init: softplus([-0.5, 1.0, ...]) ≈ [0.47, 1.31, ...]
        if fusion == "scaled" and n_streams > 1:
            init = [-0.5] + [1.0] * (n_streams - 1)
            self.stream_scale_logits = nn.Parameter(torch.tensor(init, dtype=torch.float32))
        else:
            self.stream_scale_logits = None
        # Gate global only: sigmoid(-2) ≈ 0.12 so local dominates at start.
        if fusion == "gated" and n_streams > 1:
            self.global_gate_logit = nn.Parameter(torch.tensor(-2.0, dtype=torch.float32))
        else:
            self.global_gate_logit = None

    @classmethod
    def from_config(cls, mc: dict, *, dropout: float = 0.3, pretrained: bool = False) -> "OrdinalNet":
        """Build from a saved model/config dict (best.pth / config.json)."""
        return cls(
            backbone=mc["backbone"],
            head=mc["head"],
            dual=bool(mc.get("dual", True)),
            local_frac=float(mc.get("local_frac", 0.72)),
            dropout=float(dropout),
            pretrained=pretrained,
            sord_alpha=float(mc.get("sord_alpha", 1.5)),
            oa_threshold=float(mc.get("oa_threshold", 0.5)),
            n_aux=int(mc.get("n_aux", 0)),
            stream_mode=str(mc.get("stream_mode", "center")),
            medial_is_image_left_for_r=bool(mc.get("medial_is_image_left_for_r", True)),
            fusion=str(mc.get("fusion", "concat")),
            periphery_suppress=float(mc.get("periphery_suppress", 0.0)),
            global_roi_frac=float(mc.get("global_roi_frac", 0.85)),
            spatial_aux=bool(mc.get("spatial_aux", False)),
            spatial_aux_global=bool(mc.get("spatial_aux_global", False)),
        )

    def global_gate(self) -> torch.Tensor | None:
        if self.global_gate_logit is None:
            return None
        return torch.sigmoid(self.global_gate_logit)

    def _fuse_pooled(self, zs: list[torch.Tensor]) -> torch.Tensor:
        if self.stream_scale_logits is not None:
            scales = F.softplus(self.stream_scale_logits)
            zs = [z * scales[i] for i, z in enumerate(zs)]
        elif self.global_gate_logit is not None:
            g = torch.sigmoid(self.global_gate_logit)
            zs = [zs[0] * g] + list(zs[1:])
        return torch.cat(zs, dim=1)

    # ── forward ──────────────────────────────────────────────────────────────
    def feature_maps(
        self, x: torch.Tensor, side: torch.Tensor | None = None
    ) -> list[tuple[torch.Tensor, float | None]]:
        x_global = periphery_mask(x, self.global_roi_frac, self.periphery_suppress)
        maps = [(self.body(x_global), None)]
        if not self.dual:
            return maps
        if self.stream_mode == "medlat":
            if side is None:
                raise ValueError("stream_mode=medlat requires side (0=L, 1=R)")
            med, lat = medlat_crops(x, side, self.local_frac, self.medial_is_image_left_for_r)
            maps.append((self.body(med), self.local_frac))
            maps.append((self.body(lat), self.local_frac))
            return maps
        maps.append((self.body(center_zoom(x, self.local_frac)), self.local_frac))
        return maps

    def forward_from_maps(self, acts: list[torch.Tensor]) -> torch.Tensor:
        z = self._fuse_pooled([self.pool(a) for a in acts])
        return self.fc(self.dropout(z))

    def forward(self, x: torch.Tensor, side: torch.Tensor | None = None) -> torch.Tensor:
        return self.forward_from_maps([a for a, _ in self.feature_maps(x, side)])

    def _spatial_aux_logits(self, act: torch.Tensor, side: torch.Tensor) -> torch.Tensor:
        """Med/lat halves of a stream map → aux logits (B, 12) ordered jsn_med, jsn_lat, ost_med, ost_lat."""
        if side is None:
            raise ValueError("spatial_aux requires side (0=L, 1=R)")
        _b, _c, _h, w = act.shape
        mid = w // 2
        left = self.pool(act[:, :, :, :mid])
        right = self.pool(act[:, :, :, mid:])
        # R: medial on image-left iff medial_is_image_left_for_r; L is the mirror.
        med_is_left = torch.where(
            side.long() == 1,
            torch.full((act.shape[0],), self.medial_is_image_left_for_r, device=act.device, dtype=torch.bool),
            torch.full((act.shape[0],), not self.medial_is_image_left_for_r, device=act.device, dtype=torch.bool),
        )
        z_med = torch.where(med_is_left[:, None], left, right)
        z_lat = torch.where(med_is_left[:, None], right, left)
        med = self.aux_med(z_med)  # (B, 6) = jsn_med(3) + ost_med(3)
        lat = self.aux_lat(z_lat)
        jsn_med, ost_med = med[:, :3], med[:, 3:]
        jsn_lat, ost_lat = lat[:, :3], lat[:, 3:]
        return torch.cat([jsn_med, jsn_lat, ost_med, ost_lat], dim=1)

    def forward_with_aux(
        self, x: torch.Tensor, side: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """KL logits and aux logits from one trunk pass. Aux is (B, n_aux * 3)."""
        raw, aux, _acts = self.forward_train(x, side)
        if aux is None:
            raise RuntimeError("forward_with_aux called but model has no aux heads")
        return raw, aux

    def forward_train(
        self, x: torch.Tensor, side: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor | None, list[torch.Tensor]]:
        """KL logits, optional aux logits, and stream feature maps (for attention prior)."""
        maps = self.feature_maps(x, side)
        acts = [a for a, _ in maps]
        z = self.dropout(self._fuse_pooled([self.pool(a) for a in acts]))
        raw = self.fc(z)
        if self.spatial_aux:
            aux_local = self._spatial_aux_logits(acts[1], side)
            if self.spatial_aux_global:
                # Shared med/lat heads on global halves push the rim stream toward JSN/ost.
                aux = 0.5 * (self._spatial_aux_logits(acts[0], side) + aux_local)
            else:
                aux = aux_local
        elif self.aux_fc is not None:
            aux = self.aux_fc(z)
        else:
            aux = None
        return raw, aux, acts

    def head_parameters(self):
        params = list(self.fc.parameters())
        if self.aux_fc is not None:
            params += list(self.aux_fc.parameters())
        if self.aux_med is not None:
            params += list(self.aux_med.parameters())
        if self.aux_lat is not None:
            params += list(self.aux_lat.parameters())
        if self.stream_scale_logits is not None:
            params.append(self.stream_scale_logits)
        if self.global_gate_logit is not None:
            params.append(self.global_gate_logit)
        return params

    def backbone_parameters(self):
        # Include pool (ConvNeXt classifier LayerNorm lives here; ResNet/EffNet pool has no params).
        return list(self.body.parameters()) + list(self.pool.parameters())

    # ── outputs ──────────────────────────────────────────────────────────────
    def to_probs(self, raw: torch.Tensor) -> torch.Tensor:
        raw = raw.float()
        if self.head_name == "cascade":
            return s6_class_probs(raw[:, :1], raw[:, 1:])
        if self.head_name in ("cumbce", "coral"):
            return corn_probs(raw)
        return F.softmax(raw, dim=1)

    def decode(self, raw: torch.Tensor) -> torch.Tensor:
        raw = raw.float()
        if self.head_name == "cascade":
            return s6_predict_with_threshold(raw[:, :1], raw[:, 1:], self.oa_threshold)
        if self.head_name in ("cumbce", "coral"):
            return corn_predict(raw)
        return raw.argmax(dim=1)

    def threshold_probs(self, raw: torch.Tensor) -> torch.Tensor | None:
        """Cumulative P(y > k) for threshold heads (for rank-consistency checks)."""
        raw = raw.float()
        if self.head_name in ("cumbce", "coral"):
            return torch.sigmoid(raw)
        if self.head_name == "cascade":
            return torch.sigmoid(raw[:, 1:])
        return None

    # ── loss ─────────────────────────────────────────────────────────────────
    def loss(
        self,
        raw: torch.Tensor,
        labels: torch.Tensor,
        soft_kl12: float = 0.0,
        soft_oa01: float = 0.0,
    ) -> torch.Tensor:
        raw = raw.float()
        labels = labels.long()
        eps12 = float(soft_kl12)
        eps01 = float(soft_oa01)
        if self.head_name == "cascade":
            # OA gate: optional soft targets for KL0/KL1; KL2+ stay hard-positive.
            oa_tgt = (labels > 0).float()
            if eps01 > 0:
                oa_tgt = oa_tgt.clone()
                oa_tgt[labels == 0] = eps01
                oa_tgt[labels == 1] = 1.0 - eps01
            loss = F.binary_cross_entropy_with_logits(raw[:, 0], oa_tgt)
            oa = labels > 0
            if oa.any():
                y = labels[oa]
                yr = (y - 1).clamp(min=0)
                p_gt = torch.stack([(yr > k).float() for k in range(3)], dim=1)
                if eps12 > 0:
                    m1, m2 = y == 1, y == 2
                    p_gt[m1] = torch.tensor([eps12, 0.0, 0.0], device=raw.device, dtype=raw.dtype)
                    p_gt[m2] = torch.tensor([1.0 - eps12, 0.0, 0.0], device=raw.device, dtype=raw.dtype)
                loss = loss + F.binary_cross_entropy_with_logits(raw[oa, 1:], p_gt)
            return loss
        if self.head_name in ("cumbce", "coral"):
            p_gt = torch.stack([(labels > k).float() for k in range(4)], dim=1)
            if eps01 > 0:
                # Soften P(y>0) for KL0/KL1 only.
                p_gt = p_gt.clone()
                p_gt[labels == 0, 0] = eps01
                p_gt[labels == 1, 0] = 1.0 - eps01
            if eps12 > 0:
                m1, m2 = labels == 1, labels == 2
                # P(y>0), P(y>1), P(y>2), P(y>3) under soft KL1/KL2
                row1 = torch.tensor([1.0 - eps01 if eps01 > 0 else 1.0, eps12, 0.0, 0.0],
                                    device=raw.device, dtype=raw.dtype)
                p_gt[m1] = row1
                p_gt[m2] = torch.tensor([1.0, 1.0 - eps12, 0.0, 0.0], device=raw.device, dtype=raw.dtype)
            return F.binary_cross_entropy_with_logits(raw, p_gt)
        k = torch.arange(5, device=raw.device, dtype=raw.dtype)
        soft = F.softmax(-self.sord_alpha * (k[None, :] - labels[:, None].to(raw.dtype)) ** 2, dim=1)
        return -(soft * F.log_softmax(raw, dim=1)).sum(dim=1).mean()

    def aux_loss(
        self,
        raw_aux: torch.Tensor,
        aux_targets: torch.Tensor,
        feat_weights: torch.Tensor | list[float] | None = None,
    ) -> torch.Tensor:
        """Cumulative BCE per feature. aux_targets is (B, n_aux) with -1 where the grade is missing.

        feat_weights: optional length-n_aux positive weights (e.g. boost jsn_med). Default all 1.
        When weights are all 1, matches unweighted mean over valid threshold elements.
        """
        raw_aux = raw_aux.float().view(-1, self.n_aux, 3)
        y = aux_targets.long()
        if feat_weights is None:
            w = torch.ones(self.n_aux, device=raw_aux.device, dtype=raw_aux.dtype)
        else:
            w = torch.as_tensor(feat_weights, device=raw_aux.device, dtype=raw_aux.dtype)
            if w.numel() != self.n_aux:
                raise ValueError(f"feat_weights length {w.numel()} != n_aux {self.n_aux}")
        terms, weights = [], []
        for j in range(self.n_aux):
            valid = y[:, j] != -1
            if not valid.any():
                continue
            yj = y[valid, j]
            tgt = torch.stack([(yj > k).float() for k in range(3)], dim=1)
            t = F.binary_cross_entropy_with_logits(raw_aux[valid, j], tgt, reduction="none").reshape(-1)
            terms.append(t)
            weights.append(torch.full_like(t, w[j]))
        if not terms:
            return raw_aux.sum() * 0
        tcat = torch.cat(terms)
        wcat = torch.cat(weights)
        return (tcat * wcat).sum() / wcat.sum().clamp_min(1e-8)

    def decode_aux(self, raw_aux: torch.Tensor) -> torch.Tensor:
        """Hard 0-3 grade per feature: count of cumulative thresholds with prob > 0.5."""
        logits = raw_aux.float().view(-1, self.n_aux, 3)
        return (torch.sigmoid(logits) > 0.5).sum(dim=-1).long()

    def aux_probs(self, raw_aux: torch.Tensor) -> torch.Tensor:
        """(B, n_aux, 3) cumulative P(feature > k)."""
        return torch.sigmoid(raw_aux.float().view(-1, self.n_aux, 3))

    def config(self) -> dict:
        return {
            "backbone": self.backbone_name,
            "head": self.head_name,
            "dual": self.dual,
            "local_frac": self.local_frac,
            "sord_alpha": self.sord_alpha,
            "oa_threshold": self.oa_threshold,
            "n_aux": self.n_aux,
            "stream_mode": self.stream_mode,
            "medial_is_image_left_for_r": self.medial_is_image_left_for_r,
            "fusion": self.fusion,
            "periphery_suppress": self.periphery_suppress,
            "global_roi_frac": self.global_roi_frac,
            "spatial_aux": self.spatial_aux,
            "spatial_aux_global": self.spatial_aux_global,
        }
