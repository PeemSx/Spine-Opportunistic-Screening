from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class CenterNetFocalLoss(nn.Module):
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred = pred.clamp(min=1e-6, max=1.0 - 1e-6)
        pos_inds = target.eq(1).float()
        neg_inds = target.lt(1).float()
        neg_weights = torch.pow(1 - target, 4)

        pos_loss = torch.log(pred) * torch.pow(1 - pred, 2) * pos_inds
        neg_loss = torch.log(1 - pred) * torch.pow(pred, 2) * neg_weights * neg_inds

        num_pos = pos_inds.sum()
        pos_loss = pos_loss.sum()
        neg_loss = neg_loss.sum()
        if num_pos == 0:
            return -neg_loss
        return -(pos_loss + neg_loss) / num_pos


class RegL1Loss(nn.Module):
    @staticmethod
    def _gather_feat(feat: torch.Tensor, ind: torch.Tensor) -> torch.Tensor:
        dim = feat.size(2)
        ind = ind.unsqueeze(2).expand(ind.size(0), ind.size(1), dim)
        return feat.gather(1, ind)

    def forward(
        self,
        output: torch.Tensor,
        mask: torch.Tensor,
        ind: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        pred = output.permute(0, 2, 3, 1).contiguous()
        pred = pred.view(pred.size(0), -1, pred.size(3))
        pred = self._gather_feat(pred, ind)

        mask = mask.unsqueeze(2).expand_as(pred).float()
        loss = F.l1_loss(pred * mask, target * mask, reduction="sum")
        return loss / (mask.sum() + 1e-4)


class CenterNetLoss(nn.Module):
    def __init__(self, hm_weight: float = 1.0, reg_weight: float = 1.0, wh_weight: float = 0.1) -> None:
        super().__init__()
        self.hm_loss = CenterNetFocalLoss()
        self.reg_loss = RegL1Loss()
        self.wh_loss = RegL1Loss()
        self.hm_weight = float(hm_weight)
        self.reg_weight = float(reg_weight)
        self.wh_weight = float(wh_weight)

    def loss_dict(
        self,
        outputs: dict[str, torch.Tensor],
        batch: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        hm = self.hm_loss(outputs["hm"], batch["hm"])
        reg = self.reg_loss(outputs["reg"], batch["reg_mask"], batch["ind"], batch["reg"])
        wh = self.wh_loss(outputs["wh"], batch["reg_mask"], batch["ind"], batch["wh"])
        total = self.hm_weight * hm + self.reg_weight * reg + self.wh_weight * wh
        return {
            "loss": total,
            "hm_loss": hm,
            "reg_loss": reg,
            "wh_loss": wh,
            "weighted_hm_loss": self.hm_weight * hm,
            "weighted_reg_loss": self.reg_weight * reg,
            "weighted_wh_loss": self.wh_weight * wh,
        }

    def forward(
        self,
        outputs: dict[str, torch.Tensor],
        batch: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        return self.loss_dict(outputs, batch)["loss"]
