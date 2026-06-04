from __future__ import annotations
import torch, torch.nn as nn, torch.nn.functional as F

def dice_loss(logits, target, eps=1.0):
    p = torch.sigmoid(logits).flatten(1)
    t = target.flatten(1)
    return (1-(2*(p*t).sum(1)+eps)/((p.sum(1)+t.sum(1))+eps)).mean()

class MultiTaskLoss(nn.Module):
    def __init__(self, lambda_cls=1.0, lambda_pres=2.0, lambda_loc=1.0, fn_pos_weight=5.0):
        super().__init__()
        self.lc,self.lp,self.ll = lambda_cls,lambda_pres,lambda_loc
        self.ce = nn.CrossEntropyLoss()
        self.register_buffer("pos_weight", torch.tensor(float(fn_pos_weight)))
    def forward(self, out, batch):
        dev = next(iter(out.values())).device
        total = torch.zeros((),device=dev); logs={}
        if "cls" in out and "label" in batch:
            l=self.ce(out["cls"],batch["label"]); total=total+self.lc*l; logs["loss_cls"]=float(l.detach())
        if "pres" in out and "contam" in batch:
            l=F.binary_cross_entropy_with_logits(out["pres"],batch["contam"].float(),pos_weight=self.pos_weight)
            total=total+self.lp*l; logs["loss_pres"]=float(l.detach())
        if "loc" in out and "mask" in batch:
            valid=batch.get("has_mask",torch.ones(out["loc"].shape[0],dtype=torch.bool,device=dev))
            if valid.any():
                l=F.binary_cross_entropy_with_logits(out["loc"][valid],batch["mask"][valid].float())+dice_loss(out["loc"][valid],batch["mask"][valid].float())
                total=total+self.ll*l; logs["loss_loc"]=float(l.detach())
        logs["loss_total"]=float(total.detach()); return total,logs