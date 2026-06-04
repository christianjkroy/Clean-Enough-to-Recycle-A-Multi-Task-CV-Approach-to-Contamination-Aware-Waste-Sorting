from __future__ import annotations
import numpy as np, torch, torch.nn.functional as F

class PresenceGradCAM:
    def __init__(self,model):
        self.model=model; self.acts=self.grads=None
        t=model.backbone_target_layer()
        t.register_forward_hook(lambda m,i,o: setattr(self,"acts",o.detach()))
        t.register_full_backward_hook(lambda m,gi,go: setattr(self,"grads",go[0].detach()))
    @torch.enable_grad()
    def __call__(self,image):
        self.model.zero_grad(); out=self.model(image); out["pres"].sum().backward()
        w=self.grads.mean(dim=(2,3),keepdim=True)
        cam=F.relu((w*self.acts).sum(1))
        cam=F.interpolate(cam[None],size=image.shape[-2:],mode="bilinear",align_corners=False)[0,0]
        cam=cam-cam.min(); cam=cam/(cam.max()+1e-6)
        return cam.cpu().numpy()