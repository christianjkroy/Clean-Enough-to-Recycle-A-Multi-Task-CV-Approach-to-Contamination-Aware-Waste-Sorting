from __future__ import annotations
import torch, torch.nn as nn, torchvision

class _SegDecoder(nn.Module):
    def __init__(self, in_ch=2048):
        super().__init__()
        chans = [in_ch, 512, 256, 128, 64, 32]
        self.blocks = nn.Sequential(*[
            nn.Sequential(nn.Upsample(scale_factor=2,mode="bilinear",align_corners=False),
                nn.Conv2d(ci,co,3,padding=1), nn.BatchNorm2d(co), nn.ReLU(True))
            for ci,co in zip(chans[:-1],chans[1:])])
        self.out = nn.Conv2d(32,1,1)
    def forward(self,x): return self.out(self.blocks(x))

class MultiHeadResNet(nn.Module):
    def __init__(self, num_classes=4, pretrained=True, active_heads=("cls","pres","loc")):
        super().__init__()
        self.active_heads = set(active_heads)
        w = torchvision.models.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
        b = torchvision.models.resnet50(weights=w)
        self.stem = nn.Sequential(b.conv1,b.bn1,b.relu,b.maxpool,b.layer1,b.layer2,b.layer3,b.layer4)
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.cls_head = nn.Linear(2048, num_classes)
        self.pres_head = nn.Linear(2048, 1)
        self.loc_head = _SegDecoder(2048)
    def forward(self, x):
        feat = self.stem(x)
        pooled = self.gap(feat).flatten(1)
        out = {}
        if "cls"  in self.active_heads: out["cls"]  = self.cls_head(pooled)
        if "pres" in self.active_heads: out["pres"] = self.pres_head(pooled).squeeze(1)
        if "loc"  in self.active_heads: out["loc"]  = self.loc_head(feat).squeeze(1)
        return out
    def backbone_target_layer(self): return self.stem[-1]