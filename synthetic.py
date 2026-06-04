from __future__ import annotations
import numpy as np
from PIL import Image

KINDS = ("grease","residue","liquid")

def _blob_mask(h,w,rng):
    yy,xx = np.mgrid[0:h,0:w]; mask=np.zeros((h,w),dtype=np.float32)
    for _ in range(rng.integers(2,6)):
        cy,cx=rng.uniform(0.2,0.8)*h,rng.uniform(0.2,0.8)*w
        sy,sx=rng.uniform(0.05,0.22)*h,rng.uniform(0.05,0.22)*w
        mask=np.maximum(mask,np.exp(-(((yy-cy)**2)/(2*sy**2)+((xx-cx)**2)/(2*sx**2))).astype(np.float32))
    return np.clip(mask*(0.7+0.3*rng.random((h,w)).astype(np.float32)),0,1)

def _texture(kind,h,w,rng):
    b=rng.random((h,w,1)).astype(np.float32)
    if kind=="grease": t=np.array([40,35,30],dtype=np.float32)*(0.6+0.5*b)
    elif kind=="residue":
        s=(rng.random((h,w,1))>0.85).astype(np.float32)
        t=np.array([110,80,50],dtype=np.float32)*(0.7+0.4*b)+s*30
    else: t=np.array([90,110,130],dtype=np.float32)*(0.8+0.3*b)
    return np.clip(t,0,255).astype(np.uint8)

class SyntheticContaminator:
    def __init__(self, seed=None): self.rng=np.random.default_rng(seed)
    def __call__(self, clean, kind=None):
        clean=clean.convert("RGB"); w,h=clean.size
        kind=kind or KINDS[self.rng.integers(len(KINDS))]
        arr=np.asarray(clean,dtype=np.float32)
        alpha=_blob_mask(h,w,self.rng)*self.rng.uniform(0.55,0.95)
        tex=_texture(kind,h,w,self.rng).astype(np.float32)
        out=Image.fromarray(np.clip(arr*(1-alpha[...,None])+tex*alpha[...,None],0,255).astype(np.uint8))
        return out,(alpha>0.25).astype(np.float32)