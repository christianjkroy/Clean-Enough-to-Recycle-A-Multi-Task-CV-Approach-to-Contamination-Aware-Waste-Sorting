from __future__ import annotations
import torch

class PresenceMeter:
    def __init__(self,fn_cost=5.0,thresh=0.5): self.fn_cost=fn_cost; self.thresh=thresh; self.tp=self.fp=self.tn=self.fn=0
    def update(self,logits,target):
        pred=(torch.sigmoid(logits)>=self.thresh).long(); tgt=target.long()
        self.tp+=int(((pred==1)&(tgt==1)).sum()); self.fp+=int(((pred==1)&(tgt==0)).sum())
        self.tn+=int(((pred==0)&(tgt==0)).sum()); self.fn+=int(((pred==0)&(tgt==1)).sum())
    def compute(self):
        tp,fp,fn=self.tp,self.fp,self.fn
        p=tp/(tp+fp) if (tp+fp) else 0.0; r=tp/(tp+fn) if (tp+fn) else 0.0
        f1=2*p*r/(p+r) if (p+r) else 0.0
        return {"precision":p,"recall":r,"f1":f1,"cost_weighted_err":(self.fn_cost*fn+fp)/max(1,tp+fp+self.tn+fn)}

class SegMeter:
    def __init__(self,thresh=0.5,eps=1e-6): self.thresh=thresh; self.eps=eps; self.inter=self.union=self.tp=self.fp=self.fn=0.0
    def update(self,logits,target):
        pred=(torch.sigmoid(logits)>=self.thresh).float(); tgt=(target>=0.5).float()
        self.inter+=float((pred*tgt).sum()); self.union+=float(((pred+tgt)>=1).float().sum())
        self.tp+=float((pred*tgt).sum()); self.fp+=float((pred*(1-tgt)).sum()); self.fn+=float(((1-pred)*tgt).sum())
    def compute(self):
        miou=self.inter/(self.union+self.eps); p=self.tp/(self.tp+self.fp+self.eps); r=self.tp/(self.tp+self.fn+self.eps)
        return {"mIoU":miou,"pixel_f1":2*p*r/(p+r+self.eps)}

class ClsMeter:
    def __init__(self,num_classes=4): self.n=num_classes; self.correct=self.total=0; self.per_tp=[0]*num_classes; self.per_fp=[0]*num_classes; self.per_fn=[0]*num_classes
    def update(self,logits,target):
        pred=logits.argmax(1); self.correct+=int((pred==target).sum()); self.total+=target.numel()
        for c in range(self.n):
            self.per_tp[c]+=int(((pred==c)&(target==c)).sum()); self.per_fp[c]+=int(((pred==c)&(target!=c)).sum()); self.per_fn[c]+=int(((pred!=c)&(target==c)).sum())
    def compute(self):
        f1s=[]
        for c in range(self.n):
            tp,fp,fn=self.per_tp[c],self.per_fp[c],self.per_fn[c]; p=tp/(tp+fp) if (tp+fp) else 0.0; r=tp/(tp+fn) if (tp+fn) else 0.0
            f1s.append(2*p*r/(p+r) if (p+r) else 0.0)
        return {"accuracy":self.correct/max(1,self.total),"macro_f1":sum(f1s)/self.n}