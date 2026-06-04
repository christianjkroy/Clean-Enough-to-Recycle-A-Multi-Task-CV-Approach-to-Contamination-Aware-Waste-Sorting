from __future__ import annotations
import argparse, math, torch
from torch.utils.data import DataLoader
from datasets import ZeroWasteClean, TacoContam
from losses import MultiTaskLoss
from metrics import PresenceMeter, SegMeter, ClsMeter
from models import MultiHeadResNet

HEADS={"B0":("cls",),"B1":("cls","pres","loc"),"B2":("cls","pres","loc"),"B3":("cls","pres","loc")}

def cosine_warmup(step,total,warmup):
    if step<warmup: return step/max(1,warmup)
    p=(step-warmup)/max(1,total-warmup); return 0.5*(1+math.cos(math.pi*p))

def move(batch,device): return {k:(v.float().to(device) if torch.is_tensor(v) and v.is_floating_point() else v.to(device) if torch.is_tensor(v) else v) for k,v in batch.items()}

def evaluate(model,loader,device):
    model.eval(); pm,sm,cm=PresenceMeter(),SegMeter(),ClsMeter()
    with torch.no_grad():
        for batch in loader:
            batch=move(batch,device); out=model(batch["image"])
            if "pres" in out and "contam" in batch: pm.update(out["pres"],batch["contam"])
            if "loc" in out and "mask" in batch:
                v=batch.get("has_mask"); sel=v if v is not None else slice(None)
                if v is None or v.any(): sm.update(out["loc"][sel],batch["mask"][sel])
            if "cls" in out and "label" in batch: cm.update(out["cls"],batch["label"])
    res={}
    res.update({f"pres/{k}":v for k,v in pm.compute().items()})
    res.update({f"loc/{k}":v for k,v in sm.compute().items()})
    res.update({f"cls/{k}":v for k,v in cm.compute().items()})
    return res

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--condition",choices=list(HEADS),required=True)
    ap.add_argument("--zerowaste_root",required=True)
    ap.add_argument("--taco_test_index",default=None)
    ap.add_argument("--epochs",type=int,default=30)
    ap.add_argument("--batch_size",type=int,default=32)
    ap.add_argument("--lr",type=float,default=1e-4)
    ap.add_argument("--weight_decay",type=float,default=0.05)
    ap.add_argument("--lambda_cls",type=float,default=1.0)
    ap.add_argument("--lambda_pres",type=float,default=2.0)
    ap.add_argument("--lambda_loc",type=float,default=1.0)
    ap.add_argument("--fn_pos_weight",type=float,default=5.0)
    ap.add_argument("--workers",type=int,default=4)
    ap.add_argument("--seed",type=int,default=0)
    ap.add_argument("--wandb",action="store_true")
    args=ap.parse_args()
    torch.manual_seed(args.seed)
    if torch.cuda.is_available(): device="cuda"
    elif torch.backends.mps.is_available(): device="mps"
    else: device="cpu"
    print(f"using device: {device}")
    run=None
    if args.wandb:
        import wandb; run=wandb.init(project="contam-cv",config=vars(args),name=f"{args.condition}-seed{args.seed}")
    model=MultiHeadResNet(active_heads=HEADS[args.condition]).to(device)
    synthesize=args.condition in ("B2","B3")
    train_ds=ZeroWasteClean(args.zerowaste_root,train=True,synthesize=synthesize,seed=args.seed)
    train_ld=DataLoader(train_ds,batch_size=args.batch_size,shuffle=True,num_workers=args.workers,drop_last=True)
    test_ld=None
    if args.taco_test_index:
        import json
        with open(args.taco_test_index) as f: idx=json.load(f)
        test_ld=DataLoader(TacoContam(idx),batch_size=args.batch_size,num_workers=args.workers)
    criterion=MultiTaskLoss(args.lambda_cls,args.lambda_pres,args.lambda_loc,args.fn_pos_weight).to(device)
    opt=torch.optim.AdamW(model.parameters(),lr=args.lr,weight_decay=args.weight_decay)
    total_steps=args.epochs*max(1,len(train_ld)); warmup=max(1,len(train_ld)); step=0; best_f1=-1.0
    for epoch in range(args.epochs):
        model.train()
        for batch in train_ld:
            batch=move(batch,device)
            for g in opt.param_groups: g["lr"]=args.lr*cosine_warmup(step,total_steps,warmup)
            out=model(batch["image"]); loss,logs=criterion(out,batch)
            opt.zero_grad(); loss.backward(); opt.step(); step+=1
            if run and step%20==0: run.log({**logs,"lr":opt.param_groups[0]["lr"]},step=step)
        if test_ld:
            res=evaluate(model,test_ld,device)
            print(f"[epoch {epoch}] "+" ".join(f"{k}={v:.4f}" for k,v in res.items()))
            if run: run.log(res,step=step)
            if res.get("pres/f1",0)>best_f1:
                best_f1=res["pres/f1"]; torch.save(model.state_dict(),f"best_{args.condition}_s{args.seed}.pt")
    print("done. best presence F1:",best_f1)
    if run: run.finish()

if __name__=="__main__": main()