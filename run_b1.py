import sys, time, torch, json
sys.path.insert(0, "/Users/christianroy/Desktop/contam_cv")
from datasets import ZeroWasteClean, TacoContam
from models import MultiHeadResNet
from losses import MultiTaskLoss
from metrics import PresenceMeter, SegMeter
from torch.utils.data import DataLoader

device = "mps" if torch.backends.mps.is_available() else "cpu"
print("device:", device, flush=True)
torch.manual_seed(0)
model = MultiHeadResNet(active_heads=("cls","pres","loc")).to(device)
train_ds = ZeroWasteClean("./zerowaste", train=True, synthesize=True, synth_prob=0.5, seed=0)
train_ld = DataLoader(train_ds, batch_size=16, shuffle=True, num_workers=0, drop_last=True)
with open("taco_manual.json") as f: idx = json.load(f)
test_ld = DataLoader(TacoContam(idx, train=False), batch_size=16, num_workers=0)
print(f"test set: {len(idx)} images, {sum(1 for x in idx if x['contam'])} contaminated", flush=True)
crit = MultiTaskLoss().to(device)
opt = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.05)
for epoch in range(15):
    model.train(); t0=time.time()
    for batch in train_ld:
        batch={k:(v.float().to(device) if torch.is_tensor(v) and v.is_floating_point() else v.to(device) if torch.is_tensor(v) else v) for k,v in batch.items()}
        out=model(batch["image"]); loss,_=crit(out,batch)
        opt.zero_grad(); loss.backward(); opt.step()
    model.eval(); pm=PresenceMeter(); sm=SegMeter()
    with torch.no_grad():
        for batch in test_ld:
            imgs=batch["image"].float().to(device); out=model(imgs)
            pm.update(out["pres"], batch["contam"].float().to(device))
            v=batch.get("has_mask")
            if v is None or v.any():
                sel=v if v is not None else slice(None)
                sm.update(out["loc"][sel], batch["mask"][sel].float().to(device))
    rp=pm.compute(); rs=sm.compute()
    print(f"epoch {epoch+1}/15 — pres_f1 {rp['f1']:.3f} recall {rp['recall']:.3f} mIoU {rs['mIoU']:.3f} — {time.time()-t0:.0f}s", flush=True)
torch.save(model.state_dict(), "b1_trained.pt")
print("saved b1_trained.pt", flush=True)
