from __future__ import annotations
import os, json, torch
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as T
from synthetic import SyntheticContaminator

MEAN=[0.485,0.456,0.406]; STD=[0.229,0.224,0.225]

def tf(train=True):
    aug=[T.Resize((224,224))]
    if train: aug+=[T.RandomHorizontalFlip(),T.ColorJitter(0.2,0.2,0.2,0.05)]
    return T.Compose(aug+[T.ToTensor(),T.Normalize(MEAN,STD)])

class ZeroWasteClean(Dataset):
    def __init__(self,root,train=True,synthesize=False,synth_prob=0.5,seed=0):
        self.root=root; self.train=train; self.synthesize=synthesize; self.synth_prob=synth_prob
        self.tf=tf(train); self.contaminator=SyntheticContaminator(seed=seed); self.samples=self._index()
    def _index(self):
        split="train" if self.train else "test"
        ann_path=os.path.join(self.root,"splits_final_deblurred",split,"labels.json")
        img_dir=os.path.join(self.root,"splits_final_deblurred",split,"data")
        with open(ann_path) as f: coco=json.load(f)
        cat_ids=sorted(c["id"] for c in coco["categories"])
        cat_map={cid:i for i,cid in enumerate(cat_ids)}
        id_to_path={img["id"]:os.path.join(img_dir,img["file_name"]) for img in coco["images"]}
        seen={}
        for ann in coco["annotations"]:
            iid=ann["image_id"]
            if iid not in seen: seen[iid]=(id_to_path[iid],cat_map[ann["category_id"]])
        return list(seen.values())
    def __len__(self): return len(self.samples)
    def __getitem__(self,i):
        path,label=self.samples[i]; img=Image.open(path).convert("RGB").resize((224,224))
        s={"label":int(label)}
        if self.synthesize and torch.rand(1).item()<self.synth_prob:
            ci,mask=self.contaminator(img); s["image"]=self.tf(ci)
            s["contam"]=1.0; s["mask"]=torch.from_numpy(mask); s["has_mask"]=True
        else:
            s["image"]=self.tf(img); s["contam"]=0.0; s["mask"]=torch.zeros(224,224); s["has_mask"]=True
        return s

class TacoContam(Dataset):
    def __init__(self,ann_index,train=False): self.items=ann_index; self.tf=tf(train)
    def __len__(self): return len(self.items)
    def __getitem__(self,i):
        it=self.items[i]; img=Image.open(it["image_path"]).convert("RGB").resize((224,224))
        s={"image":self.tf(img),"contam":float(it["contam"])}
        if it.get("mask_path"):
            import numpy as np
            m=Image.open(it["mask_path"]).convert("L").resize((224,224))
            s["mask"]=(torch.from_numpy(np.asarray(m))>127).float(); s["has_mask"]=True
        else:
            s["mask"]=torch.zeros(224,224); s["has_mask"]=False
        if "label" in it: s["label"]=int(it["label"])
        return s
