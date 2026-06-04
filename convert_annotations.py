"""
Convert CVAT COCO exports to B1 training index.
Merges both JSONs from different batches.

Run from your contam_cv folder:
    /opt/miniconda3/bin/python convert_annotations.py
"""

import json, os, re
from PIL import Image, ImageDraw

CVAT_JSONS = [
    os.path.expanduser("~/Desktop/contam_cv/instances_default_1__2__4_.json"),
    os.path.expanduser("~/Desktop/contam_cv/instances_default_5__6_.json"),
]
TACO_ROOT   = os.path.expanduser("~/Desktop/contam_cv/TACO/data")
MASK_DIR    = os.path.expanduser("~/Desktop/contam_cv/masks")
OUTPUT_JSON = os.path.expanduser("~/Desktop/contam_cv/taco_manual.json")

os.makedirs(MASK_DIR, exist_ok=True)

def find_on_disk(cvat_name):
    base = re.sub(r"_\d+(\.\w+)$", r"\1", cvat_name)
    for batch in sorted(os.listdir(TACO_ROOT)):
        batch_dir = os.path.join(TACO_ROOT, batch)
        if not os.path.isdir(batch_dir): continue
        for fname in [base, base.upper(), base.lower(),
                      base.replace(".jpg",".JPG"), base.replace(".JPG",".jpg")]:
            c = os.path.join(batch_dir, fname)
            if os.path.exists(c): return c
    return None

index = []; found = not_found = total_contam = 0; seen = set()

for jp in CVAT_JSONS:
    if not os.path.exists(jp):
        print(f"WARNING: {jp} not found, skipping"); continue
    with open(jp) as f: coco = json.load(f)
    id_to_name = {img["id"]: img["file_name"] for img in coco["images"]}
    id_to_anns = {}
    for ann in coco["annotations"]:
        id_to_anns.setdefault(ann["image_id"], []).append(ann)
    for img in coco["images"]:
        iid = img["id"]; cvat_name = img["file_name"]
        path = find_on_disk(cvat_name)
        if path is None: print(f"  NOT FOUND: {cvat_name}"); not_found += 1; continue
        if path in seen: continue
        seen.add(path); found += 1
        anns = id_to_anns.get(iid, []); contam = 1 if anns else 0; mask_path = None
        if anns:
            total_contam += 1
            try: pil = Image.open(path); w, h = pil.size
            except: w, h = img["width"], img["height"]
            mask = Image.new("L", (w, h), 0); draw = ImageDraw.Draw(mask)
            for ann in anns:
                x, y, bw, bh = ann["bbox"]
                draw.rectangle([x, y, x+bw, y+bh], fill=255)
            safe = re.sub(r"[^\w.]", "_", cvat_name) + "_mask.png"
            mp = os.path.join(MASK_DIR, safe); mask.save(mp); mask_path = mp
        index.append({"image_path": path, "contam": contam, "mask_path": mask_path})

with open(OUTPUT_JSON, "w") as f: json.dump(index, f, indent=2)
print(f"\nDone.")
print(f"  Images found:     {found}")
print(f"  Images not found: {not_found}")
print(f"  Contaminated:     {total_contam}")
print(f"  Clean:            {found - total_contam}")
print(f"  Output:           {OUTPUT_JSON}")
