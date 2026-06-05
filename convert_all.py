import json, os, re
from PIL import Image, ImageDraw

CVAT_JSONS = [
    os.path.expanduser("~/Desktop/contam_cv/instances_default_batch_new.json"),
    os.path.expanduser("~/Desktop/contam_cv/instances_default_batch1.json"),
    os.path.expanduser("~/Desktop/contam_cv/instances_default_batch124.json"),
    os.path.expanduser("~/Desktop/contam_cv/instances_default_batch378.json"),
    os.path.expanduser("~/Desktop/contam_cv/instances_default_batch56.json"),
    os.path.expanduser("~/Desktop/contam_cv/instances_default_batch91015.json"),
]
TACO_ROOT   = os.path.expanduser("~/Desktop/contam_cv/TACO/data")
MASK_DIR    = os.path.expanduser("~/Desktop/contam_cv/masks")
OUTPUT_JSON = os.path.expanduser("~/Desktop/contam_cv/taco_manual.json")

os.makedirs(MASK_DIR, exist_ok=True)

disk_index = {}
for batch in sorted(os.listdir(TACO_ROOT)):
    batch_dir = os.path.join(TACO_ROOT, batch)
    if not os.path.isdir(batch_dir): continue
    for fname in os.listdir(batch_dir):
        disk_index.setdefault(fname.lower(), []).append(os.path.join(batch_dir, fname))

def find_on_disk(cvat_name, used):
    base = re.sub(r"_\d+(\.\w+)$", r"\1", cvat_name).lower()
    for c in disk_index.get(base, []):
        if c not in used: return c
    return disk_index.get(base, [None])[0]

index = []; found = not_found = total_contam = 0
seen_paths = set(); used_per_base = {}

for jp in CVAT_JSONS:
    with open(jp) as f: coco = json.load(f)
    id_to_anns = {}
    for ann in coco["annotations"]:
        id_to_anns.setdefault(ann["image_id"], []).append(ann)
    for img in coco["images"]:
        iid = img["id"]; cvat_name = img["file_name"]
        base = re.sub(r"_\d+(\.\w+)$", r"\1", cvat_name).lower()
        used = used_per_base.get(base, set())
        path = find_on_disk(cvat_name, used)
        if path is None: not_found += 1; continue
        if path in seen_paths: continue
        seen_paths.add(path); used_per_base.setdefault(base, set()).add(path); found += 1
        anns = id_to_anns.get(iid, []); contam = 1 if anns else 0; mask_path = None
        if anns:
            total_contam += 1
            try: pil = Image.open(path); w, h = pil.size
            except: w, h = img["width"], img["height"]
            mask = Image.new("L", (w, h), 0); draw = ImageDraw.Draw(mask)
            for ann in anns:
                if ann.get("segmentation") and ann["segmentation"]:
                    pts = ann["segmentation"][0]
                    poly = [(pts[i], pts[i+1]) for i in range(0, len(pts), 2)]
                    draw.polygon(poly, fill=255)
                else:
                    x, y, bw, bh = ann["bbox"]
                    draw.rectangle([x, y, x+bw, y+bh], fill=255)
            safe = re.sub(r"[^\w.]", "_", cvat_name) + "_mask.png"
            mp = os.path.join(MASK_DIR, safe); mask.save(mp); mask_path = mp
        index.append({"image_path": path, "contam": contam, "mask_path": mask_path})

with open(OUTPUT_JSON, "w") as f: json.dump(index, f, indent=2)
print(f"Done.")
print(f"  Images found:     {found}")
print(f"  Images not found: {not_found}")
print(f"  Contaminated:     {total_contam}")
print(f"  Clean:            {found - total_contam}")
