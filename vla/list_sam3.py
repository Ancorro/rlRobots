"""List first, then check with SAM 3: each name a vision-language model listed from the whole picture (vla/list_check.py
ask) is looked for by SAM 3 (text prompt, facebook/sam3 as in so101/see.py); a name SAM 3 cannot find is dropped.
Targets listing's weak point, invented names (owner 2026-10-06).

A name counts as found when SAM 3 returns an instance with score >= --thr whose mask is under 10 % of the picture (see.py's
"not the table or the robot" rule) and at most 30 % robot/fixture pixels (vla/crop_list.py's fixture filter, known.npz).
--thr defaults to 0.3, see.py's value, set before these runs; the report also sweeps 0.2-0.7.

  python vla/list_sam3.py runs/listing/<stem>.jsonl ...   (open answers: pass the -judged.jsonl copy; its guess/truth
                                                           pairs are kept for the names that stay, so no new judging)
  -> runs/listing/<stem>-sam3.jsonl (rows as list_check's, plus "sam3": best score per listed name), scored by
     list_check report.                                              cluster: sbatch vla/list_sam3.sbatch <stems>
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from transformers import Sam3Model, Sam3Processor

OUT = Path("runs/listing")
SAM3 = "facebook/sam3"
ARGS = dict(a[2:].split("=", 1) for a in sys.argv[1:] if a.startswith("--") and "=" in a)
THR = float(ARGS.get("thr", 0.3))
SWEEP = (0.2, 0.3, 0.4, 0.5, 0.6, 0.7)


def main():
    proc = Sam3Processor.from_pretrained(SAM3)
    model = Sam3Model.from_pretrained(SAM3).to("cuda").eval()
    kn = dict(np.load("runs/croplist/known.npz"))  # robot/fixture pixels per picture (crop_list truthboxes)

    @torch.inference_mode()
    def best(img, image, phrase):
        inputs = proc(images=img, text=phrase, return_tensors="pt").to("cuda")
        r = proc.post_process_instance_segmentation(model(**inputs), threshold=min(SWEEP), mask_threshold=0.5,
                                                    target_sizes=inputs.get("original_sizes").tolist())[0]
        known = np.unpackbits(kn[image])[:640 * 640].reshape(640, 640).astype(bool)
        s = 0.0
        for m, sc in zip(r["masks"].cpu().numpy().astype(bool), r["scores"].float().cpu().numpy()):
            if m.sum() and m.sum() < m.size * 0.1 and known[m].mean() <= 0.3:
                s = max(s, float(sc))
        return s

    for src in [Path(a) for a in sys.argv[1:] if not a.startswith("--")]:
        rows = [json.loads(l) for l in src.read_text().splitlines() if l.strip()]
        cache = {}
        for r in rows:
            img = Image.open(OUT / f"{r['image']}.png").convert("RGB")
            r["sam3"] = []
            for name in r["listed"] or []:
                k = (r["image"], name)
                if k not in cache:
                    cache[k] = best(img, r["image"], name)
                r["sam3"].append(round(cache[k], 4))
        for thr in SWEEP:
            out = []
            for r in rows:
                o = dict(r)
                if r["listed"] is not None:
                    keep = [i for i, s in enumerate(r["sam3"]) if s >= thr]
                    o["listed"] = [r["listed"][i] for i in keep]
                    if "pairs" in r:
                        new = {old: new for new, old in enumerate(keep)}
                        o["pairs"] = [[new[g], t] for g, t in r["pairs"] if g in new]
                out.append(o)
            stem = src.stem + ("-sam3" if thr == THR else f"-sam3t{round(thr * 100):02d}")
            (OUT / f"{stem}.jsonl").write_text("".join(json.dumps(o) + "\n" for o in out))
        n = sum(len(r["listed"] or []) for r in rows)
        kept = sum(s >= THR for r in rows for s in r["sam3"])
        print(f"{src.stem}: {n} listed names, {kept} kept by SAM 3 at {THR} ({len(cache)} SAM 3 calls)")


if __name__ == "__main__":
    main()
