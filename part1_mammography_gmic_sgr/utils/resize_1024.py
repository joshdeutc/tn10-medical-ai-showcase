import os
from pathlib import Path
from multiprocessing import Pool, cpu_count
from PIL import Image
from tqdm import tqdm

SRC = Path("/home/joshua/breast_cancer/data/preprocess_image/cropped_images")
DST = Path("/home/joshua/breast_cancer/data/preprocess_image/cropped_1024")
SIZE = (1024, 1024)

def resize_one(args):
    src_path, dst_path = args
    try:
        dst_path.parent.mkdir(parents=True, exist_ok=True)
        if dst_path.exists():
            return "skip"
        img = Image.open(src_path).resize(SIZE, Image.LANCZOS)
        img.save(dst_path, format="PNG", optimize=False)
        return "ok"
    except Exception as e:
        return f"err:{e}"

if __name__ == "__main__":
    pairs = [
        (p, DST / p.relative_to(SRC))
        for p in SRC.rglob("*.png")
    ]
    print(f"{len(pairs)} images à traiter avec {cpu_count()} cœurs")

    with Pool(cpu_count()) as pool:
        results = list(tqdm(pool.imap_unordered(resize_one, pairs), total=len(pairs)))

    ok = results.count("ok")
    skip = results.count("skip")
    err = [r for r in results if r.startswith("err")]
    print(f"Terminé — ok:{ok} skip:{skip} erreurs:{len(err)}")
    if err:
        for e in err[:5]:
            print(e)
