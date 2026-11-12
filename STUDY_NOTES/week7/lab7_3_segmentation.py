"""Lab 7.3 - Instance segmentation with YOLOv8m-seg + classical (low-level) segmentation.

Part A: deep instance segmentation (what the lab does):  results = model(img)
Part B: lecture's low-level methods - grey-level thresholding (Otsu), k-means colour
        clustering, region growing - to contrast with the deep approach.
"""
import os
os.environ.setdefault("YOLO_CONFIG_DIR", "/tmp/ultra")
import cv2, numpy as np
from skimage import data as skd
from ultralytics import YOLO

HERE = os.path.dirname(os.path.abspath(__file__))
MODELS = os.environ.get("MODELS", os.path.join(HERE, "..", "..", "..", "models"))
OUT = os.path.join(HERE, "out"); os.makedirs(OUT, exist_ok=True)
model = YOLO(os.path.join(MODELS, "yolov8m-seg.pt"))

def bgr(a): return cv2.cvtColor(a, cv2.COLOR_RGB2BGR)
imgs = {
    "bus": cv2.imread(os.path.join(HERE, "data", "bus.jpg")),
    "zidane": cv2.imread(os.path.join(HERE, "data", "zidane.jpg")),
    "cat": bgr(skd.chelsea()),
    "coffee": bgr(skd.coffee()),
}

print("=== Part A: YOLOv8m-seg instance segmentation ===")
for name, im in imgs.items():
    r = model(im, verbose=False)[0]
    cv2.imwrite(os.path.join(OUT, f"seg_{name}.jpg"), r.plot())       # = results[0].plot()
    names = [r.names[int(c)] for c in r.boxes.cls]
    confs = [round(float(c), 2) for c in r.boxes.conf]
    print(f"\n{name}: {len(names)} instances   speed(ms) {({k: round(v,1) for k,v in r.speed.items()})}")
    if r.masks is not None:
        areas = r.masks.data.sum(dim=(1, 2)).tolist()
        mh, mw = r.masks.data.shape[1:]
        for i, (n, c, a) in enumerate(zip(names, confs, areas)):
            print(f"   #{i} {n:10s} conf={c:.2f} mask area={a/(mh*mw)*100:5.1f}% of image")
        # semantic vs instance: merge instance masks of the same class
        from collections import Counter
        cnt = Counter(names)
        print("   instance view :", dict(cnt))
        print("   semantic view :", {k: "1 region (merged)" for k in cnt}, "-> semantic output cannot tell how many of each (e.g. %d separate persons -> one 'person' region)" % cnt.get("person", 0) if cnt.get("person", 0) > 1 else "-> counts lost")
        # mask overlap between persons (instances stay separate even when they touch)
        pm = [r.masks.data[i].cpu().numpy() > 0.5 for i, n in enumerate(names) if n == "person"]
        if len(pm) > 1:
            ov = sum(int((pm[i] & pm[j]).sum()) for i in range(len(pm)) for j in range(i + 1, len(pm)))
            print(f"   overlapping pixels between different person masks: {ov}")

print("\n=== Part B: classical segmentation (grey-level / colour / region growing) ===")
# 1. grey-level threshold on skimage 'coins' (24 coins; uneven lighting + textured coin faces)
coins = skd.coins()
T, bw = cv2.threshold(coins, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
n, lab = cv2.connectedComponents(bw)
print(f"Otsu threshold T={T:.0f}; foreground fraction={bw.mean()/255:.2f}; connected components={n-1} (true coins=24)")
ad = cv2.adaptiveThreshold(coins, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 51, -5)
n2, _ = cv2.connectedComponents(ad)
print(f"adaptive threshold components={n2-1}")
cv2.imwrite(os.path.join(OUT, "classical_otsu_coins.png"), np.hstack([coins, bw, ad]))

# 2. clustering in colour space (k-means) on coffee
im = imgs["coffee"]
Z = im.reshape(-1, 3).astype(np.float32)
for k in (2, 3, 5):
    crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)
    comp, lab_, ctr = cv2.kmeans(Z, k, None, crit, 3, cv2.KMEANS_PP_CENTERS)
    sizes = np.bincount(lab_.ravel()) / len(Z) * 100
    print(f"k-means colour clustering k={k}: segment sizes % = {np.round(np.sort(sizes)[::-1],1).tolist()} compactness={comp:.3g}")
    cv2.imwrite(os.path.join(OUT, f"classical_kmeans_k{k}.png"), ctr.astype(np.uint8)[lab_.ravel()].reshape(im.shape))

# 3. region growing from a seed on coins (tolerance on grey level)
def region_grow(g, seed, tol):
    h, w = g.shape; seg = np.zeros((h, w), bool); st = [seed]; ref = float(g[seed]); s = 0
    while st:
        y, x = st.pop()
        if seg[y, x] or abs(float(g[y, x]) - ref) > tol: continue
        seg[y, x] = True
        for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            ny, nx = y + dy, x + dx
            if 0 <= ny < h and 0 <= nx < w and not seg[ny, nx]: st.append((ny, nx))
    return seg
blur = cv2.GaussianBlur(coins, (5, 5), 0)
for tol in (5, 15, 30, 60):
    seg = region_grow(blur, (45, 80), tol)
    print(f"region growing seed=(45,80) grey={blur[45,80]} tol={tol:>2}: region = {seg.mean()*100:5.1f}% of image")
