"""Lab 7.1 - Haar cascade face detection (Viola & Jones 2001) - verified run.

Usage:  python lab7_1_face_haar.py            (MODELS env var = folder with the xml)
Colab-only bits of the original notebook (google.colab.patches.cv2_imshow, Drive mount,
brunel.jpg) are replaced by: cv2.imwrite() of annotated results + bundled sample images.
"""
import os, time, xml.etree.ElementTree as ET
import cv2, numpy as np
from skimage import data as skd

HERE = os.path.dirname(os.path.abspath(__file__))
MODELS = os.environ.get("MODELS", os.path.join(HERE, "..", "..", "..", "models"))
OUT = os.path.join(HERE, "out"); os.makedirs(OUT, exist_ok=True)
XML = os.path.join(MODELS, "haarcascade_frontalface_default.xml")
face_cascade = cv2.CascadeClassifier(XML)
assert not face_cascade.empty(), "cascade failed to load"

def bgr(a):  # skimage gives RGB (or gray)
    return cv2.cvtColor(a, cv2.COLOR_RGB2BGR) if a.ndim == 3 else cv2.cvtColor(a, cv2.COLOR_GRAY2BGR)

def detect(img, save=None, **kw):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)          # cascade works on GRAY
    t = time.perf_counter()
    faces = face_cascade.detectMultiScale(gray, **kw)
    ms = (time.perf_counter() - t) * 1000
    if save:
        vis = img.copy()
        for (x, y, w, h) in faces:
            cv2.rectangle(vis, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.imwrite(os.path.join(OUT, save), vis)
    return faces, ms

# ---------------------------------------------------------------- images
imgs = {
    "zidane (2 frontal faces)": cv2.imread(os.path.join(HERE, "data", "zidane.jpg")),
    "bus (many people, small)": cv2.imread(os.path.join(HERE, "data", "bus.jpg")),
    "astronaut (1 frontal)": bgr(skd.astronaut()),
    "cat (animal face)": bgr(skd.chelsea()),
    "coffee (NO face)": bgr(skd.coffee()),
    "camera (man, side/back view)": bgr(skd.camera()),
}

print("=== Task 1: lab default  detectMultiScale(gray, minNeighbors=5) ===")
for name, im in imgs.items():
    f, ms = detect(im, save="".join(c if c.isalnum() else "_" for c in name.split("(")[0]) + "haar.jpg", minNeighbors=5)
    print(f"{name:32s} size={im.shape[1]}x{im.shape[0]}  faces={len(f)}  {ms:6.1f} ms  boxes={[tuple(map(int,b)) for b in f]}")

# ---------------------------------------------------------------- conditions
print("\n=== Task 2: varying conditions on the astronaut (true faces = 1) ===")
base = bgr(skd.astronaut())
h, w = base.shape[:2]
def gamma(im, g):
    return (255 * (im / 255.0) ** g).astype(np.uint8)
def rot(im, a):
    M = cv2.getRotationMatrix2D((im.shape[1] / 2, im.shape[0] / 2), a, 1.0)
    return cv2.warpAffine(im, M, (im.shape[1], im.shape[0]))
variants = {
    "original": base,
    "dark (gamma 3)": gamma(base, 3.0),
    "very dark (gamma 6)": gamma(base, 6.0),
    "over-bright (gamma .3)": gamma(base, 0.3),
    "blurred (k=25)": cv2.GaussianBlur(base, (25, 25), 0),
    "rotated 15 deg": rot(base, 15),
    "rotated 30 deg": rot(base, 30),
    "rotated 45 deg": rot(base, 45),
    "upside down": cv2.rotate(base, cv2.ROTATE_180),
    "mirror (flip)": cv2.flip(base, 1),
    "tiny (64px wide)": cv2.resize(base, (64, 64)),
    "face partly occluded": (lambda im: (cv2.rectangle(im.copy(), (150, 40), (260, 120), (0, 0, 0), -1)))(base),
}
for name, im in variants.items():
    f, ms = detect(im, save="astro_" + "".join(c if c.isalnum() else "_" for c in name) + ".jpg", minNeighbors=5)
    print(f"{name:24s} faces={len(f)}  {[tuple(map(int,b)) for b in f]}")

# ---------------------------------------------------------------- parameters
print("\n=== Task 3: parameter sweep (zidane = 2 true faces / coffee = 0 true faces / bus = small people) ===")
print("minNeighbors  scaleFactor | zidane  coffee  bus")
for sf in (1.05, 1.1, 1.3):
    for mn in (0, 1, 3, 5, 8):
        r = [len(detect(imgs[k], scaleFactor=sf, minNeighbors=mn)[0])
             for k in ("zidane (2 frontal faces)", "coffee (NO face)", "bus (many people, small)")]
        print(f"{mn:12d}  {sf:11.2f} | {r[0]:6d}  {r[1]:6d}  {r[2]:3d}")
f, ms = detect(imgs["zidane (2 frontal faces)"], scaleFactor=1.2, minNeighbors=5, minSize=(50, 50))
print("lab's commented variant (scaleFactor=1.2,minSize=50):", len(f), "faces")

# ---------------------------------------------------------------- quantitative eval
print("\n=== Task 4: quantitative test on LFW subset (24x24-ish cropped faces) + non-face images ===")
lfw = skd.lfw_subset()      # (200, 25, 25) float, 100 faces then 100 non-faces
labels = np.r_[np.ones(100), np.zeros(100)]
tp = fp = 0
for im, lab in zip(lfw, labels):
    big = cv2.resize((im * 255).astype(np.uint8), (100, 100), interpolation=cv2.INTER_CUBIC)
    big = cv2.copyMakeBorder(big, 30, 30, 30, 30, cv2.BORDER_REPLICATE)   # give a 'context' border
    n = len(face_cascade.detectMultiScale(big, scaleFactor=1.1, minNeighbors=5))
    if n > 0 and lab == 1: tp += 1
    if n > 0 and lab == 0: fp += 1
print(f"LFW subset: faces detected {tp}/100 (recall {tp}%)   non-face patches flagged {fp}/100 (false-positive rate {fp}%)")

# ---------------------------------------------------------------- inside the cascade
print("\n=== Task 5: what is inside haarcascade_frontalface_default.xml ===")
root = ET.parse(XML).getroot()
cas = root.find("cascade")
stages = cas.find("stages")
counts = [len(s.find("weakClassifiers")) for s in stages]
print("window size:", cas.find("width").text, "x", cas.find("height").text)
print("number of stages:", len(counts), " total weak classifiers (features):", sum(counts))
print("weak classifiers per stage:", counts)
print("first 3 stages:", counts[:3], "-> stage 1 needs only", counts[0], "feature tests; later stages are far bigger (last =", counts[-1], ")")
feats = root.find(".//features")
print("distinct Haar features (rectangles) stored:", len(feats))

# integral image demo (the trick that makes each feature O(1))
print("\n=== Integral image demo ===")
A = np.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16]], dtype=np.uint8)
ii = cv2.integral(A)       # (5x5) with zero first row/col
print("image:\n", A, "\nintegral image (padded):\n", ii)
# sum of region rows1-2, cols1-2 = [[6,7],[10,11]] = 34 via 4 lookups
r0, c0, r1, c1 = 1, 1, 3, 3
print("sum rows1-2/cols1-2 via 4 look-ups: ", ii[r1, c1] - ii[r0, c1] - ii[r1, c0] + ii[r0, c0], " (direct:", A[1:3, 1:3].sum(), ")")
# 2-rectangle Haar feature: left-half minus right-half on a 2x4 strip
left = ii[2, 2] - ii[0, 2] - ii[2, 0] + ii[0, 0]
right = ii[2, 4] - ii[0, 4] - ii[2, 2] + ii[0, 2]
print("2-rect Haar feature on rows0-1: left", left, "- right", right, "=", left - right)
