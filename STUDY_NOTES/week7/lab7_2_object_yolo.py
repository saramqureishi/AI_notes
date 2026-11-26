"""Lab 7.2 - Object detection with YOLOv3 (full) vs YOLOv3-tiny - verified run.

cvlib.object_detection.YOLO is a thin wrapper around cv2.dnn; this script does the same
steps explicitly so you can see them (blob -> forward -> filter by confidence -> NMS):
    bbox, label, conf = yolo.detect_objects(img, confidence=0.5, nms_thresh=0.3)
"""
import os, time
import cv2, numpy as np
from skimage import data as skd

HERE = os.path.dirname(os.path.abspath(__file__))
MODELS = os.environ.get("MODELS", os.path.join(HERE, "..", "..", "..", "models"))
OUT = os.path.join(HERE, "out"); os.makedirs(OUT, exist_ok=True)
LABELS = open(os.path.join(MODELS, "yolov3.txt")).read().strip().split("\n")

class YOLO:
    def __init__(self, weights, cfg):
        self.net = cv2.dnn.readNetFromDarknet(cfg, weights)
        self.out = self.net.getUnconnectedOutLayersNames()
    def detect_objects(self, img, confidence=0.5, nms_thresh=0.3, size=416):
        H, W = img.shape[:2]
        blob = cv2.dnn.blobFromImage(img, 1 / 255.0, (size, size), swapRB=True, crop=False)
        self.net.setInput(blob)
        outs = self.net.forward(self.out)               # ONE forward pass = "You Only Look Once"
        boxes, confs, ids = [], [], []
        for o in outs:                                   # each row: cx,cy,w,h,objectness,class scores...
            for d in o:
                s = d[5:]; c = int(np.argmax(s)); p = float(s[c])
                if p > confidence:
                    cx, cy, w, h = d[0] * W, d[1] * H, d[2] * W, d[3] * H
                    boxes.append([int(cx - w / 2), int(cy - h / 2), int(w), int(h)])
                    confs.append(p); ids.append(c)
        raw = len(boxes)
        keep = cv2.dnn.NMSBoxes(boxes, confs, confidence, nms_thresh)
        keep = np.array(keep).flatten() if len(keep) else []
        return ([boxes[i] for i in keep], [LABELS[ids[i]] for i in keep], [confs[i] for i in keep], raw)

def draw(img, bbox, label, conf):
    v = img.copy()
    for (x, y, w, h), l, c in zip(bbox, label, conf):
        cv2.rectangle(v, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(v, f"{l} {c:.2f}", (x, max(y - 5, 12)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    return v

def bgr(a): return cv2.cvtColor(a, cv2.COLOR_RGB2BGR)

full = YOLO(os.path.join(MODELS, "yolov3.weights"), os.path.join(MODELS, "yolov3.cfg"))
tiny = YOLO(os.path.join(MODELS, "yolov3-tiny.weights"), os.path.join(MODELS, "yolov3-tiny.cfg"))

imgs = {
    "bus": cv2.imread(os.path.join(HERE, "data", "bus.jpg")),
    "zidane": cv2.imread(os.path.join(HERE, "data", "zidane.jpg")),
    "cat": bgr(skd.chelsea()),
    "coffee": bgr(skd.coffee()),
    "rocket": bgr(skd.rocket()),
    "horse(bw silhouette)": cv2.cvtColor((skd.horse() * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR),
}

def summary(lbl, conf):
    from collections import Counter
    return dict(Counter(lbl))

print("=== Task: default settings (confidence 0.5, NMS 0.3) full vs tiny ===")
tot = {"full": 0.0, "tiny": 0.0}
for name, im in imgs.items():
    for tag, m in (("full", full), ("tiny", tiny)):
        t = time.perf_counter(); b, l, c, raw = m.detect_objects(im); ms = (time.perf_counter() - t) * 1000
        tot[tag] += ms
        cv2.imwrite(os.path.join(OUT, f"yolo_{tag}_{name.split('(')[0]}.jpg"), draw(im, b, l, c))
        print(f"{name:22s} {tag:4s} {ms:7.0f} ms  raw boxes={raw:3d} after NMS={len(b):2d}  {summary(l, c)}  top conf={max(c) if c else 0:.2f}")
print(f"total time: full {tot['full']:.0f} ms  vs tiny {tot['tiny']:.0f} ms  (tiny ~{tot['full']/tot['tiny']:.1f}x faster)")

print("\n=== Task: effect of confidence / NMS thresholds on bus.jpg (full model) ===")
for conf, nms in ((0.9, 0.3), (0.5, 0.3), (0.25, 0.3), (0.25, 0.2), (0.25, 0.9), (0.05, 0.3)):
    b, l, c, raw = full.detect_objects(imgs["bus"], confidence=conf, nms_thresh=nms)
    print(f"confidence={conf:<4} nms={nms:<4} raw={raw:3d} kept={len(b):3d}  {summary(l, c)}")

print("\n=== Task: per-box detail, bus.jpg, full vs tiny ===")
for tag, m in (("full", full), ("tiny", tiny)):
    b, l, c, _ = m.detect_objects(imgs["bus"])
    for bb, ll, cc in sorted(zip(b, l, c), key=lambda t: -t[2]):
        print(f"  {tag:4s} {ll:12s} {cc:.2f} box(x,y,w,h)={bb}")

print("\n=== Task: can lowering the threshold rescue YOLOv3-tiny? ===")
for n in ("zidane", "cat", "coffee", "horse(bw silhouette)"):
    for c in (0.5, 0.3, 0.15):
        b, l, cf, raw = tiny.detect_objects(imgs[n], confidence=c)
        print(f"{n:22s} tiny confidence={c:<4} -> {list(zip(l, [round(x, 2) for x in cf]))}")
