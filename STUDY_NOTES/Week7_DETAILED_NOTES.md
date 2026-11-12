# CS5707 — WEEK 7: IMAGE PROCESSING / COMPUTER VISION (Complete Detailed Notes)

> Self-contained. If you read only this file, you have Week 7.
> Lecturer: Yongmin Li. Reading: Russell & Norvig §27 (Computer Vision), Viola & Jones 2001,
> Long 2015 (FCN), Krizhevsky 2012 (AlexNet), Redmon 2016 (YOLO).
> Labs use OpenCV (Haar cascade), YOLOv3 / YOLOv3-tiny (via cv2.dnn, what `cvlib` wraps) and
> Ultralytics YOLOv8m-seg.
>
> **Honest gaps**
> - The lecture PDF (51 slides) is mostly pictures; I read every slide's text but not the image
>   content. Where a slide is only a figure I say so.
> - The lab data (`brunel.jpg`, `objects.jpg`, `segment.jpg`, Drive folder) is NOT in the repo.
>   I ran the labs on other images (zidane, bus, scikit-image samples). Numbers below are
>   for those images, not Brunel's. The **lessons** transfer; the exact counts won't.
> - The **Week 7 quiz has not been pasted yet** -> no `Week7_QUIZ_with_answers.md` (see end).
> - Exam value: this is a "describe / compare / explain why" week, not a calculation week.
>   The only hand-calculable bits are: image storage size, integral image + Haar feature,
>   cascade arithmetic, IoU/NMS, conv output size. They are in Section 9 — learn them.

---

## 0. The big idea

A computer sees an image as a grid of numbers. **Computer vision = turning that grid into
decisions**: *is there a face? where? what objects? which pixels belong to which object?*
The week walks one story in four steps, each harder than the last:

| Task | Question answered | Output | Lab |
|---|---|---|---|
| Image fundamentals | How is an image stored? | pixel array | — |
| **Face detection** | Where are the faces? | boxes | 7.1 Haar cascade |
| **Object detection / recognition** | What objects, where? | boxes + class + confidence | 7.2 YOLOv3 |
| **Segmentation** | Which pixels belong to what? | per-pixel mask | 7.3 YOLOv8-seg |

The second story is **traditional (hand-designed features + a classifier)** versus
**modern (one end-to-end neural net learns the features)**. This is the contrast the exam
loves (Seminar Q5). The turning point is **AlexNet, ILSVRC 2012**.

Important framing (HANDOVER): you already did ML and DL. The lecturer says so too — the labs
use "built" models and skip training. So focus on **what each method does, why it works, and
its trade-offs**, not on how to train.

---

## 1. Image fundamentals (Seminar Q1)

### 1.1 Image formation (slide 5; figure only)
Light -> lens -> sensor (camera) just like the eye's lens -> retina. The image is a 2-D
**projection** of a 3-D world, so depth is lost (R&N §27.2: pinhole camera, lens,
perspective projection, scaled orthographic projection). One line is enough for the exam.

### 1.2 Colour depth — THE table to memorise

| Type | Bits / pixel | Values per pixel | Number of colours/levels | Also called |
|---|---|---|---|---|
| **1-bit** | 1 bit | 0 or 1 | 2 | binary image, 1-bit monochrome (no colour) |
| **8-bit grey-level** | 8 bits = 1 byte | 0 – 255 | 256 grey levels (dark ≈ 10, bright ≈ 230) | greyscale |
| **24-bit colour** | 24 bits = 3 bytes | three values 0–255 (R,G,B) | 256³ = **16,777,216** | true colour / RGB |

- 24-bit = **three 8-bit channels** (Red, Green, Blue) stacked; each channel is itself a grey-level
  image (slide 10: forestfire.bmp and its R, G, B channels).
- Pixel (colour) = a vector `(R,G,B)`; (255,0,0) red, (0,0,0) black, (255,255,255) white.
- **Storage** for a W×H image = W·H·bits/8 bytes (uncompressed). Example 1024×768 = 786,432 px:

```
 1-bit   : 786,432 / 8      =    98,304 bytes  =  96 KB
 8-bit   : 786,432 x 1      =   786,432 bytes  = 768 KB
 24-bit  : 786,432 x 3      = 2,359,296 bytes  = 2.25 MB
```
- Where each is used: binary = thresholded masks, scanned text, segmentation output; grey =
  medical imaging, and what the Haar detector actually uses; colour = photos.
- **Gotcha in the lab:** OpenCV loads colour images as **BGR, not RGB**. That's why the lab does
  `cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)` before face detection (the Haar cascade works on grey).

---

## 2. Face recognition and face detection (lecture slides 11–18, Seminar Q2)

### 2.1 The family of face problems (slide 13)
| Problem | Question |
|---|---|
| **Face detection** | Is there a face, and where? (first stage of everything else) |
| **Verification** | Is this the SAME person? **1 vs 1** (phone unlock) |
| **Recognition / identification** | Who is this? **1 vs many** |
| Expression recognition | Smiling / angry? |
| Face editing | Modify the face |

Applications (slide 12): biometrics / identity checks, border control, video surveillance, Apple Face ID,
Facebook **DeepFace — 97.35% accuracy, human-level**.

Generic pipeline (slide 14): **find faces -> locate landmarks -> encode the face -> train/apply model.**

### 2.2 Haar cascade detector (Viola & Jones, 2001) — HIGH EXAM VALUE
First real-time face detector; "still one of the fastest". Three ideas, and *all three* are needed
to answer "why is it effective **and** computationally efficient":

| # | Idea | What it gives | Why it helps |
|---|---|---|---|
| 1 | **Haar features** + **integral image** | Cheap, meaningful features | Any rectangle sum = 4 look-ups, so a feature costs a constant, regardless of size |
| 2 | **AdaBoost** feature selection | A small strong classifier from a huge pool | Picks the few useful features out of ~180,000 per 24×24 window |
| 3 | **Attentional cascade** | Early rejection of non-faces | Most windows are not faces and are discarded after a handful of tests |

#### (1) Haar features (slide 16)
Differences of sums of pixel values in adjacent rectangles:
```
 A: 2-rectangle (left/right)     B: 2-rectangle (top/bottom)
 C: 3-rectangle (centre vs sides) D: 4-rectangle (diagonal)

 feature value = (sum of pixels in WHITE rectangles) - (sum of pixels in BLACK rectangles)
```
**Why they detect faces (slide 17):** faces share a brightness structure that survives
identity, skin tone and expression:
- **A: eye region is darker than the cheeks** (2-rectangle, horizontal band).
- **B: the two eyes are darker than the nose bridge** (3-rectangle).
So a handful of such contrasts is a good face signature. Also fast, and insensitive to overall
brightness because they are *differences* (Viola–Jones also variance-normalises each window).

The base detector window is **24×24 pixels**; with all positions/scales/types, one window
has **over 180,000 possible rectangle features** (the paper's figure; far more than the 576
pixels) — hence we need AdaBoost to pick.

#### Integral image (the efficiency trick)
`ii(x,y)` = sum of all pixels above and to the left of (x,y), inclusive. Computed in ONE pass.
Any rectangle sum = **4 array references**:
```
 sum(rect) = ii(D) - ii(B) - ii(C) + ii(A)      (A,B,C,D = the four corners)
```
A 2-rectangle feature uses 6 look-ups, a 3-rectangle 8, a 4-rectangle 9 (adjacent
rectangles share corners). Worked in Section 9.1.

#### (2) AdaBoost
Each **weak classifier** = one Haar feature + threshold (+ polarity); it only needs to be a bit better
than chance (error < 0.5). AdaBoost repeatedly: re-weights the training examples so mistakes count
more, picks the best single feature, and finally combines them as a weighted vote = a **strong classifier**.
Paper: a 200-feature classifier gives **95% detection at 1 false positive in 14,084**, but is
too slow on its own -> hence the cascade. Selected features are interpretable (eye band, nose bridge).

#### (3) The cascade (slide 18)
"A series of detectors." Every sub-window of the scan goes through stage 1, 2, 3 ...:
```
 sub-window -> [Stage 1: few features] --pass--> [Stage 2] --pass--> ... --pass--> [Stage N] -> FACE
                     | reject                        | reject                          
                     v                               v                              
                 NOT a face (stop, no more work)   NOT a face
```
- Early stages: tiny classifiers tuned for **~100% detection, accepting many false positives**
  (paper: a 2-feature first stage detects 100% of faces with a 40% false-positive rate).
- Later stages: bigger and stricter; only the "most likely candidates" reach them.
- A window is a face only if it passes **all** stages.
- Because >99% of windows in a photo are NOT faces, almost all are rejected within the first
  one or two stages. **Paper: average of ~10 feature evaluations per window (out of 6,061 total).**
- **Speed (paper):** 384×288 image in ~0.067 s on a 700 MHz Pentium III (~15 fps), ~15× faster
  than Rowley–Baluja–Kanade, ~600× faster than Schneiderman–Kanade.
- Paper's final cascade: **38 stages, 6,061 features**; the first five layers have 1, 10, 25, 25, 50 features.
  (The OpenCV model file used in the lab is a different, later-trained cascade: see Section 7.)

#### Limitations (critical-evaluation marks)
- Mostly **frontal, upright** faces (one cascade per pose); no rotation invariance.
- Sensitive to occlusion, strong shadows, and small faces (24×24 minimum window).
- Higher false-positive rate than modern CNN detectors; but far cheaper -> good for embedded/real-time.
- Needs a labelled training set of thousands of faces/non-faces (training is slow, running is fast).

### 2.3 Scanning detail: how `detectMultiScale` uses the 24×24 window
The detector slides the 24×24 window over the image, then **repeats on scaled versions** (image
pyramid / `scaleFactor`). Windows that overlap on the same face give several hits; `minNeighbors`
keeps only those confirmed by that many neighbouring hits (merging duplicates and suppressing
isolated false alarms). Section 7.1 shows the numbers.

---

## 3. Image segmentation (lecture slides 19–30, Seminar Q3 & Q4)

**Definition:** partition an image into **meaningful regions**, based on measurements taken from the
image: grey level, colour, texture, depth or motion. (R&N: group pixels with similar brightness/colour/texture;
note that *edges alone are not enough* — a tiger in grass produces edges on every stripe and blade.)

**Applications (slides 21–22):** automated driving (road, vehicles, pedestrians, buildings, trees, lights,
signs); medical imaging (fundus image: blood vessels, optic disc; OCT: retinal layers).

### 3.1 Low-level methods (slides 23–26) — "pixel statistics only"
| Method | How it works | Good for | Fails when |
|---|---|---|---|
| **Grey-level / colour thresholding** | Histogram of grey levels is bimodal (dark object, bright background); choose threshold **T** in the valley; pixel < T = object, else background. Otsu picks T automatically by maximising between-class variance | High-contrast, even lighting | Uneven lighting, overlapping histograms, textured objects |
| **Texture** | Describe each local neighbourhood by texture statistics (filter responses); group pixels with similar texture | Regions defined by pattern (grass, fabric) not colour | Fine-scale boundaries are blurry |
| **Region growing** | Start from a **seed** pixel; add neighbours whose value is similar (within tolerance); stop at the boundary | Connected homogeneous regions | Sensitive to seed and tolerance; leaks through weak edges |
| Clustering (k-means on colour) | Treat each pixel as a point in colour space; cluster; label = cluster | Quick colour regions | No spatial reasoning; k must be chosen |

### 3.2 High-level methods (slide 27)
Clustering, compression, **Markov Random Fields (MRFs)**, **graph cut** (nodes = pixels, edges weighted by similarity;
minimise cut cost — R&N cites Shi & Malik's *normalised cut*), **level sets**, and **neural networks**:
R-CNN -> Fast R-CNN -> Faster R-CNN -> YOLO (slide 27 lists them under segmentation / detection).
R&N: purely low-level segmentation cannot give the true object boundaries; you need
**high-level knowledge of objects**. Over-segmentation into **superpixels** (hundreds instead of millions of
pixels) is a common preprocessing step.

### 3.3 Neural-network segmentation: Fully Convolutional Networks (Long et al. 2015, slide 28)
- Idea: take a classification CNN (AlexNet / VGG / GoogLeNet) and make it **fully convolutional**
  so it takes an image of **any size** and outputs a **per-pixel class map** — "end-to-end,
  pixels-to-pixels".
- Step 1 **Convert fully-connected layers into convolutions** (the 1×1 / spatial conv) so the
  net outputs a *coarse spatial heat-map* instead of a single label.
- Step 2 **In-network upsampling** = *backwards strided convolution* ("deconvolution", learnable) to bring the
  coarse map back to full resolution.
- Step 3 **Skip architecture:** fuse the deep, coarse, *semantic* layer with shallower, fine, *appearance*
  layers -> sharper boundaries. Variants **FCN-32s, FCN-16s, FCN-8s** (fewer skips = coarser; 8s is best).
- Result: PASCAL VOC 2012 **62.2% mean IU** (20% relative improvement), inference **< 1/5 second**.
- Lab note: the Lab PDF's requirement asks to "describe how you would segment using a ResNet model"
  — the same recipe applies to any backbone (ResNet features -> upsample -> per-pixel softmax).

### 3.4 Semantic vs instance segmentation (slide 29, Seminar Q4) — certain exam question
| | Semantic segmentation | Instance segmentation |
|---|---|---|
| Label per pixel | **class** only | class **and which object instance** |
| Three people | ONE "person" region | THREE separate person masks |
| Can you count objects? | No | **Yes** |
| Typical models | FCN | Mask R-CNN, YOLOv8-seg (Lab 7.3) |
| Rule of thumb | "stuff" (sky, road, grass) | "things" (people, cars) |

*Lab proof (Section 7.3):* on the bus image YOLOv8-seg returned 4 separate person masks which
overlap each other by 316 pixels — a semantic output would have merged them into one region.
A third type, **panoptic**, combines both (not in the lecture; mention only as extra).

---

## 4. Object recognition & detection (slides 31–50, Seminar Q5)

### 4.1 Terminology (get this precise)
| Task | Output |
|---|---|
| **Classification** | One label for the whole image |
| **Detection** | Many objects: **class + bounding box (+ confidence)** |
| **Segmentation** | Pixel mask per object / per class |

R&N: a detector scans windows, classifies each, keeps high scorers, resolves overlaps.
There are O(n⁴) possible windows in an n×n image, so you can't test them all -> need proposals
or a grid.

### 4.2 Evolution of recognition (slide 32)
Early: **edges, boundaries, shape, sketches** -> **2½-D**: depth and discontinuities, only the projection into
the image plane -> **full 3-D models**. (Marr-style; one sentence is enough.)

### 4.3 TRADITIONAL, feature-based recognition (slides 33–40)
Two **separate** stages, each hand-designed:

**Stage 1 — Feature extraction.** Good features are: *local, representative, robust, invariant*.
| Feature | What it captures | Slide |
|---|---|---|
| **Edges** (Sobel/Canny: image gradient + smoothing, R&N: convolve with Gaussian derivative) | Intensity discontinuities | 34 |
| **Corners** (**Harris** detector) | Points whose neighbourhood changes in all directions: stable, repeatable | 35 |
| **Haar features** | Rectangle contrast (Viola–Jones) | 36 |
| **Shapes** (**Hough transform**) | Votes in parameter space for lines/circles | 37 |
| **HOG** (Histogram of Oriented Gradients, Dalal & Triggs 2005) | Local gradient direction histograms; strong for pedestrians | 38 |
| **SIFT** (Lowe 2004) | Scale-invariant keypoint descriptors; match across scale/rotation/viewpoint | 39 |

**Stage 2 — Classification** (slide 40): Bayesian methods, genetic algorithms, decision trees/forests,
nearest neighbours, **SVM**, regression, neural networks. (Notice: earlier weeks' topics reappear
here — Bayes (W4), GAs (W3).)

*Weakness:* features are designed by **humans** for each problem; two-stage pipeline is not
jointly optimised; weak to clutter, lighting and viewpoint; performance plateaued before 2012.

### 4.4 ILSVRC and AlexNet (slides 41–43) — the turning point
- **ILSVRC** = ImageNet Large Scale Visual Recognition Challenge: ~1.2 M training images,
  **1000 categories** (R&N: 1.4 M images in lab text), top-5 error as the metric.
  R&N: in 2010 the best systems could not beat ~70% top-5 accuracy.
- **AlexNet (Krizhevsky, Sutskever, Hinton), ILSVRC 2012 winner**:
  - **Top-5 error 15.3%** vs **26.2%** for second place (lecture: ">10.8% lower than the runner-up").
  - Architecture: **5 convolutional + 3 fully-connected layers**, final 1000-way **softmax**;
    **60 million parameters, 650,000 neurons**.
  - Key ingredients (paper): **ReLU** activations (train several times faster than tanh), training
    on **two GPUs**, **dropout** in the first two FC layers (fights overfitting), **data augmentation**,
    local response normalisation, **overlapping pooling**.
  - Why it mattered: it showed that with big data (ImageNet) + GPUs + deep CNNs, **learned features beat
    hand-designed ones** -> "new era of AI" (lecture slide 42).
- **LeCun 1998 (LeNet) vs AlexNet 2012** (slide 43): same idea (conv -> pool -> FC); AlexNet is much
  deeper/larger, uses ReLU + GPUs + ImageNet-scale data. (The figure slide has no text; I'd say: "same
  recipe, scaled up thanks to data and compute".)

### 4.5 MODERN deep-learning recognition (slides 44–48)
- **One single end-to-end neural network.**
- **No separate feature extraction:** the first few layers *are* the feature extractor.
- **No separate classifier:** the last few layers can approximate any classification function.
- Both parts are **learned jointly from data** by back-propagation.

**Why CNNs work (R&N §27.4.2):** a convolution layer measures how much each local window looks like a
learned kernel; many kernels find many patterns; stacking layers composes patterns into bigger
patterns; weights are **shared across positions** (translation equivariance) -> far fewer
parameters than fully-connected nets.

**What is inside the net (slides 45–48, figures):**
- First layer conv masks look like **edge / colour-blob detectors** (Gabor-like) — what traditional
  vision designed by hand, now learned.
- Going up: edges -> textures/corners -> parts -> whole objects (Lee 2012).
- "Highly activating patches": pick a layer and channel, run many images, show patches with the highest
  activation -> each channel is a detector for a concept.
- Final layer: nearest neighbours in the feature space return semantically similar images.

### 4.6 Object detection with deep nets: R-CNN family vs YOLO (slides 27, 49–50; R&N §27.5)

**Two-stage (R-CNN family)**
| Model | Idea |
|---|---|
| **R-CNN** | ~2,000 region proposals -> run a CNN on each separately -> classify |
| **Fast R-CNN** | Run the CNN ONCE on the whole image, then ROI-pool features per proposal |
| **Faster R-CNN** | A **Region Proposal Network** (objectness scores for **9 anchor boxes** — 3 scales × 3 aspect ratios — at grid points), **ROI pooling**, classifier, **bounding-box regression**, **NMS** |

Accurate but "scanning goes several rounds" (lab text).

**YOLO — You Only Look Once (Redmon 2016)** — treat detection as a single **regression** problem:
- Divide the image into an **S×S grid** (S = 7). Each cell predicts **B = 2 boxes** (x, y, w, h,
  confidence) and **C = 20 class probabilities** (PASCAL VOC).
- Output tensor **S × S × (B·5 + C) = 7 × 7 × 30** (= 1,470 numbers).
- Confidence = `Pr(Object) × IOU(truth, pred)`; class score = `Pr(Class|Object) × Pr(Object) × IOU`.
- Network: **24 conv layers + 2 FC** (inspired by GoogLeNet; 1×1 reduction layers).
- Loss = sum-squared error with `λcoord = 5` (up-weight box coordinates) and `λnoobj = 0.5`
  (down-weight empty cells, which otherwise dominate); sqrt of w,h so small boxes matter.
- **Speed:** 45 fps (Fast YOLO **155 fps**); mAP 63.4% on VOC 2007 (real-time) — **more than twice** the mAP
  of earlier real-time detectors (52.7% for Fast YOLO).
- **Strengths:** very fast; sees the **whole image** (global context) so makes fewer *background* errors
  than Fast R-CNN. **Weaknesses:** more **localisation errors**; struggles with **small objects
  in groups** (each cell predicts only 2 boxes, one class); poor on unusual aspect ratios.
- **Post-processing: confidence threshold, then non-maximum suppression (NMS)** (adds 2–3% mAP).
- The lab uses **YOLOv3** (multi-scale) and **YOLOv3-tiny**; Lab 7.3 uses **YOLOv8** — same
  single-pass idea. Do not quote the 7×7×30 tensor as being YOLOv3's; it is the 2016 original.

### 4.7 Key detection concepts (R&N §27.5)
| Term | Meaning |
|---|---|
| Bounding box | Axis-aligned rectangle (x, y, w, h) |
| **Confidence** | How sure the model is that box contains class c |
| **IoU** (intersection over union) | `area(overlap) / area(union)` of two boxes; 1 = identical, 0 = disjoint |
| **NMS** | Greedy: sort by score, keep best, **delete boxes overlapping it more than a threshold**, repeat |
| Anchor boxes | Preset shapes the net adjusts |
| **Recall / Precision** | found-all-objects vs no-false-alarms; detector evaluation must balance both (mAP) |
| Bounding-box regression | Refine rough window to tight box |

### 4.8 Seminar Q5 in one table — traditional vs deep learning
| | Traditional feature-based | Deep learning (CNN/YOLO) |
|---|---|---|
| Features | **Hand-designed** (Haar, HOG, SIFT, edges, corners) | **Learned** from data (first layers) |
| Classifier | Separate (SVM, boosting, trees, NN...) | Learned jointly (last layers) |
| Pipeline | Two stages, optimised separately | **End-to-end**, one network |
| Data needed | Little to moderate | **Huge labelled datasets** (ImageNet) |
| Compute | Cheap at run-time (Haar: CPU, 15 fps in 2001) | Needs GPU for training and, for large models, inference |
| Accuracy | Good in constrained settings; plateaued | State of the art (AlexNet 15.3% vs 26.2%) |
| Robustness | Brittle (pose, lighting, clutter) | Much more robust |
| Interpretability | Features understandable | Black-box (but can visualise filters/activations) |
| Examples | Viola–Jones, HOG+SVM, SIFT | AlexNet, R-CNN family, YOLO |

**"Critically evaluate" marks:** deep nets win on accuracy but cost data, compute and explainability;
Haar still sensible on a low-power device; both can be biased by training data.

---

## 5. Where the lecture and the papers differ (so you don't get tripped)
- Lecture says "Face recognition (Facebook DeepFace 97.35%)": verification-type benchmark, not detection.
- Lecture slide 27 lists R-CNN/YOLO under "high-level segmentation methods"; strictly these are **detection**
  networks (box output). YOLOv8-**seg** (Lab 7.3) and Mask R-CNN add a mask head.
- Lab text for 7.3 mentions "PixelLib" and "ResNet"; the actual notebook uses Ultralytics YOLOv8.
- Lab text says ILSVRC has "1.4 million images"; the AlexNet paper says 1.2 million training images.

---

## 6. WORKED EXAMPLES (by hand — most valuable for the written exam)

### 6.1 Integral image and a Haar feature
Image (4×4):
```
  1  2  3  4
  5  6  7  8
  9 10 11 12
 13 14 15 16
```
Integral image `ii` (cumulative sum over rows and columns; shown with a zero border):
```
  0   0   0   0   0
  0   1   3   6  10
  0   6  14  24  36
  0  15  33  54  78
  0  28  60  96 136
```
(e.g. ii at 2nd row/2nd col = 1+2+5+6 = 14 ✔.) *Verified by `cv2.integral`, lab script.*

**Rectangle sum, rows 1–2, cols 1–2 (values 6,7,10,11) = 34:**
`ii(3,3) − ii(1,3) − ii(3,1) + ii(1,1) = 54 − 6 − 15 + 1 = 34` ✔ (direct: 6+7+10+11 = 34). 4 look-ups.

**2-rectangle Haar feature on the top two rows** (left half columns 0–1 minus right half columns 2–3):
left = 1+2+5+6 = 14, right = 3+4+7+8 = 22, feature = 14 − 22 = **−8**.
A large |value| means a strong vertical edge/contrast there.

### 6.2 Why a cascade is fast (and the arithmetic of stage rates)
If every stage has detection rate d and false-positive rate f, **an N-stage cascade has overall rates dᴺ and fᴺ**.
Example (my numbers, in the spirit of the paper): 10 stages, each d = 0.99 and f = 0.30:
```
 overall detection rate  = 0.99^10 = 0.904   (lose 10% of faces)
 overall false-positive  = 0.30^10 = 5.9 x 10^-6   (about 1 in 170,000 windows)
```
So each stage can be **weak on false positives** (30% pass!) but must be **very good on detection**
(99%): errors multiply to nearly zero while detection only decays slowly. Cost: if a non-face passes stage k
with probability 0.3, the expected number of stages evaluated is ≈ 1 + 0.3 + 0.09 + ... ≈ 1.43, so
the expensive late stages almost never run. (Paper measured **~10 features per window** vs 6,061 total.)

### 6.3 IoU and NMS by hand
Boxes as (x1,y1,x2,y2):  A=(10,10,50,50) score 0.9;  B=(12,12,52,52) 0.8;  C=(100,100,140,140) 0.7;
D=(14,8,54,48) 0.6. NMS threshold = 0.5.

```
 IoU(A,B): overlap = (50-12) x (50-12) = 38 x 38 = 1444
           union   = 1600 + 1600 - 1444 = 1756  -> IoU = 0.822
 IoU(A,D): overlap = (50-14) x (48-10) = 36 x 38 = 1368
           union   = 3200 - 1368 = 1832         -> IoU = 0.747
 IoU(A,C): no overlap -> 0
```
NMS trace: sort by score: A, B, C, D.
1. Keep **A** (highest). Remove boxes with IoU(A, ·) > 0.5: **B** (0.822) and **D** (0.747) go. C stays (0).
2. Keep **C**. Nothing left.
**Result: A and C** (two objects, even though four boxes fired).
Lab proof: bus.jpg, YOLOv3, confidence 0.25: with `nms_thresh = 0.3` -> **6 boxes**; with `nms_thresh = 0.9`
(almost no suppression) -> **28 boxes** (20 'person' and 7 'bus' for one bus and four people!).

### 6.4 YOLO output size
S = 7, B = 2, C = 20 -> `7 × 7 × (2·5 + 20) = 7 × 7 × 30 = 1470` numbers. If you change to C = 80 classes
(COCO) with S=7, B=2: `7×7×(10+80) = 4410`.

### 6.5 CNN arithmetic (cheap marks if asked)
Output size `= (W − F + 2P)/S + 1`.
- AlexNet conv1: W = 227, F = 11, S = 4, P = 0 -> (227−11)/4 + 1 = **55**.
- 3×3 conv, padding 1, stride 1 on 224 -> (224−3+2)/1 + 1 = **224** (size preserved).
- Parameters of one conv layer = (F·F·C_in + 1) · C_out; 3×3 on RGB -> 64 filters = (27+1)·64 = **1,792**.
- Weight sharing is why convolution needs few parameters compared with fully-connected layers.

---

## 7. THE LABS (all run & verified)

> Environment: OpenCV **4.10.0.84** (the lab pins this version; OpenCV 5 removed `CascadeClassifier`),
> `ultralytics 8.4`, CPU only. Scripts: `week7/lab7_1_face_haar.py`, `lab7_2_object_yolo.py`,
> `lab7_3_segmentation.py`; raw outputs in `week7/lab7_*_output.txt`; annotated images in `week7/out/`.
> Model files are big and are NOT in the repo (download links in the scripts' docstrings / below).
> Colab-only lines (`google.colab.patches.cv2_imshow`, Drive mount) are replaced with `cv2.imwrite`.

### 7.1 Face detection (Haar cascade)
Method (lab code): `gray = cv2.cvtColor(img, COLOR_BGR2GRAY)`, then
`faces = face_cascade.detectMultiScale(gray, minNeighbors=5)`; draw rectangles.

**Results with the lab's default (`minNeighbors=5`):**
| Image | True faces | Detected | Verdict |
|---|---|---|---|
| zidane.jpg (1280×720) | 2 | 2 | ✔ 78 ms |
| bus.jpg (people, small, in street) | ~2 visible faces (both with sunglasses) | 2 | ✔ I checked the picture: both boxes are on faces |
| astronaut (frontal portrait) | 1 | 1 | ✔ |
| cat (animal face) | 0 | 0 | ✔ correct rejection |
| coffee cup (no face) | 0 | 0 | ✔ |
| camera man (side **profile**) | 1 (not frontal) | 0 | ✘ miss: **non-frontal** (checked by eye) |

**Conditions on one portrait (true faces = 1):**
| Condition | Detected | Verdict |
|---|---|---|
| original | 1 | ✔ |
| darker (gamma 3) | 1 | ✔ |
| **very dark (gamma 6)** | 2 | face ✔ + **1 false positive** (box on the suit) |
| over-bright (gamma 0.3) | 1 | ✔ |
| blurred (25×25 Gaussian) | 1 | ✔ robust to blur |
| rotated 15° | 1 | ✔ small tilt tolerated |
| **rotated 30°** | 0 | ✘ miss (face clearly visible, just tilted) |
| **rotated 45°** | 1 | ✘ **wrong**: the box sits on the mouth/chin, not the face |
| upside down | 1 | ✘ the face was **missed**; the single box is a **false positive on the shuttle** (checked by eye) |
| mirror image | 2 | face ✔ + **tiny false positive** on the helmet |
| shrunk to 64×64 | 0 | ✘ too small for the 24×24 window after scale steps |
| eyes/forehead covered by a black box | 0 | ✘ occlusion of the eye region (the key Haar cue) kills detection |

**Parameter sweep** (counts on zidane (true 2) / coffee (true 0)):
| minNeighbors | scaleFactor 1.05 | 1.10 | 1.30 |
|---|---|---|---|
| 0 | 188 / 9 | 89 / 2 | 28 / 1 |
| 1 | 9 / 2 | 4 / 0 | 2 / 0 |
| 3 | 3 / 0 | 2 / 0 | 2 / 0 |
| 5 | 3 / 0 | **2 / 0** | 1 / 0 |
| 8 | 2 / 0 | 2 / 0 | 1 / 0 |
- With `minNeighbors=0` the same two faces become **89–188 overlapping boxes** and a no-face image gets false alarms.
- Larger `scaleFactor` (1.3) is faster but **misses** a face (1 instead of 2): coarse scale steps skip the right window size.
- The lab's commented-out variant (`scaleFactor=1.2, minSize=(50,50)`) found only **1** of the 2 zidane faces.

**Quantitative test (scikit-image LFW subset, 100 face crops + 100 non-face crops, resized to 100×100 + border):**
recall **97/100**, false positives **0/100**.
(Cautious interpretation: tightly cropped, centred, frontal faces = the best case for this detector.)

**Inside the OpenCV model file:** window **24×24**, **25 stages**, **2,913** weak classifiers;
features per stage: `9, 16, 27, 32, 52, 53, 62, 72, 83, 91, 99, 115, 127, 135, 136, 137, 159, 155, 169, 196, 197, 181, 199, 211, 200`.
Stage 1 uses **9** features and stage 25 uses **200**: cheap early gate, expensive late check — exactly the cascade
idea. (It's a different, later cascade from the paper's 38-stage/6,061-feature one.)

**What the lab report wants (model answers):**
1. *Process + why Haar works + how the cascade saves computation* -> Section 2.2 (eye band darker than cheeks; nose bridge lighter than eyes;
   integral image 4 look-ups; ~10 features per window on average; reject early).
2. *Try your own images* -> table above: the detector is good on frontal/upright/well-lit faces and fails on rotation >~30°,
   profile views, occlusion, tiny faces; animal faces and objects mostly rejected, but dark/odd textures can give false positives.
3. *Evaluate* -> speed is excellent (20–170 ms per image on CPU, even 1280×720), accuracy conditional on pose; tune
   `minNeighbors` (precision) and `scaleFactor` (speed vs recall).

### 7.2 Object detection (YOLOv3 vs YOLOv3-tiny)
Method (as `cvlib` does): `blobFromImage(img, 1/255, (416,416), swapRB=True)` -> one forward pass -> keep boxes whose class score >
confidence (default **0.5**) -> **NMS** (default **0.3**) -> draw labels. Labels file = 80 COCO classes.

| Image | YOLOv3 (full) | time | YOLOv3-tiny | time |
|---|---|---|---|---|
| bus.jpg | 4 persons + 1 bus (best conf 1.00) | 922 ms (first call) | 3 persons | 141 ms |
| zidane.jpg | 2 persons | 185 ms | **nothing** | 23 ms |
| cat | cat 0.99 | 180 ms | **nothing** | 23 ms |
| coffee | cup + spoon | 180 ms | **nothing** | 28 ms |
| horse silhouette | horse 0.98 | 192 ms | **nothing** | 24 ms |
| rocket (no COCO class) | nothing | 182 ms | nothing | 28 ms |
Total: full 1840 ms vs tiny 267 ms -> **tiny ≈ 6.9× faster** (timing includes the slow first call for the full model, so the real gap per image
is ~8×: ~185 ms vs ~25 ms).

**Can I just lower the threshold for tiny?** (confidence 0.5 / 0.3 / 0.15):
zidane -> none / person 0.33 / two persons (0.33, 0.25); cat -> none at all 3 settings; coffee -> none /
**'toilet' 0.42** / 'toilet' 0.42 (wrong); horse -> none. So tiny's problem is **low recall and wrong labels**, not just a threshold.

**Threshold experiment on bus.jpg (full model):**
| confidence | NMS | raw boxes | kept | classes |
|---|---|---|---|---|
| 0.9 | 0.3 | 15 | 4 | 3 person + bus |
| 0.5 | 0.3 | 25 | 5 | 4 person + bus |
| 0.25 | 0.3 | 31 | 6 | 4 person + bus + handbag |
| 0.25 | 0.2 | 31 | 6 | same |
| 0.25 | **0.9** | 31 | **28** | 20 person + 7 bus + handbag (duplicates!) |
| 0.05 | 0.3 | 34 | 6 | same as 0.25 |
Lesson: confidence trades **recall vs precision** (0.9 drops the 0.83-confidence person at the edge; 0.25 adds a handbag);
NMS removes **duplicate** boxes (31 raw -> 6). In the full-model run 25 raw boxes -> 5 final.

**Report answers:**
1. *How to perform recognition with a pre-built model:* load config + weights + labels -> pre-process (resize 416×416, scale 1/255, BGR->RGB)
   -> single forward pass (grid predictions of boxes, objectness, class scores) -> threshold on confidence -> NMS -> draw boxes + labels.
2. *Full vs tiny:* full = more accurate, slower (250 MB); tiny = 35 MB, ~7–8× faster, **misses most objects at normal thresholds**.
   Choose tiny for embedded/real-time, full when accuracy matters.
3. *Evaluate:* only classes among COCO's 80 are found (rocket -> nothing); small/overlapping objects are the weak spot;
   no ground truth here, so these are qualitative; a proper evaluation uses mAP/precision/recall against labelled boxes.

### 7.3 Instance segmentation (YOLOv8m-seg) + classical segmentation
Method (lab code): `model = YOLO('yolov8m-seg.pt'); results = model(img); annotated = results[0].plot()`.
The net outputs, for every detected object: class, confidence, box **and a mask**.

| Image | Instances found | Notes |
|---|---|---|
| bus | 4 person (0.93, 0.91, 0.89, 0.58) + bus 0.92 + tie 0.29 | masks of different persons overlap by 316 px; bus mask = 30% of the frame; inference ~230 ms |
| zidane | person 0.92, person 0.92, tie 0.85, tie 0.31, **person 0.26** | the 0.26 person and the second tie look like low-confidence extras (I did not check these two by eye) |
| cat | cat 0.94 | mask covers ~73% of the frame |
| coffee | cup 0.97, spoon 0.84, **dining table 0.75** | table mask ≈ 66%: instance models also give "stuff" masks for COCO classes |
(Mask areas are measured on the model's internal letterboxed mask, so percentages are approximate.)
Speed: ~170–230 ms/image CPU inference.

**Classical (low-level) segmentation for contrast (scikit-image "coins": 24 coins):**
- **Otsu global threshold:** T = **107**, foreground 39%, but **96 connected components** (true coins: 24). The coins have
  textured faces and the lighting is uneven, so the foreground fragments.
- **Adaptive (local) threshold:** **755** components — worse (it responds to texture).
- **k-means colour clustering on the coffee image:** k=2 -> 52.8% / 47.2%; k=3 -> 54.9 / 22.6 / 22.4; k=5 -> 28.3 / 26.7 / 20.0 / 17.1 / 7.9 %.
  It groups by colour only; no concept of "cup".
- **Region growing** from a seed of grey 105 on blurred coins: tolerance 5 -> 3.8% of the image; 15 -> 11.7%; 30 -> 24.8%; **60 -> 67.0%**
  (leaks into the background). Small tolerance = under-segment; large = **leaks across weak boundaries**.
=> The lesson of R&N: low-level methods split/merge by pixel statistics; **high-level knowledge of objects** (the CNN) gives
meaningful regions (cup, person) rather than "regions of similar grey".

**Report answers:**
1. *Segment with a ResNet:* use ResNet (or any backbone) as encoder (conv layers extract features at decreasing resolution), replace the
   final FC classifier by a 1×1-conv classifier (as FCN), upsample (transpose conv / skip connections from earlier layers) to input size,
   per-pixel softmax; train with pixel-wise cross-entropy. For **instances**, add a box/mask head (Mask R-CNN) or use YOLOv8-seg's prototype masks.
2. *Evaluate:* metrics are **IoU / mean IoU** for masks, mAP for instances; qualitatively check mask edges on touching objects (bus people overlap),
   low-confidence false instances (zidane), speed.

---

## 8. Exam-style checklist
- [ ] Define 1-bit, 8-bit grey, 24-bit colour images; give levels/colours and bytes per pixel; compute file size.
- [ ] List the 5 face problems; distinguish **verification (1:1)** vs **recognition (1:N)**.
- [ ] Explain **Haar features** (white − black rectangles), what A and B capture on a face, the **integral image** (4 look-ups).
- [ ] Explain **AdaBoost** (weak classifier = one feature; weighted vote) and the **attentional cascade** (early reject, ~10 of 6,061 features).
- [ ] Say why Haar cascade is **fast** and list its **limitations** (frontal, rotation, occlusion, lighting).
- [ ] Define segmentation; name three methods with how they work (threshold, region growing, clustering/graph cut/FCN).
- [ ] **Semantic vs instance** segmentation; give the "3 people" example.
- [ ] Explain FCN: fully-convolutional + upsampling + skip connections.
- [ ] Traditional pipeline: feature extraction (edges, Harris, Haar, Hough, HOG, SIFT) + classifier (SVM, trees, NN...).
- [ ] Modern pipeline: single end-to-end network, features learned in early layers; AlexNet 2012 (15.3% vs 26.2%, ReLU, dropout, GPUs).
- [ ] R-CNN -> Fast -> Faster R-CNN (proposals, ROI pooling, anchors) vs **YOLO** (grid, one pass, 7×7×30, 45 fps).
- [ ] Do **IoU** and **NMS** by hand; explain confidence threshold vs NMS threshold.
- [ ] Compare traditional vs deep learning (table in 4.8) and **critically evaluate**.

---

## 9. Seminar questions (practise) — answer outlines

**Q1. Explain different image types by colour depth (1-bit, 8-bit grey, 24-bit colour).**
1-bit: one bit per pixel, 0/1, two levels, binary/monochrome image. 8-bit grey: one byte per pixel, 0–255, 256 grey levels (10 dark,
230 bright). 24-bit: three bytes per pixel, usually R,G,B each 0–255, 16.7 M colours, three channels each a grey image. Give size formula
W×H×bits/8 and the 1024×768 numbers (96 KB / 768 KB / 2.25 MB).

**Q2. Describe face detection; why Haar cascade is effective and efficient.**
Pipeline: grey-scale image -> slide 24×24 window at many positions/scales -> for each window evaluate the cascade -> windows surviving all stages = faces
-> merge overlapping hits (`minNeighbors`) -> draw boxes. Effective: Haar features capture face-specific brightness structure (eyes darker than cheeks,
eyes darker than nose bridge); AdaBoost selects the most discriminative few from a huge set. Efficient: integral image (any rectangle in 4 look-ups),
tiny early stages reject most non-face windows (~10 features per window), only promising ones get the full test. Mention limits (frontal only etc.).

**Q3. Name three segmentation methods and how they work.**
(a) **Thresholding** on grey-level/colour histogram: pick T in the valley between modes; Otsu automatic. (b) **Region growing**: seed + add similar neighbours until a boundary.
(c) **Clustering / k-means** on colour (or texture) features; or **graph cut** (min cut on pixel-similarity graph); or **FCN** (CNN with per-pixel output). Add one limitation each.

**Q4. Semantic vs instance segmentation.**
Semantic: class label per pixel, multiple objects of the same class form one region (no counting). Instance: separates individual objects of the same class (own mask each).
Example: crowd -> one "person" region vs N person masks. Instance = detection + segmentation (Mask R-CNN, YOLOv8-seg).

**Q5. Traditional feature-based vs deep-learning object detection.**
Use the Section 4.8 table: hand-crafted features (edges, Harris corners, Haar, Hough, HOG, SIFT) + separate classifier (SVM, trees, Bayes...) vs end-to-end CNN (AlexNet -> R-CNN family -> YOLO)
learning features and classifier jointly. Mention data/compute/accuracy/robustness/interpretability trade-off, AlexNet's 2012 jump, and that YOLO reframes detection as one regression pass (speed).

---

## 10. How to reproduce
```
python3 -m venv venv && . venv/bin/activate
pip install numpy matplotlib scikit-image "opencv-python-headless==4.10.0.84" ultralytics
export MODELS=/path/to/models        # folder with the files listed below
python week7/lab7_1_face_haar.py
python week7/lab7_2_object_yolo.py
python week7/lab7_3_segmentation.py
```
Model files expected in `$MODELS` (not stored in git): `haarcascade_frontalface_default.xml`, `yolov3.cfg`, `yolov3.weights`,
`yolov3-tiny.cfg`, `yolov3-tiny.weights`, `yolov3.txt` (80 labels), `yolov8m-seg.pt`.
Sources: https://github.com/yongminli/data (xml, tiny cfg/weights, labels), https://github.com/pjreddie/darknet/blob/master/cfg/yolov3.cfg,
https://pjreddie.com/media/files/yolov3.weights, https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8m-seg.pt.
Test images used: `week7/data/bus.jpg`, `zidane.jpg` (Ultralytics samples) and scikit-image sample images.
