# Assignment 1 — Multimodal Image Classification (Traditional Features)


## Overview


This assignment implements a multimodal image classification pipeline using traditional features: Canny edge features for images and Word2Vec embeddings for captions. The chosen dataset is MS COCO 2017 (validation set). The goal is to compare unimodal (image-only and text-only) and multimodal (fused) classification performance across the 80 COCO categories.


## Dataset and Preprocessing


- **Dataset:** COCO 2017 validation images and captions. For reproducibility, images are read from `val2017/` and annotations from `annotations/captions_val2017.json` and `annotations/instances_val2017.json`.


- **Label assignment:** Each image is assigned a single class by selecting the most frequent object category appearing in that image's instance annotations. This produces an image-level label in the 80 COCO label space.


- **Image preprocessing & features:** Images are converted to grayscale, resized to 224×224, smoothed with a Gaussian filter, and Canny edge detection is applied. The resulting edge map is downsampled to 28×28 and flattened to produce a 784-dimensional feature vector. This provides a compact, traditional edge-based descriptor.


- **Text preprocessing & features:** Captions are lowercased, stripped of punctuation, and tokenized. A Word2Vec model (either pretrained or trained on the COCO captions) produces 300-dimensional word vectors. For each caption, token vectors are averaged to form a caption embedding; the final image-level text feature is the average of all captions for the image (typically 5 captions). This yields a 300-dim representation per image.


## Fusion & Model


- **Fusion strategy:** Feature concatenation (image 784-d + text 300-d → 1084-d fused vector). This simple early fusion preserves both modalities.


- **Classifier:** Multinomial Logistic Regression (solver: saga) trained on fused features. For unimodal comparisons, the same classifier is trained on image-only (784-d) and text-only (300-d) features.


- **Splits:** 70% training, 10% validation, 20% testing (applied to the full set as specified). For quick experiments, the notebook allows limiting to a subset of images (e.g., 2000 images).


## Experiments & Evaluation


- **Metrics:** Classification accuracy and confusion matrix are reported for each model (image-only, text-only, fused).


- **Expected outcomes:** Multimodal models should outperform unimodal baselines when captions provide complementary information about the object class. However, limitations exist due to noisy captions, class imbalance, and coarse label assignment.


## Strengths, Limitations & Future Work


**Strengths:**
- Simple, interpretable features (Canny edges, Word2Vec) demonstrate classical multimodal fusion ideas.
- Concise pipeline easy to reproduce and extend.

**Limitations:**
- Label assignment heuristic (most frequent object) may be noisy when multiple objects exist.
- Canny edges discard color and texture information useful for some classes.
- Training Word2Vec from scratch on a small dataset produces lower-quality embeddings compared to large pretrained models.

**Future improvements:**
- Use pretrained image features (ResNet) combined with caption embeddings.
- Try more advanced fusion (learned gating, attention-based fusion).
- Handle multi-label classification instead of forcing a single label per image.


## Reproducibility & Deliverables

Files generated:
- `Assignment1.ipynb` — Jupyter notebook implementing the full pipeline (saved in the submission zip).
- This report (2–3 pages) — include with code in the final zip `MML_assignment1_UEN_Name.zip`.


---

*Prepared automatically by ChatGPT — edit details (paths, experimental numbers) after running the notebook locally.*
