# Insight Hunters

Third-year Eduvos research project investigating TB classification, medical-domain transfer learning, Grad-CAM explainability and image-content validation using NIH ChestX-ray14, Shenzhen and Montgomery chest radiographs.

> **Research prototype only:** Not clinically validated; do not use for diagnosis or treatment decisions.

## Project Overview

This project investigates the application of deep learning techniques to chest X-ray image analysis. The project uses three chest X-ray datasets: the NIH Chest X-ray dataset, Shenzhen Hospital chest X-ray dataset and Montgomery County chest X-ray dataset.

The project is being developed as a third-year group project by the Insight Hunters team.

The project has completed dataset preparation, four-architecture CNN comparison, NIH-based ResNet50 pretraining, Grad-CAM explainability, MobileNetV2 image validation and controlled backend integration tests. Streamlit end-to-end integration and debugging remain ongoing.

---

## Datasets

Three chest X-ray datasets are used within the project.

### NIH Chest X-ray Dataset

- Total images: **112,120**
- Image format: PNG
- Image dimensions: 1024 × 1024
- Number of diagnostic categories: 15
- Used for dataset exploration **and 14-label medical-domain pretraining of ResNet50** before TB fine-tuning

### Shenzhen Hospital Dataset

- Total images: **662**
- TB Negative: **326**
- TB Positive: **336**
- Image format: PNG
- Image dimensions vary across the dataset

### Montgomery County Dataset

- Total images: **138**
- TB Negative: **80**
- TB Positive: **58**
- Image format: PNG
- Primary image dimensions:
  - 4020 × 4892
  - 4892 × 4020

### Combined Tuberculosis Dataset

The Shenzhen and Montgomery datasets form the binary tuberculosis classification dataset used in the current model-preparation pipeline.

| Class | Number of Images |
|---|---:|
| TB Negative | 406 |
| TB Positive | 394 |
| **Total** | **800** |

The original image datasets are **not stored in this GitHub repository** because of their size. They are maintained separately within the group's shared Google Drive project storage.

---

## Dataset Exploration and Quality Assessment

Dataset exploration was performed before model development to establish the structure, characteristics and quality of the available data.

The analysis included:

- Dataset size and structure
- Class distributions
- Image formats
- Image dimensions
- Image modes
- Demographic information where available
- Image integrity checking
- Exact duplicate detection
- Random visual inspection

A total of **112,920 images** across the NIH, Shenzhen and Montgomery datasets were included in the image-integrity assessment.

No unreadable or corrupted images were detected during the completed integrity checks.

Exact duplicate checking using MD5 hashing did not identify exact duplicate image files.

Random visual inspection was also performed using samples from each dataset.

---

## Dataset Splitting

The combined Shenzhen and Montgomery tuberculosis dataset contains **800 images**.

A reproducible stratified split was created using a **70% training / 15% validation / 15% testing** configuration.

| Split | TB Negative | TB Positive | Total |
|---|---:|---:|---:|
| Training | 284 | 276 | 560 |
| Validation | 61 | 59 | 120 |
| Testing | 61 | 59 | 120 |
| **Total** | **406** | **394** | **800** |

A random state of **42** was used to support reproducibility.

The generated split was verified for:

- Correct overall image count
- Correct class distributions
- Stratification
- Correct filename-based TB labels
- No image-level overlap between generated split assignments

The physical training, validation and testing directories were subsequently verified and contained all **800 expected images**.

No overlapping filenames were detected between the physical training, validation and testing sets.

---

## Data Preprocessing

A TensorFlow input pipeline was created to prepare the verified tuberculosis dataset for later deep learning models.

### Input Configuration

- Image dimensions: **224 × 224**
- Channels: **3 (RGB)**
- Batch size: **32**
- Classification type: **Binary**
- Class 0: **TB Negative**
- Class 1: **TB Positive**
- Random seed: **42**

The original image files are not permanently resized or modified.

Images are loaded and converted into consistent **224 × 224 × 3** tensors through the TensorFlow input pipeline.

TensorFlow successfully loaded:

- **560 training images**
- **120 validation images**
- **120 testing images**

The resulting datasets contain:

- 18 training batches
- 4 validation batches
- 4 testing batches

Training data is shuffled, while validation and testing data are kept unshuffled.

Dataset prefetching is enabled to improve input-pipeline efficiency.

The base datasets retain their loaded pixel-value representation. Model-specific preprocessing and normalization will be applied during model development.

Data augmentation will be applied to the **training data only** during the model-development phase. Validation and testing images will not be augmented.

---

## Repository Structure

The following structure reflects the notebooks uploaded during the October 2026 repository update. Notebook 10 has not been identified, so its number is intentionally unused. Verify filenames against the live repository if any uploads were renamed differently.

```text
Insight-Hunters/
├── notebooks/
│   ├── 01_Dataset_Structure_and_Statistics.ipynb
│   ├── 02_Data_Quality_and_Cleaning.ipynb
│   ├── 03_Dataset_Splitting.ipynb
│   ├── 04_Data_Preprocessing.ipynb
│   ├── 05_VGG16_Model_Development.ipynb
│   ├── 06_ResNet50_Model_Development.ipynb
│   ├── 07_DenseNet121_Model_Development.ipynb
│   ├── 08_EfficientNetB0_Model_Development.ipynb
│   ├── 09_Final_Model_Evaluation_and_Comparison.ipynb
│   ├── 11_NIH_Domain_Specific_Pretraining.ipynb
│   ├── 12_GradCAM_Explainability.ipynb
│   └── 13_Model_Integration_Testing.ipynb
├── .gitignore
├── README.md
└── requirements.txt
```

Large datasets, generated data splits, model checkpoints and exported models are intentionally excluded from the repository. The Streamlit application and model inference handover have been developed separately; their presence in this GitHub repository has not yet been verified.

---

## Notebooks

### 01 — Dataset Structure and Statistics

`01_Dataset_Structure_and_Statistics.ipynb`

Explores the structure and characteristics of the NIH, Shenzhen and Montgomery datasets.

The notebook includes:

- Dataset image counts
- Dataset classes
- Class distributions
- Image formats
- Image dimensions
- Image modes
- Demographic information
- Dataset summary statistics
- Data visualisations

---

### 02 — Data Quality and Cleaning

`02_Data_Quality_and_Cleaning.ipynb`

Assesses the quality and integrity of the three chest X-ray datasets.

The notebook includes:

- Image-integrity checking
- Corrupted/unreadable image detection
- Exact duplicate detection using MD5 hashing
- Random visual inspection
- Data-quality summary

The exact-duplicate analysis detects files with identical binary content and should not be interpreted as a near-duplicate or visual-similarity analysis.

---

### 03 — Dataset Splitting

`03_Dataset_Splitting.ipynb`

Creates and verifies the stratified training, validation and testing split for the combined Shenzhen and Montgomery tuberculosis dataset.

The notebook includes:

- Filename-based TB label verification
- Stratified 70/15/15 splitting
- Reproducible random state
- Class-distribution verification
- Split-overlap checking
- Split summary generation
- Complete split-manifest generation

---

### 04 — Data Preprocessing

`04_Data_Preprocessing.ipynb`

Integrates the prepared tuberculosis dataset into the machine-learning input pipeline.

The notebook includes:

- Project environment verification
- Physical dataset-split verification
- Class-folder verification
- Filename-label verification
- TensorFlow configuration
- TensorFlow dataset loading
- 224 × 224 × 3 image conversion
- Binary-label verification
- Batch verification
- Visual preprocessing sanity checking
- Dataset prefetching
- Dependency verification

This notebook completed the initial dataset exploration and preparation phase; subsequent model development and integration were documented separately.

---

### 05 — VGG16 Model Development

`05_VGG16_Model_Development.ipynb`

Documents the ImageNet-pretrained VGG16 baseline, frozen-backbone training, selective fine-tuning, checkpointing and validation analysis.

### 06 — ResNet50 Model Development

`06_ResNet50_Model_Development.ipynb`

Documents the original ImageNet-pretrained ResNet50 experiment, including its training and fine-tuning stages. This is distinct from the later NIH-pretrained ResNet50.

### 07 — DenseNet121 Model Development

`07_DenseNet121_Model_Development.ipynb`

Documents the standardized DenseNet121 transfer-learning experiment, training configuration, fine-tuning, checkpoints and validation results.

### 08 — EfficientNetB0 Model Development

`08_EfficientNetB0_Model_Development.ipynb`

Documents the standardized EfficientNetB0 transfer-learning experiment, including preprocessing, staged training, checkpoint selection and validation analysis.

### 09 — Final Model Evaluation and Comparison

`09_Final_Model_Evaluation_and_Comparison.ipynb`

Evaluates the four original CNN architectures on the held-out TB test set and records the selection of ImageNet-pretrained ResNet50 for further experimentation. This is a historical selection stage, not the final deployed-model decision.

### 11 — NIH Domain-Specific Pretraining

`11_NIH_Domain_Specific_Pretraining.ipynb`

Documents NIH patient-level splitting, Google Drive data-loading bottlenecks, local Colab image staging, TFRecord development, 14-label chest X-ray pretraining, NIH-to-TB ResNet50 transfer learning and evaluation. The NIH pretraining task is not TB classification.

### 12 — Grad-CAM Explainability

`12_GradCAM_Explainability.ipynb`

Documents Grad-CAM generation and visualization for the ResNet50 TB classifier, including heatmap overlays and exploratory comparisons. Heatmaps do not constitute clinical explanations.

### 13 — Model Integration Testing

`13_Model_Integration_Testing.ipynb`

Documents MobileNetV2 chest-image validation, robustness checks, integration with TB classification and Grad-CAM, controlled inference testing, and the backend handover. The notebook includes intermediate development iterations; the final validated entry point is `run_complete_inference()`.

---

## Shared Project Storage

Large project files are maintained through the group's shared Google Drive storage rather than GitHub.

The shared project structure includes (illustrative top-level layout):

```text
Insight_Hunters_Project/
│
├── 01_Datasets/
│   ├── Raw/
│   ├── Metadata/
│   └── Processed/
│
├── 02_Data_Splits/
│   ├── train/
│   ├── val/
│   └── test/
│
├── 03_Model_Checkpoints/
├── 04_Results/
├── 05_Exports/
├── 06_Notebooks/
├── 07_App_Integration/
├── 08_Image_Validation/
└── requirements.txt
```

This separation allows GitHub to manage source code and project documentation while Google Drive manages large datasets, generated files, model checkpoints and exports.

---

## Environment and Dependencies

The project's core Python dependencies are recorded in:

`requirements.txt`

The repository's current dependency list includes:

- TensorFlow
- NumPy
- Pandas
- Matplotlib
- Pillow
- Scikit-learn
- Streamlit
- OpenCV (`opencv-python-headless`)

To install the required dependencies in a compatible Python environment:

```bash
pip install -r requirements.txt
```

Google Colab is currently used as the primary notebook environment.

---

## Reproducibility

Several measures have been implemented to improve reproducibility throughout the project:

- Shared project directory structure
- Centralised notebooks
- Version-controlled source code
- `requirements.txt` dependency record
- Random seed of 42
- Stratified dataset splitting
- Split manifest
- Verified class mapping
- Physical split verification
- Explicit TensorFlow input configuration

---

## Data and Repository Policy

The following files are intentionally excluded from GitHub:

- Raw chest X-ray datasets
- Processed medical images
- Physical training, validation and testing image directories
- Model checkpoints
- Trained model files
- Final model exports
- Temporary notebook files

These exclusions help prevent large medical-image datasets and generated model files from being accidentally committed to the repository.

---

## Model Development and Evaluation

Four CNN architectures were evaluated on the held-out TB test set (120 images). These are the **initial ImageNet-based model comparison** results:

| Architecture | Accuracy | Sensitivity | Specificity | ROC-AUC |
|---|---:|---:|---:|---:|
| VGG16 | 83.33% | 71.19% | 95.08% | 92.41% |
| **ResNet50** | **87.50%** | 79.66% | **95.08%** | **93.64%** |
| DenseNet121 | 82.50% | 74.58% | 90.16% | 92.28% |
| EfficientNetB0 | 74.17% | 79.66% | 68.85% | 86.30% |

ResNet50 was selected for further experimentation. NIH domain-specific pretraining was then used to investigate whether additional chest X-ray experience improved TB classification.

### NIH Data Engineering and TFRecords

NIH ChestX-ray14 comprises 112,120 images and created substantial storage and data-loading challenges. The team developed a TFRecord pipeline and staged 86,524 training/validation images locally in Colab (approximately 32.45 GB; approximately 132.9 minutes to transfer in the recorded run). An initial Google Drive benchmark measured approximately **12.28 seconds/batch**; a subsequent local-storage benchmark measured approximately **0.37 seconds/batch** over ten batches. These different benchmarks do not isolate the effect of TFRecords alone.

NIH splits were **73,916 training**, **12,608 validation** and **25,596 reserved test** images. Recorded checks found no patient or image overlap across the NIH partitions. NIH pretraining predicted 14 thoracic disease labels; this was **not direct TB training**.

### Final NIH-Pretrained ResNet50

| Metric | ImageNet → TB | NIH → TB (selected) |
|---|---:|---:|
| Accuracy | 87.50% | **87.50%** |
| Sensitivity | 79.66% | **83.05%** |
| Specificity | **95.08%** | 91.80% |
| ROC-AUC | 93.64% | **93.89%** |

The NIH-pretrained model was selected for its higher sensitivity, despite lower specificity and unchanged accuracy. Its held-out confusion matrix was **TN=56, FP=5, FN=10, TP=49**. The experiments do not establish a statistically significant advantage for NIH pretraining.

### Grad-CAM Explainability

Grad-CAM heatmaps use the selected ResNet50's `conv5_block3_out` layer (7×7 spatial map), resized and overlaid on chest radiographs. These visualizations show influential image regions but do not verify pathology or clinical correctness.

### MobileNetV2 Image Validator

A separate model screens uploads before TB classification to reduce misleading predictions for non-chest images. It was trained using **560 chest X-rays and 630 non-chest photographs**, validated on **120 chest and 134 non-chest images**, and tested on **120 chest and 134 non-chest images**. The original held-out test produced **254/254 correct decisions**. Additional exploratory checks correctly handled 100 pet photos, 20 abdominal CT images, 21 hand X-rays, 30 NIH chest radiographs, and grayscale conversions of the same 100 pet photos. These limited samples do not guarantee generalization.

The validator outputs a non-chest score: **below 0.5 accepts**, while **0.5 or above rejects**. The score is not a clinically calibrated probability.

### Backend and Application Integration

The unified `tb_model_inference.py` module provides `run_complete_inference(tb_model, validator_model, image_input, threshold=0.5)` and returns `accepted`, `rejected` or `error`. Only accepted images receive TB classification and Grad-CAM through this validated entry point. Controlled Colab tests covered chest X-rays, hand radiographs, invalid bytes and Streamlit-style uploaded files. **Actual Streamlit integration is still being debugged:** an apparent chest X-ray rejection and a non-chest-image prediction screenshot require investigation. Do not describe the full application as validated.

## Project Status — October 2026

- [x] Dataset exploration, integrity checks, splitting and preprocessing
- [x] VGG16, ResNet50, DenseNet121 and EfficientNetB0 comparison
- [x] NIH storage benchmarking, local staging and TFRecord work
- [x] NIH-to-TB ResNet50 transfer learning and held-out evaluation
- [x] Grad-CAM implementation
- [x] MobileNetV2 image validator and exploratory robustness testing
- [x] Unified backend and controlled integration testing
- [ ] Complete Streamlit integration and resolve upload/validation issues
- [ ] Final end-to-end application testing
- [ ] Final demonstration and oral presentation (20 November 2026)

## Limitations

The TB dataset contains only 800 images and the held-out test contains only 120. The selected model missed 10 TB-positive cases in testing. Differences among data sources, limited validator image diversity, and lack of external clinical validation constrain generalizability. Grad-CAM is not a medical explanation. The prototype must not be used for clinical diagnosis.

---

## Team

**Insight Hunters**

Third-year group deep learning project.

### Dataset Exploration and Preparation Responsibilities

- **Dataset Structure and Statistics** — dataset exploration, descriptive statistics and visualisations
- **Data Quality and Cleaning** — integrity checking, duplicate detection and visual quality inspection
- **Dataset Splitting** — stratified training, validation and testing split
- **Project Setup and Integration** — shared project structure, notebooks, dependencies, dataset access, GitHub and preparation-pipeline integration

---

## Current Phase

Dataset preparation, model comparison, NIH pretraining, image validation, Grad-CAM and backend testing have been completed. Streamlit integration and final end-to-end verification are ongoing.
