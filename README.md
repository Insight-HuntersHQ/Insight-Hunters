# Insight Hunters

Deep learning project for chest X-ray analysis using the NIH Chest X-ray, Shenzhen and Montgomery datasets.

## Project Overview

This project investigates the application of deep learning techniques to chest X-ray image analysis. The project uses three chest X-ray datasets: the NIH Chest X-ray dataset, Shenzhen Hospital chest X-ray dataset and Montgomery County chest X-ray dataset.

The project is being developed as a third-year group project by the Insight Hunters team.

The dataset exploration and preparation phase includes dataset structure and statistical analysis, data quality assessment, dataset splitting, preprocessing and the development of a reproducible input pipeline for later model training and evaluation.

---

## Datasets

Three chest X-ray datasets are used within the project.

### NIH Chest X-ray Dataset

- Total images: **112,120**
- Image format: PNG
- Image dimensions: 1024 × 1024
- Number of diagnostic categories: 15
- Used for broader chest X-ray analysis and dataset exploration

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

The GitHub repository contains the code and reproducibility material required for the project.

```text
Insight-Hunters/
│
├── notebooks/
│   ├── 01_Dataset_Structure_and_Statistics.ipynb
│   ├── 02_Data_Quality_and_Cleaning.ipynb
│   ├── 03_Dataset_Splitting.ipynb
│   └── 04_Data_Preprocessing.ipynb
│
├── .gitignore
├── README.md
└── requirements.txt
```

Large datasets, generated data splits, model checkpoints and exported models are intentionally excluded from the repository.

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

This notebook represents the final stage of the current dataset exploration and preparation pipeline.

---

## Shared Project Storage

Large project files are maintained through the group's shared Google Drive storage rather than GitHub.

The shared project structure is organised as:

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
└── requirements.txt
```

This separation allows GitHub to manage source code and project documentation while Google Drive manages large datasets, generated files, model checkpoints and exports.

---

## Environment and Dependencies

The project's core Python dependencies are recorded in:

`requirements.txt`

Core libraries currently include:

- TensorFlow
- NumPy
- Pandas
- Matplotlib
- Pillow
- Scikit-learn

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

## Project Status

### Dataset Exploration and Preparation

- [x] Dataset structure and statistical analysis
- [x] Class-distribution analysis
- [x] Image-property analysis
- [x] Data quality and integrity checking
- [x] Exact duplicate checking
- [x] Visual data inspection
- [x] Stratified dataset splitting
- [x] Split verification
- [x] Physical split verification
- [x] TensorFlow dataset loading
- [x] Input-pipeline verification
- [x] Shared project storage structure
- [x] Notebook organisation
- [x] Dependency configuration
- [x] GitHub repository setup

### Model Development and Evaluation

- [ ] Data augmentation
- [ ] Baseline model development
- [ ] Transfer-learning model comparison
- [ ] VGG16 development
- [ ] Model training
- [ ] Fine-tuning
- [ ] Performance evaluation
- [ ] Confusion-matrix analysis
- [ ] Model comparison
- [ ] Final model selection
- [ ] Final model export

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

The **Dataset Exploration and Preparation** phase has been completed.

The project is progressing into **Model Development and Evaluation**, where multiple deep learning architectures will be developed and compared before final model selection.
