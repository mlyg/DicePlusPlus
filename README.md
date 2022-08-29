# Calibrating the Dice loss for biomedical image segmentation

Link to the arXiv preprint version of the paper: https://arxiv.org/abs/2111.00528

## Reproducing results from the paper
To reproduce the results from the paper:
1. Download Datasets
2. Clone the MIScnn repository (or use the Google Colab scripts provided in this repository)
3. Set up file directories
4. Run Google Colab scripts (or modify to run locally)

## Datasets
1. **Digital Retinal Images for Vessel Extraction (DRIVE)** \
**Description:** 40 coloured fundus photographs obtained from diabetic retinopathy screening in the Netherlands \
**Link:** https://computervisiononline.com/dataset/1105138662

2. **Breast UltraSound 2017 (BUS2017)** \
**Description:** Dataset B consists of 163 ultrasound images and associated ground truth segmentations collected from the UDIAT Diagnostic Centre of the Parc Tauli Corporation, Sabadell, Spain \
**Link:** http://www2.docm.mmu.ac.uk/STAFF/m.yap/dataset.php

3. **2018 Data Science Bowl (2018DSB)** \
**Description:** 670 light microscopy images for nuclei segmentation** \
**Link:** https://www.kaggle.com/c/data-science-bowl-2018

4. **ISIC2018: Skin Lesion Analysis Towards Melanoma Detection grand challenge** \
**Description:** 2,594 images of skin lesions \
**Link:** https://challenge.isic-archive.com/data/#2018

5. **CVC-ClinicDB** \
**Description:**  612 frames containing polyps generated from 23 video sequences from 13 different patients using standard colonoscopy \
**Link:** https://polyp.grand-challenge.org/CVCClinicDB/

6. **Kidney Tumour Segmentation 2019 (KiTS19)** \
**Description:** 300 arterial phase abdominal CT scans. These are from patients who underwent partial removal of the tumour and surrounding kidney or complete removal of the kidney including the tumour at the University of Minnesota Medical Center, USA. \
**Link:** https://github.com/neheller/kits19

## Cloning the MIScnn repository
The MIScnn pipeline is used in these experiments. The repository can be found at: https://github.com/frankkramer-lab/MIScnn. 

In our experiments, we make small but significant modifications to the MIScnn code. Specifically we:
1. replace Batch Normalisation with Instance Normalisation
2. replace Adam with SGD with momentum
Both modifications are found in the src directory under 'modified_files'.

## Setting up file directories
Both the data and dataset split information require a specific setup to be compatible with the MIScnn pipeline. 

### Data
The data must be structured similar to the KiTS19 dataset:
```
data
├── case_00000
|   ├── imaging.nii.gz
|   └── segmentation.nii.gz
├── case_00001
|   ├── imaging.nii.gz
|   └── segmentation.nii.gz
...
├── case_00209
|   ├── imaging.nii.gz
|   └── segmentation.nii.gz
```

### Dataset split
The training and testing dataset splits are found as json files in this repository. These must be contained in a folder called 'fold_0' for the Google Colab scripts to work. More information can be found in the MIScnn github repository.
