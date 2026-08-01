# DSR-Gen
Registration‑Free Contrast-Enhancement Generation Framework Based on Deformation Mask and Self‑Supervised Depth Prior
Official PyTorch implementation of DSR-Gen: A registration‑free framework for synthesizing contrast‑enhanced CT (CECT) from non‑contrast CT (NCCT) without requiring paired or registered training data.

# Overview
DSR-Gen addresses the fundamental challenge of anatomical misalignment between NCCT and CECT acquisitions—caused by respiratory motion, cardiac pulsation, and intestinal peristalsis—which renders perfectly paired training data unattainable in clinical practice.
Our framework introduces three key innovations:
1.Deformation Mask: A threshold‑based mask (100–400 HU) that selectively supervises only anatomically consistent regions, effectively circumventing the misalignment problem without requiring explicit registration.
2.Explicit Enhancement Component Learning: An Enhancement Component Learning and Embedding Module (ECLEM) that decouples contrast pattern modeling from structural preservation by explicitly learning the enhancement component。
3.Self‑Supervised Depth Prior: A Self‑Reconstruction Module (SRM) that shares network weights with the Enhancement Generation Module (EGM), anchoring the synthesis process to the input anatomy and preserving structural fidelityeven in misaligned regions.
<img width="9179" height="4218" alt="figure1" src="https://github.com/user-attachments/assets/a6e68008-6f3b-4ab1-96a8-3fd9f9852f93" />

<img width="3134" height="975" alt="figure2" src="https://github.com/user-attachments/assets/9c60f7f7-705a-4ef7-a5d0-b891240e562e" />

<img width="4821" height="4124" alt="figure3" src="https://github.com/user-attachments/assets/225826be-2321-4d5e-bcc4-983ad99db046" />

# Prerequisites
Python 3.8+
PyTorch 1.12+
CUDA 11.3+

# Setup
# Clone the repository
git clone https://github.com/zkp-521/DSR-Gen-main.git
cd DSR-Gen
# Create virtual environment
conda create -n dsrgen python=3.9
conda activate dsrgen
# Install dependencies
pip install -r requirements.txt

# Requirements
dynamic-network-architectures==0.2
imageio==2.37.0
matplotlib==3.10.1
ml-collections==1.1.0
networkx==3.3
nibabel==5.3.2
numpy==2.1.2
opencv-python==4.11.0.86
pandas==2.2.3
pillow==11.0.0
pip==25.0
pydicom==3.0.1
scikit-image==0.25.2
scikit-learn==1.6.1
scipy==1.15.2
simpleitk==2.4.1
six==1.17.0
surface-distance==0.1
torch==2.7.0+cu118
torchaudio==2.7.0+cu118
torchvision==0.22.0+cu118
tqdm==4.67.1

# Data Preparation
# CT Datasets
Center 1 (In‑house): Multi‑phase CT (non‑contrast, arterial, venous, delayed) from 516 patients, acquired at 120 kV, 181 mA, with slice thickness 1.5 mm and matrix size 512×512.
Center 2 (MCT‑LTDiag): Publicly available multi‑phase CT dataset [1] from 520 patients, with slice thickness 5 mm, 120 kVp, pixel spacing 0.8125 mm, matrix size 512×512.
# MRI Dataset
MU‑Glioma Post: Public dataset [2] comprising 594 MRI sequences from 203 patients with paired routine T1‑weighted and T1‑weighted post‑contrast images.

# Training
# Train DSR‑Gen on CT Dataset
python train_ourmodel.py

# Train DSR‑Gen on MRI Dataset
python train_ourmodel_mrhead.py

# Testing & Evaluation
# test CT
python test_ct.py
# test MRI
python test_MRI.py
# Evaluation
python metric.py

# Metrics computed
Edge‑SSIM: Structural similarity of edge maps
MAE‑deformation: Mean absolute error in high‑difference regions (>400 HU)
Mask‑MAE: Masked MAE excluding misaligned regions
LPIPS(VGG): Learned Perceptual Image Patch Similarity

# Acknowledgments
MCT‑LTDiag Dataset [1]
MU‑Glioma Post Dataset [2]

# References
[1] Wu, X. et al. "A Multi-phase CT Dataset for Automated Differential Diagnosis of Liver Tumors." Scientific Data, 2025.
[2] Mahmoud, E. et al. "MU-Glioma Post: A comprehensive dataset of automated MR multi-sequence segmentation and clinical features." Scientific Data, 12, 1847 (2025).
