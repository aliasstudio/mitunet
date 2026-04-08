# **MitUNet: Enhancing Floor Plan Recognition**

**Official implementation of the paper:**
> [**Enhancing Floor Plan Recognition: A Hybrid Mix-Transformer and U-Net Approach for Precise Wall Segmentation**](https://doi.org/10.1007/s00138-026-01815-y)

[![Paper](https://img.shields.io/badge/Springer-Article-blue)](https://doi.org/10.1007/s00138-026-01815-y)
[![arXiv](https://img.shields.io/badge/arXiv-2512.02413-b31b1b.svg)](https://arxiv.org/abs/2512.02413)

## **Overview**

**MitUNet** is a hybrid deep learning architecture designed for
high-precision semantic segmentation of walls in
2D floor plans. It addresses the challenge of vectorizing thin, complex
structures by combining the global context awareness of Transformers
with the local spatial precision of CNNs.

### **Key Features**

-   **Hybrid Encoder--Decoder:** Combines a **Mix-Transformer (MiT-b4)**
    encoder from SegFormer with a **U-Net** decoder for fine-grained
    restoration.
-   **Attention Mechanism:** Uses **scSE** (Spatial & Channel
    Squeeze--Excitation) modules to suppress background noise.
-   **Optimized Loss:** Trained with asymmetric **Tversky Loss** (α =
    0.6, β = 0.4) for balanced precision and recall, reducing staircase
    artifacts.

------------------------------------------------------------------------

## **Project Structure**

    .
    ├── datasets/               # Local dataset copies
    ├── experiments/            # Checkpoints and logs
    │   ├── models/             # Trained models
    │   └── experiments.xlsx    # Quantitative results
    ├── images/                 # Figures and visualizations
    ├── notebooks/              # Jupyter notebooks
    │   └── MitUNet.ipynb       # Main pipeline
    ├── requirements.txt        # Dependencies
    └── README.md               # Documentation

------------------------------------------------------------------------

## **Installation & Requirements**

This project uses **Python 3.10+** and **PyTorch**.

Install dependencies:

``` bash
pip install -r requirements.txt
```

**Key Dependencies:**

-   torch, torchvision
-   segmentation-models-pytorch==0.5.0
-   albumentations
-   roboflow

------------------------------------------------------------------------

## **Datasets**

Two datasets are used in this work:

1.  **CubiCasa5k (Pre-training)**
    Source: https://github.com/CubiCasa/CubiCasa5k
2.  **Floor Plan CIS (Fine-tuning)**
    Download: https://doi.org/10.5281/zenodo.17871079

> **Note:** `notebooks/MitUNet.ipynb` can download preprocessed versions
> of these datasets via the Roboflow API (API key required).

------------------------------------------------------------------------

## **Usage**

### **Option 1: Local Runtime with Docker (Recommended)**

1.  Install Docker and NVIDIA Container Toolkit.
2.  Launch your Jupyter environment with this repo mounted.
3.  Run `notebooks/MitUNet.ipynb`.

### **Option 2: Google Colab**

1.  Open `notebooks/MitUNet.ipynb` in Colab.
2.  Set **Runtime → GPU**.
3.  Run the notebook.

> Batch size 4 with 512×512 resolution requires sufficient VRAM.

------------------------------------------------------------------------

## **Inference Example**

1.  Download the best model:
    https://github.com/aliasstudio/mitunet/blob/master/experiments/models/mitunet_finetune_a6_mit_b4_tversky_8864_28E.pth
2.  Place it into `experiments/models/` (or adjust the path).
3.  Run:

``` python
import torch
import cv2
import numpy as np
import segmentation_models_pytorch as smp
import albumentations as A
from albumentations.pytorch import ToTensorV2

# 1. Setup device and paths
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
WEIGHTS_PATH = "experiments/models/mitunet_finetune_a6_mit_b4_tversky_8864_28E.pth"
IMAGE_PATH = "path/to/your/image.jpg"

# 2. Define Model Architecture (preserving original training logic)
# We instantiate a Segformer to extract its encoder, ensuring strict compatibility
aux_segformer = smp.Segformer(encoder_name="mit_b4", encoder_weights=None)
model = smp.Unet(
    encoder_name="mit_b4",
    encoder_weights=None,
    in_channels=3,
    classes=1,
    decoder_attention_type="scse"
)
# Transplant the encoder
model.encoder = aux_segformer.encoder

# 3. Load trained weights
state_dict = torch.load(WEIGHTS_PATH, map_location=DEVICE)
model.load_state_dict(state_dict)
model.to(DEVICE)
model.eval()

# 4. Define Preprocessing (exact inference transforms)
transform = A.Compose([
    A.Resize(512, 512),
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ToTensorV2(),
])

# 5. Load and Preprocess Image
image = cv2.imread(IMAGE_PATH)
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

augmented = transform(image=image_rgb)
input_tensor = augmented['image'].unsqueeze(0).to(DEVICE)

# 6. Predict
with torch.no_grad():
    logits = model(input_tensor)
    probs = torch.sigmoid(logits)
    mask = (probs > 0.5).float()

# 7. Convert result back to numpy (H, W)
result_mask = mask.squeeze().cpu().numpy()

# Optional: Save output
cv2.imwrite("output_mask.png", result_mask * 255)
```

------------------------------------------------------------------------

## **Results**

Comparison of the best-performing configuration for each architecture on the Regional Dataset.
The Boundary IoU (B-IoU) metric is included to demonstrate the accuracy of structural edge preservation. Additionally, MitUNet requires less VRAM compared to heavier CNN baselines like UNet++.

| Model | Encoder | Loss | Recall (%) | Precision (%) | Accuracy (%) | mIoU (%) | B-IoU (%) | VRAM (MiB) |
| :---- | :---- | :---- | :---- | :---- | :---- | :---- | :---- | :---- |
| **MitUNet (Ours)** | **mit_b4** | **Tversky** | 92.35 | **94.82** | **98.86** | **87.91** | **85.01** | 1751 |
| UNet++ | resnet50 | Lovasz | 93.74 | 92.65 | 98.77 | 87.25 | 84.68 | 3311 |
| UNet scSE | resnet50 | Dice | 93.63 | 92.34 | 98.73 | 86.87 | 84.25 | 1503 |
| UPerNet | mit_b4 | Dice | 92.97 | 92.75 | 98.71 | 86.67 | 82.91 | 2211 |
| SegFormer | mit_b4 | Lovasz | **93.88** | 91.75 | 98.69 | 86.57 | 83.26 | 1270 |
| DeepLabV3+ | resnet50 | Lovasz | 91.42 | 92.56 | 98.57 | 85.16 | 81.59 | **947** |

*VRAM measured on RTX 4060 Ti (16 GB), batch size 4. Tversky parameters used: α=0.6, β=0.4.*

------------------------------------------------------------------------

## **Citation**

``` bibtex
@Article{mitunet2026,
  author   = {Parashchuk, Dmitriy and Kaspshitskiy, Alexey and Karyakin, Yuriy},
  journal  = {Machine Vision and Applications},
  title    = {Enhancing floor plan recognition: a hybrid mix-transformer and U-Net approach for precise wall segmentation},
  year     = {2026},
  number   = {3},
  pages    = {53},
  volume   = {37},
  doi      = {10.1007/s00138-026-01815-y},
  url      = {https://doi.org/10.1007/s00138-026-01815-y}
}
```

------------------------------------------------------------------------

## **License**

This project is licensed under the MIT License see the [LICENSE](LICENSE) file.

> **Note:** The pre-trained models provided in this repository were trained on the CubiCasa5k dataset. Therefore, the model weights are subject to the Creative Commons Attribution-NonCommercial 4.0 International License (CC-BY-NC 4.0). Commercial use of the pre-trained weights is restricted.

------------------------------------------------------------------------

## **Acknowledgments**

Supported by the Ministry of Science and Higher Education of the Russian
Federation (State assignment **FEWZ-2024-0052**).
