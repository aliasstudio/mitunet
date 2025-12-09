# Floor Plan CIS Dataset

## Overview
This dataset contains **500 original, high-resolution floor plan images** collected from real estate listings within the Russian Federation and CIS region. These images serve as the source domain data for the research project **MitUNet** and the associated manuscript **arXiv:2512.02413** ([https://arxiv.org/abs/2512.02413](https://arxiv.org/abs/2512.02413)).

**Note:** This dataset contains **raw images** without resizing or pre-applied augmentations. This allows researchers to apply their own preprocessing pipelines suitable for their specific model architectures.

## Content & Challenges
The dataset features distinct regional architectural styles that challenge standard segmentation models:
* **Texture-based Material Encoding:** Differentiation between load-bearing walls (solid fills) and partition walls (hatching/textures).
* **Non-Manhattan Geometry:** Presence of curved walls and angled structures.
* **Semantic Clutter:** Heavy presence of dimension lines, text, and furniture outlines overlapping with structural elements.

## Dataset Structure & Specifications
* **Image Format:** JPG/PNG (Original resolution, variable dimensions).
* **Preprocessing:** None (Images are provided "as is").

## Reproduction of MitUNet Results
To reproduce the results reported in the MitUNet paper, the following pipeline should be applied programmatically during training:
1.  **Resize:** Scale images to fixed **512x512 pixels**.
2.  **Augmentations:** Apply geometric (rotation, perspective) and photometric (brightness, CLAHE) transformations. We recommend using the `Albumentations` library.
3.  **Refinement:** Ground truth masks in this dataset have already been cleaned (door/window openings subtracted) for better topological consistency.

## Citation
If you utilize this dataset, please cite the associated manuscript **arXiv:2512.02413**.

Please refer to the official arXiv page for the most up-to-date title and citation details (BibTeX):
[https://arxiv.org/abs/2512.02413](https://arxiv.org/abs/2512.02413)