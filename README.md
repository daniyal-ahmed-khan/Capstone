# Chest X-Ray Disease Classification
## OpenCV + Deep Learning Pipeline with Fairness Analysis

### Project Overview
An end-to-end computer vision pipeline for early detection of Pneumonia and Infiltration from chest X-rays using the NIH Chest X-Ray14 dataset.

### Classes
| Label | Count |
|---|---|
| No Finding | 60,361 |
| Infiltration | 19,894 |
| Pneumonia | 1,431 |

### Tech Stack
- OpenCV, PyTorch, ResNet-50, EfficientNet-B3, Grad-CAM, Streamlit

### Repository Structure
- `notebooks/` — EDA, training, and fairness analysis notebooks
- `src/` — Python scripts for preprocessing and modeling
- `models/` — Saved model weights
- `results/` — Metrics, plots, Grad-CAM visualizations
- `dashboard/` — Streamlit app
