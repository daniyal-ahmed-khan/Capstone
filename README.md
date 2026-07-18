# Chest X-Ray Disease Classification
## OpenCV + Deep Learning Pipeline with Fairness Analysis

**CIS-627 Capstone | Daniyal Ahmed Khan | St. Thomas University**

## 🚀 Live Demo
👉 **[Try the app here](https://capstone-chestxray.streamlit.app)**

## Project Overview
A deep learning pipeline for chest X-ray disease classification using:
- **Model:** DenseNet-121 pretrained on ImageNet
- **Dataset:** NIH Chest X-Ray14 + Kaggle Pneumonia (74,503 images)
- **Classes:** No Finding, Infiltration, Pneumonia
- **Overall Accuracy:** 84%
- **Pneumonia F1:** 0.96

## Results
| Class | Precision | Recall | F1 |
|---|---|---|---|
| No Finding | 0.88 | 0.93 | 0.90 |
| Infiltration | 0.34 | 0.23 | 0.28 |
| Pneumonia | 0.99 | 0.92 | 0.96 |

## Fairness Analysis
- **Gender gap:** 2.07% (Female 84.11% vs Male 82.03%)
- **Age gap:** 6.77% (41-60 best at 84.19%, 0-20 worst at 77.42%)

## Weekly Progress
- **Week 1:** Dataset setup, Kaggle download
- **Week 2:** EDA
- **Week 3:** OpenCV preprocessing (CLAHE + Gaussian blur)
- **Week 4:** Augmentation, patient-level splits, class imbalance
- **Week 5:** DenseNet-121 training, CheXNet comparison, SMOTE
- **Week 6:** Fairness analysis by gender and age
- **Week 7:** Streamlit web app deployment

## Tech Stack
PyTorch, DenseNet-121, OpenCV, Albumentations, Streamlit
