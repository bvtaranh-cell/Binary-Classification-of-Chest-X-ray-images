# GROUP 8: TERENCE MPOFU, BLESSING TARANHIKE
# Binary Classification of Pneumonia vs Normal Chest X-Rays Using Deep Learning

## Overview
This project builds a deep learning pipeline to classify chest X-ray images into two classes: **NORMAL** and **PNEUMONIA**. The work combines images from the Kaggle Chest X-Ray dataset and the NIH Chest X-ray dataset, then applies preprocessing, quality checks, and model training using both a baseline CNN and transfer learning models.

## Objectives
- Build a clean binary chest X-ray dataset from Kaggle and NIH
- Compare multiple deep learning models for pneumonia classification
- Improve model interpretability using **Grad-CAM**
- Identify which image regions most influenced the model’s prediction

## Datasets
- **Kaggle Chest X-Ray Pneumonia Dataset:** https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia
- **NIH Chest X-ray Dataset:** https://www.kaggle.com/datasets/nih-chest-xrays/data
- **Cleaned Combined Dataset (Google Drive):** https://drive.google.com/file/d/13jbZEaVAWHZk246WSngMfhJ71wdLZ5mu/view?usp=drive_link

The Kaggle dataset was already organized for pneumonia classification and contains chest X-ray images grouped into two main classes:
- **NORMAL** = normal chest X-ray images
- **PNEUMONIA** = chest X-ray images showing pneumonia cases

The NIH dataset was filtered into a strict binary setup where:
- **NORMAL** = `No Finding`
- **PNEUMONIA** = `Pneumonia`

## Data Preparation
The dataset preparation pipeline included:
- extracting NIH binary classes from the metadata CSV
- combining Kaggle and NIH images into one dataset
- checking for corrupted images
- checking for duplicate files
- checking image sizes
- flagging suspicious low-quality images
- checking for cross-class duplicates in the final combined dataset

The final dataset was organized into:
- `combined_dataset/NORMAL`
- `combined_dataset/PNEUMONIA`

## Models
The following models were explored:
- Baseline CNN
- ResNet50
- DenseNet121
- EfficientNetB0

## Results
Among the models tested, **ResNet50 performed better than the other models** on this task. It gave the strongest overall performance for distinguishing pneumonia from normal chest X-rays.

To improve interpretability, **Grad-CAM** was used to visualize the image regions that contributed most to the model’s predictions. This helped show whether the model was focusing on clinically relevant lung areas rather than unrelated parts of the image.

## Presentation Video
- **YouTube Link:** https://youtu.be/fA6UpMl4Hmg

## Repository Contents
This repository includes:
- project README
- final code with comments
- presentation slides
- dataset links
- results and visualizations

## Conclusion
This project demonstrates how deep learning can be applied to binary chest X-ray classification for pneumonia detection. Among the tested models, ResNet50 achieved the best overall performance, while Grad-CAM improved explainability by highlighting the image regions influencing predictions.
