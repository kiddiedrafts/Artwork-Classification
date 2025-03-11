# Artworks Classification

This project focuses on classifying artworks by famous artists using deep learning. The dataset contains images of artworks from 51 renowned artists, and the goal is to build a model that can accurately classify these artworks based on the artist.

## Dataset

The dataset used is **Best Artworks of All Time** from Kaggle. It includes images of artworks by 51 famous artists, including **Vincent van Gogh, Pablo Picasso, and Claude Monet**. The images are labeled by artist names, making it suitable for supervised learning tasks. The dataset is available under the **CC-BY-NC-SA 4.0** license.

- **Dataset Source:** [Best Artworks of All Time](https://www.kaggle.com/datasets/ikarus777/best-artworks-of-all-time)

## Data Preprocessing

- **Image Preprocessing:** Images were resized to **224x224 pixels** and normalized using the **ResNet50 built preprocessing function**, and converted into tensors suitable for deep learning models.  
- **Label Encoding:** Artist names (categorical labels) were encoded into numeric values using `LabelEncoder`.  

## Model Architecture

Two models were implemented:

1. **ResNet50 from Scratch:** A ResNet50 model trained without pre-trained weights.  
2. **ResNet50 with Pre-trained Weights:** A ResNet50 model initialized with **ImageNet weights**, with additional dense layers for fine-tuning. The base model was frozen during training.  

## Training

- Both models were trained using the **Adam optimizer** and **categorical cross-entropy loss**.  
- **Early stopping** and **model checkpointing** were used to prevent overfitting and save the best model.  
- Training was performed for **15 epochs**.  

## Model Performance

The best-performing model was the **ResNet50 with pre-trained weights**. Its performance on the dataset is as follows:

- **Training Accuracy:** 0.96  
- **Validation Accuracy:** 0.94 

Training and validation curves for **loss** and **accuracy** were plotted to monitor the model's performance.

## Requirements

*   pandas
*   numpy
*   splitfolders
*   matplotlib
*   plotly
*   keras
*   scikit-learn
