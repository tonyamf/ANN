# Emotion Recognition with Artificial Neural Networks

This repository contains the code and resources for the research paper, "Geometric and Appearance approaches with Artificial Neural Network for Discrete Human Emotion Recognition from Static Face Images." The project explores and implements two primary approaches for recognizing human emotions from static facial images.

## Table of Contents
- [Project Overview](#project-overview)
- [Dataset](#dataset)
- [Methodology](#methodology)
  - [Geometric-Based Approach](#geometric-based-approach)
  - [Appearance-Based Approach](#appearance-based-approach)
- [Code Structure](#code-structure)
- [Setup and Installation](#setup-and-installation)
- [Usage](#usage)
- [Corrections and Improvements](#corrections-and-improvements)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)

## Project Overview

This project aims to recognize discrete human emotions from static face images using Artificial Neural Networks. It delves into two main feature extraction techniques:
- **Geometric-based:** Analyzing key facial features and their geometric properties.
- **Appearance-based:** Utilizing the entire facial image to learn features.

The project uses the Fer2013 dataset and implements both a Multi-Layer Perceptron (MLP) for geometric features and a Convolutional Neural Network (CNN) for appearance-based features.

## Dataset

The project uses the **Fer2013 dataset**, which was part of a Kaggle competition. The dataset consists of 48x48 pixel grayscale images of faces. The task is to categorize each face based on the emotion shown in the facial expression into one of seven categories:
- 0: Angry
- 1: Disgust
- 2: Fear
- 3: Happy
- 4: Sad
- 5: Surprise
- 6: Neutral

## Methodology

### Geometric-Based Approach

This approach involves:
1.  **Facial Landmark Detection:** Identifying key points on the face, such as the corners of the eyes and mouth.
2.  **Feature Extraction:** Calculating geometric properties from these landmarks:
    - **Distance Signature:** Distances between all pairs of keypoints.
    - **Shape Signature:** Signatures derived from triangles formed by triplets of keypoints.
    - **Texture Signature:** A custom Local Binary Pattern (LBP) around keypoints.
3.  **Statistical Analysis:** Computing statistical measures like skewness, kurtosis, moments, and entropy for these signatures.
4.  **Model Training:** Training a Multi-Layer Perceptron (MLP) on these extracted features.

### Appearance-Based Approach

This method leverages the power of deep learning:
1.  **Image Preprocessing:** The 48x48 pixel grayscale images are used directly as input.
2.  **Feature Extraction and Classification:** A Convolutional Neural Network (CNN) is used to automatically learn and extract relevant features from the images and classify them into the seven emotion categories.

## Code Structure

The implementation is provided in the `cwANN_.ipynb` Jupyter Notebook. The key sections of the notebook are:
- **Library Imports:** Necessary libraries such as `dlib`, `face_recognition`, `OpenCV`, `TensorFlow/Keras`, and `scikit-learn` are imported.
- **Data Loading:** The Fer2013 dataset is loaded and preprocessed.
- **Feature Extraction:** Implementation of both geometric and appearance-based feature extraction.
- **Model Implementation:**
  - **MLP:** With Bayesian optimization for hyperparameter tuning.
  - **CNN:** For end-to-end image classification.

## Setup and Installation

To run the code in this repository, you need to have Python installed, along with the necessary libraries. You can install the required libraries using pip:

```bash
pip install -r requirements.txt
```

**requirements.txt:**
```
dlib
face_recognition
opencv-python
tensorflow
keras
scikit-learn
scikit-optimize
pandas
numpy
matplotlib
seaborn
```

## Usage

You can run the Jupyter Notebook `cwANN_.ipynb` to see the implementation of the emotion recognition models. The notebook is self-contained and includes all the steps from data loading to model evaluation.

## Improvements to the code

The following improvement have been made to the original code for better structure and performance:

1.  **Added `if __name__ == "__main__"` block:** Encapsulated the main script logic in a `if __name__ == "__main__"` block to ensure it runs only when the script is executed directly.

2.  **Removed Redundant Data Loading:** The dataset is now loaded only once to avoid redundancy and improve code clarity.

3.  **Optimized Bayesian Search:** The `n_jobs` parameter in `BayesSearchCV` is set to `-1` to utilize all available CPU cores for more efficient hyperparameter tuning.

## Results

The paper and the code demonstrate that the appearance-based approach using a CNN generally outperforms the geometric-based approach with an MLP on the Fer2013 dataset. The combination of both approaches, however, can lead to even better performance, suggesting that both geometric and appearance features contain valuable information for emotion recognition.

## Contributing

Contributions are welcome! If you have any suggestions, improvements, or bug fixes, please feel free to open an issue or submit a pull request.
