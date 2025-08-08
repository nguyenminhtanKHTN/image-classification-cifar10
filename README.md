# Image Classification with CIFAR-10

   This project classifies images from the CIFAR-10 dataset using Convolutional Neural Networks (CNN). It is part of my ML learning journey to master deep learning and computer vision.

   ## Learning Goals
   - Understand Convolutional Neural Networks (CNN) and deep learning.
   - Preprocess image data (normalization, data augmentation).
   - Apply transfer learning with pre-trained models (e.g., VGG16).
   - Deploy models via Flask API.
   - Manage ML projects with GitHub.

   ## Directory Structure
   ```
   image-classification-cifar10/
   ├── data/
   │   ├── raw/
   │   ├── processed/
   │   └── external/
   ├── notebooks/
   │   ├── data_exploration.ipynb
   │   ├── preprocessing.ipynb
   │   └── model_training.ipynb
   ├── src/
   │   ├── __init__.py
   │   ├── preprocess.py
   │   ├── train_model.py
   │   ├── predict.py
   │   └── app.py
   ├── models/
   │   ├── cnn_model.h5
   │   └── vgg16_transfer_model.h5
   ├── results/
   ├── requirements.txt
   ├── README.md
   ├── LICENSE
   └── .gitignore
   ```

   ## Progress
   - [x] Project setup and initial structure
   - [ ] Data exploration
   - [ ] Data preprocessing
   - [ ] Model training
   - [ ] API deployment

   ## Setup
   1. Install dependencies: `pip install -r requirements.txt`
   2. Run notebooks: `jupyter notebook`