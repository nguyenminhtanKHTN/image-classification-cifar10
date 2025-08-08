import tensorflow as tf
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from ydata_profiling import ProfileReport

# Download the CIFAR-10 dataset from TensorFlow Datasets
(X_train, y_train), (X_test, y_test) = tf.keras.datasets.cifar10.load_data()
# Class names for CIFAR-10 dataset
class_names = [
    'airplane', 'automobile', 'bird', 'cat', 'deer',
    'dog', 'frog', 'horse', 'ship', 'truck'
]
# Check the shape of the dataset
print(f"Training data shape: {X_train.shape}, Training labels shape: {y_train.shape}")
print(f"Test data shape: {X_test.shape}, Test labels shape: {y_test.shape}")

# Visualize some images from the dataset
plt.figure(figsize=(10, 10))
for i in range(10):
    ax = plt.subplot(5, 5, i + 1)
    plt.imshow(X_train[i])
    plt.title(class_names[y_train[i][0]])
    plt.axis("off")
plt.savefig('../results/cifar10_sample_images.png')
plt.show()

# Visualize the distribution of labels in the training set
label_counts = pd.Series(y_train.flatten()).value_counts().sort_index()
plt.figure(figsize=(10, 5))
label_counts.plot(kind='bar')
plt.title('Distribution of Labels in CIFAR-10 Training Set')
plt.xlabel('Class')
plt.ylabel('Count')
plt.xticks(ticks=np.arange(10), labels=class_names, rotation=45)
plt.savefig('../results/cifar10_label_distribution.png')
plt.show()

