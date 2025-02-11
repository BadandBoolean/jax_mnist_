# CNN
import jax.numpy as jnp
from jax import grad, jit, vmap, pmap
from jax import random
import tensorflow as tf
import tensorflow_datasets as tfds
from pylab import rcParams
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedShuffleSplit
import cv2
import albumentations as albu
from skimage.transform import resize
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pylab import rcParams
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from keras.callbacks import Callback, EarlyStopping, ReduceLROnPlateau
import tensorflow as tf
import keras
from keras.models import Sequential, load_model
from keras.layers import Dropout, Dense, GlobalAveragePooling2D
from keras.optimizers import Adam
import efficientnet.keras as efn

tf.config.set_visible_devices([], device_type="GPU")
import time
from jax.scipy.special import logsumexp


data_dir = "/tmp/tfds"

cifar_data, info = tfds.load(
    name="cifar100", batch_size=-1, data_dir=data_dir, with_info=True
)
cifar_data = tfds.as_numpy(cifar_data)
train_data, test_data = cifar_data["train"], cifar_data["test"]
num_labels = info.features["label"].num_classes
h, w, c = info.features["image"].shape
# label names
names = info.features["label"].names
coarse_names = info.features["coarse_label"].names

# Full train set
train_images, train_labels, train_coarse_labels = (
    train_data["image"],
    train_data["label"],
    train_data["coarse_label"],
)
train_images = jnp.reshape(train_images, (len(train_images), h, w, c))
# If batch_size = 32, then:
# 	•	The array has 32 rows, one for each image.
#   •   Each row is a 1D vector of length 784 (one value per
#   pixel).

# Full test set
test_images, test_labels = test_data["image"], test_data["label"]
# test_images = jnp.reshape(test_images, (len(test_images), num_pixels))


if __name__ == "__main__":
    rcParams["figure.figsize"] = 2, 2
    imageId = np.random.randint(0, len(train_images))
    plt.imshow(train_images[imageId])
    plt.axis("off")
    print("Image id selected:", imageId)
    print("Shape of image:", train_images[imageId].shape)
    print("coarse label:", train_coarse_labels[imageId])
    print("coarse label name:", coarse_names[train_coarse_labels[imageId]])

    print("fine label:", train_labels[imageId])
    print("fine label name:", names[train_labels[imageId]])
    plt.show()
