"""
I was used to PyTorch. It took me more time to understand and install TensorFlow.
"""

import numpy as np
from custom_data_generator import get_train_dataset
import tensorflow as tf
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D
import os

batch_size = 32
num_classes = 5
epochs = 100
data_augmentation = False
num_predictions = 20
save_dir = os.path.join(os.getcwd(), "saved_models")
model_name = "keras_cifar10_trained_model.h5"

train_dataset = "../dataset/train/train"
test_dataset = "../dataset/test"

train_ids = np.genfromtxt(
    os.path.join(train_dataset, "..", "train.truth.csv"),
    delimiter=",",
    skip_header=True,
    dtype=str,
)
train_ids = {x[0]: x[1] for x in train_ids}

x_train, y_train = get_train_dataset(train_dataset, train_ids)

# The data, split between train and test sets:
# (x_train, y_train), (x_test, y_test) = cifar10.load_data()
# print("x_train shape:", x_train.shape)
# print(y_train[0], "train samples")
# print(x_test.shape[0], "test samples")

# Convert class vectors to binary class matrices.
y_train = tf.keras.utils.to_categorical(y_train, num_classes)
# y_test = tf.keras.utils.to_categorical(y_test, num_classes)

model = Sequential()
model.add(Conv2D(32, (3, 3), padding="same", input_shape=x_train.shape[1:]))
model.add(Activation("relu"))
model.add(Conv2D(32, (3, 3)))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(64, (3, 3), padding="same"))
model.add(Activation("relu"))
model.add(Conv2D(64, (3, 3)))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(512))
model.add(Activation("relu"))
model.add(Dropout(0.5))
model.add(Dense(num_classes))
model.add(Activation("softmax"))

# initiate RMSprop optimizer
opt = tf.keras.optimizers.RMSprop(learning_rate=0.0001, decay=1e-6)

# Let's train the model using RMSprop
model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])

x_train = x_train.astype("float32")
# x_test = x_test.astype("float32")
x_train /= 255
# x_test /= 255

# Save model and weights
if not os.path.isdir(save_dir):
    os.makedirs(save_dir)
model_path = os.path.join(save_dir, model_name)
model.save(model_path)
print("Saved trained model at %s " % model_path)

# Score trained model.
scores = model.evaluate(x_train, y_train, verbose=1)
print("Test loss:", scores[0])
print("Test accuracy:", scores[1])
