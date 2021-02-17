import tensorflow as tf
import os
from dataset_loader import load_image
import numpy as np
import cv2

save_dir = os.path.join(os.getcwd(), "saved_models")
model_name = "keras_cifar10_trained_model.h5"
model_path = os.path.join(save_dir, model_name)

test_dataset = "../dataset/test"
test_dataset_rotated = "../dataset/test-rotated"
if not os.path.isdir(test_dataset_rotated):
    os.makedirs(test_dataset_rotated)

model = tf.keras.models.load_model(model_path)


def get_rotation_angle(c):
    if c == 0:
        return cv2.cv2.ROTATE_90_CLOCKWISE
    elif c == 1:
        return cv2.cv2.ROTATE_90_COUNTERCLOCKWISE
    elif c == 3:
        return cv2.cv2.ROTATE_180
    return None


def rotate_image(image_path, image_rotated_path, angle):
    image = cv2.imread(image_path)
    image_rotated = cv2.rotate(image, angle)
    cv2.imwrite(image_rotated_path, image_rotated)


for image_file in os.listdir(test_dataset):
    image_path = os.path.join(test_dataset, image_file)
    image_rotated_path = os.path.join(test_dataset_rotated, image_file)
    image = load_image(image_path)
    image = image.reshape(1, 32, 32, 3)
    result = np.argmax(model.predict(image), axis=-1)[0]
    rotation_angle = get_rotation_angle(result)
    if rotation_angle:
        rotate_image(image_path, image_rotated_path, rotation_angle)
