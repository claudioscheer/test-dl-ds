import os
import numpy as np
import cv2

save_dir = os.path.join(os.getcwd(), "saved_models")
model_name = "keras_cifar10_trained_model.h5"
model_path = os.path.join(save_dir, model_name)

test_dataset = "../dataset/test-rotated"

all_images = []

for image_file in os.listdir(test_dataset):
    image_path = os.path.join(test_dataset, image_file)
    image = cv2.imread(image_path)
    all_images.append(image)

print(np.array(all_images).shape)
np.save("all-images.npy", np.array(all_images))
