import tensorflow as tf
import os
from dataset_loader import load_image, transform_class
import csv
import numpy as np

save_dir = os.path.join(os.getcwd(), "saved_models")
model_name = "keras_cifar10_trained_model.h5"
model_path = os.path.join(save_dir, model_name)

test_dataset = "../dataset/test"

model = tf.keras.models.load_model(model_path)

with open("test.preds.csv", "w") as file:
    csv_writer = csv.writer(
        file, delimiter=",", quotechar='"', quoting=csv.QUOTE_MINIMAL
    )
    csv_writer.writerow(["fn", "label"])

    for image_file in os.listdir(test_dataset):
        image_path = os.path.join(test_dataset, image_file)
        image = load_image(image_path)
        image = image.reshape(1, 32, 32, 3)
        result = np.argmax(model.predict(image), axis=-1)

        csv_writer.writerow([image_file, transform_class(result[0])])
