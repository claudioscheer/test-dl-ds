import numpy as np
import os
import cv2

target_dict = {
    "rotated_left": 0,
    "rotated_right": 1,
    "upside": 2,
    "upside_down": 3,
    "upright": 4,
}


def load_dataset(image_folder, train_ids, validation_split):
    "I should use data generators."
    img_data_array = []
    class_name = []

    for image_file in os.listdir(image_folder):
        image_path = os.path.join(image_folder, image_file)
        image = cv2.imread(image_path, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (32, 32), interpolation=cv2.INTER_AREA)
        image = np.array(image)
        image = image.astype("float32")
        image /= 255
        img_data_array.append(image)
        class_name.append(target_dict[train_ids[image_file]])

    validation_index = int(len(class_name) * validation_split)
    return (
        np.array(img_data_array[0:validation_index]),
        np.array(class_name[0:validation_index]),
    ), (
        np.array(img_data_array[validation_index:]),
        np.array(class_name[validation_index:]),
    )