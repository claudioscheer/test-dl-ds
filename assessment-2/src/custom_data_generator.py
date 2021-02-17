import tensorflow as tf
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


def get_train_dataset(image_folder, train_ids):
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
    return np.array(img_data_array), np.array(class_name)


# class DataGenerator(tf.keras.utils.Sequence):
#     def __init__(
#         self, ids, labels, batch_size=32, dim=(32, 32), n_channels=3, n_classes=4
#     ):
#         self.dim = dim
#         self.batch_size = batch_size
#         self.labels = labels
#         self.list_IDs = list_IDs
#         self.n_channels = n_channels
#         self.n_classes = n_classes
#         self.on_epoch_end()

#     def __len__(self):
#         return int(np.floor(len(self.list_IDs) / self.batch_size))

#     def __getitem__(self, index):
#         # Generate indexes of the batch
#         indexes = self.indexes[index * self.batch_size : (index + 1) * self.batch_size]

#         # Find list of IDs
#         list_IDs_temp = [self.list_IDs[k] for k in indexes]

#         # Generate data
#         X, y = self.__data_generation(list_IDs_temp)

#         return X, y

#     def on_epoch_end(self):
#         self.indexes = np.arange(len(self.list_IDs))

#     def __data_generation(self, list_IDs_temp):
#         X = np.empty((self.batch_size, *self.dim, self.n_channels))
#         y = np.empty((self.batch_size), dtype=int)

#         # Generate data
#         for i, ID in enumerate(list_IDs_temp):
#             # Store sample
#             X[
#                 i,
#             ] = np.load("data/" + ID + ".npy")

#             # Store class
#             y[i] = self.labels[ID]

#         return X, tf.keras.utils.to_categorical(y, num_classes=self.n_classes)
