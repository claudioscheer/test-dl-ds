# How to run

The train and test datasets must be extract to the `dataset` folder. All source code files are inside `src` folder; make sure you are inside this folder when running the scripts.

## `cifar10_cnn.py`

This script train and save the trained model to `saved_models/keras_cifar10_trained_model.h5`.

```bash
python cifar10_cnn.py
```

## `test.py`

This script loads the trained model, executes predictions on the test dataset and creates the `test.preds.csv` file with the images and the predicted position for each image.

```bash
python test.py
```

## `rotate_images.py`

This script loads the trained model and rotate the images according to the position predicted. The rotated images are save to `dataset/test-rotated`.

```bash
python rotate_images.py
```

## `create_numpy_array.py`

This script creates a NumPy array with all images inside `dataset/test-rotated`. The array is saved to `src/all-images.npy`.

```bash
python create_numpy_array.py
```

# Results

The model trained, the `test.preds.csv` file, the zip file with roteated images and the NumPy array can be accessed [here](https://github.com/claudioscheer/test-dl-position/releases/tag/assignment-2).

# Approach

First, I had to understand a little bit about how TensorFlow works. I was used to PyTorch. Then, I focused on understanding the dataset and how it was structured. I used OpenCV to load the images and NumPy to load the ground truth CSV. I splitted the train dataset into train and validation. The validation dataset gives a better idea of the model performance on the test dataset. I used 10% of the train dataset to validate the model.

Since the model was already defined, I just adjusted the script to my needs. To have a faster training process, I increased the batch size and checked if the model was running on GPU. PyTorch requires calling `model.cuda()`. It looks like you do not need that on TensorFlow 2.

Finally, I focused on loading the model and transforming the images, according to the requirements, using OpenCV library.

# Next steps

- Data augmentation (black and white images, flipped images, etc);
- Instead of 32x32, use images of 64x64 as input;
- Add more convolutional layers to the topology, taking care of overfitting and underfitting problems;
- Test other optimizers, such as Adam;
- Increase the learning rate;
- Explore parallelism to process the images;
