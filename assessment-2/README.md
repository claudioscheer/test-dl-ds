# How to run

All source code files are inside `src` folder. The `train` and `test` datasets must be extract to the `dataset` folder.

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

All files created when running the scripts above can be accessed [here](https://github.com/claudioscheer/test-dl-position/releases/tag/assignment-2).
