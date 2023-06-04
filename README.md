# Image_Classification_using_notebook
# CIFAR-10 Image Classification

This project implements an image classification model using a Convolutional Neural Network (CNN) on the CIFAR-10 dataset. The model is trained to classify images into 10 different classes.

## Dataset

The CIFAR-10 dataset consists of 60,000 32x32 color images in 10 classes, with 6,000 images per class. The classes include airplanes, cars, birds, cats, deer, dogs, frogs, horses, ships, and trucks.

The CIFAR-10 dataset can be downloaded from the [official CIFAR-10 website](http://www.cs.toronto.edu/~kriz/cifar.html). Please refer to the website for the dataset download instructions and terms of use.

## Model Architecture

The implemented model architecture is as follows:

- Convolutional layer (32 filters, kernel size 3x3, ReLU activation)
- Max pooling layer (pool size 2x2)
- Convolutional layer (64 filters, kernel size 3x3, ReLU activation)
- Max pooling layer (pool size 2x2)
- Convolutional layer (128 filters, kernel size 3x3, ReLU activation)
- Max pooling layer (pool size 2x2)
- Flatten layer
- Dense layer (128 units, ReLU activation)
- Dropout layer (dropout rate 0.5)
- Dense layer (10 units, softmax activation)

## Training

The model is trained using the Adam optimizer and the categorical cross-entropy loss function. The training is performed for 20 epochs with a batch size of 128. The training and validation loss and accuracy are monitored during the training process.

## Model Evaluation

After training, the model is evaluated on the test set to assess its performance. The following metrics are calculated:

- Test loss
- Test accuracy
- Precision
- Recall
- F1-score

## Usage

To use the trained model for image classification:

1. Install the required dependencies mentioned in the `requirements.txt` file.
2. Load the saved model using `load_model` from `keras.models`.
3. Preprocess the input image using the `preprocess_image` function.
4. Make predictions on the preprocessed image using the loaded model.
5. Get the predicted class label using `np.argmax(predictions[0])`.

Refer to the code and documentation for detailed implementation steps and examples.

## Repository Structure

- `cifar-10-batches-py/`: Directory containing the CIFAR-10 dataset
- `model.h5`: Saved model file
- `image_classification.ipynb`: Jupyter Notebook with the code implementation
- `README.md`: Project README file

## Acknowledgments

The CIFAR-10 dataset is collected by Alex Krizhevsky, Vinod Nair, and Geoffrey Hinton. More information about the dataset can be found at [CIFAR-10](http://www.cs.toronto.edu/~kriz/cifar.html).

## License

This project is licensed under the [MIT License](LICENSE).
