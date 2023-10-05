# TensorFlow - learning by doing
This repo contains my whole journey from a TF no-name to (hopefully) a certified TensorFlow Developer. I feel stoked about how much one 
can learn by themselves if there's enough motivation and curiosity involved.

You can find a full journey with the one and only [@mrdbourke](https://github.com/mrdbourke) containing countless hours put into his courses. A 
big shoutout to him for what he does for the AI/ML community sharing all this wisdom with great enthusiasm and limitless humour!

**Note:** All notebooks without `exercise` in the filename are coded during the course, they will have more than some similarities 
to the original ones.

Going in chronological (and logical) order you can find here:

[Notebook 00 - TensorFlow Fundamentals](https://github.com/pawelkiszczak/tensorflow/blob/main/colab_notebooks/00_tensorflow_fundamentals.ipynb)
 * Introduction to tensors (scalars, vectors, matrixes)
 * Creating tensors with `tf.Variable()` and `tf.constant()`
 * Manipulating tensors (calculations, shuffling, expanding etc.)
 * Using `NumPy` to create tensors
 * Changing datatypes of a tensor
 * Aggregating tensors
 * One-hot encoding
 * Accessing the GPU for more tensor-computing power

[Notebook 01 - Neural Network Regression with TensorFlow](https://github.com/pawelkiszczak/tensorflow/blob/main/colab_notebooks/01_neural_network_regression_with_tensorflow.ipynb)
* Architecture of a neural network regression model
* Importance of input & output shapes
* Create custom data to view and fit to the model
* Whole modelling process and its steps
* Different evaluation methods
* Tackle first realand large dataset (insurance prediction)

[Notebook 02 - Neural Network Classificaion with TensorFlow](https://github.com/pawelkiszczak/tensorflow/blob/main/colab_notebooks/02_neural_network_classification%20with_TensorFlow.ipynb)
* Different types of classification problems (binary, multiclass, multilabel)
* Activation functions and their importance in modelling (`relu`, `tanh`, `sigmoid` and others)
* Finding the best learning rate for given problem
* Classification evaluation metrics (confusion matrix)
* Working with `fashion_mnist` dataset for multiclass classification
* Normalizataion/scaling
* What patterns is the model learning?

[Notebook 03 - Introduction to computer vision with TensorFlow](https://github.com/pawelkiszczak/tensorflow/blob/main/colab_notebooks/03_introduction_to_computer_vision_with_tensorflow.ipynb)
* Building a simple end-to-end CNN model (image classification: pizza or steak)
* Batching data for the model
* New layers introduced: `Dropout`, `Activation`
* Overfitting vs underfitting
* Data augmentation techniques for images
* Making predictions on custom data to test our model
* Working with Food101 dataset: preprocessing, modelling, evaluating
* Hyperparameter tuning to reduce overfitting

[Notebook 04 - Transfer Learning in TensorFlow Part 1: Feature Extraction](https://github.com/pawelkiszczak/tensorflow/blob/main/colab_notebooks/04_transfer_learning_in_tensorflow_feature_extraction.ipynb)
* Working with Foof101 dataset (again)
* Using callbacks: `TensorBoard`, `ModelCheckpoint`
* Creating data loaders with `ImageDataGenerator`
* Using a pretrained model from [TensorFlow Hub](https://tfhub.dev) being EfficientNetB0 and ResNET50 V2
* Freezing all the layers of pretrained model with `KerasLayer(trainable=False)`
* Compering the results of mentioned models with `TensorBoard`

[Notebook 05 - Transfer Learning with TensorFlow Part 2: Fine-Tuning](https://github.com/pawelkiszczak/tensorflow/blob/main/colab_notebooks/05_transfer_learning_in_tensorflow_part_2_fine_tuning.ipynb)
* Reusing the earlier written `helper_functions` script (plot curves, unzip data, walk through directory etc.)
* Building `Model0` with TensorFlow's `FunctionalAPI` and EfficientNetB0
* Understanding the basic principles behind `GlobalAveragePooling2D` layer
* Running a series of experiments - from `Model1` to `Model4` (differentiaton between feature extraction/fine tuned EfficientNetB0 model and amount of training data)
* Learning about **data augmentation** and its unparalelled strenghts for training deep neural networks
* Fine-tuning an earlier used EfficientNetB0 model to compare the performance versus the base version of the model

[Notebook 06 - Transfer Learning with TensorFlow Part 3: Scaling up (Food Vision mini)](https://github.com/pawelkiszczak/tensorflow/blob/main/colab_notebooks/06_transfer_learning_in_tensorflow_part_3_scaling_up.ipynb)
* 