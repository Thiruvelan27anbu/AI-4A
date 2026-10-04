# EX NO. 4(a)

# CONVOLUTIONAL NEURAL NETWORK USING CIFAR-10 DATASET

## AIM

To build and train a **Convolutional Neural Network (CNN)** using TensorFlow for image classification using the **CIFAR-10 dataset**.

---

# OBJECTIVES

The objectives of this experiment are:

1. To understand the concept of Convolutional Neural Networks.
2. To load the CIFAR-10 image dataset.
3. To preprocess and normalize image data.
4. To convert class labels into one-hot encoded format.
5. To visualize sample images from the dataset.
6. To design a CNN architecture using convolution and pooling layers.
7. To train the CNN model using the training dataset.
8. To evaluate the model using accuracy and loss.
9. To predict the classes of unseen test images.

---

# SOFTWARE REQUIREMENTS

* Python
* Google Colab / Jupyter Notebook
* TensorFlow
* Keras
* NumPy
* Matplotlib

# HARDWARE REQUIREMENTS

* Computer or Laptop
* Minimum 4 GB RAM
* Internet connection for downloading the dataset

---

# INTRODUCTION

A **Convolutional Neural Network (CNN)** is a type of Deep Learning model mainly used for image processing and image classification tasks.

CNNs are particularly suitable for image-based applications because they can automatically learn important visual features directly from images.

These features may include:

* Edges
* Shapes
* Textures
* Patterns
* Object parts

Unlike traditional machine learning approaches, CNNs can automatically extract useful image features instead of requiring all features to be manually designed.

A CNN processes an image through multiple layers. The earlier layers generally learn simple features, while deeper layers learn more complex patterns.

The basic CNN process can be represented as:

**Input Image → Convolution → ReLU → Pooling → Convolution → Fully Connected Layer → Output Class**

CNNs are widely used in:

* Image classification
* Face recognition
* Object detection
* Medical image analysis
* Self-driving cars
* Security systems
* Handwriting recognition

The main advantage of CNN is its ability to automatically learn useful spatial features from images.

---

# CIFAR-10 DATASET

## ABOUT THE DATASET

The **CIFAR-10 dataset** is a commonly used dataset for image classification experiments.

It contains a total of **60,000 color images**.

## DATASET DISTRIBUTION

| Dataset      | Number of Images |
| ------------ | ---------------: |
| Training Set |           50,000 |
| Test Set     |           10,000 |
| Total        |           60,000 |

Each image has a size of:

**32 × 32 pixels**

Each image contains three color channels:

* Red
* Green
* Blue

Therefore, the input image shape is:

**(32, 32, 3)**

---

# CIFAR-10 CLASSES

The CIFAR-10 dataset contains ten different classes:

1. Airplane
2. Automobile
3. Bird
4. Cat
5. Deer
6. Dog
7. Frog
8. Horse
9. Ship
10. Truck

The objective of the CNN model is to classify an input image into one of these ten categories.

---

# IMPORTING REQUIRED LIBRARIES

TensorFlow and Keras are used for building the CNN model. NumPy is used for numerical operations, while Matplotlib is used for displaying images and graphs.

```python
import tensorflow as tf
from tensorflow.keras import datasets, layers, models
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt
import numpy as np
```

## DESCRIPTION OF LIBRARIES

### TensorFlow

TensorFlow is a machine learning and deep learning framework used to develop and train neural network models.

### Keras

Keras provides a high-level interface for creating and training neural networks.

### NumPy

NumPy is used for numerical calculations and array manipulation.

### Matplotlib

Matplotlib is used to visualize images, graphs, accuracy, and loss.

### datasets

The Keras datasets module provides functions for loading standard datasets such as CIFAR-10.

### layers

The layers module provides CNN layers such as:

* Conv2D
* MaxPooling2D
* Dropout
* Flatten
* Dense

### models

The models module is used to construct the neural network architecture.

### to_categorical()

The `to_categorical()` function converts class labels into one-hot encoded vectors.

---

# LOADING THE CIFAR-10 DATASET

The CIFAR-10 dataset can be loaded directly using TensorFlow/Keras.

```python
(X_train, y_train), (X_test, y_test) = datasets.cifar10.load_data()
```

Here:

* `X_train` contains training images.
* `y_train` contains training labels.
* `X_test` contains test images.
* `y_test` contains test labels.

The training dataset is used to teach the CNN model, while the test dataset is used to evaluate its performance on unseen images.

---

# DATA PREPROCESSING

Data preprocessing is an important step before training a neural network.

The original image pixels contain integer values between **0 and 255**.

For efficient neural network training, these values are normalized to a range between **0 and 1**.

---

# NORMALIZATION OF IMAGE DATA

The image data is normalized using the following code:

```python
X_train = X_train.astype('float32') / 255.0
X_test = X_test.astype('float32') / 255.0
```

The conversion to `float32` changes the image data into a suitable floating-point format.

Dividing by 255 converts the pixel values from:

**0–255 → 0–1**

Normalization helps improve:

* Training stability
* Convergence
* Numerical efficiency
* Model performance

---

# ONE-HOT ENCODING

The class labels are converted into one-hot encoded vectors.

```python
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)
```

Since there are ten classes, each label is represented using a vector containing ten elements.

For example, an Airplane label can be represented as:

```text
[1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
```

A Cat label can be represented as:

```text
[0, 0, 0, 1, 0, 0, 0, 0, 0, 0]
```

One-hot encoding allows the CNN model to perform multi-class classification.

---

# VISUALIZATION OF SAMPLE IMAGES

The different classes of CIFAR-10 images can be visualized using Matplotlib.

```python
class_names = [
    'Airplane', 'Automobile', 'Bird', 'Cat', 'Deer',
    'Dog', 'Frog', 'Horse', 'Ship', 'Truck'
]

plt.figure(figsize=(10, 10))

for i in range(16):
    plt.subplot(4, 4, i + 1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(X_train[i])
    plt.xlabel(class_names[np.argmax(y_train[i])])

plt.show()
```

This program displays a **4 × 4 grid containing 16 sample images**.

The class name corresponding to each image is displayed below the image.

---

# CNN ARCHITECTURE

The CNN model consists of multiple layers.

The major components are:

1. Convolutional layers
2. ReLU activation
3. Max pooling
4. Dropout
5. Flatten
6. Dense layer
7. Softmax output layer

The architecture can be represented as:

**Input Image**

↓

**Convolution Layer**

↓

**ReLU Activation**

↓

**Convolution Layer**

↓

**Max Pooling**

↓

**Dropout**

↓

**Convolution Layer**

↓

**Max Pooling**

↓

**Flatten**

↓

**Dense Layer**

↓

**Softmax Output**

---

# BUILDING THE CNN MODEL

A Sequential model is used to create the CNN.

```python
model = models.Sequential()
```

The Sequential model allows layers to be added one after another.

---

# FIRST CONVOLUTIONAL BLOCK

```python
model.add(layers.Conv2D(
    32, (3, 3),
    activation='relu',
    padding='same',
    input_shape=(32, 32, 3)
))

model.add(layers.Conv2D(
    32, (3, 3),
    activation='relu',
    padding='same'
))

model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Dropout(0.25))
```

The first convolutional layers learn basic visual features such as:

* Edges
* Lines
* Simple textures
* Basic patterns

The MaxPooling layer reduces the spatial dimensions of the feature maps.

The Dropout layer helps reduce overfitting during training.

---

# SECOND CONVOLUTIONAL BLOCK

```python
model.add(layers.Conv2D(
    64, (3, 3),
    activation='relu',
    padding='same'
))

model.add(layers.Conv2D(
    64, (3, 3),
    activation='relu',
    padding='same'
))

model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Dropout(0.25))
```

The second convolutional block contains 64 filters.

These layers can learn more detailed patterns from the images.

---

# THIRD CONVOLUTIONAL BLOCK

```python
model.add(layers.Conv2D(
    128, (3, 3),
    activation='relu',
    padding='same'
))

model.add(layers.Conv2D(
    128, (3, 3),
    activation='relu',
    padding='same'
))

model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Dropout(0.25))
```

The third convolutional block uses 128 filters.

The deeper layers learn increasingly complex features related to objects, shapes, and textures.

---

# FLATTEN AND DENSE LAYERS

After convolution and pooling operations, the extracted feature maps are converted into a one-dimensional vector using the Flatten layer.

```python
model.add(layers.Flatten())

model.add(layers.Dense(
    512,
    activation='relu'
))

model.add(layers.Dropout(0.5))

model.add(layers.Dense(
    10,
    activation='softmax'
))
```

## FUNCTIONS OF THE LAYERS

### Flatten Layer

The Flatten layer converts multidimensional feature maps into a one-dimensional vector.

### Dense Layer

The Dense layer learns relationships between the extracted features.

### Dropout Layer

Dropout randomly disables a portion of neurons during training. This helps reduce overfitting.

### Softmax Layer

The final Dense layer contains 10 neurons because the CIFAR-10 dataset has ten classes.

The Softmax activation produces probability values for the ten classes.

---

# MODEL COMPILATION

The CNN model is compiled using the Adam optimizer and categorical cross-entropy loss.

```python
model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()
```

## PARAMETERS USED

| Parameter                | Value                     |
| ------------------------ | ------------------------- |
| Optimizer                | Adam                      |
| Loss Function            | Categorical Cross-Entropy |
| Performance Metric       | Accuracy                  |
| Number of Output Classes | 10                        |

---

# MODEL TRAINING

The CNN model is trained using the training dataset.

```python
history = model.fit(
    X_train,
    y_train,
    epochs=30,
    batch_size=64,
    validation_split=0.2
)
```

The model is trained for:

**30 epochs**

with:

**Batch Size = 64**

and:

**Validation Split = 20%**

During training, the model learns features from the training images and adjusts its parameters to reduce classification error.

---

# TRAINING PARAMETERS

### Epoch

An epoch represents one complete pass through the training dataset.

In this experiment:

**Epochs = 30**

### Batch Size

Batch size represents the number of training samples processed before the model updates its parameters.

In this experiment:

**Batch Size = 64**

### Validation Split

A portion of the training dataset is used for validation.

In this experiment:

**Validation Split = 0.2**

---

# MODEL PERFORMANCE ANALYSIS

After training, the model's performance can be analyzed using:

* Training accuracy
* Validation accuracy
* Training loss
* Validation loss

The training history contains these values for every epoch.

---

# ACCURACY GRAPH

```python
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)

plt.plot(
    history.history['accuracy'],
    label='Train Accuracy'
)

plt.plot(
    history.history['val_accuracy'],
    label='Validation Accuracy'
)

plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
```

The accuracy graph shows how the classification accuracy changes during training.

Generally, training accuracy increases as the model learns useful features.

Validation accuracy indicates how well the model performs on validation data.

---

# LOSS GRAPH

```python
plt.subplot(1, 2, 2)

plt.plot(
    history.history['loss'],
    label='Train Loss'
)

plt.plot(
    history.history['val_loss'],
    label='Validation Loss'
)

plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.show()
```

The loss graph shows how the error changes during training.

A decreasing training loss generally indicates that the model is learning from the training data.

---

# OVERFITTING

Overfitting occurs when a model learns the training data too closely and does not generalize well to unseen data.

For example, if training accuracy becomes very high while validation accuracy remains significantly lower, the model may be overfitting.

Dropout layers are included in this CNN architecture to help reduce overfitting.

---

# PREDICTING TEST IMAGES

After training, the CNN model can be used to predict the class of unseen test images.

```python
def plot_predictions(index):

    img = X_test[index]

    true_label = class_names[
        np.argmax(y_test[index])
    ]

    pred_probs = model.predict(
        np.expand_dims(img, axis=0),
        verbose=0
    )

    pred_label = class_names[
        np.argmax(pred_probs)
    ]

    plt.imshow(img)

    plt.title(
        f"True: {true_label} | Pred: {pred_label}"
    )

    plt.axis('off')
    plt.show()
```

The function displays:

* Test image
* Actual class
* Predicted class

---

# PREDICTING MULTIPLE TEST IMAGES

The first five test images can be predicted using:

```python
for i in range(5):
    plot_predictions(i)
```

The output displays the actual and predicted class for each test image.

---

# COMPLETE PROGRAM

```python
import tensorflow as tf
from tensorflow.keras import datasets, layers, models
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt
import numpy as np

# Load CIFAR-10 dataset
(X_train, y_train), (X_test, y_test) = datasets.cifar10.load_data()

# Normalize image data
X_train = X_train.astype('float32') / 255.0
X_test = X_test.astype('float32') / 255.0

# One-hot encode labels
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

# Class names
class_names = [
    'Airplane', 'Automobile', 'Bird', 'Cat', 'Deer',
    'Dog', 'Frog', 'Horse', 'Ship', 'Truck'
]

# Visualize sample images
plt.figure(figsize=(10, 10))

for i in range(16):
    plt.subplot(4, 4, i + 1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(X_train[i])
    plt.xlabel(class_names[np.argmax(y_train[i])])

plt.show()

# Create CNN model
model = models.Sequential()

# First convolutional block
model.add(layers.Conv2D(
    32, (3, 3),
    activation='relu',
    padding='same',
    input_shape=(32, 32, 3)
))

model.add(layers.Conv2D(
    32, (3, 3),
    activation='relu',
    padding='same'
))

model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Dropout(0.25))

# Second convolutional block
model.add(layers.Conv2D(
    64, (3, 3),
    activation='relu',
    padding='same'
))

model.add(layers.Conv2D(
    64, (3, 3),
    activation='relu',
    padding='same'
))

model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Dropout(0.25))

# Third convolutional block
model.add(layers.Conv2D(
    128, (3, 3),
    activation='relu',
    padding='same'
))

model.add(layers.Conv2D(
    128, (3, 3),
    activation='relu',
    padding='same'
))

model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Dropout(0.25))

# Fully connected layers
model.add(layers.Flatten())

model.add(layers.Dense(
    512,
    activation='relu'
))

model.add(layers.Dropout(0.5))

model.add(layers.Dense(
    10,
    activation='softmax'
))

# Compile model
model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Display model architecture
model.summary()

# Train model
history = model.fit(
    X_train,
    y_train,
    epochs=30,
    batch_size=64,
    validation_split=0.2
)

# Plot accuracy and loss
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)

plt.plot(
    history.history['accuracy'],
    label='Train Accuracy'
)

plt.plot(
    history.history['val_accuracy'],
    label='Validation Accuracy'
)

plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.subplot(1, 2, 2)

plt.plot(
    history.history['loss'],
    label='Train Loss'
)

plt.plot(
    history.history['val_loss'],
    label='Validation Loss'
)

plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.show()

# Prediction function
def plot_predictions(index):

    img = X_test[index]

    true_label = class_names[
        np.argmax(y_test[index])
    ]

    pred_probs = model.predict(
        np.expand_dims(img, axis=0),
        verbose=0
    )

    pred_label = class_names[
        np.argmax(pred_probs)
    ]

    plt.imshow(img)

    plt.title(
        f"True: {true_label} | Pred: {pred_label}"
    )

    plt.axis('off')
    plt.show()

# Predict first five test images
for i in range(5):
    plot_predictions(i)
```

---

# OUTPUT

The CIFAR-10 dataset was successfully loaded using TensorFlow and Keras.

The image data was normalized from the range **0–255 to 0–1**, and the class labels were converted into one-hot encoded vectors.

Sixteen sample images were successfully visualized in a 4 × 4 grid.

A CNN model containing convolutional, pooling, dropout, flatten, and dense layers was successfully created.

The model was trained for **30 epochs** with a batch size of **64**.

Training and validation accuracy and loss graphs were generated.

The trained CNN model was also used to predict the classes of unseen test images.

---

# OBSERVATION

| Parameter         | Observation               |
| ----------------- | ------------------------- |
| Dataset           | CIFAR-10                  |
| Total Images      | 60,000                    |
| Training Images   | 50,000                    |
| Test Images       | 10,000                    |
| Image Size        | 32 × 32                   |
| Color Channels    | 3                         |
| Number of Classes | 10                        |
| Epochs            | 30                        |
| Batch Size        | 64                        |
| Optimizer         | Adam                      |
| Loss Function     | Categorical Cross-Entropy |
| Output Activation | Softmax                   |

---

# RESULT

The CIFAR-10 dataset was successfully loaded and preprocessed.

A Convolutional Neural Network containing convolutional, pooling, dropout, flatten, and dense layers was successfully constructed using TensorFlow and Keras.

The model was trained using the training dataset, and its performance was analyzed using training and validation accuracy and loss graphs.

The trained model was also successfully used to predict the classes of unseen test images.

